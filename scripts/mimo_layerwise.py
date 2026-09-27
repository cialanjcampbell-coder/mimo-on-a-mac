#!/usr/bin/env python3
"""Layer-by-layer forward pass of the MLX MiMo-V2.6-Flash conversion, one layer resident at a time.

  mimo_layerwise.py <model_dir> <text_file> [--max-tokens 2048] [--routes out.npz]

Teacher-forced (prefill-style) pass over a text: reports perplexity and next-token top-1 accuracy,
which validates the conversion (a wrong nibble order or qkv slicing gives perplexity in the hundreds).
With --routes, saves every token's top-8 expert ids per MoE layer. Teacher forcing gives exactly the
routes that decoding the same text would take, so this is also the routing-trace tool.
Uses the vendored mimo_v2 modules (engine/mimo_stream/mimo_v2.py, from mlx-lm PR #1219).
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for engine.*
from engine.mimo_stream import mimo_v2
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tokenizer_utils import load as load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("text")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--routes")
    ap.add_argument("--mtp", action="store_true", help="teacher-forced per-depth accuracy of the 3 MTP heads")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    args = mimo_v2.ModelArgs.from_dict(cfg)
    from pathlib import Path
    tok = load_tokenizer(Path(a.model))
    ids = tok.encode(open(a.text).read())[: a.max_tokens]
    T = len(ids)
    dense = mx.load(os.path.join(a.model, "dense.safetensors"))

    def sub(prefix):
        return [(k[len(prefix):], v) for k, v in dense.items() if k.startswith(prefix)]

    t0 = time.time()
    emb = nn.Embedding(args.vocab_size, args.hidden_size)
    emb.load_weights(sub("model.embed_tokens."))
    h = emb(mx.array(ids)[None]).astype(mx.bfloat16)
    mx.eval(h)
    full_mask = create_attention_mask(h, None)
    swa_mask = create_attention_mask(h, None, window_size=args.sliding_window_size)

    routes = {}
    for L in range(args.num_hidden_layers):
        is_moe, is_swa = bool(args.moe_layer_freq[L]), bool(args.hybrid_layer_pattern[L])
        layer = mimo_v2.DecoderLayer(args, is_moe=is_moe, is_sliding_window=is_swa)
        w = sub(f"model.layers.{L}.")
        if is_moe:
            nn.quantize(layer, group_size=32, bits=4, mode="mxfp4",
                        class_predicate=lambda p, m: "switch_mlp" in p and hasattr(m, "to_quantized"))
            ex = mx.load(os.path.join(a.model, f"experts-L{L:02d}.safetensors"))
            w += [(k[len(f"model.layers.{L}."):], v) for k, v in ex.items()]
        layer.load_weights(w, strict=True)
        mask = swa_mask if is_swa else full_mask
        x = h + layer.self_attn(layer.input_layernorm(h), mask, None)
        r = layer.post_attention_layernorm(x)
        if is_moe:
            inds, scores = layer.mlp.gate(r)
            y = layer.mlp.switch_mlp(r, inds)
            y = (y * scores[..., None]).sum(axis=-2).astype(r.dtype)
            routes[f"L{L:02d}"] = inds
        else:
            y = layer.mlp(r)
        h = x + y
        mx.eval(h, *( [routes[f"L{L:02d}"]] if is_moe else [] ))
        if is_moe:
            routes[f"L{L:02d}"] = np.array(routes[f"L{L:02d}"][0]).astype(np.uint8)
        del layer, w
        if is_moe:
            del ex
        mx.clear_cache()
        print(f"  layer {L:2d} {'moe' if is_moe else 'dense'} {'swa' if is_swa else 'full'}  {time.time()-t0:6.1f}s  "
              f"peak {mx.get_peak_memory()/1e9:5.1f} GB", flush=True)

    norm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
    norm.load_weights(sub("model.norm."))
    head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    head.load_weights(sub("lm_head."))
    logits = head(norm(h)).astype(mx.float32)[0]
    tgt = mx.array(ids[1:])
    lp = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
    nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).squeeze(-1)
    acc = (mx.argmax(logits[:-1], axis=-1) == tgt).astype(mx.float32)
    mx.eval(nll, acc)
    print(f"tokens {T}  ppl {float(mx.exp(nll.mean())):.3f}  top1 {float(acc.mean())*100:.1f}%  "
          f"(2nd half ppl {float(mx.exp(nll[T//2:].mean())):.3f})  time {time.time()-t0:.1f}s")
    print("next-token prediction after text:", repr(tok.decode([int(mx.argmax(logits[-1]))])))
    if a.mtp:
        mtp_eval(a.model, args, emb, head, norm, h, ids)
    if a.routes:
        np.savez_compressed(a.routes, ids=np.array(ids, dtype=np.int32), **routes)
        print("routes saved to", a.routes)


class MTPLayer(nn.Module):
    """MiMo-V2 MTP head (vLLM mimo_v2_mtp.py / oMLX): eh_proj([enorm(emb), hnorm(hidden)]) -> SWA block -> final norm."""

    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        self.enorm = nn.RMSNorm(H, eps=args.layernorm_epsilon)
        self.hnorm = nn.RMSNorm(H, eps=args.layernorm_epsilon)
        self.eh_proj = nn.Linear(2 * H, H, bias=False)
        self.input_layernorm = nn.RMSNorm(H, eps=args.layernorm_epsilon)
        self.self_attn = mimo_v2.Attention(args, is_sliding_window=True)
        self.pre_mlp_layernorm = nn.RMSNorm(H, eps=args.layernorm_epsilon)
        self.mlp = mimo_v2.MLP(args)
        self.final_layernorm = nn.RMSNorm(H, eps=args.layernorm_epsilon)
        self.window = args.sliding_window_size

    def __call__(self, hidden, tok_emb):
        x = self.eh_proj(mx.concatenate([self.enorm(tok_emb), self.hnorm(hidden)], axis=-1))
        mask = create_attention_mask(x, None, window_size=self.window)
        h = x + self.self_attn(self.input_layernorm(x), mask, None)
        h = h + self.mlp(self.pre_mlp_layernorm(h))
        self.last_prenorm = h
        return self.final_layernorm(h)


def mtp_eval(model_dir, args, emb, head, norm, h, ids):
    w = mx.load(os.path.join(model_dir, "mtp.safetensors"))
    heads = []
    for k in range(args.num_nextn_predict_layers if hasattr(args, "num_nextn_predict_layers") else 3):
        m = MTPLayer(args)
        pre = f"model.mtp.layers.{k}."
        m.load_weights([(n[len(pre):], v) for n, v in w.items() if n.startswith(pre)], strict=True)
        heads.append(m)
    T = len(ids)
    tokens = mx.array(ids)[None]
    main = norm(h)

    def step(m, hid, k):
        n = T - (k + 2)
        out = m(hid[:, :n], emb(tokens[:, k + 1 : k + 1 + n]).astype(mx.bfloat16))
        pred = mx.argmax(head(out).astype(mx.float32)[0], axis=-1)
        return out, float((pred == tokens[0, k + 2 : k + 2 + n]).astype(mx.float32).mean()) * 100

    def chain(order, use_prenorm):
        hid, res = main, []
        for k, hi in enumerate(order):
            out, acc = step(heads[hi], hid, k)
            res.append(acc)
            hid = heads[hi].last_prenorm if use_prenorm else out
        return ", ".join(f"d{k+1} {r:.1f}%" for k, r in enumerate(res))

    print("MTP chain heads 0,1,2 (normed out) :", chain([0, 1, 2], False))
    print("MTP chain heads 0,1,2 (pre-norm out):", chain([0, 1, 2], True))
    print("MTP chain head 0 x3   (normed out) :", chain([0, 0, 0], False))
    print("MTP chain head 0 x3   (pre-norm out):", chain([0, 0, 0], True))
    for hi in (1, 2):
        print(f"MTP head {hi} used at depth 1        :", f"{step(heads[hi], main, 0)[1]:.1f}%")
    # (e) no hidden chaining: head k gets main hidden[t] + token[t+1+k], predicts token[t+2+k]
    res = []
    for k in range(3):
        n = T - (k + 2)
        out = heads[k](main[:, :n], emb(tokens[:, k + 1 : k + 1 + n]).astype(mx.bfloat16))
        pred = mx.argmax(head(out).astype(mx.float32)[0], axis=-1)
        res.append(float((pred == tokens[0, k + 2 : k + 2 + n]).astype(mx.float32).mean()) * 100)
    print("MTP main-hidden + shifted token      :", ", ".join(f"d{k+1} {r:.1f}%" for k, r in enumerate(res)))
    # (f) chain, but hidden realigned: head k+1 at t uses out_k[t+1] (the state that has already seen token t+1+k)
    hid, res = main, []
    for k in range(3):
        n = T - (k + 2)
        out = heads[k](hid[:, :n], emb(tokens[:, k + 1 : k + 1 + n]).astype(mx.bfloat16))
        pred = mx.argmax(head(out).astype(mx.float32)[0], axis=-1)
        res.append(float((pred == tokens[0, k + 2 : k + 2 + n]).astype(mx.float32).mean()) * 100)
        hid = mx.concatenate([out[:, 1:], out[:, -1:]], axis=1)
    print("MTP chain shifted by one position    :", ", ".join(f"d{k+1} {r:.1f}%" for k, r in enumerate(res)))


if __name__ == "__main__":
    main()
