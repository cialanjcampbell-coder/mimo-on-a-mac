#!/usr/bin/env python3
"""Routing traces for MiMo-V2.6-Flash over a corpus, layer-major (each layer's weights loaded once).

  mimo_route_trace.py <model_dir> <corpus_dir> <out_dir>

For every session .npz (ids, role) in <corpus_dir>/{train,test}, runs a teacher-forced forward pass and
writes <out_dir>/<split>/<name>.npz with the input arrays plus, per MoE layer L:
  L{L:02d}  uint8[T,8]  experts chosen at layer L
  P{L:02d}  uint8[T,8]  prediction from layer L-1's post-attention residual through layer L's norm+router
                        (available at layer L-1's routing sync, one MoE step ahead; "Fate"-style)
  Q{L:02d}  uint8[T,8]  prediction from layer L's input residual (available after layer L-1's MoE)
Teacher forcing gives exactly the routes decoding the same text would take.
"""
import glob, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for engine.*
from engine.mimo_stream import mimo_v2
from mlx_lm.models.base import create_attention_mask
import json


def main():
    model_dir, corpus, out = sys.argv[1:4]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    args = mimo_v2.ModelArgs.from_dict(cfg)
    dense = mx.load(os.path.join(model_dir, "dense.safetensors"))
    sub = lambda pre: [(k[len(pre):], v) for k, v in dense.items() if k.startswith(pre)]

    sessions = []
    for split in ("train", "test"):
        for f in sorted(glob.glob(os.path.join(corpus, split, "*.npz"))):
            if os.path.exists(os.path.join(out, split, os.path.basename(f))):
                continue  # already traced (resume)
            d = np.load(f)
            sessions.append({"split": split, "name": os.path.basename(f), "ids": d["ids"], "role": d["role"], "rec": {}})
    print(f"{len(sessions)} sessions, {sum(len(s['ids']) for s in sessions)} tokens", flush=True)

    emb = nn.Embedding(args.vocab_size, args.hidden_size)
    emb.load_weights(sub("model.embed_tokens."))
    for s in sessions:
        s["h"] = emb(mx.array(s["ids"])[None]).astype(mx.bfloat16)
        s["xprev"] = None
        mx.eval(s["h"])
    t0 = time.time()
    for L in range(args.num_hidden_layers):
        is_moe, is_swa = bool(args.moe_layer_freq[L]), bool(args.hybrid_layer_pattern[L])
        layer = mimo_v2.DecoderLayer(args, is_moe=is_moe, is_sliding_window=is_swa)
        w = sub(f"model.layers.{L}.")
        if is_moe:
            nn.quantize(layer, group_size=32, bits=4, mode="mxfp4",
                        class_predicate=lambda p, m: "switch_mlp" in p and hasattr(m, "to_quantized"))
            ex = mx.load(os.path.join(model_dir, f"experts-L{L:02d}.safetensors"))
            w += [(k[len(f"model.layers.{L}."):], v) for k, v in ex.items()]
        layer.load_weights(w, strict=True)
        mx.eval(layer.parameters())
        for s in sessions:
            h = s["h"]
            if is_moe:
                pn = layer.post_attention_layernorm
                q_inds, _ = layer.mlp.gate(pn(h))
                p_inds = layer.mlp.gate(pn(s["xprev"]))[0] if s["xprev"] is not None else q_inds
            mask = create_attention_mask(h, None, window_size=args.sliding_window_size if is_swa else None)
            x = h + layer.self_attn(layer.input_layernorm(h), mask, None)
            r = layer.post_attention_layernorm(x)
            if is_moe:
                inds, scores = layer.mlp.gate(r)
                y = (layer.mlp.switch_mlp(r, inds) * scores[..., None]).sum(axis=-2).astype(r.dtype)
            else:
                y = layer.mlp(r)
            s["h"] = x + y
            s["xprev"] = x
            if is_moe:
                mx.eval(s["h"], s["xprev"], inds, p_inds, q_inds)
                s["rec"][f"L{L:02d}"] = np.array(inds[0]).astype(np.uint8)
                s["rec"][f"P{L:02d}"] = np.array(p_inds[0]).astype(np.uint8)
                s["rec"][f"Q{L:02d}"] = np.array(q_inds[0]).astype(np.uint8)
            else:
                mx.eval(s["h"], s["xprev"])
        del layer, w
        if is_moe:
            del ex
        mx.clear_cache()
        print(f"layer {L:2d} done {time.time()-t0:7.1f}s peak {mx.get_peak_memory()/1e9:5.1f} GB", flush=True)
    for s in sessions:
        d = os.path.join(out, s["split"])
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, s["name"] + ".partial.npz")
        np.savez(tmp, ids=s["ids"], role=s["role"], **s["rec"])
        os.replace(tmp, os.path.join(d, s["name"]))
    print(f"wrote {len(sessions)} traces to {out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
