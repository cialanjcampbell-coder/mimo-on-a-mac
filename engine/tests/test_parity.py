"""Engine parity + first speed numbers (needs the model, ~40-95 GB; take the GPU lock first).

  python engine/tests/test_parity.py --slots 64 [--text FILE] [--gen 48]

(a) prefill quality on a text vs the layer-by-layer reference (scripts/mimo_layerwise.py: ppl 2.805,
    top-1 77.2% on scripts/convert-mimo-v26-mlx.py, 1500 tokens);
(b) decode path == prefill path: logits from token-by-token decode_step over positions P..P+D must match
    a single prefill pass over the same tokens (checks KV / sliding-window caches and the M=1 path);
(c) greedy generation, decode tok/s, prefill tok/s, expert hit rate.
Run twice with different --slots: results must not depend on the budget.
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import numpy as np
    import mlx.core as mx
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from engine.mimo_stream.model import Engine, E
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

MODEL = os.environ.get("MIMO_MODEL_DIR", "")
ROUTES = os.environ.get("MIMO_ROUTES_DIR", "")


def profile(routes_dir=ROUTES):
    prof = {}
    for f in glob.glob(f"{routes_dir}/*.npz"):
        z = np.load(f)
        m = z["role"] == 1
        for k in z.files:
            if k.startswith("L"):
                L = int(k[1:])
                prof.setdefault(L, np.zeros(E, np.int64))
                np.add.at(prof[L], z[k][m].ravel(), 1)
    return prof


def main():
    ap = argparse.ArgumentParser(description="MiMo-V2.6-Flash parity and speed benchmark")
    ap.add_argument("--slots", type=int, default=64, help="resident experts per MoE layer (default: 64)")
    ap.add_argument("--model", default=MODEL, help="path to converted MiMo MLX model (default: MIMO_MODEL_DIR)")
    ap.add_argument("--routes", default=ROUTES, help="path to routing traces directory (default: MIMO_ROUTES_DIR)")
    ap.add_argument("--text", default=str(Path(__file__).resolve().parents[2] / "scripts/convert-mimo-v26-mlx.py"), help="input text file for prefill and decode parity check")
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--decode-check", type=int, default=24)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--out")
    a = ap.parse_args()

    if not HAS_DEPS:
        print("MLX or dependencies not available. Please run in an MLX environment.")
        return

    model_ok = bool(a.model) and Path(a.model).exists()
    routes_ok = bool(a.routes) and Path(a.routes).exists()
    if not model_ok or not routes_ok:
        missing = []
        if not model_ok:
            missing.append(f"model at '{a.model}'")
        if not routes_ok:
            missing.append(f"routes at '{a.routes}'")
        print(f"Skipping test_parity: {' and '.join(missing)} not found.\n"
              f"Usage: python -m engine.tests.test_parity [options]\n\n"
              f"Flags:\n"
              f"  --slots SLOTS   Number of resident expert slots per MoE layer (e.g. 64, 125)\n"
              f"  --model DIR     Path to converted model directory (or set MIMO_MODEL_DIR)\n"
              f"  --routes DIR    Path to route traces directory (or set MIMO_ROUTES_DIR)\n"
              f"  --text FILE     Path to input text file for prefill/decode parity\n"
              f"  --tokens N      Prefill token count (default: {a.tokens})\n"
              f"  --gen N         Greedy generation token count (default: {a.gen})")
        return

    if not Path(a.text).exists():
        print(f"Input text file '{a.text}' not found. Please specify an existing file with --text FILE.")
        return

    res = {"slots": a.slots}
    eng = Engine(a.model, a.slots, profile=profile(a.routes))
    tok = load_tokenizer(Path(a.model))
    ids = tok.encode(open(a.text).read())[: a.tokens]

    # (a) prefill quality
    t0 = time.time()
    logits = eng.forward(ids, eng.make_cache(), all_logits=True)
    t = time.time() - t0
    tgt = mx.array(ids[1:])
    lp = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
    nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).squeeze(-1)
    res["ppl"] = float(mx.exp(nll.mean()))
    res["top1"] = float((mx.argmax(logits[:-1], axis=-1) == tgt).astype(mx.float32).mean())
    res["prefill_tok_s"] = len(ids) / t
    print(f"(a) {len(ids)} tokens: ppl {res['ppl']:.3f} top1 {res['top1'] * 100:.1f}%  "
          f"(reference 2.805 / 77.2%)  prefill {t:.1f}s = {res['prefill_tok_s']:.0f} tok/s", flush=True)

    # (b) decode path == prefill path
    P, D = 256, a.decode_check
    cache = eng.make_cache()
    eng.forward(ids[:P], cache)
    dec = [eng.decode_step(i, cache) for i in ids[P:P + D]]
    ref = logits[P: P + D]
    dec = mx.stack(dec)
    agree = float((mx.argmax(dec, -1) == mx.argmax(ref, -1)).astype(mx.float32).mean())
    rel = float(mx.abs(dec - ref).max() / mx.abs(ref).max())
    res["decode_vs_prefill_argmax_agree"] = agree
    res["decode_vs_prefill_max_rel"] = rel
    print(f"(b) decode vs prefill over {D} positions: argmax agree {agree * 100:.1f}%, max rel diff {rel:.4f}", flush=True)

    # (c) greedy generation from a code prompt
    prompt = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that parses an ISO-8601 "
                                       "date string into a datetime, with tests. Be brief."}],
                                     add_generation_prompt=True, tokenize=False)
    pids = tok.encode(prompt)
    h0, m0 = eng.hit_rate()
    io0, ios0 = eng.io.bytes, eng.io.seconds
    out, st = eng.generate(pids, a.gen, eos={tok.eos_token_id} if tok.eos_token_id is not None else None)
    h1, m1 = eng.hit_rate()
    res.update({f"gen_{k}": v for k, v in st.items()})
    res["gen_misses_per_token"] = (m1 - m0) / max(len(out), 1)
    res["gen_io_GB"] = (eng.io.bytes - io0) / 1e9
    res["gen_io_s"] = eng.io.seconds - ios0
    print(f"(c) prefill {len(pids)} tok {st['prefill_tok_s']:.0f} tok/s; decode {len(out)} tok at {st['decode_tok_s']:.2f} tok/s; "
          f"misses/token {res['gen_misses_per_token']:.1f}; io {res['gen_io_GB']:.2f} GB in {res['gen_io_s']:.1f}s", flush=True)
    print("---\n" + tok.decode(out) + "\n---")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
