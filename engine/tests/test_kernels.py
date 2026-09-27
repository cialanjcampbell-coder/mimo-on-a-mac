"""Correctness + speed of engine.mimo_stream.kernels against MLX's own MXFP4 dequantisation.

  python -m unittest engine/tests/test_kernels.py -v   (correctness)
  python engine/tests/test_kernels.py --bench         (speed vs stock gather_qmm)
"""
import sys, time, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import numpy as np
    import mlx.core as mx
    from engine.mimo_stream.kernels import moe_experts, group_routes
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

H, N = 4096, 2048


def make_pool(S, seed=0):
    mx.random.seed(seed)
    p = {}
    for name, (o, i) in {"gate": (N, H), "up": (N, H), "down": (H, N)}.items():
        w = (mx.random.normal((S, o, i)) * 0.02).astype(mx.bfloat16)
        q, s = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
        p[f"{name}_w"], p[f"{name}_s"] = q, s
    mx.eval(p)
    return p


def reference(x, pool, inds, scores):
    """float32 reference: dequantize with MLX, SwiGLU, weighted sum."""
    deq = {n: mx.dequantize(pool[f"{n}_w"], pool[f"{n}_s"], group_size=32, bits=4, mode="mxfp4").astype(mx.float32)
           for n in ("gate", "up", "down")}
    xf = x.astype(mx.float32)
    out = []
    for m in range(x.shape[0]):
        acc = mx.zeros((H,), mx.float32)
        for e, s in zip(inds[m], scores[m]):
            g = deq["gate"][int(e)] @ xf[m]
            u = deq["up"][int(e)] @ xf[m]
            hh = (g * mx.sigmoid(g)) * u
            acc = acc + float(s) * (deq["down"][int(e)] @ hh)
        out.append(acc)
    return mx.stack(out)


def run(pool, x, inds, scores, R=8):
    uids, urows, ucnt, uw, gidx = group_routes(inds, scores, R)
    return moe_experts(x, pool, uids.astype(np.int32), urows, ucnt, uw, R, gather_idx=mx.array(gidx))


def rel_err(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


class TestKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAS_MLX:
            raise unittest.SkipTest("mlx or numpy not available")

    def test_matches_mlx_dequant(self):
        S = 24
        pool = make_pool(S)
        rng = np.random.default_rng(0)
        for M in (1, 2, 4, 8):
            x = (mx.random.normal((M, H)) * 1.0).astype(mx.bfloat16)
            inds = np.stack([rng.choice(S, 8, replace=False) for _ in range(M)])
            if M > 1:
                inds[1, :4] = inds[0, :4]  # shared experts across rows
            scores = rng.random((M, 8)).astype(np.float32)
            scores /= scores.sum(1, keepdims=True)
            y = run(pool, x, inds, scores)
            ref = reference(x, pool, inds, scores)
            e = rel_err(y, ref)
            self.assertLess(e, 2e-2, f"M={M}, error={e}")   # bf16 activations/outputs vs float32 reference

    def test_single_expert_many_rows(self):
        S = 4
        pool = make_pool(S, seed=1)
        x = mx.random.normal((8, H)).astype(mx.bfloat16)
        inds = np.tile(np.arange(4), (8, 1))  # all 8 rows route to the same 4 experts (R = 8 rows each)
        scores = np.full((8, 4), 0.25, np.float32)
        y = run(pool, x, inds, scores)
        ref = reference(x, pool, inds, scores)
        self.assertLess(rel_err(y, ref), 2e-2)


def bench(S=128, reps=30):
    if not HAS_MLX:
        print("mlx or numpy not available")
        return
    pool = make_pool(S)
    rng = np.random.default_rng(1)
    print(f"slot pool S={S}; expert = 12.75 MiB (weights+scales)")
    exp_bytes = 3 * (N * H // 2 + N * H // 32)
    for M in (1, 2, 4, 8):
        x = mx.random.normal((M, H)).astype(mx.bfloat16)
        inds = np.stack([rng.choice(S, 8, replace=False) for _ in range(M)])
        scores = np.full((M, 8), 1 / 8, np.float32)
        uids, urows, ucnt, uw, _ = group_routes(inds, scores)
        args = [mx.array(a) for a in (uids.astype(np.int32), urows, ucnt, uw)]
        f = lambda: moe_experts(x, pool, *args)
        idx = mx.array(inds.astype(np.uint32)).reshape(M, 1, 8)
        x4 = x.reshape(M, 1, 1, H)

        def stock():
            a = mx.gather_qmm(x4, pool["gate_w"], pool["gate_s"], None, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4")
            b = mx.gather_qmm(x4, pool["up_w"], pool["up_s"], None, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4")
            hh = a * mx.sigmoid(a) * b
            y = mx.gather_qmm(hh, pool["down_w"], pool["down_s"], None, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4")
            return (y.squeeze(-2) * mx.array(scores)[..., None]).sum(-2)

        for name, fn in (("ours", f), ("stock", stock)):
            for _ in range(3):
                mx.eval(fn())
            t0 = time.time()
            for _ in range(reps):
                mx.eval(fn())
            ms = (time.time() - t0) / reps * 1e3
            U = len(uids)
            print(f"M={M} {name:5s} {ms:6.3f} ms  distinct={U:2d}  {U * exp_bytes / (ms / 1e3) / 1e9:5.0f} GB/s effective")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        unittest.main()
