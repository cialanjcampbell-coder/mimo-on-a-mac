"""Check stock MLX sorted gather_qmm against a float32 per-expert reference (the NAX row-overflow bug).

  python bench/scripts/gather_qmm_sorted_check.py

MLX <= 0.32.2 on M5 (NAX kernels): affine sorted gather_qmm leaves the *leading* rows wrong when
M > 32,767 and M % 64 != 0 (ml-explore/mlx#3856, fixed by #3922 in 0.32.3). MXFP4 is unaffected.
Measurement script: bench/scripts/gather_qmm_sorted_check.py. A write-up can be added under bench/reports when the characterization is complete. Small (<2 GB); still take the GPU lock.
"""
import numpy as np, mlx.core as mx
print("mlx", mx.__version__, mx.device_info().get("device_name"))

def run(M, N, K, E, mode, bits, gs, sorted_=True, seed=0):
    rng = np.random.default_rng(seed)
    W = mx.array(rng.standard_normal((E, N, K)).astype(np.float32) * 0.05).astype(mx.bfloat16)
    q = mx.quantize(W, group_size=gs, bits=bits, mode=mode)
    w, s = q[0], q[1]; b = q[2] if mode == "affine" else None
    Wd = mx.dequantize(w, s, b, group_size=gs, bits=bits, mode=mode).astype(mx.float32)
    idx = np.sort(rng.integers(0, E, M)).astype(np.uint32)
    x = mx.array(rng.standard_normal((M, 1, K)).astype(np.float32)).astype(mx.bfloat16)
    y = mx.gather_qmm(x, w, s, b, rhs_indices=mx.array(idx), transpose=True, group_size=gs, bits=bits,
                      mode=mode, sorted_indices=sorted_)[:, 0, :].astype(mx.float32)
    xf = x[:, 0, :].astype(mx.float32)
    ref = mx.concatenate([xf[int(a):int(c)] @ Wd[e].T for e in range(E)
                          for a, c in [np.searchsorted(idx, [e, e + 1])] if c > a], axis=0)
    mx.eval(y, ref)
    y, ref = np.array(y), np.array(ref)
    err = np.abs(y - ref).max(axis=1)
    scale = np.abs(ref).max()
    bad = np.nonzero(err > 0.05 * scale)[0]
    zero = np.nonzero(np.abs(y).max(axis=1) == 0)[0]
    rng_s = f"rows {bad.min()}..{bad.max()}" if len(bad) else ""
    print(f"{mode:6s} b{bits} gs{gs} {'sorted' if sorted_ else 'unsort'} M={M:6d} N={N:4d} K={K:4d} E={E:3d}  "
          f"bad={len(bad):6d} zero_rows={len(zero):6d} {rng_s}", flush=True)
    return len(bad)

Ms = [32767, 32768, 32769, 33010, 40010, 40064, 50000]
for mode, bits, gs in [("affine", 4, 64), ("affine", 8, 64), ("mxfp4", 4, 32)]:
    for M in Ms:
        run(M, 64, 128, 64, mode, bits, gs)
print("-- larger N/K (Flash-Next-like gate: N=640 K=2560, 512 experts)")
for mode, bits, gs in [("affine", 4, 64), ("mxfp4", 4, 32)]:
    for M in [32768, 32769, 40010]:
        run(M, 640, 2560, 512, mode, bits, gs)
print("-- unsorted control")
run(40010, 64, 128, 64, "affine", 4, 64, sorted_=False)
