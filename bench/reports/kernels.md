# MXFP4 expert kernels for MiMo, distinct-expert layout (2026-09-27)

Code: `engine/mimo_stream/kernels.py` (adapted from mlx-serve's MIT gather-QMV kernels); tests in `engine/tests/test_kernels.py`.
- Kernel A fuses the expert-side gate projection, up projection, and SwiGLU activation into one producer kernel. Kernel B performs the down projection, scales each expert contribution by its router (gating) weight, and uses a row-wise scatter-add to accumulate the mixture-of-experts output.
- Both kernels use a **distinct-expert dispatch** layout, the schedule is organized around unique experts rather than individual token–expert assignments. For each expert, MXFP4 weights are decoded once per 32-value block (packed uint4 nibbles plus one e8m0 block scale), then reused in dot products with every token row routed to that expert. Expert weights are selected by slot index from a resident device-memory pool, avoiding per-token weight gathers and redundant dequantization.

## Correctness
Against `mx.dequantize(mode="mxfp4")` + FP32 SwiGLU + weighted sum (random weights ×0.02, 24-slot pool, shared experts across rows):

| Rows | Max relative error |
|---|---|
| 1 | 0.40% |
| 4 | 0.44% |
| 8 | 0.45% |

These errors are BF16 activation rounding. Negative control (nibble order swapped): 130%.

## Speed
M5 Max, 128-slot pool (1.6 GB), fresh random experts every call, 20 calls per eval (the way a real graph runs):

| M rows | distinct experts | ours | stock `gather_qmm` ×3 | speedup |
|---|---|---|---|---|
| 1 | 8 | 0.222 ms / 482 GB/s | 0.200 ms / 534 GB/s | 0.90× |
| 2 | 15.2 | 0.405 ms / 503 GB/s | 0.661 ms / 308 GB/s | 1.63× |
| 4 | 29.3 | 0.771 ms / 508 GB/s | 2.481 ms / 158 GB/s | 3.22× |
| 8 | 51.4 | 1.444 ms / 476 GB/s | 9.341 ms / 74 GB/s | **6.47×** |

- A single `mx.eval` has about 0.16 ms of fixed overhead, so one-call timings are misleading. Kernel A alone, amortised: 577 GB/s.
- **Dispatch rule: stock `gather_qmm` at M=1, these kernels at M≥2.**
