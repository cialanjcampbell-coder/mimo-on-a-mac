# M5 Max hardware limits: GPU bandwidth, SSD and mmap reads, quantized-matmul verify cost (2026-09-26)

Machine: MacBook Pro Mac17,6, M5 Max (40-core GPU), 128 GB, 2 TB SSD, macOS Darwin 27.0.
Scripts were run from `/tmp/hwbench` (scratch venv, MLX 0.32.2). mlx-serve was idle with no model loaded; nothing else was on the GPU.
`mx.device_info()`: max_recommended_working_set_size 115.4 GB (107.5 GiB), max_buffer_length 86.6 GB.

## GPU memory bandwidth (MLX)
| Test | GB/s |
|---|---|
| `sum` over 4 GB fp32 (read) | 570 |
| elementwise 4 GB fp32 (read + write) | 388 |
| 4-bit `quantized_matmul`, 1 token (matrix-vector), 4.8 GB of weights | 574 |

The decode ceiling for resident weights is about **570 GB/s ÷ bytes read per token**.

## SSD random reads (uncached, `F_NOCACHE` + `pread`, Qwen3.8-27B BF16 shards, 55 GB)
| Block | qd 1 | qd 4 | qd 16 | qd 32 |
|---|---|---|---|---|
| 256 KB | 1.9 | 6.8 | 11.1 | 11.8 GB/s |
| 1 MB | 4.3 | 12.7 | 13.3 | 13.3 |
| 4 MB | 8.7 | 13.8 | 14.0 | 13.9 |
| 16 MB | 12.0 | 15.4 | 15.2 | 15.2 |

## mmap page-fault path (Qwen3-32B-4bit + Qwen-Image shards, 18 GB, Python threads)
| | 1 MB | 8 MB |
|---|---|---|
| plain faults, qd 1 / 16 | 1.1 / 1.2 GB/s | 1.4 / 1.5 GB/s |
| `madvise(WILLNEED)` first, qd 1 / 16 | 5.5 / 5.8 | 6.1 / 7.2 |

Plain mmap faults reach about **10% of what the SSD can do**. Streaming experts needs explicit parallel `pread` of expert-sized blocks (≥1 MB, ≥4 in flight) into GPU-visible buffers. Unified memory means no extra copy.

Caveat: the Python harness is GIL-bound, so the fault numbers may be understated for a native multithreaded reader. The ordering pread ≫ madvise ≫ faults agrees with oMLX PR #3672 (preadv 7–8× faster than mmap gathers).

## Verify cost vs batch rows, 4-bit `quantized_matmul` (8 × 65536×8192, gs 64)
| rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 16 | 24 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| × of 1 row | 1.00 | 1.01 | 1.21 | 1.41 | 1.93 | 2.59 | 3.42 | 3.40 | 4.58 | 4.39 | 5.34 | 1.87 | 1.97 | 1.97 | 2.75 |

The small-batch `qmv` path grows almost linearly from 5 to 12 rows; the tiled `qmm` path starts at 16. For speculative decoding on dense projections, drafts of ≤3 tokens (≤4 rows) are nearly free, while 5–12 rows hit a kernel cliff. Either pad to 16 or fix the dispatch threshold. This is stock MLX 0.32.2; mlx-serve has its own `verifyQmm` lanes, so re-check there.
