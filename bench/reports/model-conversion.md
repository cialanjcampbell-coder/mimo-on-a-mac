# MiMo-V2.6-Flash-RL → MLX: conversion and validation (2026-09-26)

- Source: `XiaomiMiMo/MiMo-V2.6-Flash-RL` @ `5711b268` (164 GiB; the audio tokenizer was not downloaded).
- Output: 166 GiB, written by `scripts/convert-mimo-v26-mlx.py`, which took around 2 mins.
  - **Experts:** 47 files `experts-LNN.safetensors`, MXFP4 copied byte-for-byte and stacked `[256, out, in/8]` u32 + `[256, out, in/32]` u8 e8m0. Each data section starts on a 16 KiB boundary, so every expert of every projection sits at `base + e*stride`, aligned.
  - **Dense:** `dense.safetensors`, 427 tensors. FP8 128×128 block weights are dequantised to BF16 (fp8 × scale_inv, computed in fp32). The fused qkv is split with the TP=4 slice layout `[q_t | k_t | v_t]`, whose scale blocks are padded per slice. o_proj, routers, embeddings, lm_head, norms and sinks are copied as stored.
  - **MTP:** `mtp.safetensors`, 3 heads, same treatment.
  - **DFlash:** `dflash/` holds the drafter weights (BF16, copied), `config.json`, and `mask_embedding.safetensors`. The mask embedding is 4096 BF16 values read from the torch zip's raw storage; nothing was unpickled.
- Environment: MLX 0.32.2 and mlx-lm from PR #1219 (`kernelpool/mlx-lm@add-mimo-v2`).

## Checks
| Check | Result |
|---|---|
| 60 random expert tensors (weights and scales) vs source bytes | 0 mismatches; data sections aligned |
| MTP q/v slices vs an independent NumPy e4m3 decode | max relative error 0.19–0.30%, which is BF16 rounding |
| Layer-by-layer teacher-forced forward (`scripts/mimo_layerwise.py`), 1,500 tokens of unseen Python (the converter itself) | **ppl 2.805, top-1 77.2%, second-half ppl 1.657** |
| Wall time / peak memory of that pass, one layer resident at a time | 14.2–14.7 s / 15.2 GB |

## MTP heads: they are parallel, not chained
Teacher-forced top-1 accuracy on the same 1,500 tokens. `main` is the main model's final hidden state; head k predicts token t+2+k.

| Wiring | d1 | d2 | d3 |
|---|---|---|---|
| Chained, heads 0→1→2, each head's normed output as the next head's hidden state (DeepSeek-V3 style; how I read oMLX's `MiMoV2MultiTokenPredictor`) | 72.6 | 1.5 | 0.6 |
| Same, pre-norm output passed on | 72.6 | 3.6 | 1.7 |
| Head 0 reused three times (vLLM `num_mtp_layers=1`) | 72.6 | 2.2–3.5 | 0.6–1.3 |
| Chained, shifted one position | 72.6 | 3.5 | 0.7 |
| **Head k = main hidden[t] + embedding of token[t+1+k]** | **72.6** | **68.3** | **65.8** |
| Head 1 / head 2 used at depth 1 | 68.8 / 64.0 | | |

Every chained form collapses at depth 2. With the parallel form, all three heads are useful. At inference, head k takes the last verified main-model hidden state plus the draft from head k-1. So the heads are sequential in tokens but share one hidden state.

Caveats:
- The accuracies are teacher-forced (conditional). Measured on novel code rather than the model's own output.
- The tech report says MTP-3 comes from SFT. But the RL checkpoint's default drafter is block-6 DFlash, which has 31.3% longer average accepted length than MTP (report §RL, "Draft Model Acceleration").
