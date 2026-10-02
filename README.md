# MiMo on a Mac

[![tests](https://github.com/cialanjcampbell-coder/mimo-on-a-mac/actions/workflows/tests.yml/badge.svg)](https://github.com/cialanjcampbell-coder/mimo-on-a-mac/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/cialanjcampbell-coder/mimo-on-a-mac)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-171f2c?logo=apple&logoColor=white)


**Running a 309B MoE model as a local coding agent on a 128 GB MacBook.**

![mimo-agent finding and fixing a month-end bug in the bugfix-report eval task, then passing its hidden checks](docs/demo.gif)

*The `bugfix-report` eval task run on an M5 Max. Pauses are shortened; the full run took about 2 minutes.*

A 160 GB model doesn't fit in 128 GB of memory. This repo runs one anyway, well enough to use as a daily coding agent. Xiaomi's **MiMo-V2.6-Flash** (309B total / 15B active parameters, MXFP4 experts) runs on an Apple M5 Max through a custom MLX inference engine. The engine keeps about half of the experts in memory and streams the rest from the SSD. It serves an OpenAI-compatible API, and the repo includes a lightweight coding agent (`scripts/mimo-agent`) that uses it.

| | |
|---|---|
| Model | MiMo-V2.6-Flash-RL at its **released precision** (MXFP4 experts; no pruning or extra quantisation) |
| Memory | 125 of 256 experts per layer in memory (~92 GB wired); the rest read from SSD on demand |
| Quality | Perplexity 2.77–2.81 on unseen code, matching a layer-by-layer reference forward pass (2.805) |
| Decode | 12–14 tok/s in real agent sessions |
| Prefill | 380–500 tok/s; about 1 s to first token on cached follow-up turns |

Informally, in daily use, it gave better answers with a lower thermal footprint than Qwen3.8-Flash-Next, a smaller model that fits entirely in memory. 

* The agent will automatically run shell commands and edit files. Use in commited repos or a sandbox*

## How it was built

Each step was measured before the next was built. The benchmark reports and raw measurements are indexed in [`bench/README.md`](bench/README.md).

1. **Measure the hardware.** GPU bandwidth is about 570 GB/s. The SSD delivers 13–15 GB/s with parallel `pread`s but only 1.2–1.5 GB/s through mmap page faults, so plain mmap wastes about 90% of the drive. SSD reads cost the GPU only 3% of its bandwidth. ([write-up](bench/reports/hardware.md))
2. **Choose the model with arithmetic:** bytes read per token ÷ bandwidth, and what fits in about 100 GB. MiMo was the only strong candidate whose *released* weights I could run here without further quantisation.
3. **Convert and validate.** Expert weights were copied byte for byte (60/60 samples identical) into per-layer files with every expert 16 KiB-aligned for direct I/O. The dense FP8 weights were dequantised, including the fused QKV tensor stored as four tensor-parallel slices with padded scales.
4. **Simulate before building.** Routing traces came from 52 real coding-agent sessions (279k tokens), split by project into train and test sets, and fed to an expert-cache simulator. Decayed LFU reached 91% hits at 49% resident, against 96% for the offline optimum (Belady). The engine was only built after this passed a 90% gate.
5. **Build the engine** (`engine/mimo_stream/`):
   - wired slot pools with decayed-LFU eviction, filled by a direct-I/O `preadv` pool (`F_NOCACHE`)
   - next-layer expert prediction with top-4 prefetch
   - double-buffered expert staging for prefill
   - custom Metal kernels for small batches
   - fixed-order reductions, so output is deterministic
6. **Serve it.** The OpenAI-compatible server has a prompt renderer that is byte-identical to the model's chat template, so agent tool loops reuse the whole KV cache. It parses MiMo's native tool-call format and rolls back the sliding-window caches when a prompt diverges.

Optimisation log (decode, tok/s): 8.65 → 10.3 (8-bit dense weights, wired memory) → 11.1 (prefetch) → 13.9 (removed hidden GPU syncs) → 14.4 (queue resident experts before waiting on the SSD).

## Findings

- **Expert caching on agentic traffic.** Fixed "hot sets" of experts don't carry over between projects (76% hits vs 91% for decayed LFU). Consecutive tokens share only 2.6 of 8 experts. The model's own output is more cache-friendly than text written by another model (29 vs 44 misses per token).
- **Speculative decoding doesn't spread the SSD cost.** Every row of a verify batch brings about 33 new expert misses, whatever the batch width. Drafts only pay off when acceptance is very high, as with copy-heavy output.
- **Small-batch MXFP4 kernels.** Stock MLX `gather_qmm` reaches ~534 GB/s at 1 row but falls to 74 GB/s at 8 rows. Kernels that read each distinct expert once (adapted from mlx-serve, MIT) hold ~500 GB/s from 1 to 8 rows, 6.5× faster at 8 rows. ([write-up](bench/reports/kernels.md))
- **MiMo's MTP heads are parallel.** Head k takes the main model's hidden state plus the token k ahead; teacher-forced accuracy is 72.6 / 68.3 / 65.8% by depth. Chaining them DeepSeek-style drops accuracy to about 1% at depth 2.
- **Routing is chaotic under ~1e-3 numerical noise.** Near-tied top-8 choices flip and the differences compound: two correct code paths agree on the top token at only about 90% of positions. Implementations therefore have to be compared by perplexity and KL divergence, not by token equality.
- **Dead ends, recorded so nobody repeats them:**
  - Using the page cache as a second expert tier overcommitted RAM, and decode fell to about 10 s/token.
  - Unwired MLX buffers got compressed by macOS (70 GB).
  - Prefetching all 8 predicted experts cost more SSD time than it saved.

## Related work

Streaming experts from flash storage is an active area, and this project builds on it:
- Apple's *LLM in a flash* (2023)
- flash-moe
- SwiftLM, streamlx, paged-moe and ssd-moe/deepseek-v4-flash-mlx
- oMLX's expert offload, including MiMo support in [PR #3865](https://github.com/jundot/omlx/pull/3865), measured at ~1.3–1.6 tok/s on a 48 GB M4 Pro
- MTPLX and antirez/ds4 for MTP and Metal kernels on Apple silicon

Several issues hit here were also found independently by others: the MiMo fused-QKV tensor-parallel layout (vllm-backport #106, oMLX #3976), and the MLX 0.32.2 sorted `gather_qmm` bug past 32,767 rows (ml-explore/mlx#3856, fixed in #3922; our characterisation and check of the fix: [measurement script](bench/scripts/gather_qmm_sorted_check.py)). What this repo adds:
- a working engine for this model, measured end to end on a 128 GB machine, used every day
- cache-policy data from real agentic coding sessions, with train and test split by project
- the finding that speculation doesn't pay under SSD streaming

## Limitations

- Tested on a single machine (M5 Max, 128 GB, internal SSD).
- One conversation is cached at a time.

## Layout

- `engine/mimo_stream/`: the SSD streaming engine (direct I/O preadv pool, decayed-LFU slot cache, custom Metal kernels, model architecture)
- `engine/server.py`: OpenAI-compatible completions server with prefix caching
- `agent/`: lightweight stdlib-only coding agent, interactive and headless (launched by `scripts/mimo-agent`)
- `scripts/`: coding agent launcher (`mimo-agent`), model conversion, server daemon control, layerwise verification, routing trace analysis, and cache simulation
- `evals/`: 6 coding-agent tasks with hidden checks, run by `scripts/eval-task` ([details](evals/README.md))
- `bench/`: benchmark reports and raw machine-readable results

## Running it

### Prerequisites
- Apple silicon Mac with 128 GB of unified memory (tested only on an M5 Max).
- A fast internal SSD with ~170 GB free for the converted weights.
- Python 3.10+ with MLX dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 1. Convert the model weights
Download `XiaomiMiMo/MiMo-V2.6-Flash-RL` from Hugging Face and convert into the 16 KiB-aligned SSD layout:
```bash
python scripts/convert-mimo-v26-mlx.py /path/to/MiMo-V2.6-Flash-RL /path/to/converted-model
```

### 2. Start the server
```bash
export MIMO_MODEL_DIR=/path/to/converted-model
scripts/mimo-server-ctl start
```
The server listens on `http://127.0.0.1:8082/v1` with an OpenAI-compatible API. Status and metrics:
```bash
scripts/mimo-server-ctl status
```

### 3. Run the coding agent
`scripts/mimo-agent` is a lightweight coding agent (stdlib-only Python, code in `agent/`) with tools `read`, `edit`, `write`, `grep`, `find`, `ls`, `bash`.

Interactive REPL (`/reset` clears the conversation, `/exit` quits, Ctrl-C interrupts a turn):
```bash
scripts/mimo-agent
```
Headless, one prompt then exit (`--json` for machine-readable output):
```bash
scripts/mimo-agent -p "fix the failing test in tests/test_foo.py" [--json]
```
Flags:
- `--tools read,edit,...`: restrict the tool set (default: all)
- `--thinking on|off`: toggle MiMo's reasoning
- `--base-url URL`, `--model ID`: use any OpenAI-compatible server (default: the local MiMo server on port 8082)
- `--max-turns N`: cap the number of model turns
- `--no-start`: don't start the server

If `/health` fails, the agent starts the server with `scripts/mimo-server-ctl start` (needs `MIMO_MODEL_DIR`), unless `--no-start` is given.

To use it from any directory, symlink it onto your PATH:
```bash
ln -s "$PWD/scripts/mimo-agent" ~/.local/bin/mimo-agent
```

### Testing
Run the unit test suites:
```bash
python3 -m unittest discover -s engine/tests -v
python3 -m unittest discover -s scripts/tests -v
python3 -m unittest discover -s agent/tests -v
```
The engine tests that need the model skip unless `MIMO_MODEL_DIR` is set.

## License

MIT (see `LICENSE`). `engine/mimo_stream/kernels.py` adapts kernels from [mlx-serve](https://github.com/ddalcu/mlx-serve) (MIT), and `engine/mimo_stream/mimo_v2.py` is vendored from [mlx-lm](https://github.com/ml-explore/mlx-lm) PR #1219 (MIT); see `THIRD_PARTY_NOTICES.md`. Model weights are not included and are under Xiaomi's license.
