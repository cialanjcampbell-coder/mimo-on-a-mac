# Engine A/B: eviction policy + 4 GB more resident experts (2026-09-28)

> **Rolled back.** The per-layer budgets (this section) and the priority I/O scheduler ([below](#prefetch-and-io-scheduling)) were measured and then reverted on 2026-09-29 in favour of the simpler baseline. The shipped engine uses a uniform 125-slot budget, decay every 16 tokens and FIFO I/O.

`engine/tests/test_parity.py` with the server's settings (8-bit dense, top-4 prefetch), GPU lock held, MiMo-V2.6-Flash MLX. Same code version; the old configuration reproduced with `--slots 125 --decay-every 16`. JSON: `engine-evict-ab-{old,new}-servercfg.json` (and the `-old`/`-new` runs with the test's defaults: BF16 dense, top-8 prefetch, which show the same direction).

| | Old: uniform 125/layer, decay every 16 | New: per-layer budgets (6,174 total, 120-164/layer), decay every 32 | Change |
|---|---|---|---|
| Teacher-forced decode, 256 tokens | 11.61 tok/s | **12.50 tok/s** | +7.7% |
| Demand misses/token | 31.6 | 27.7 | -12% |
| SSD read/token | 704 MB | 618 MB | -12% |
| Greedy generation, 48 tokens | 10.07 tok/s | 10.67 tok/s | +6% |
| ppl / top-1 (1,500 tokens) | 2.771 / 77.1% | 2.771 / 77.1% | identical |
| Expert pools | 78.5 GB | 82.5 GB (+4 GB, user-approved) | |
| Staging | 2 x 131 slots | 2 x 136 slots | +0.07 GB |

- **Policy source:** [the cache simulation results](../results/raw/mimo-cache-sim.json). At equal memory the simulator predicted -6% misses from the policy alone; with +4 GB it predicted -15%. Measured: -12%.
- **Absolute speed is text-dependent:** this passage misses more than the baseline passage (31.6 vs 23.4/token), hence 11.6 rather than 14.4 tok/s for the old configuration.
- **Small prefill is visible here too:** a 32-token prompt prefilled at 13-14 tok/s (~2.4 s), because short chunks read nearly all non-resident experts. This is the next target, along with deeper, confidence-gated prefetch.
- **Server:** during the experiment it defaulted to `--slots-total 6174`. After the rollback, the server uses the uniform 125-slot budget.

## Baseline (first engine build)

The first engine measurements used 125 slots per MoE layer, 8-bit dense weights, and two staging banks on an M5 Max with MLX 0.32.2.

### Quality

| Configuration | ppl (1,500 tokens of unseen Python) | top-1 |
|---|---|---|
| Reference, layer-by-layer | 2.805 | 77.2% |
| Engine, BF16 dense, 64 slots | 2.781 | 77.4% |
| Engine, BF16 dense, 125 slots | 2.805 | 77.1% |
| Engine, 8-bit dense, 125 slots | 2.771 | - |

Runs were bit-identical. Numerically equivalent paths agreed on the argmax at 88-95% of positions, with mean KL about 0.02; routing sensitivity means paths should be compared with distribution metrics rather than token identity.

### Decode progression

| Step | tok/s | demand misses/token | MB read/token |
|---|---:|---:|---:|
| BF16 dense, no wiring | 8.65 | 44.0 | 589 |
| 8-bit dense, wired memory | 10.28 | 44.3 | 592 |
| Top-4 prefetch | 11.05 | 23.1 | 506 |
| Removed hidden slicing syncs | 13.94 | 22.9 | 501 |
| Resident experts queued before SSD wait | **14.37** | 23.4 | 506 |

The go/no-go gate for the engine was decode within 30% of the corrected 18-24 tok/s estimate. At 14.4 tok/s, the gate passed. Prefill reached 500 tok/s for a 4,096-token chunk with double-buffered staging.

Failed experiments are retained as context: using the page cache as a second tier caused compression and swapping, and unwired MLX buffers produced a 103 GB footprint with 70 GB compressed.

## Prefetch and I/O scheduling

One-layer-ahead routing prediction reached 86-87% top-4 precision and 69-70% top-8 recall. Two- and three-layer look-ahead were less precise.

The experimental I/O pool (later rolled back) prioritized demand reads, waited only for needed prefetches, protected in-flight slots from eviction, and invalidated failed prefetches. On a 256-token teacher-forced decode, priority plus selective waits improved throughput from 12.52 to 12.79 tok/s while preserving perplexity and decode-vs-prefill checks bit-for-bit.

More prefetch was counterproductive because the SSD is the binding constraint:

| Prefetch per layer | Decode | Demand misses/token | SSD read/token |
|---:|---:|---:|---:|
| 4 | **12.79 tok/s** | 27.6 | 605 MB |
| 6 | 12.52 tok/s | 22.2 | 703 MB |
| 8 | 11.97 tok/s | 17.5 | 839 MB |

At 12.8 tok/s the SSD is busy about 46 of 78 ms per token. More speculative reads delay demand reads, so deeper look-ahead was not pursued. The remaining practical levers are additional resident memory and reducing the roughly 35 ms of GPU and synchronization time per token.
