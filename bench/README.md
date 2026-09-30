# Benchmark records

This directory contains the benchmark reports and the raw measurements behind them.

## Reports

- [Hardware limits](reports/hardware.md): M5 Max GPU bandwidth, SSD reads, mmap, and quantized matmul costs.
- [Model conversion](reports/model-conversion.md): MiMo-V2.6-Flash conversion and validation.
- [Engine performance](reports/engine-performance.md): first engine measurements, eviction policy, and prefetch/I/O scheduling.
- [MXFP4 kernels](reports/kernels.md): distinct-expert kernel correctness and speed.
- [Operational trials](reports/operational-trials.md): an end-to-end MiMo smoke trial.

## Raw results

The JSON files in [results/raw](results/raw/) are machine-readable benchmark artifacts. Name suffixes identify configurations or A/B variants; the reports give the date of each experiment. They let results be re-plotted or compared without parsing prose.

The main groups are:

- `engine-*`: engine parity and configuration experiments.
- `engine-evict-*`: old/new eviction-policy comparisons.
- `engine-prefetch-*`: prefetch depth and I/O scheduler comparisons.
- `mimo-cache-sim.json`: routing and cache simulation output.
- `initial-bench.json`: hardware and kernel microbenchmarks.
