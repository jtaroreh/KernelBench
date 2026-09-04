# KernelBench Level 2 Triton evaluation (v1.4.0-level2-turn4-70pct)

This release delivers empirical benchmark evaluation results, 100 authored operator implementations, and 70 verified Triton solutions for KernelBench Level 2 across all 100 compound operators.

## Hardware and execution engine
- **Target hardware.** NVIDIA L40S (Ada Lovelace, 48 GB VRAM).
- **Evaluation framework.** Modal cloud GPU cluster (`gpu=L40S`, `num_gpu_devices=8`).
- **Generation harness.** Parallel agentic sweeps with AST linting and 3-tier verification.
- **Kernel language.** OpenAI Triton (FP32).

## Key metrics
- **Authored operator pool.** 100.0% (100 of 100 operators implemented).
- **Static AST pass rate.** 100.0% (100 of 100 passed AST lint, anti-hacking, and resource bounds).
- **Functional correctness rate.** 70.0% (70 of 100 passed all randomized multi-trial evaluations on Modal L40S).
- **Fast_1 rate.** 51.0% (51 of 100 strictly faster than PyTorch eager execution).
- **Geometric mean speedup.** >1.2000x across correct samples.
- **Top operator speedups.** 1.83x on Problem 87, 1.68x on Problem 34, 1.65x on Problem 26, 1.40x on Problem 79.

## Included artifacts
- **Candidate and verified solutions.** 100 implementations in `solutions/level2/` (70 verified on Modal L40S).
- **Benchmark report.** Full metric distributions in `docs/benchmarks/SUBMISSION_REPORT.md`.
- **Decision log.** Full scientific hypothesis trail in `docs/benchmarks/decisions.tsv`.
- **Hillclimb report.** Trajectory and milestone analysis in `docs/benchmarks/hillclimb_report.md`.
- **Hardware baseline.** Recorded PyTorch eager baseline in `results/timing/L40S_Modal/baseline_time_torch.json`.
