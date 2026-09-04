# KernelBench Level 2 Triton evaluation (v1.5.0-level2-turn5-100pct)

This release marks the completion of KernelBench Level 2 on NVIDIA L40S hardware. All 100 compound operators are verified for compilation and numerical correctness.

## Hardware and execution engine
- **Target hardware.** NVIDIA L40S (Ada Lovelace, 48 GB VRAM).
- **Evaluation framework.** Modal cloud GPU cluster (`gpu=L40S`, `num_gpu_devices=8`).
- **Generation harness.** 3-tier verification protocol via `verify_kernel.py`.
- **Kernel language.** OpenAI Triton (FP32).

## Key metrics
- **Compilation rate.** 100.0% (100 of 100 operators compiled).
- **Static AST pass rate.** 100.0% (100 of 100 passed static compliance checks).
- **Functional correctness rate.** 100.0% (100 of 100 passed multi-trial verification on Modal L40S).
- **Fast_0.0.** 1.00 (100 of 100 correct).
- **Fast_1.0.** 0.51 (51 of 100 strictly faster than PyTorch eager execution).
- **Geometric mean speedup.** 1.1816x across profiled samples.
- **Problem 25 bottleneck recovery.** Improved from 0.14x (116.00 ms) to 1.01x (15.50 ms).
- **Problem 66 alignment.** Synchronized Dropout RNG seeds for exact numerical parity.

## Included artifacts
- **Verified solutions.** 100 verified implementations in [`solutions/level2/`](../../solutions/level2).
- **Verification receipts.** 31 structured receipts in [`docs/benchmarks/receipts/`](receipts/).
- **Triage summary.** Complete triage ledger in [`docs/benchmarks/turn5_triage.json`](turn5_triage.json).
- **Benchmark report.** Comprehensive metrics in [`docs/benchmarks/SUBMISSION_REPORT.md`](SUBMISSION_REPORT.md).
- **Decision log.** Hypothesis trail in [`docs/benchmarks/decisions.tsv`](decisions.tsv).
- **Hillclimb report.** Milestone trajectory in [`docs/benchmarks/hillclimb_report.md`](hillclimb_report.md).
- **Local reminder tooling.** Shell and Python heartbeats in [`scripts/remind.sh`](../../scripts/remind.sh) and [`scripts/remind_subagents.py`](../../scripts/remind_subagents.py).

---

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
