# KernelBench Level 2 Triton Evaluation (v1.0.0-level2-l40s)

This release delivers empirical benchmark evaluation results and 39 verified Triton solutions for KernelBench Level 2 across all 100 compound operators.

## Hardware and Execution Engine
- **Target hardware.** NVIDIA L40S (Ada Lovelace, 48 GB VRAM).
- **Evaluation framework.** Modal cloud GPU cluster (`gpu=L40S`, `num_gpu_devices=8`).
- **Generation agent.** Gemini 3.8 Flash with multi-model fallback.
- **Kernel language.** OpenAI Triton (FP32).

## Key Metrics
- **Compilation rate.** 100.0% (100 of 100 compiled cleanly).
- **Functional correctness rate.** 39.0% (39 of 100 passed all randomized numerical trials).
- **Fast_1 rate.** 30.0% (30 of 100 strictly faster than PyTorch eager execution).
- **Geometric mean speedup.** 1.1580x across correct samples.

## Included Artifacts
- **Verified solutions.** 39 memory-coalesced, fused Triton kernels in `solutions/level2/`.
- **Benchmark report.** Full metric distributions in `docs/benchmarks/SUBMISSION_REPORT.md`.
- **Hardware baseline.** Recorded PyTorch eager baseline in `results/timing/L40S_Modal/baseline_time_torch.json`.
- **Evaluation data.** Raw per-kernel execution logs attached in `eval_results.json`.
