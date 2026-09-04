---
name: verify-kernelbench
description: "Verify KernelBench GPU kernels for compilation, numerical correctness, static compliance, and speedup on Modal cloud GPUs or local hardware. Use when proving kernel implementations, validating solutions, running doctor checks, or testing benchmark runs."
---

# Verify KernelBench

This skill provides the verification harness and standards for KernelBench. It enables autonomous cloud agents to reliably verify GPU kernels for numerical correctness, compilation integrity, static safety, and performance speedup on cloud NVIDIA GPUs via Modal.

## Launch

KernelBench does not run as a persistent server. Verification uses on-demand execution on local Python 3.10 and remote Modal GPU containers.

To prepare the environment for verification:
1. Ensure dependencies are synced:
   ```bash
   uv sync
   ```
2. Confirm Modal authentication is active:
   ```bash
   uv run modal profile current
   ```
   Ready when output prints the active profile name (e.g. `hypixelk`).

## Doctor

Before attempting any kernel evaluation, run the doctor script to confirm environment integrity:
```bash
uv run python .agents/skills/verify-kernelbench/scripts/doctor.py
```
Ready when the script outputs `[SUCCESS] Environment is fully ready for kernel verification.` and exits with code 0.
The doctor verifies:
- Python 3.10 runtime.
- Core packages (`torch`, `modal`, `pydra`, `numpy`, `kernelbench`).
- Modal credentials and cloud connection.
- Dataset files across Level 1, 2, and 3.
- Static anti-hacking validator functionality.

If the doctor fails, do not proceed with evaluations. Fix the reported issues first.

## Doctor Quick-Fixes
- If `modal profile current` fails: run `modal token new` or configure `~/.modal.toml`.
- If packages fail to import: run `uv sync --extra dev`.

## Drive

### 1. Single Kernel Verification (Recommended)
Use the unified verification helper to validate static safety, compile, check numerical correctness, and measure speedup against PyTorch eager:
```bash
uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py \
    --level <level> \
    --problem-id <problem_id> \
    --kernel <path_to_kernel.py> \
    --gpu L40S \
    --backend triton \
    --num-correct-trials 5 \
    --num-perf-trials 50 \
    --json-out <evidence_json_path>
```
Output reports `Compiled: True`, `Correctness: True`, and `Speedup vs Eager: <N>x`. Exits 0 on pass, 1 on failure.

### 2. Static Anti-Hacking Check
Verify that a kernel contains no prohibited bypass patterns (try-except fallbacks, empty pass statements, non-default CUDA streams, input mutation, or reference caching):
```bash
uv run python -c "from kernelbench.kernel_static_checker import validate_kernel_static; valid, errors, warnings = validate_kernel_static(open(\"<path>\").read(), backend=\"triton\"); print(valid, errors); assert valid"
```

### 3. Batch Evaluation
Run batch evaluation across multiple problems on Modal:
```bash
uv run python scripts/eval_from_generations.py \
    run_name=<unique_agent_run_name> \
    dataset_src=local \
    level=<level> \
    eval_mode=modal \
    gpu=L40S \
    backend=triton \
    timeout=180
```

### 4. Benchmark Scoring and Analysis
Aggregate results and compute `fast_p` and geometric mean speedup against baseline timings:
```bash
uv run python scripts/benchmark_eval_analysis.py \
    run_name=<unique_agent_run_name> \
    level=<level> \
    hardware=L40S_Modal \
    baseline=baseline_time_torch \
    baseline_file=results/timing/L40S_Modal/baseline_time_torch.json \
    eval_results_dir=runs
```

## Evidence

A valid proof must capture the following:
1. **Static Validation**: Output of `validate_kernel_static` showing zero errors.
2. **Execution Results**: JSON output from `verify_kernel.py` containing:
   - `compiled: true`
   - `correctness: true` across 5 randomized trials
   - `kernel_time_ms`: positive floating point runtime
   - `speedup_vs_eager`: speedup ratio greater than 1.0x for performance claims
3. **Log Artifacts**: Store verification evidence in `runs/<unique_agent_id>/` or pass `--json-out /tmp/verify_<agent_id>_<problem_id>.json`.

## Cleanup

- Modal containers automatically terminate and spin down after execution (idle timeout is 5 seconds).
- Temporary files created during compilation are scoped to unique process IDs.
- For local builds, remove cache directories if needed:
  ```bash
  rm -rf test_build/
  ```
- Evidence JSON files should be retained in the agent run directory as receipts.

## Helpers

The skill ships with two executable scripts:
- `.agents/skills/verify-kernelbench/scripts/doctor.py`: Environment diagnostic tool.
- `.agents/skills/verify-kernelbench/scripts/verify_kernel.py`: Automated end-to-end kernel validator with static checking, cloud compilation, correctness checks, and speedup profiling.
