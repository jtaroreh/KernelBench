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

#### Command-Line Flags
- `--lint-only`: Run local static and Triton AST checks only without cloud compute.
- `--quick`: Quick mode: 1 correctness trial, 0 perf trials, skip baseline timing.
- `--use-cached-baseline`: Use cached eager baseline from `results/timing/` if available (default: enabled).
- `--no-cached-baseline`: Force remote baseline timing on Modal even if cached baseline exists.

### 2. Diagnostic Triage Workflow
Follow this 3-tier escalation workflow when implementing and verifying kernels:
1. **Local AST & Static Lint (`--lint-only`)**: Fast zero-cost local check validating syntax, anti-hacking rules, `ModelNew` class existence, unsupported Triton syntax (`tl.tanh`, `continue`/`break`), and unclamped `tl.exp()`.
   ```bash
   uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --lint-only --level <level> --problem-id <id> --kernel <path>
   ```
2. **Quick 1-Trial Functional Test (`--quick`)**: Fast cloud validation executing 1 correctness trial and 0 perf trials, bypassing baseline timing on Modal.
   ```bash
   uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --quick --level <level> --problem-id <id> --kernel <path>
   ```
3. **Full Verification (`--num-correct-trials 5 --num-perf-trials 50`)**: Complete verification verifying correctness over 5 randomized trials and profiling 50 timing iterations against PyTorch eager.
   ```bash
   uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --level <level> --problem-id <id> --kernel <path> --num-correct-trials 5 --num-perf-trials 50 --json-out <evidence_json_path>
   ```

### 3. Static Anti-Hacking Check
Verify that a kernel contains no prohibited bypass patterns (try-except fallbacks, empty pass statements, non-default CUDA streams, input mutation, or reference caching):
```bash
uv run python -c "from kernelbench.kernel_static_checker import validate_kernel_static; valid, errors, warnings = validate_kernel_static(open(\"<path>\").read(), backend=\"triton\"); print(valid, errors); assert valid"
```

### 4. Batch Evaluation
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

### 5. Benchmark Scoring and Analysis
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

## Triton Implementation Cheatsheet & Pitfall Guard

- **Tanh**: `tl.tanh` does not exist in Triton. Use `tl.math.tanh(x)` or custom rational approximation `(tl.exp(2*x)-1)/(tl.exp(2*x)+1)`.
- **Mish & GELU**: Exponentials overflow easily in float32. Always clamp inputs to `tl.exp`: use `tl.clamp(x, -88.0, 88.0)`.
- **Flow Control**: `continue` and `break` statements are unsupported inside `@triton.jit` kernels. Use boolean masks and `tl.where` instead.
- **Initialization Order**: Match the reference `Model.__init__` attribute creation sequence identically so PyTorch's RNG initialization yields bitwise-identical weights.

## Evidence

A valid proof must capture the following:
1. **Static Validation**: Output of `validate_kernel_static` and `lint_kernel_ast` showing zero errors.
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
- `.agents/skills/verify-kernelbench/scripts/verify_kernel.py`: Automated end-to-end kernel validator with static checking, AST linting, cloud compilation, correctness checks, and speedup profiling.
