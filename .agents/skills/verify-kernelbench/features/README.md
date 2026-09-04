# KernelBench Verification Map

This directory is the maintained source for verifying GPU kernels and benchmark evaluations in KernelBench. Read this index before running verification, then use the matching feature file as the recipe.

## Baseline preconditions

- Active Python 3.10 virtual environment with dependencies synchronized via `uv sync`.
- Modal CLI installed and authenticated with active profile (`uv run modal profile current`).
- Dataset directory present at `KernelBench/` with `level1`, `level2`, and `level3` problems.
- Run `uv run python .agents/skills/verify-kernelbench/scripts/doctor.py` and require all checks to pass before starting work.
- For concurrent cloud agents, assign a unique `AGENT_ID` or timestamp to avoid colliding on output files or run directories.

## Driving conventions

- Always run static validation (`validate_kernel_static`) before dispatching to cloud GPUs.
- Prefer `.agents/skills/verify-kernelbench/scripts/verify_kernel.py` for single kernel verification.
- Specify exact backend (`triton`, `cuda`, `tilelang`, `cute`) matching the kernel implementation.
- Capture structured JSON output via `--json-out <path>` for machine verification.
- Never write output files to shared directories without agent-specific prefixes.

## Proof and skip reporting

- A kernel is verified only when:
  1. Static anti-hacking checks pass with zero errors.
  2. Compilation succeeds on the target GPU architecture (`compiled == True`).
  3. Numerical correctness succeeds across all randomized input trials (`correctness == True`).
  4. Runtime measurement collects at least 10 trials using `cuda_event` synchronization.
- Record problem ID, level, runtime, reference eager runtime, and speedup ratio in evidence logs.

## Feature entry contract

Each feature file describes:
1. `Sub-features`: component verification capabilities.
2. `How to get to it (user POV)`: user-facing or agent-facing entry points.
3. `Driving it with <harness>`: step-by-step verification commands with preconditions and observable outputs.
4. `Gotchas`: failure modes and traps that invalidate runs.

## Features

- [Single kernel evaluation](./single-kernel-eval.md) covers verifying a single candidate kernel against reference.
- [Static checker](./static-checker.md) covers pattern validation to prevent banned reward hacks.
- [Batch evaluation](./batch-eval.md) covers running batch evaluation across problem sets on Modal.
- [Benchmark scoring](./benchmark-scoring.md) covers computing fast_p scores and geometric mean speedups.
