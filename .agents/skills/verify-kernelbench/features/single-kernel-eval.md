# Single kernel evaluation

Single kernel evaluation validates that a candidate GPU kernel compiles, produces numerically accurate outputs matching the PyTorch reference, and measures wall-clock speedup on a target hardware device.

## Sub-features

- `eval-static-gate` verifies no prohibited patterns exist before launching cloud compute via static regex checks and AST linting (`lint_kernel_ast`).
- `eval-cloud-compile` compiles the custom kernel on the target cloud GPU architecture.
- `eval-correctness` verifies numerical equivalence across randomized input trials within dtype tolerance.
- `eval-speedup` compares wall-clock execution time against eager PyTorch and torch.compile.
- `eval-diagnostics` surfaces structured diagnostic breakdowns for compilation failures, runtime exceptions with tracebacks, and numerical discrepancies.

## How to get to it (user POV)

- Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --level <level> --problem-id <id> --kernel <path> --use-cached-baseline`.
- Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --lint-only --level <level> --problem-id <id> --kernel <path>` for immediate zero-compute linting.
- Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --quick --level <level> --problem-id <id> --kernel <path>` for fast 1-trial functional testing without baseline profiling.
- Run `uv run python scripts/run_and_check.py ref_origin=kernelbench dataset_src=local level=<level> problem_id=<id> kernel_src_path=<path> eval_mode=modal gpu=L40S backend=triton`.

## Driving it with verify-kernelbench

Preconditions:
- Environment doctor reports all checks passing (`uv run python .agents/skills/verify-kernelbench/scripts/doctor.py`).
- Candidate kernel file exists and defines class `ModelNew`.
- Modal authentication is active.

- **Check static compliance.** Run `uv run python -c "from kernelbench.kernel_static_checker import validate_kernel_static; assert validate_kernel_static(open(\"<path>\").read(), backend=\"<backend>\")[0]"`. Result prints without assertion error.
- **Run local AST & static lint.** Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --lint-only --level <level> --problem-id <id> --kernel <path>`. Exits with code 0 on pass or code 1 with specific AST/static violation messages.
- **Run quick functional test.** Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --quick --level <level> --problem-id <id> --kernel <path>`. Validates compilation and 1 trial correctness without timing overhead.
- **Run full single kernel verification.** Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --level <level> --problem-id <id> --kernel <path> --gpu L40S --backend triton --num-correct-trials 5 --num-perf-trials 50 --use-cached-baseline --json-out /tmp/verify_<id>.json`. Exit code is 0 and output reports `Compiled: True` and `Correctness: True`.
- **Verify structured output.** Inspect `/tmp/verify_<id>.json`. JSON contains a uniform evidence contract across all modes (`compiled`, `correctness`, `kernel_time_ms`, `ref_eager_time_ms`, `speedup_vs_eager`, `metadata`, `error`, `max_difference`, and `avg_difference`).

## Diagnostic Output Fields

On failures, detailed diagnostic blocks are printed and recorded in structured JSON:
- **Compilation Failure**: Prints `--- COMPILATION FAILURE ---` with `compilation_error_name` and `compilation_error`.
- **Runtime Exception**: Prints `--- RUNTIME EXCEPTION ---` with `runtime_error_name`, `runtime_error`, and full `runtime_error_traceback`.
- **Numerical Mismatch**: Prints `--- NUMERICAL MISMATCH ---` with `correctness_trials`, `correctness_issue`, `max_difference`, and `avg_difference`.

## Gotchas

- Calling PyTorch operations inside `ModelNew` that are compute layers (e.g. `nn.Conv2d`) is flagged as bypass by static checker if not allowed for that level.
- Ensure input shapes and tensor contiguity (`x.contiguous()`) are respected in kernel wrappers.
- For Triton kernels, decorators must use `@triton.jit` or `@triton.autotune`.
- Neither `tl.tanh` nor `tl.math.tanh` exist in Triton on Modal; module aliases (`tlm.tanh`) and direct imports (`from triton.language import tanh`) are rejected. Always compute tanh via `2.0 * tl.sigmoid(2.0 * x) - 1.0`.
- `tl.pow` is not supported in `triton.language`; use `x * x` or `**`.
- GELU approximation: Standard Triton tanh-based GELU yields ~4.7e-4 error vs PyTorch exact GELU; use Abramowitz & Stegun 7.1.26 rational Chebyshev polynomial erf approximation for < 2.5e-7 error.
- Shared memory allocations must not exceed 100 KB (101,376 bytes) on NVIDIA L40S GPUs. Size block tiles $\le 64 \times 64$ and `num_stages \le 2` accordingly.
- Loop unrolling PTX compiler hang: Avoid large nested loops or multi-stage reductions in Triton kernels; keep heavy normalizations in PyTorch/cuDNN and fuse elementwise epilogues.
- Reduction accumulator poisoning: Initialize with `-float('inf')` for `tl.max` and `float('inf')` for `tl.min`.
- Loop-carried variable types: Initialize tensor accumulators in `tl.range` loops with `tl.zeros([BLOCK_SIZE], dtype=tl.float32)`, not scalar `0.0`.
- Kernel launches & argument bounds: Never pass launch arguments both positionally and by keyword, never pass more positional arguments than the kernel defines, and never invoke `@triton.jit` functions directly from host Python without launch grid brackets `[grid](...)`.
- Nested functions: Helper functions defined inside `@triton.jit` kernels (including inside `if` or loop blocks) cannot be compiled by Triton; hoist helper functions to module level decorated with `@triton.jit`.
- Exponentials must be clamped (`tl.clamp(x, -88.0, 88.0)`) to avoid Inf/NaN errors.
