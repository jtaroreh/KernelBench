# Single kernel evaluation

Single kernel evaluation validates that a candidate GPU kernel compiles, produces numerically accurate outputs matching the PyTorch reference, and measures wall-clock speedup on a target hardware device.

## Sub-features

- `eval-static-gate` verifies no prohibited patterns exist before launching cloud compute.
- `eval-cloud-compile` compiles the custom kernel on the target cloud GPU architecture.
- `eval-correctness` verifies numerical equivalence across randomized input trials within dtype tolerance.
- `eval-speedup` compares wall-clock execution time against eager PyTorch and torch.compile.

## How to get to it (user POV)

- Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --level <level> --problem-id <id> --kernel <path>`.
- Run `uv run python scripts/run_and_check.py ref_origin=kernelbench dataset_src=local level=<level> problem_id=<id> kernel_src_path=<path> eval_mode=modal gpu=L40S backend=triton`.

## Driving it with verify-kernelbench

Preconditions:
- Environment doctor reports all checks passing.
- Candidate kernel file exists and defines class `ModelNew`.
- Modal authentication is active.

- **Check static compliance.** Run `uv run python -c "from kernelbench.kernel_static_checker import validate_kernel_static; assert validate_kernel_static(open("<path>").read(), backend="<backend>")[0]"`. Result prints without assertion error.
- **Run automated single kernel verification.** Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --level <level> --problem-id <id> --kernel <path> --gpu L40S --backend triton --json-out /tmp/verify_<id>.json`. Exit code is 0 and output reports `Compiled: True` and `Correctness: True`.
- **Verify structured output.** Inspect `/tmp/verify_<id>.json`. JSON contains `compiled: true`, `correctness: true`, `kernel_time_ms`, and `speedup_vs_eager > 0`.

## Gotchas

- Calling PyTorch operations inside `ModelNew` that are compute layers (e.g. `nn.Conv2d`) is flagged as bypass by static checker if not allowed for that level.
- Ensure input shapes and tensor contiguity (`x.contiguous()`) are respected in kernel wrappers.
- For Triton kernels, decorators must use `@triton.jit` or `@triton.autotune`.
