# Static checker

The static checker scans candidate kernel source code using regex patterns to catch banned reward hacking behavior, fallback bypasses, and unallowed PyTorch layer wrapping.

## Sub-features

- `check-try-except` catches try-except fallback wrappers that mask broken custom kernels.
- `check-pass-statement` catches empty pass statements in inherited classes.
- `check-pytorch-layers` blocks unauthorized use of `torch.nn` compute layers.
- `check-stream-divergence` detects non-default CUDA streams used to evade timing synchronization.

## How to get to it (user POV)

- Call `from kernelbench.kernel_static_checker import validate_kernel_static`.
- Run `verify_kernel.py` which executes static validation as step 1.

## Driving it with verify-kernelbench

Preconditions:
- `kernelbench` package is installed in virtual environment.

- **Run direct python check.** Run `uv run python -c "from kernelbench.kernel_static_checker import validate_kernel_static; ok, errs, warns = validate_kernel_static(open("<path>").read(), backend="triton"); print(ok, errs); assert ok"`. Output is `True []`.
- **Catch bypass.** Check a file containing `try: ... except: pass`. Validator returns `valid=False` with error naming try-except block.

## Gotchas

- Comments containing keywords can sometimes trigger false matches if not stripped; `validate_kernel_static` strips comments automatically.
- Different backends (`cuda`, `triton`, `tilelang`) enforce different mandatory decorators or includes.
