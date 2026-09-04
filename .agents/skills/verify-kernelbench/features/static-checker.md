# Static checker

The static checker scans candidate kernel source code using regex patterns and AST parsing to catch banned reward hacking behavior, fallback bypasses, unallowed PyTorch layer wrapping, and illegal Triton kernel semantics.

## Sub-features

- `check-try-except` catches try-except fallback wrappers that mask broken custom kernels.
- `check-pass-statement` catches empty pass statements in inherited classes.
- `check-pytorch-layers` blocks unauthorized use of `torch.nn` compute layers.
- `check-stream-divergence` detects non-default CUDA streams used to evade timing synchronization.
- `check-ast-model-new` verifies class `ModelNew` is explicitly defined.
- `check-ast-tanh` catches nonexistent `tl.tanh` or `triton.language.tanh` attribute calls.
- `check-ast-math-tanh` catches nonexistent `tl.math.tanh` or `triton.language.math.tanh` attribute calls.
- `check-ast-pow` catches nonexistent `tl.pow` or `triton.language.pow` attribute calls.
- `check-ast-duplicate-args` catches parameters passed both positionally and as keyword arguments to kernel launches.
- `check-ast-nested-jit` catches nested functions defined inside functions decorated with `@triton.jit`.
- `check-ast-host-jit-call` catches direct host invocations of `@triton.jit` functions without launch grids `[grid](...)`.
- `check-ast-control-flow` flags unsupported `continue` or `break` statements inside `@triton.jit` / `@triton.autotune` functions.
- `check-ast-unclamped-exp` warns when `tl.exp` arguments are not protected with `clamp` or `minimum`.

## How to get to it (user POV)

- Call `from kernelbench.kernel_static_checker import validate_kernel_static`.
- Call `from verify_kernel import lint_kernel_ast`.
- Run `verify_kernel.py --lint-only` which executes both regex static checks and AST linting locally without cloud execution.

## Driving it with verify-kernelbench

Preconditions:
- `kernelbench` package is installed in virtual environment.

- **Run direct python check.** Run `uv run python -c "from kernelbench.kernel_static_checker import validate_kernel_static; ok, errs, warns = validate_kernel_static(open(\"<path>\").read(), backend=\"triton\"); print(ok, errs); assert ok"`. Output is `True []`.
- **Run AST and static lint.** Run `uv run python .agents/skills/verify-kernelbench/scripts/verify_kernel.py --lint-only --level <level> --problem-id <id> --kernel <path>`. Prints `[PASS] Static and AST checks passed cleanly.` and exits with 0 on clean kernels.
- **Catch bypass.** Check a file containing `try: ... except: pass`. Validator returns `valid=False` with error naming try-except block.
- **Catch Triton AST pitfalls.** Run with `--lint-only` against kernels containing `tl.tanh`, `tl.math.tanh`, `tl.pow`, duplicate launch arguments, host JIT calls, or `continue`/`break` inside `@triton.jit`. Returns exit code 1 and prints specific error recommendations.

## Gotchas

- Comments containing keywords can sometimes trigger false matches if not stripped; `validate_kernel_static` strips comments automatically.
- Different backends (`cuda`, `triton`, `tilelang`) enforce different mandatory decorators or includes.
- Both `tl.tanh` and `tl.math.tanh` are flagged as nonexistent errors; use the sigmoid identity `2.0 * tl.sigmoid(2.0 * x) - 1.0`.
