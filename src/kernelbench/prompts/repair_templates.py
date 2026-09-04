"""Repair prompt generation templates for Triton kernel iterative refinement."""

from __future__ import annotations

from typing import Any, Dict, Optional

TRITON_PITFALL_GUARDS = """### Triton Pitfall Guards & Implementation Constraints
1. **Tanh**: Neither `tl.tanh` nor `tl.math.tanh` exist in Modal's Triton runtime. You MUST compute tanh using the sigmoid identity: `2.0 * tl.sigmoid(2.0 * x) - 1.0`. Prohibit `tl.math.tanh` and `tl.tanh`.
2. **Exponentials & fp32 reductions**: float32 exponentials overflow easily. Always clamp inputs to `tl.exp`: use `tl.clamp(x, -88.0, 88.0)`. Accumulate reductions (sums, norms) in float32 before downcasting.
3. **No Nested Functions**: Nested function definitions inside `@triton.jit` kernels cause `UnsupportedLanguageConstruct`. Place all helper functions at module level outside the main kernel and decorate each helper with `@triton.jit`.
4. **Epilogue Fusion**: For complex operations (such as 3D convolutions or transposed 3D convolutions with multiple spatial axes), use PyTorch eager to compute the spatial convolution/matmul, then dispatch a fused Triton kernel to compute all subsequent bias additions, activations, scaling, and reductions in a single memory pass.
5. **Flow Control**: `continue` and `break` statements are unsupported inside `@triton.jit` kernels. Use boolean masks and `tl.where` instead.
6. **Initialization Order & Interface**: Match reference `Model.__init__` attribute creation order identically for PyTorch RNG bitwise weight parity. Retain `ModelNew` interface and forward arguments.
7. **Output Format**: Only output valid Python code enclosed in ```python ... ```."""


def format_error_section(diagnostic_info: Dict[str, Any]) -> str:
    """Format failure diagnostics into a clear diagnostic report."""
    parts = []
    category = diagnostic_info.get("failure_category", "other")
    parts.append(f"**Failure Category**: `{category}`")

    runtime_error_name = diagnostic_info.get("runtime_error_name")
    if runtime_error_name:
        parts.append(f"**Error Type**: `{runtime_error_name}`")

    runtime_error = diagnostic_info.get("runtime_error")
    if runtime_error:
        parts.append(f"**Error Message**:\n```\n{runtime_error}\n```")

    traceback = diagnostic_info.get("runtime_error_traceback")
    if traceback:
        parts.append(f"**Traceback**:\n```\n{traceback}\n```")

    max_diff = diagnostic_info.get("max_difference")
    avg_diff = diagnostic_info.get("avg_difference")
    issue = diagnostic_info.get("correctness_issue")

    mismatch_details = []
    if max_diff is not None:
        mismatch_details.append(f"- Max Difference: {max_diff}")
    if avg_diff is not None:
        mismatch_details.append(f"- Avg Difference: {avg_diff}")
    if issue:
        mismatch_details.append(f"- Correctness Issue: {issue}")
    if diagnostic_info.get("has_nan"):
        mismatch_details.append("- Numerical Issue: NaN / Inf overflow detected in kernel output.")

    if mismatch_details:
        parts.append("**Numerical / Correctness Details**:\n" + "\n".join(mismatch_details))

    return "\n\n".join(parts)


def generate_repair_prompt(
    problem_name: str,
    reference_code: str,
    previous_kernel: str,
    diagnostic_info: Dict[str, Any],
    target_model: str = "Gemini 3.8 Flash",
) -> str:
    """Generate a targeted repair prompt for a failing Triton kernel."""
    error_section = format_error_section(diagnostic_info)

    prompt = f"""You are an expert GPU kernel engineer tasked with repairing a failing Triton kernel for {target_model}.
Problem: {problem_name}

The previous implementation failed during compilation, execution, or numerical verification against the reference PyTorch implementation.
Your goal is to inspect the error, identify the root cause, and produce a fully functional, bitwise-accurate, high-performance Triton kernel implementation in `ModelNew`.

### Reference PyTorch Architecture
```python
{reference_code.strip()}
```

### Previous Failing Implementation
```python
{previous_kernel.strip()}
```

### Diagnostic Failure Information
{error_section}

{TRITON_PITFALL_GUARDS}

Please provide the corrected Python code defining `ModelNew` that fixes the error and matches the reference output.
Output only the Python code within ```python ... ``` code block."""
    return prompt.strip()
