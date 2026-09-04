"""
Unit tests for the KernelBench verification skill.

Validates:
1. AST linting rules in verify_kernel.py (disallowed tanh, pow, nested JIT, continue/break, host JIT calls, duplicate args, unclamped exp).
2. Cached baseline entry resolution across naming formats.
3. Environment doctor diagnostic checks.
4. Static validation across verified kernel solutions.

Run with:
    uv run pytest .agents/skills/verify-kernelbench/tests/test_verify_kernel.py -v
"""

import ast
import os
import sys
import pytest

# Ensure scripts directory is importable
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from verify_kernel import lint_kernel_ast, _find_baseline_entry
from doctor import (
    check_python_version,
    check_dependencies,
    check_helpers,
    check_static_checker,
    check_baseline_cache,
)


# ============================================================================
# 1. AST Linting: ModelNew Presence
# ============================================================================

def test_ast_missing_model_new():
    code = """
import torch
import torch.nn as nn

class OtherModel(nn.Module):
    def forward(self, x):
        return x * 2
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("ModelNew class definition not found" in e for e in errors)


def test_ast_present_model_new():
    code = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return x * 2
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert not any("ModelNew class definition not found" in e for e in errors)


# ============================================================================
# 2. AST Linting: Nonexistent Tanh Functions
# ============================================================================

@pytest.mark.parametrize("tanh_call", [
    "tl.tanh(x)",
    "triton.language.tanh(x)",
    "tl.math.tanh(x)",
    "triton.language.math.tanh(x)",
])
def test_ast_disallowed_tanh_calls(tanh_call):
    code = f"""
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def kernel(x_ptr, y_ptr, N):
    x = tl.load(x_ptr)
    y = {tanh_call}
    tl.store(y_ptr, y)

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Neither tl.tanh nor tl.math.tanh exist" in e for e in errors), f"Failed to catch: {tanh_call}"


def test_ast_disallowed_tanh_from_import():
    code = """
import triton
from triton.language import tanh
import torch.nn as nn

@triton.jit
def kernel(x_ptr, y_ptr, N):
    x = tl.load(x_ptr)
    y = tanh(x)
    tl.store(y_ptr, y)

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Neither tl.tanh nor tl.math.tanh exist" in e for e in errors)


def test_ast_disallowed_tanh_aliased_import():
    code = """
import triton
import triton.language.math as tlm
import torch.nn as nn

@triton.jit
def kernel(x_ptr, y_ptr, N):
    x = tl.load(x_ptr)
    y = tlm.tanh(x)
    tl.store(y_ptr, y)

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Neither tl.tanh nor tl.math.tanh exist" in e for e in errors)


def test_ast_allowed_sigmoid_tanh_identity():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def kernel(x_ptr, y_ptr, N):
    x = tl.load(x_ptr)
    # Correct sigmoid identity for tanh
    y = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    tl.store(y_ptr, y)

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert not any("tanh" in e.lower() for e in errors)


# ============================================================================
# 3. AST Linting: Nonexistent Power Function
# ============================================================================

@pytest.mark.parametrize("pow_call", [
    "tl.pow(x, 2.0)",
    "triton.language.pow(x, 2.0)",
])
def test_ast_disallowed_pow_calls(pow_call):
    code = f"""
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def kernel(x_ptr, y_ptr, N):
    x = tl.load(x_ptr)
    y = {pow_call}
    tl.store(y_ptr, y)

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("tl.pow does not exist in triton.language" in e for e in errors), f"Failed to catch: {pow_call}"


def test_ast_disallowed_pow_from_import():
    code = """
import triton
from triton.language import pow
import torch.nn as nn

@triton.jit
def kernel(x_ptr, y_ptr, N):
    x = tl.load(x_ptr)
    y = pow(x, 2)
    tl.store(y_ptr, y)

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("tl.pow does not exist in triton.language" in e for e in errors)


# ============================================================================
# 4. AST Linting: Nested Functions inside @triton.jit
# ============================================================================

def test_ast_nested_jit_function():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def outer_kernel(x_ptr):
    def helper(v):
        return v + 1.0
    val = helper(tl.load(x_ptr))

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Nested function 'helper' inside @triton.jit" in e for e in errors)


def test_ast_nested_jit_function_in_block():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def outer_kernel(x_ptr):
    if True:
        def block_helper(v):
            return v + 2.0
        val = block_helper(tl.load(x_ptr))

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Nested function 'block_helper' inside @triton.jit" in e for e in errors)


# ============================================================================
# 5. AST Linting: Unsupported Control Flow (continue / break)
# ============================================================================

@pytest.mark.parametrize("bad_stmt", ["continue", "break"])
def test_ast_control_flow(bad_stmt):
    code = f"""
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def kernel(x_ptr):
    for i in range(10):
        {bad_stmt}

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("continue/break statements are unsupported" in e for e in errors)


# ============================================================================
# 6. AST Linting: Clamped vs Unclamped tl.exp()
# ============================================================================

def test_ast_unclamped_exp_warning():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def kernel(x_ptr, y_ptr):
    x = tl.load(x_ptr)
    y = tl.exp(x)
    tl.store(y_ptr, y)

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, warnings = lint_kernel_ast(code, backend="triton")
    assert any("Unclamped tl.exp() detected" in w for w in warnings)


def test_ast_clamped_exp_no_warning():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def kernel(x_ptr, y_ptr):
    x = tl.load(x_ptr)
    y = tl.exp(tl.clamp(x, -88.0, 88.0))
    tl.store(y_ptr, y)

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    errors, warnings = lint_kernel_ast(code, backend="triton")
    assert not any("Unclamped tl.exp()" in w for w in warnings)


# ============================================================================
# 7. AST Linting: Direct Host Invocations of @triton.jit Functions
# ============================================================================

def test_ast_host_jit_invocation():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def my_triton_kernel(x_ptr, y_ptr):
    pass

class ModelNew(nn.Module):
    def forward(self, x):
        # Direct call without [grid] launch brackets
        my_triton_kernel(x.data_ptr(), x.data_ptr())
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Direct host invocation of @triton.jit function 'my_triton_kernel'" in e for e in errors)


def test_ast_host_jit_invocation_qualified():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def my_triton_kernel(x_ptr, y_ptr):
    pass

class ModelNew(nn.Module):
    def forward(self, x):
        self.my_triton_kernel(x.data_ptr(), x.data_ptr())
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Direct host invocation of @triton.jit function 'my_triton_kernel'" in e for e in errors)


def test_ast_proper_grid_launch():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def my_triton_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pass

class ModelNew(nn.Module):
    def forward(self, x):
        grid = lambda meta: (1,)
        my_triton_kernel[grid](x.data_ptr(), x.data_ptr(), BLOCK_SIZE=64)
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert not any("Direct host invocation" in e for e in errors)


# ============================================================================
# 8. AST Linting: Duplicate & Invalid Launch Arguments
# ============================================================================

def test_ast_duplicate_keyword_args():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def my_kernel(x_ptr, BLOCK: tl.constexpr):
    pass

class ModelNew(nn.Module):
    def forward(self, x):
        grid = (1,)
        my_kernel[grid](x, BLOCK=64, BLOCK=128)
        return x
"""
    try:
        errors, _ = lint_kernel_ast(code, backend="triton")
        assert any("Duplicate keyword arguments" in e for e in errors)
    except SyntaxError:
        pass


def test_ast_duplicate_positional_and_keyword():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def my_kernel(x_ptr, y_ptr, N):
    pass

class ModelNew(nn.Module):
    def forward(self, x):
        grid = (1,)
        my_kernel[grid](x.data_ptr(), x.data_ptr(), 1024, x_ptr=x.data_ptr())
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Duplicate argument 'x_ptr' passed both positionally and as keyword" in e for e in errors)


def test_ast_duplicate_args_qualified_launch():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def my_kernel(x_ptr, y_ptr, N):
    pass

class ModelNew(nn.Module):
    def forward(self, x):
        grid = (1,)
        self.my_kernel[grid](x.data_ptr(), x.data_ptr(), 1024, x_ptr=x.data_ptr())
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Duplicate argument 'x_ptr' passed both positionally and as keyword" in e for e in errors)


def test_ast_too_many_positional_args():
    code = """
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def my_kernel(x_ptr, y_ptr):
    pass

class ModelNew(nn.Module):
    def forward(self, x):
        grid = (1,)
        my_kernel[grid](x.data_ptr(), x.data_ptr(), 1024, 2048)
        return x
"""
    errors, _ = lint_kernel_ast(code, backend="triton")
    assert any("Too many positional arguments (4) passed to kernel 'my_kernel'" in e for e in errors)


# ============================================================================
# 9. Baseline Timing Entry Resolution
# ============================================================================

class DummyProblem:
    def __init__(self, name, path, problem_id):
        self.name = name
        self.path = path
        self.problem_id = problem_id


def test_find_baseline_entry_by_name():
    baseline_data = {
        "level2": {
            "29_Matmul_Mish_Mish.py": {"mean": 3.56, "std": 0.05},
            "40_Matmul_Scaling_ResidualAdd.py": {"mean": 17.30, "std": 0.12},
        }
    }
    prob = DummyProblem(name="29_Matmul_Mish_Mish.py", path="level2/29_Matmul_Mish_Mish.py", problem_id=29)
    val = _find_baseline_entry(baseline_data, level=2, problem=prob)
    assert val == 3.56


def test_find_baseline_entry_without_extension():
    baseline_data = {
        "level2": {
            "29_Matmul_Mish_Mish.py": {"mean": 3.56},
        }
    }
    prob = DummyProblem(name="29_Matmul_Mish_Mish", path="level2/29_Matmul_Mish_Mish", problem_id=29)
    val = _find_baseline_entry(baseline_data, level=2, problem=prob)
    assert val == 3.56


def test_find_baseline_entry_by_problem_id():
    baseline_data = {
        "level2": {
            "29": {"mean": 4.15},
        }
    }
    prob = DummyProblem(name="Unknown", path="level2/Unknown.py", problem_id=29)
    val = _find_baseline_entry(baseline_data, level=2, problem=prob)
    assert val == 4.15


def test_find_baseline_entry_missing():
    baseline_data = {
        "level2": {
            "10_SomeOtherProblem.py": {"mean": 12.0},
        }
    }
    prob = DummyProblem(name="99_NotFound.py", path="level2/99_NotFound.py", problem_id=99)
    val = _find_baseline_entry(baseline_data, level=2, problem=prob)
    assert val is None


# ============================================================================
# 10. Environment Doctor Diagnostics
# ============================================================================

def test_doctor_diagnostics():
    ok_py, msg_py = check_python_version()
    assert ok_py, f"Python version check failed: {msg_py}"

    ok_dep, msg_dep = check_dependencies()
    assert ok_dep, f"Dependencies check failed: {msg_dep}"

    ok_hlp, msg_hlp = check_helpers()
    assert ok_hlp, f"Helper scripts check failed: {msg_hlp}"

    ok_stc, msg_stc = check_static_checker()
    assert ok_stc, f"Static checker diagnostic check failed: {msg_stc}"

    ok_bsl, msg_bsl = check_baseline_cache()
    assert ok_bsl, f"Baseline cache check failed: {msg_bsl}"


# ============================================================================
# 11. Real Kernel AST Check (Problem 29, 40, 87, etc.)
# ============================================================================

def test_real_kernels_ast_clean():
    from verify_kernel import _find_git_repo_root
    git_root = _find_git_repo_root(os.path.dirname(__file__))
    candidates = [
        os.path.join(git_root, ".turn4_worktree/solutions/level2/29_Matmul_Mish_Mish.py"),
        os.path.join(git_root, ".turn4_worktree/solutions/level2/40_Matmul_Scaling_ResidualAdd.py"),
        os.path.join(git_root, ".turn4_worktree/solutions/level2/87_Conv2d_Subtract_Subtract_Mish.py"),
        os.path.join(git_root, ".turn4_worktree/solutions/level2/26_ConvTranspose3d_Add_HardSwish.py"),
        os.path.join(git_root, ".turn4_worktree/solutions/level2/34_ConvTranspose3d_LayerNorm_GELU_Scaling.py"),
        os.path.abspath("solutions/level2/29_Matmul_Mish_Mish.py"),
        os.path.abspath("solutions/level2/40_Matmul_Scaling_ResidualAdd.py"),
        os.path.abspath("solutions/level2/87_Conv2d_Subtract_Subtract_Mish.py"),
    ]
    tested = 0
    seen = set()
    for path in candidates:
        real_path = os.path.realpath(path)
        if os.path.exists(path) and real_path not in seen:
            seen.add(real_path)
            with open(path, "r") as f:
                src = f.read()
            errors, warnings = lint_kernel_ast(src, backend="triton")
            assert len(errors) == 0, f"Errors in {path}: {errors}"
            tested += 1
    assert tested >= 3, f"Expected to verify at least 3 distinct real kernel files, verified {tested}"


# ============================================================================
# 12. JSON Evidence Schema Consistency
# ============================================================================

def test_json_evidence_schema_contract(tmp_path):
    import subprocess
    import json

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts/verify_kernel.py"))
    out_json = str(tmp_path / "evidence.json")

    kernel_path = os.path.abspath(".turn4_worktree/solutions/level2/40_Matmul_Scaling_ResidualAdd.py")
    if not os.path.exists(kernel_path):
        kernel_path = os.path.abspath("solutions/level2/40_Matmul_Scaling_ResidualAdd.py")

    res = subprocess.run([
        sys.executable,
        script_path,
        "--lint-only",
        "--level", "2",
        "--problem-id", "40",
        "--kernel", kernel_path,
        "--json-out", out_json,
    ], capture_output=True, text=True)

    assert res.returncode == 0, f"Lint-only failed: {res.stderr}\n{res.stdout}"
    assert os.path.exists(out_json)

    with open(out_json, "r") as f:
        data = json.load(f)

    required_keys = ["kernel_time_ms", "ref_eager_time_ms", "speedup_vs_eager", "static_passed", "compiled", "correctness"]
    for k in required_keys:
        assert k in data, f"Required evidence schema key '{k}' missing from JSON: {data}"
