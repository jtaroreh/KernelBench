#!/usr/bin/env python3
import argparse
import ast
import json
import os
import sys
import time


def _get_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        val = _get_attr_name(node.value)
        return f"{val}.{node.attr}" if val else node.attr
    return None


def _is_clamp_or_min(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func_name = _get_attr_name(node.func)
    if not func_name:
        return False
    base_name = func_name.split(".")[-1]
    return base_name in ("clamp", "minimum")


def lint_kernel_ast(kernel_src: str, backend: str = "triton") -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        tree = ast.parse(kernel_src)
    except SyntaxError as e:
        return ([f"AST parse error: {e}"], [])

    has_model_new = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            has_model_new = True
            break
    if not has_model_new:
        errors.append("ModelNew class definition not found")

    if backend == "triton":
        jit_funcs = {}
        triton_modules = {"tl", "triton.language", "triton.language.math", "tl.math"}
        disallowed_imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("triton.language", "triton.language.math"):
                        triton_modules.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in ("triton.language", "triton.language.math", "tl", "tl.math") or mod.startswith("triton.language."):
                    for alias in node.names:
                        local_name = alias.asname or alias.name
                        if alias.name == "tanh":
                            disallowed_imports.add((local_name, "Neither tl.tanh nor tl.math.tanh exist in Modal's Triton runtime; compute tanh via the sigmoid identity: 2.0 * tl.sigmoid(2.0 * x) - 1.0"))
                        elif alias.name == "pow":
                            disallowed_imports.add((local_name, "tl.pow does not exist in triton.language; compute powers via x * x or Python ** operator"))

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    dec_name = _get_attr_name(target)
                    if dec_name in ("triton.jit", "triton.autotune", "jit", "autotune", "tl.jit"):
                        params = [arg.arg for arg in node.args.args]
                        jit_funcs[node.name] = params
                        break

        for local_name, err in disallowed_imports:
            if err not in errors:
                errors.append(err)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr_name = _get_attr_name(node)
                if attr_name:
                    base_mod = attr_name.rsplit(".", 1)[0]
                    attr_tail = attr_name.split(".")[-1]
                    if attr_tail == "tanh" and (base_mod in triton_modules or "triton" in base_mod or "tl" in base_mod.split(".")):
                        err = "Neither tl.tanh nor tl.math.tanh exist in Modal's Triton runtime; compute tanh via the sigmoid identity: 2.0 * tl.sigmoid(2.0 * x) - 1.0"
                        if err not in errors:
                            errors.append(err)
                    elif attr_tail == "pow" and (base_mod in triton_modules or "triton" in base_mod or "tl" in base_mod.split(".")):
                        err = "tl.pow does not exist in triton.language; compute powers via x * x or Python ** operator"
                        if err not in errors:
                            errors.append(err)

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_triton = node.name in jit_funcs
                if is_triton:
                    for subnode in ast.walk(node):
                        if subnode is not node and isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            err = f"Nested function '{subnode.name}' inside @triton.jit kernel '{node.name}' is unsupported. Hoist helper functions to module level decorated with @triton.jit."
                            if err not in errors:
                                errors.append(err)

                    for subnode in ast.walk(node):
                        if isinstance(subnode, (ast.Continue, ast.Break)):
                            err = "continue/break statements are unsupported in @triton.jit kernels; use conditional masks or tl.where instead"
                            if err not in errors:
                                errors.append(err)
                        elif isinstance(subnode, ast.Call):
                            call_name = _get_attr_name(subnode.func)
                            if call_name:
                                if call_name.endswith(".tanh") or call_name == "tanh":
                                    err = "Neither tl.tanh nor tl.math.tanh exist in Modal's Triton runtime; compute tanh via the sigmoid identity: 2.0 * tl.sigmoid(2.0 * x) - 1.0"
                                    if err not in errors:
                                        errors.append(err)
                            if call_name in ("tl.exp", "triton.language.exp", "tl.math.exp") or (call_name and call_name.endswith(".exp") and call_name.rsplit(".", 1)[0] in triton_modules):
                                is_clamped = False
                                if subnode.args:
                                    arg = subnode.args[0]
                                    if _is_clamp_or_min(arg):
                                        is_clamped = True
                                if not is_clamped:
                                    warn = "Unclamped tl.exp() detected; risk of NaN/Inf overflow. Consider tl.clamp(x, -88.0, 88.0)"
                                    if warn not in warnings:
                                        warnings.append(warn)
                else:
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Call):
                            func_full = _get_attr_name(subnode.func)
                            func_id = func_full.split(".")[-1] if func_full else None
                            if func_id and func_id in jit_funcs and not isinstance(subnode.func, ast.Subscript):
                                err = f"Direct host invocation of @triton.jit function '{func_id}' without launch grid '[grid](...)'; causes 'RuntimeError: Cannot call @triton.jit\'d outside of the scope of a kernel'"
                                if err not in errors:
                                    errors.append(err)

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Subscript):
                    raw_kernel_name = _get_attr_name(node.func.value)
                    kernel_name = raw_kernel_name.split(".")[-1] if raw_kernel_name else ""
                    kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]
                    if len(kw_names) != len(set(kw_names)):
                        err = f"Duplicate keyword arguments detected in launch for kernel '{raw_kernel_name}': {kw_names}"
                        if err not in errors:
                            errors.append(err)
                    if kernel_name in jit_funcs:
                        params = jit_funcs[kernel_name]
                        num_pos = len(node.args)
                        if num_pos > len(params):
                            err = f"Too many positional arguments ({num_pos}) passed to kernel '{kernel_name}' which defines {len(params)} parameters; causes TypeError in launch binder"
                            if err not in errors:
                                errors.append(err)
                        pos_params = params[:num_pos]
                        for kw in kw_names:
                            if kw in pos_params:
                                err = f"Duplicate argument '{kw}' passed both positionally and as keyword to kernel '{kernel_name}'; causes TypeError in launch binder"
                                if err not in errors:
                                    errors.append(err)

    return errors, warnings


def _find_baseline_entry(baseline_json, level: int, problem):
    level_key = f"level{level}"
    dicts_to_search = []
    if isinstance(baseline_json, dict):
        if level_key in baseline_json and isinstance(baseline_json[level_key], dict):
            dicts_to_search.append(baseline_json[level_key])
        dicts_to_search.append(baseline_json)

    search_keys = []
    if problem.name:
        search_keys.append(problem.name)
        if problem.name.endswith(".py"):
            search_keys.append(problem.name[:-3])
        else:
            search_keys.append(f"{problem.name}.py")
    if problem.path:
        search_keys.append(problem.path)
        base = os.path.basename(problem.path)
        search_keys.append(base)
        if base.endswith(".py"):
            search_keys.append(base[:-3])
        else:
            search_keys.append(f"{base}.py")
    search_keys.extend([str(problem.problem_id), problem.problem_id])

    for d in dicts_to_search:
        for k in search_keys:
            if k in d and isinstance(d[k], dict) and "mean" in d[k]:
                return float(d[k]["mean"])
    return None


def _find_git_repo_root(start_dir: str) -> str:
    curr = os.path.abspath(start_dir)
    while curr and curr != os.path.dirname(curr):
        git_dir = os.path.join(curr, ".git")
        if os.path.exists(git_dir):
            if os.path.isfile(git_dir):
                try:
                    with open(git_dir, "r") as f:
                        line = f.read().strip()
                    if line.startswith("gitdir:"):
                        git_path = line.split("gitdir:", 1)[1].strip()
                        if not os.path.isabs(git_path):
                            git_path = os.path.abspath(os.path.join(curr, git_path))
                        if ".git" in git_path:
                            main_git = git_path[:git_path.index(".git") + 4]
                            return os.path.dirname(main_git)
                except Exception:
                    pass
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def _make_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_make_serializable(item) for item in obj]
    else:
        return str(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Verify a KernelBench kernel implementation for compilation, correctness, and performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--level", type=int, required=True, choices=[1, 2, 3], help="KernelBench benchmark level (1, 2, or 3)")
    parser.add_argument("--problem-id", type=int, required=True, help="Problem index within the benchmark level (1-indexed)")
    parser.add_argument("--kernel", type=str, required=True, help="Path to Python file defining ModelNew")
    parser.add_argument("--gpu", type=str, default="L40S", help="Target Modal GPU architecture")
    parser.add_argument("--backend", type=str, default="triton", choices=["triton", "cuda", "tilelang"], help="Kernel compiler backend")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Floating point precision")
    parser.add_argument("--num-correct-trials", type=int, default=5, help="Correctness verification trials with randomized inputs")
    parser.add_argument("--num-perf-trials", type=int, default=50, help="Performance profiling trials")
    parser.add_argument("--use-cached-baseline", action=argparse.BooleanOptionalAction, default=True, help="Use cached baseline timing from results/timing/ instead of remote baseline profiling")
    parser.add_argument("--lint-only", action="store_true", help="Run local static anti-hacking and Triton AST checks only without cloud execution")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 correctness trial, 0 perf trials, skip baseline timing")
    parser.add_argument("--json-out", type=str, default=None, help="Optional output path for structured JSON execution receipt")
    parser.add_argument("--timeout", type=int, default=300, help="Modal container timeout in seconds")

    args = parser.parse_args()

    if not os.path.exists(args.kernel):
        print(f"[FAIL] Kernel file not found: {args.kernel}")
        sys.exit(1)

    with open(args.kernel, "r") as f:
        kernel_src = f.read()

    print("=" * 70)
    print(f"Verifying Kernel: Level {args.level}, Problem {args.problem_id}")
    print(f"Source file: {args.kernel}")
    print(f"Target GPU:  {args.gpu} | Backend: {args.backend} | Precision: {args.precision}")
    print("=" * 70)

    # 1. Static Anti-Hacking and AST Validation
    print("[1/3] Running static security, anti-hacking, and Triton AST validation...")
    from kernelbench.kernel_static_checker import validate_kernel_static
    static_ok, static_errors, static_warnings = validate_kernel_static(
        kernel_src,
        backend=args.backend,
        precision=args.precision,
    )
    ast_errors, ast_warnings = lint_kernel_ast(kernel_src, backend=args.backend)
    all_errors = static_errors + ast_errors
    all_warnings = static_warnings + ast_warnings
    passed = (len(all_errors) == 0) and static_ok

    if not passed:
        print(f"[FAIL] Static check rejected kernel with errors: {all_errors}")
        if all_warnings:
            print(f"[WARN] Warnings: {all_warnings}")
        if args.json_out:
            res = {
                "level": args.level,
                "problem_id": args.problem_id,
                "kernel_path": args.kernel,
                "static_passed": False,
                "static_errors": all_errors,
                "static_warnings": all_warnings,
                "compiled": False,
                "correctness": False,
                "kernel_time_ms": -1.0,
                "ref_eager_time_ms": None,
                "speedup_vs_eager": -1.0,
                "runtime_ms": -1.0,
                "speedup": -1.0,
                "metadata": {},
                "error": "; ".join(all_errors),
                "max_difference": None,
                "avg_difference": None,
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w") as f:
                json.dump(res, f, indent=2)
        sys.exit(1)

    if all_warnings:
        print(f"[WARN] Static checker warnings: {all_warnings}")
    else:
        print("[PASS] Static and AST checks passed cleanly.")

    if args.lint_only:
        if args.json_out:
            res = {
                "level": args.level,
                "problem_id": args.problem_id,
                "kernel_path": args.kernel,
                "static_passed": True,
                "static_errors": [],
                "static_warnings": all_warnings,
                "compiled": None,
                "correctness": None,
                "kernel_time_ms": -1.0,
                "ref_eager_time_ms": None,
                "speedup_vs_eager": -1.0,
                "runtime_ms": -1.0,
                "speedup": -1.0,
                "metadata": {},
                "error": None,
                "max_difference": None,
                "avg_difference": None,
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w") as f:
                json.dump(res, f, indent=2)
        sys.exit(0)

    # 2. Problem Reference Loading
    print(f"[2/3] Fetching reference problem from KernelBench level {args.level}...")
    from kernelbench.dataset import construct_kernelbench_dataset
    dataset = construct_kernelbench_dataset(level=args.level, source="local")
    problem = dataset.get_problem_by_id(args.problem_id)
    ref_arch_src = problem.code
    print(f"Loaded reference: {problem.name}")

    ref_eager_time = None
    if args.use_cached_baseline:
        baseline_path = f"results/timing/{args.gpu}_Modal/baseline_time_torch.json"
        if not os.path.exists(baseline_path):
            git_root = _find_git_repo_root(os.path.dirname(__file__))
            candidates = [
                os.path.join(git_root, baseline_path),
                os.path.join(git_root, ".turn4_worktree", baseline_path),
                os.path.join(git_root, ".turn3_worktree", baseline_path),
                os.path.join(git_root, ".turn2_worktree", baseline_path),
                os.path.abspath(f"../{baseline_path}"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    baseline_path = candidate
                    break
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)
                ref_eager_time = _find_baseline_entry(baseline_data, args.level, problem)
                if ref_eager_time is not None:
                    print(f"Loaded cached eager baseline: {ref_eager_time:.2f} ms from {baseline_path}")
            except Exception as e:
                print(f"[WARN] Failed to read cached baseline: {e}")

    # 3. Cloud GPU Execution via Modal
    print(f"[3/3] Executing on Modal cloud GPU ({args.gpu})...")
    import modal
    from scripts.run_and_check import app, EvalFunc, gpu_arch_mapping

    gpu_arch = gpu_arch_mapping.get(args.gpu, ["Ada"])
    eval_config = {
        "num_correct_trials": args.num_correct_trials,
        "num_perf_trials": args.num_perf_trials,
        "verbose": False,
        "measure_performance": True,
        "timing_method": "cuda_event",
        "precision": args.precision,
        "backend": args.backend,
    }

    if args.quick:
        eval_config["num_correct_trials"] = 1
        eval_config["num_perf_trials"] = 0
        eval_config["measure_performance"] = False

    with app.run():
        try:
            eval_result = EvalFunc.with_options(
                gpu=args.gpu,
                timeout=args.timeout,
            )().evaluate_single_sample_src_modal.remote(
                ref_arch_src=ref_arch_src,
                kernel_src=kernel_src,
                configs=eval_config,
                gpu_arch=gpu_arch,
            )
        except Exception as e:
            print(f"[FAIL] Modal remote execution crashed: {e}")
            sys.exit(1)

        if ref_eager_time is None and not args.quick:
            print(f"Measuring PyTorch eager baseline on Modal ({args.gpu})...")
            ref_eager = EvalFunc.with_options(
                gpu=args.gpu,
                timeout=args.timeout,
            )().measure_program_time_modal.remote(
                ref_arch_src=ref_arch_src,
                num_trials=args.num_perf_trials,
                use_torch_compile=False,
                torch_compile_backend=None,
                torch_compile_options=None,
                gpu_arch=gpu_arch,
                precision=args.precision,
            )
            ref_eager_time = ref_eager.get("mean", None) if ref_eager else None

    meta = getattr(eval_result, "metadata", {}) or {}
    if isinstance(eval_result, dict) and not meta:
        meta = eval_result.get("metadata", {}) or {}

    print("=" * 70)
    print(f"Compiled:          {eval_result.compiled}")
    print(f"Correctness:       {eval_result.correctness}")
    print(f"Custom Kernel:     {eval_result.runtime:.2f} ms" if eval_result.runtime > 0 else "Custom Kernel:     N/A")
    print(f"PyTorch Eager:     {ref_eager_time:.2f} ms" if ref_eager_time else "PyTorch Eager:     N/A")

    speedup = None
    if eval_result.correctness and eval_result.runtime > 0 and ref_eager_time:
        speedup = ref_eager_time / eval_result.runtime

    if eval_result.correctness:
        if speedup is not None:
            print(f"Speedup vs Eager:  {speedup:.2f}x")
        else:
            if args.quick:
                print("Speedup vs Eager:  N/A (quick mode, performance not profiled)")
            else:
                print("Speedup vs Eager:  N/A (no baseline timing)")
    else:
        print("Speedup vs Eager:  N/A (correctness failed)")
    print("=" * 70)

    if not eval_result.compiled:
        print("\n--- COMPILATION FAILURE ---")
        err_name = meta.get("compilation_error_name", "UnknownCompilationError")
        err_msg = meta.get("compilation_error", "No compilation error message provided")
        print(f"Error Name: {err_name}")
        print(f"Error Details: {err_msg}")
    elif not eval_result.correctness:
        if "runtime_error" in meta or "runtime_error_name" in meta or "runtime_error_traceback" in meta:
            print("\n--- RUNTIME EXCEPTION ---")
            err_name = meta.get("runtime_error_name", "UnknownRuntimeError")
            err_msg = meta.get("runtime_error", "No runtime error message provided")
            err_tb = meta.get("runtime_error_traceback", "")
            print(f"Error Name: {err_name}")
            print(f"Error Details: {err_msg}")
            if err_tb:
                print(f"Traceback:\n{err_tb}")
        if "correctness_issue" in meta or "max_difference" in meta:
            print("\n--- NUMERICAL MISMATCH ---")
            trials = meta.get("correctness_trials", "N/A")
            issue = meta.get("correctness_issue", "Numerical mismatch")
            max_diff = meta.get("max_difference", "N/A")
            avg_diff = meta.get("avg_difference", "N/A")
            print(f"Trials: {trials}")
            print(f"Issue: {issue}")
            print(f"Max Difference: {max_diff}")
            print(f"Avg Difference: {avg_diff}")

    error_val = None
    if not eval_result.compiled:
        error_val = meta.get("compilation_error") or meta.get("compilation_error_name")
    elif not eval_result.correctness:
        if "runtime_error" in meta:
            error_val = meta.get("runtime_error")
        elif "correctness_issue" in meta:
            error_val = meta.get("correctness_issue")
    if error_val is not None:
        error_val = str(error_val)

    res_data = {
        "level": args.level,
        "problem_id": args.problem_id,
        "problem_name": problem.name,
        "kernel_path": args.kernel,
        "gpu": args.gpu,
        "backend": args.backend,
        "static_passed": True,
        "compiled": eval_result.compiled,
        "correctness": eval_result.correctness,
        "kernel_time_ms": eval_result.runtime if eval_result.runtime > 0 else -1.0,
        "ref_eager_time_ms": ref_eager_time,
        "speedup_vs_eager": speedup,
        "runtime_ms": eval_result.runtime,
        "speedup": speedup if speedup is not None else -1.0,
        "metadata": _make_serializable(meta),
        "error": error_val,
        "max_difference": meta.get("max_difference"),
        "avg_difference": meta.get("avg_difference"),
    }

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(res_data, f, indent=2)
        print(f"Wrote structured report to {args.json_out}")

    if eval_result.compiled and eval_result.correctness:
        print("[SUCCESS] Kernel passed compilation and correctness checks.")
        sys.exit(0)
    else:
        print("[FAILED] Kernel failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
