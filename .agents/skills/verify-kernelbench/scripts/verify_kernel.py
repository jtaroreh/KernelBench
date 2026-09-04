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
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr_name = _get_attr_name(node)
                if attr_name in ("tl.tanh", "triton.language.tanh"):
                    err = "tl.tanh does not exist in Triton; use tl.math.tanh or custom rational approximation (tl.exp(2*x)-1)/(tl.exp(2*x)+1)"
                    if err not in errors:
                        errors.append(err)

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_triton = False
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    dec_name = _get_attr_name(target)
                    if dec_name in ("triton.jit", "triton.autotune", "jit", "autotune", "tl.jit"):
                        is_triton = True
                        break

                if is_triton:
                    for subnode in ast.walk(node):
                        if isinstance(subnode, (ast.Continue, ast.Break)):
                            err = "continue/break statements are unsupported in @triton.jit kernels; use conditional masks or tl.where instead"
                            if err not in errors:
                                errors.append(err)
                        elif isinstance(subnode, ast.Call):
                            call_name = _get_attr_name(subnode.func)
                            if call_name in ("tl.exp", "triton.language.exp", "tl.math.exp"):
                                is_clamped = False
                                if subnode.args:
                                    arg = subnode.args[0]
                                    if _is_clamp_or_min(arg):
                                        is_clamped = True
                                if not is_clamped:
                                    warn = "Unclamped tl.exp() detected; risk of NaN/Inf overflow. Consider tl.clamp(x, -88.0, 88.0)"
                                    if warn not in warnings:
                                        warnings.append(warn)

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
    if problem.path:
        search_keys.append(problem.path)
        search_keys.append(os.path.basename(problem.path))
    search_keys.extend([str(problem.problem_id), problem.problem_id])

    for d in dicts_to_search:
        for k in search_keys:
            if k in d and isinstance(d[k], dict) and "mean" in d[k]:
                return float(d[k]["mean"])
    return None


def _make_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    else:
        return str(obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Robust single kernel verification tool for KernelBench")
    parser.add_argument("--level", type=int, required=True, help="KernelBench level (1, 2, or 3)")
    parser.add_argument("--problem-id", type=int, required=True, help="Logical problem ID (1-indexed)")
    parser.add_argument("--kernel", type=str, required=True, help="Path to custom kernel source file")
    parser.add_argument("--gpu", type=str, default="L40S", help="Modal GPU type (L40S, H100, A100, L4, etc.)")
    parser.add_argument("--backend", type=str, default="triton", choices=["triton", "cuda", "tilelang", "cute"], help="Kernel backend")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--num-correct-trials", type=int, default=5, help="Number of randomized correctness trials")
    parser.add_argument("--num-perf-trials", type=int, default=50, help="Number of performance timing trials")
    parser.add_argument("--timeout", type=int, default=180, help="Execution timeout in seconds")
    parser.add_argument("--json-out", type=str, default=None, help="Path to write structured JSON verification results")
    parser.add_argument("--lint-only", action="store_true", help="Run local static and Triton AST checks only without cloud compute")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 correctness trial, 0 perf trials, skip baseline timing")
    parser.add_argument("--use-cached-baseline", action="store_true", default=True, help="Use cached eager baseline from results/timing/ if available")
    parser.add_argument("--no-cached-baseline", action="store_false", dest="use_cached_baseline", help="Force remote baseline timing even if cached baseline exists")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.kernel):
        print(f"[ERROR] Kernel file not found: {args.kernel}")
        sys.exit(1)

    with open(args.kernel, "r") as f:
        kernel_src = f.read()

    print("=" * 70)
    print(f"Verifying Kernel: Level {args.level}, Problem {args.problem_id}")
    print(f"Source file: {args.kernel}")
    print(f"Target GPU:  {args.gpu} | Backend: {args.backend} | Precision: {args.precision}")
    print("=" * 70)

    # 1. Static Validation and AST Lint
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

    speedup = (ref_eager_time / eval_result.runtime) if (eval_result.correctness and eval_result.runtime > 0 and ref_eager_time) else 0.0
    if eval_result.correctness:
        print(f"Speedup vs Eager:  {speedup:.2f}x")
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
        "kernel_time_ms": eval_result.runtime,
        "ref_eager_time_ms": ref_eager_time,
        "speedup_vs_eager": speedup,
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
