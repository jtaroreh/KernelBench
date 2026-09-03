#!/usr/bin/env python3
import argparse
import sys
import os
import json
import time

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

    # 1. Static Validation
    print("[1/3] Running static security and anti-hacking validation...")
    from kernelbench.kernel_static_checker import validate_kernel_static
    static_ok, errors, warnings = validate_kernel_static(
        kernel_src,
        backend=args.backend,
        precision=args.precision,
    )
    if not static_ok:
        print(f"[FAIL] Static check rejected kernel with errors: {errors}")
        if warnings:
            print(f"[WARN] Warnings: {warnings}")
        res = {
            "level": args.level,
            "problem_id": args.problem_id,
            "kernel_path": args.kernel,
            "static_passed": False,
            "static_errors": errors,
            "compiled": False,
            "correctness": False,
            "runtime_ms": -1.0,
            "speedup": -1.0
        }
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w") as f:
                json.dump(res, f, indent=2)
        sys.exit(1)
        
    if warnings:
        print(f"[WARN] Static checker warnings: {warnings}")
    else:
        print("[PASS] Static checks passed cleanly.")

    # 2. Problem Reference Loading
    print(f"[2/3] Fetching reference problem from KernelBench level {args.level}...")
    from kernelbench.dataset import construct_kernelbench_dataset
    dataset = construct_kernelbench_dataset(level=args.level, source="local")
    problem = dataset.get_problem_by_id(args.problem_id)
    ref_arch_src = problem.code
    print(f"Loaded reference: {problem.name}")

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

        # Baseline measurement
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
