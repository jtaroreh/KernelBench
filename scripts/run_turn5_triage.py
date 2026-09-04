#!/usr/bin/env python3
"""Batch triage runner for Turn 5 candidate kernels."""

import argparse
import glob
import json
import os
import subprocess
import sys


def run_triage(problem_ids: list[int], output_dir: str = "docs/benchmarks/receipts") -> None:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = "docs/benchmarks/turn5_triage.json"
    
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    script_path = ".agents/skills/verify-kernelbench/scripts/verify_kernel.py"

    passed = []
    failed = []

    for pid in problem_ids:
        pattern = f"solutions/level2/{pid}_*.py"
        files = glob.glob(pattern)
        if not files:
            print(f"[SKIP] Problem {pid}: No kernel file matching {pattern}")
            continue

        kernel_file = files[0]
        receipt_file = os.path.join(output_dir, f"verify_{pid}.json")

        cmd = [
            sys.executable,
            script_path,
            "--quick",
            "--level", "2",
            "--problem-id", str(pid),
            "--kernel", kernel_file,
            "--json-out", receipt_file,
        ]

        print(f"\n>>> Triaging Problem {pid} ({os.path.basename(kernel_file)})...", flush=True)
        print("you got this! keep going", flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        status = "UNKNOWN"
        error_msg = None
        if os.path.exists(receipt_file):
            try:
                with open(receipt_file, "r") as rf:
                    r_data = json.load(rf)
                    comp = r_data.get("compiled", False)
                    corr = r_data.get("correctness", False)
                    error_msg = r_data.get("error")
                    if comp and corr:
                        status = "PASS"
                    elif not comp:
                        status = "COMPILATION_FAIL"
                    else:
                        status = "CORRECTNESS_FAIL"
            except Exception as e:
                status = f"RECEIPT_ERROR: {e}"
        else:
            status = "NO_RECEIPT"
            error_msg = res.stderr.strip() or res.stdout.strip()

        print(f"Result P{pid:02d}: {status} (exit {res.returncode})")
        if status == "PASS":
            passed.append(pid)
        else:
            failed.append(pid)
            if error_msg:
                first_line = error_msg.split("\n")[0][:100]
                print(f"  Error: {first_line}")

        summary[str(pid)] = {
            "file": os.path.basename(kernel_file),
            "status": status,
            "error": error_msg,
            "exit_code": res.returncode,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print("\n==================================================")
    print(f"Batch Complete: {len(passed)} PASSED, {len(failed)} FAILED")
    print(f"Passed PIDs: {passed}")
    print(f"Failed PIDs: {failed}")
    print("==================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage Level 2 candidate kernels")
    parser.add_argument("--pids", type=int, nargs="+", help="Specific problem IDs to triage")
    args = parser.parse_args()

    pids = args.pids
    if not pids:
        # Default to remaining unverified candidates
        pids = [33, 36, 37, 46, 47, 48, 52, 53, 54, 64, 66, 67, 72, 73, 75, 77, 80, 82, 84, 86, 91, 92, 93, 94, 98, 99]
    run_triage(pids)


if __name__ == "__main__":
    main()
