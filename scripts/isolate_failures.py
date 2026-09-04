"""Isolate Level 2 failures and generate diagnostic repair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# Fast import of leaf prompt template without triggering heavy package root imports (litellm, torch)
SRC_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "src" / "kernelbench" / "prompts"
if str(SRC_PROMPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_PROMPTS_DIR))

try:
    from repair_templates import generate_repair_prompt
except ImportError:
    from kernelbench.prompts.repair_templates import generate_repair_prompt


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def categorize_failure(
    runtime_error: Optional[str],
    runtime_error_name: Optional[str],
    runtime_error_traceback: Optional[str],
    correctness_issue: Optional[str],
    max_difference: Any,
    avg_difference: Any,
) -> str:
    """Categorize failure mode into standard failure categories."""
    combined_err = " ".join(
        filter(
            None,
            [
                str(runtime_error_name) if runtime_error_name else "",
                str(runtime_error) if runtime_error else "",
                str(runtime_error_traceback) if runtime_error_traceback else "",
                str(correctness_issue) if correctness_issue else "",
            ],
        )
    ).lower()

    if "has no attribute" in combined_err and (
        "triton" in combined_err or "tl." in combined_err or "tanh" in combined_err
    ):
        return "missing_triton_attribute"
    if "attributeerror" in combined_err and "tanh" in combined_err:
        return "missing_triton_attribute"

    if any(
        k in combined_err
        for k in ["continue", "break", "unsupported ast", "unsupported statement"]
    ):
        return "unsupported_ast_loop"

    if (
        "typeerror" in combined_err
        or "unexpected keyword argument" in combined_err
        or "missing required positional argument" in combined_err
    ):
        return "type_or_arg_error"

    if (
        "shape mismatch" in combined_err
        or "size of tensor" in combined_err
        or "shapes cannot be multiplied" in combined_err
        or "dimension mismatch" in combined_err
    ):
        return "shape_mismatch"

    if (
        "compilationerror" in combined_err
        or "ptxas" in combined_err
        or "cudaerror" in combined_err
    ):
        return "compilation_error"

    diff_str = str(max_difference).lower() + " " + str(avg_difference).lower()
    if (
        max_difference is not None
        or avg_difference is not None
        or (
            correctness_issue
            and any(
                k in correctness_issue.lower()
                for k in ["mismatch", "nan", "inf", "overflow", "difference"]
            )
        )
        or "nan" in combined_err
        or "nan" in diff_str
        or "output mismatch" in combined_err
    ):
        return "numerical_mismatch"

    return "other"


def get_level2_problem_map(level2_dir: Path) -> Dict[int, Path]:
    """Map problem ID to file path in KernelBench/level2."""
    prob_map = {}
    for p in level2_dir.glob("*.py"):
        prefix = p.name.split("_")[0]
        if prefix.isdigit():
            prob_map[int(prefix)] = p
    return prob_map


def isolate_failures(
    eval_results_path: Path,
    level2_dir: Path,
    kernels_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Extract failing tasks, categorize failures, and generate repair prompts."""
    initial_hash = compute_file_hash(eval_results_path)

    with open(eval_results_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    passing_problems: List[int] = []
    failing_problems_data: List[Dict[str, Any]] = []
    failures_by_id: Dict[str, Dict[str, Any]] = {}
    category_counts: Dict[str, int] = {
        "missing_triton_attribute": 0,
        "unsupported_ast_loop": 0,
        "type_or_arg_error": 0,
        "shape_mismatch": 0,
        "compilation_error": 0,
        "numerical_mismatch": 0,
        "other": 0,
    }

    total_problems = len(eval_data)
    level2_map = get_level2_problem_map(level2_dir)

    for pid_str, samples in eval_data.items():
        problem_id = int(pid_str)
        sample = samples[0] if isinstance(samples, list) and samples else samples
        if not isinstance(sample, dict):
            continue

        is_correct = sample.get("correctness") is True
        if is_correct:
            passing_problems.append(problem_id)
            continue

        metadata = sample.get("metadata", {})
        runtime_error = sample.get("runtime_error") or metadata.get("runtime_error")
        runtime_error_name = sample.get("runtime_error_name") or metadata.get(
            "runtime_error_name"
        )
        runtime_error_traceback = sample.get(
            "runtime_error_traceback"
        ) or metadata.get("runtime_error_traceback")
        max_difference = sample.get("max_difference") or metadata.get("max_difference")
        avg_difference = sample.get("avg_difference") or metadata.get("avg_difference")
        correctness_issue = sample.get("correctness_issue") or metadata.get(
            "correctness_issue"
        )

        has_nan = (
            "nan" in str(max_difference).lower()
            or "nan" in str(avg_difference).lower()
            or "nan" in str(runtime_error).lower()
            or "nan" in str(runtime_error_traceback).lower()
        )

        category = categorize_failure(
            runtime_error,
            runtime_error_name,
            runtime_error_traceback,
            correctness_issue,
            max_difference,
            avg_difference,
        )
        category_counts[category] = category_counts.get(category, 0) + 1

        ref_file = level2_map.get(problem_id)
        problem_name = ref_file.stem if ref_file else f"level_2_problem_{problem_id}"
        ref_code = (
            ref_file.read_text(encoding="utf-8")
            if ref_file and ref_file.exists()
            else "# Reference code not found"
        )

        prev_kernel_file = (
            kernels_dir / f"level_2_problem_{problem_id}_sample_0_kernel.py"
        )
        prev_kernel_code = (
            prev_kernel_file.read_text(encoding="utf-8")
            if prev_kernel_file.exists()
            else "# Previous kernel not found"
        )

        diag_info = {
            "failure_category": category,
            "runtime_error": runtime_error,
            "runtime_error_name": runtime_error_name,
            "runtime_error_traceback": runtime_error_traceback,
            "max_difference": max_difference,
            "avg_difference": avg_difference,
            "correctness_issue": correctness_issue,
            "has_nan": has_nan,
        }

        prompt = generate_repair_prompt(
            problem_name=problem_name,
            reference_code=ref_code,
            previous_kernel=prev_kernel_code,
            diagnostic_info=diag_info,
        )

        prompt_file = prompts_dir / f"level_2_problem_{problem_id}_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        record = {
            "problem_id": problem_id,
            "problem_name": problem_name,
            "failure_category": category,
            "runtime_error": runtime_error,
            "runtime_error_name": runtime_error_name,
            "runtime_error_traceback": runtime_error_traceback,
            "max_difference": max_difference,
            "avg_difference": avg_difference,
            "correctness_issue": correctness_issue,
            "has_nan": has_nan,
            "prompt_file": str(prompt_file.relative_to(output_dir.parent)),
            "prompt": prompt,
        }
        failing_problems_data.append(record)
        failures_by_id[str(problem_id)] = record

    manifest = {
        "total_problems": total_problems,
        "passing_count": len(passing_problems),
        "failing_count": len(failing_problems_data),
        "passing_problem_ids": sorted(passing_problems),
        "category_counts": category_counts,
        "failures": failures_by_id,
        "failing_problems": failing_problems_data,
    }

    manifest_path = output_dir / "diagnostic_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    final_hash = compute_file_hash(eval_results_path)
    if initial_hash != final_hash:
        raise RuntimeError("Turn 1 eval_results.json was modified during isolation!")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate Level 2 failures and build diagnostic repair harness."
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=Path("runs/cloud_agents_l2/eval_results.json"),
        help="Path to Turn 1 eval_results.json",
    )
    parser.add_argument(
        "--level2-dir",
        type=Path,
        default=Path("KernelBench/level2"),
        help="Path to KernelBench Level 2 problem reference definitions",
    )
    parser.add_argument(
        "--kernels-dir",
        type=Path,
        default=Path("runs/cloud_agents_l2"),
        help="Path to directory containing previous failing kernels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/cloud_agents_l2_turn2"),
        help="Output directory for diagnostic manifest and repair prompts",
    )
    args = parser.parse_args()

    print(f"Reading evaluation results from: {args.eval_results}")
    manifest = isolate_failures(
        eval_results_path=args.eval_results,
        level2_dir=args.level2_dir,
        kernels_dir=args.kernels_dir,
        output_dir=args.output_dir,
    )

    print("\n--- Failure Isolation Summary ---")
    print(f"Total problems evaluated: {manifest['total_problems']}")
    print(f"Passing problems: {manifest['passing_count']}")
    print(f"Failing problems: {manifest['failing_count']}")
    print("\nFailure breakdown by category:")
    for cat, count in manifest["category_counts"].items():
        print(f"  {cat}: {count}")

    print(f"\nManifest saved to: {args.output_dir / 'diagnostic_manifest.json'}")
    print(f"Individual prompts written to: {args.output_dir / 'prompts/'}")
    print("Turn 1 eval_results.json confirmed untouched.")


if __name__ == "__main__":
    main()
