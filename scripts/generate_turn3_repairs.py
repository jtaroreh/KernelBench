"""Generate Turn 3 repaired kernels for failing Level 2 tasks using Gemini 3.8 Flash."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
load_dotenv()

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.prompts.repair_templates import generate_repair_prompt
from kernelbench.utils import create_inference_server_from_presets, extract_first_code
from scripts.isolate_failures import categorize_failure, get_level2_problem_map


def save_generation_config(output_dir: Path) -> None:
    """Save generation configuration YAML to output directory."""
    config_dict = {
        "api_query_interval": 0.5,
        "backend": "triton",
        "budget_tokens": 0,
        "check_kernel": True,
        "custom_prompt_key": "turn3_targeted_repair",
        "dataset_name": "ScalingIntelligence/KernelBench",
        "dataset_src": "local",
        "hardware_gpu_name": "L40S",
        "include_hardware_info": True,
        "is_reasoning_model": False,
        "level": 2,
        "log_prompt": True,
        "max_tokens": 8192,
        "model_name": "gemini-3.8-flash",
        "num_samples": 1,
        "num_workers": 6,
        "precision": "fp32",
        "prompt_option": "targeted_diagnostic_repair",
        "reasoning_effort": "low",
        "run_name": "cloud_agents_l2_turn3",
        "runs_dir": str(REPO_ROOT / "runs"),
        "server_type": "google",
        "store_type": "local",
        "temperature": 0.0,
        "verbose": True,
    }
    config_path = output_dir / "generation_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def copy_passing_kernels(
    solutions_dir: Path,
    dst_dir: Path,
) -> Dict[int, Path]:
    """Copy existing passing kernels from solutions/level2 into Turn 3 directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied_map: Dict[int, Path] = {}
    for f in sorted(solutions_dir.glob("*.py")):
        m = re.match(r"^(\d+)_", f.name)
        if not m:
            continue
        pid = int(m.group(1))
        dst_file = dst_dir / f"level_2_problem_{pid}_sample_0_kernel.py"
        shutil.copy2(f, dst_file)
        copied_map[pid] = dst_file
    return copied_map


def extract_turn2_diagnostic(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract runtime / compilation diagnostic dictionary from eval_results sample."""
    metadata = sample.get("metadata", {}) or {}
    runtime_error = sample.get("runtime_error") or metadata.get("runtime_error")
    runtime_error_name = sample.get("runtime_error_name") or metadata.get("runtime_error_name")
    runtime_error_traceback = sample.get("runtime_error_traceback") or metadata.get("runtime_error_traceback")
    compilation_error = sample.get("compilation_error") or metadata.get("compilation_error")
    compilation_error_name = sample.get("compilation_error_name") or metadata.get("compilation_error_name")

    if not runtime_error and compilation_error:
        runtime_error = compilation_error
    if not runtime_error_name and compilation_error_name:
        runtime_error_name = compilation_error_name

    max_difference = sample.get("max_difference") or metadata.get("max_difference")
    avg_difference = sample.get("avg_difference") or metadata.get("avg_difference")
    correctness_issue = sample.get("correctness_issue") or metadata.get("correctness_issue")

    category = categorize_failure(
        runtime_error,
        runtime_error_name,
        runtime_error_traceback,
        correctness_issue,
        max_difference,
        avg_difference,
    )

    has_nan = (
        "nan" in str(max_difference).lower()
        or "nan" in str(avg_difference).lower()
        or "nan" in str(runtime_error).lower()
        or "nan" in str(runtime_error_traceback).lower()
    )

    return {
        "failure_category": category,
        "runtime_error": runtime_error,
        "runtime_error_name": runtime_error_name,
        "runtime_error_traceback": runtime_error_traceback,
        "max_difference": max_difference,
        "avg_difference": avg_difference,
        "correctness_issue": correctness_issue,
        "has_nan": has_nan,
    }


def generate_single_repair(
    problem_id: int,
    prompt: str,
    inference_fn: Any,
    output_dir: Path,
    check_kernel: bool = True,
    max_retries: int = 2,
) -> Optional[str]:
    """Query Gemini 3.8 Flash to repair kernel and validate static compliance."""
    kernel_path = output_dir / f"level_2_problem_{problem_id}_sample_0_kernel.py"

    for attempt in range(max_retries + 1):
        raw_response = inference_fn(prompt)
        if not raw_response:
            print(f"[WARN] Empty LLM response for Problem {problem_id} (attempt {attempt+1})")
            continue

        extracted_code = extract_first_code(raw_response, ["python"])
        if not extracted_code:
            print(f"[WARN] Failed to extract Python code for Problem {problem_id} (attempt {attempt+1})")
            continue

        if check_kernel:
            valid, errors, warnings = validate_kernel_static(
                extracted_code,
                backend="triton",
                precision="fp32",
            )
            if not valid:
                print(f"[WARN] Static check failed for Problem {problem_id} (attempt {attempt+1}): {errors}")
                if attempt < max_retries:
                    continue

        kernel_path.write_text(extracted_code, encoding="utf-8")
        return extracted_code

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Turn 3 repaired Triton kernels for failing Level 2 tasks."
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=Path("runs/cloud_agents_l2_turn2/eval_results.json"),
        help="Path to Turn 2 eval_results.json",
    )
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=Path("solutions/level2"),
        help="Path to solutions/level2 directory with verified passing kernels",
    )
    parser.add_argument(
        "--turn2-kernels-dir",
        type=Path,
        default=Path("runs/cloud_agents_l2_turn2"),
        help="Path to directory containing Turn 2 kernels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/cloud_agents_l2_turn3"),
        help="Output directory for Turn 3 repaired kernels",
    )
    parser.add_argument(
        "--level2-dir",
        type=Path,
        default=Path("KernelBench/level2"),
        help="Path to KernelBench Level 2 reference problem directory",
    )
    parser.add_argument(
        "--problem-ids",
        type=int,
        nargs="*",
        default=None,
        help="Specific problem IDs to repair (default: all 55 remaining failures)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip generation if output file already exists",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite of existing repaired kernels",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-3.8-flash",
        help="Gemini model name to query",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of concurrent workers for API generation",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_generation_config(args.output_dir)

    copied_solutions = copy_passing_kernels(args.solutions_dir, args.output_dir)
    passing_pids = sorted(copied_solutions.keys())
    print(f"Copied {len(copied_solutions)} passing solutions from {args.solutions_dir} to {args.output_dir}")

    all_pids = set(range(1, 101))
    failing_pids = sorted(list(all_pids - set(passing_pids)))
    print(f"Identified {len(failing_pids)} failing problem IDs remaining to repair.")

    if args.problem_ids:
        target_pids = [pid for pid in args.problem_ids if pid in failing_pids]
    else:
        target_pids = failing_pids

    with open(args.eval_results, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    level2_map = get_level2_problem_map(args.level2_dir)
    prompts_dir = args.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    items_to_generate: List[Dict[str, Any]] = []
    skipped_count = 0

    for pid in target_pids:
        kernel_path = args.output_dir / f"level_2_problem_{pid}_sample_0_kernel.py"
        if kernel_path.exists() and not args.force and args.skip_existing:
            skipped_count += 1
            continue

        ref_file = level2_map.get(pid)
        if not ref_file or not ref_file.exists():
            print(f"[WARN] Reference file not found for problem {pid}")
            continue
        ref_code = ref_file.read_text(encoding="utf-8")

        prev_kernel_path = args.turn2_kernels_dir / f"level_2_problem_{pid}_sample_0_kernel.py"
        prev_code = prev_kernel_path.read_text(encoding="utf-8") if prev_kernel_path.exists() else "# Previous kernel not found"

        sample_list = eval_data.get(str(pid), [])
        sample = sample_list[0] if sample_list else {}
        diag_info = extract_turn2_diagnostic(sample)

        prompt = generate_repair_prompt(
            problem_name=ref_file.stem,
            reference_code=ref_code,
            previous_kernel=prev_code,
            diagnostic_info=diag_info,
            target_model=args.model_name,
        )

        prompt_file = prompts_dir / f"level_2_problem_{pid}_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        items_to_generate.append({
            "problem_id": pid,
            "prompt": prompt,
            "category": diag_info["failure_category"],
        })

    print(f"Skipped {skipped_count} existing. Prepared {len(items_to_generate)} prompts for generation.")

    inference_fn = create_inference_server_from_presets(
        server_type="google",
        model_name=args.model_name,
        max_tokens=8192,
        temperature=0.0,
    )

    success_count = 0
    failure_count = 0

    def worker(item: Dict[str, Any]) -> bool:
        pid = item["problem_id"]
        res = generate_single_repair(
            problem_id=pid,
            prompt=item["prompt"],
            inference_fn=inference_fn,
            output_dir=args.output_dir,
            check_kernel=True,
        )
        return res is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_pid = {executor.submit(worker, item): item["problem_id"] for item in items_to_generate}
        for future in tqdm(concurrent.futures.as_completed(future_to_pid), total=len(items_to_generate), desc="Generating Turn 3 Repairs"):
            pid = future_to_pid[future]
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                print(f"[ERROR] Exception in generation worker for Problem {pid}: {e}")
                failure_count += 1

    total_kernels = len(list(args.output_dir.glob("level_2_problem_*_sample_0_kernel.py")))
    print("\n--- Turn 3 Repair Generation Summary ---")
    print(f"Targeted failing tasks: {len(target_pids)}")
    print(f"Successfully generated: {success_count} (skipped: {skipped_count}, failed: {failure_count})")
    print(f"Total kernels in {args.output_dir}: {total_kernels} / 100")


if __name__ == "__main__":
    main()
