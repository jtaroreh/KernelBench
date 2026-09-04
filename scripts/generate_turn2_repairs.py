"""Generate Turn 2 repaired kernels for failing Level 2 tasks using Gemini 3.8 Flash."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.utils import create_inference_server_from_presets, extract_first_code


def save_generation_config(output_dir: Path) -> None:
    """Save generation configuration YAML to output directory."""
    config_dict = {
        "api_query_interval": 0.5,
        "backend": "triton",
        "budget_tokens": 0,
        "check_kernel": True,
        "custom_prompt_key": "turn2_diagnostic_repair",
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
        "run_name": "cloud_agents_l2_turn2",
        "runs_dir": str(REPO_ROOT / "runs"),
        "server_type": "google",
        "store_type": "local",
        "temperature": 0.0,
        "verbose": True,
    }
    config_path = output_dir / "generation_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def copy_turn1_passing_kernels(
    passing_ids: List[int],
    src_dir: Path,
    dst_dir: Path,
) -> int:
    """Copy verified passing Turn 1 kernels into the Turn 2 directory."""
    copied = 0
    for pid in passing_ids:
        src_file = src_dir / f"level_2_problem_{pid}_sample_0_kernel.py"
        dst_file = dst_dir / f"level_2_problem_{pid}_sample_0_kernel.py"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            copied += 1
    return copied


def generate_single_repair(
    problem_id: int,
    prompt: str,
    inference_fn: Any,
    output_dir: Path,
    check_kernel: bool = True,
) -> Optional[str]:
    """Query Gemini 3.8 Flash to repair kernel and validate static compliance."""
    kernel_path = output_dir / f"level_2_problem_{problem_id}_sample_0_kernel.py"

    raw_response = inference_fn(prompt)
    if not raw_response:
        print(f"[WARN] Empty LLM response for Problem {problem_id}")
        return None

    extracted_code = extract_first_code(raw_response, ["python"])
    if not extracted_code:
        print(f"[WARN] Failed to extract Python code for Problem {problem_id}")
        return None

    if check_kernel:
        valid, errors, warnings = validate_kernel_static(
            extracted_code,
            backend="triton",
            precision="fp32",
        )
        if not valid:
            print(f"[WARN] Static check failed for Problem {problem_id}: {errors}")

    kernel_path.write_text(extracted_code, encoding="utf-8")
    return extracted_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Turn 2 repaired Triton kernels for failing Level 2 tasks."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/cloud_agents_l2_turn2/diagnostic_manifest.json"),
        help="Path to diagnostic manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/cloud_agents_l2_turn2"),
        help="Output directory for repaired kernels",
    )
    parser.add_argument(
        "--problem-ids",
        type=int,
        nargs="*",
        default=None,
        help="Specific problem IDs to repair (default: all failing problems in manifest)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip kernel generation if output file already exists",
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

    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_generation_config(args.output_dir)

    passing_ids = manifest_data.get("passing_problem_ids", [])
    turn1_dir = REPO_ROOT / "runs" / "cloud_agents_l2"
    copied_count = copy_turn1_passing_kernels(passing_ids, turn1_dir, args.output_dir)
    print(f"Copied {copied_count} verified Turn 1 baseline kernels into {args.output_dir}")

    failing_problems = manifest_data.get("failing_problems", [])
    if args.problem_ids:
        target_ids = set(args.problem_ids)
        failing_problems = [p for p in failing_problems if p["problem_id"] in target_ids]

    print(f"Loaded {len(failing_problems)} failing problems for Turn 2 repair.")

    inference_fn = create_inference_server_from_presets(
        server_type="google",
        model_name=args.model_name,
        max_tokens=8192,
        temperature=0.0,
    )

    success_count = 0
    skipped_count = 0
    to_generate = []

    for item in failing_problems:
        pid = item["problem_id"]
        kernel_path = args.output_dir / f"level_2_problem_{pid}_sample_0_kernel.py"
        if kernel_path.exists() and not args.force and args.skip_existing:
            skipped_count += 1
            success_count += 1
        else:
            to_generate.append(item)

    print(f"Skipped {skipped_count} existing kernels. Generating {len(to_generate)} kernels with {args.num_workers} parallel workers...")

    def worker(item):
        pid = item["problem_id"]
        prompt = item.get("prompt")
        if not prompt:
            prompt_file = REPO_ROOT / item.get("prompt_file", "")
            if prompt_file.exists():
                prompt = prompt_file.read_text(encoding="utf-8")
        if not prompt:
            return False
        return generate_single_repair(
            problem_id=pid,
            prompt=prompt,
            inference_fn=inference_fn,
            output_dir=args.output_dir,
            check_kernel=True,
        ) is not None

    if to_generate:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_pid = {executor.submit(worker, item): item["problem_id"] for item in to_generate}
            for future in tqdm(concurrent.futures.as_completed(future_to_pid), total=len(to_generate), desc="Generating Turn 2 Repairs"):
                pid = future_to_pid[future]
                try:
                    ok = future.result()
                    if ok:
                        success_count += 1
                except Exception as e:
                    print(f"[ERROR] Worker failed for Problem {pid}: {e}")

    total_kernels = len(list(args.output_dir.glob("level_2_problem_*_sample_0_kernel.py")))
    print("\n--- Turn 2 Repair Generation Summary ---")
    print(f"Total targeted failing tasks: {len(failing_problems)}")
    print(f"Successfully generated: {success_count} (skipped existing: {skipped_count})")
    print(f"Total kernels in {args.output_dir}: {total_kernels} / 100")


if __name__ == "__main__":
    main()
