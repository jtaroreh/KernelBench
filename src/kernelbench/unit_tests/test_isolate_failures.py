"""Unit tests for Level 2 failure isolation and diagnostic repair harness."""

import hashlib
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.isolate_failures import isolate_failures


@pytest.fixture(scope="module")
def paths():
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return {
        "eval_results": repo_root / "runs/cloud_agents_l2/eval_results.json",
        "level2_dir": repo_root / "KernelBench/level2",
        "kernels_dir": repo_root / "runs/cloud_agents_l2",
        "output_dir": repo_root / "runs/cloud_agents_l2_turn2",
    }


@pytest.fixture(scope="module")
def manifest(paths):
    assert paths["eval_results"].exists(), f"{paths['eval_results']} does not exist"
    initial_hash = hashlib.sha256(paths["eval_results"].read_bytes()).hexdigest()

    manifest_data = isolate_failures(
        eval_results_path=paths["eval_results"],
        level2_dir=paths["level2_dir"],
        kernels_dir=paths["kernels_dir"],
        output_dir=paths["output_dir"],
    )

    final_hash = hashlib.sha256(paths["eval_results"].read_bytes()).hexdigest()
    assert initial_hash == final_hash, "eval_results.json was modified"
    return manifest_data


def test_task_counts(manifest):
    assert manifest["failing_count"] == 61, f"Expected 61 failing tasks, got {manifest['failing_count']}"
    assert manifest["passing_count"] == 39, f"Expected 39 passing tasks, got {manifest['passing_count']}"
    assert manifest["total_problems"] == 100, f"Expected 100 total problems, got {manifest['total_problems']}"


def test_problem_6_unsupported_loop(manifest):
    record = manifest["failures"].get("6")
    assert record is not None, "Problem 6 should be in failing tasks"
    assert record["failure_category"] == "unsupported_ast_loop"
    combined_err = (
        f"{record.get('runtime_error_name', '')} {record.get('runtime_error', '')} "
        f"{record.get('runtime_error_traceback', '')}"
    ).lower()
    assert "continue" in combined_err


def test_problem_10_missing_tanh(manifest):
    record = manifest["failures"].get("10")
    assert record is not None, "Problem 10 should be in failing tasks"
    assert record["failure_category"] == "missing_triton_attribute"
    combined_err = (
        f"{record.get('runtime_error_name', '')} {record.get('runtime_error', '')} "
        f"{record.get('runtime_error_traceback', '')}"
    ).lower()
    assert "tanh" in combined_err


def test_problem_24_numerical_mismatch(manifest):
    record = manifest["failures"].get("24")
    assert record is not None, "Problem 24 should be in failing tasks"
    assert record["failure_category"] == "numerical_mismatch"
    assert record.get("has_nan") or record.get("max_difference") is not None


def test_manifest_prompts_non_empty(manifest, paths):
    assert len(manifest["failing_problems"]) == 61
    for item in manifest["failing_problems"]:
        pid = item["problem_id"]
        prompt = item["prompt"]
        assert prompt and len(prompt.strip()) > 0, f"Empty prompt for problem {pid}"
        assert "Triton Pitfall Guards" in prompt
        assert "tl.math.tanh" in prompt
        assert "tl.clamp(x, -88.0, 88.0)" in prompt
        assert "ModelNew" in prompt

        prompt_file = paths["output_dir"] / "prompts" / f"level_2_problem_{pid}_prompt.txt"
        assert prompt_file.exists(), f"Prompt file {prompt_file} does not exist"
        assert prompt_file.read_text(encoding="utf-8") == prompt


def test_eval_results_intact(paths):
    assert paths["eval_results"].exists()
    assert paths["eval_results"].stat().st_size > 0
