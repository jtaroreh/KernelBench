#!/usr/bin/env python3
import sys
import os
import subprocess

def check_step(name, fn):
    try:
        ok, msg = fn()
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status:8} {name}: {msg}")
        return ok
    except Exception as e:
        print(f"[FAIL]   {name}: Unexpected error: {e}")
        return False

def check_python_version():
    v = sys.version_info
    if v.major == 3 and v.minor == 10:
        return True, f"Python {v.major}.{v.minor}.{v.micro}"
    return False, f"Expected Python 3.10, found {v.major}.{v.minor}.{v.micro}"

def check_dependencies():
    missing = []
    for pkg in ["torch", "modal", "pydra", "numpy", "kernelbench"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return True, "All required modules import cleanly"
    return False, f"Missing packages: {missing}"

import shutil

def check_modal_auth():
    try:
        modal_cmd = shutil.which("modal")
        cmd = [modal_cmd, "profile", "current"] if modal_cmd else [sys.executable, "-m", "modal", "profile", "current"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            return True, f"Active profile: {res.stdout.strip()}"
        return False, f"Modal not authenticated: {res.stderr.strip()}"
    except Exception as e:
        return False, f"Modal CLI check failed: {e}"

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

def check_datasets():
    repo_top = _find_git_repo_root(os.path.dirname(__file__))
    kb_dir = os.path.join(repo_top, "KernelBench")
    l1 = len([f for f in os.listdir(os.path.join(kb_dir, "level1")) if f.endswith(".py")]) if os.path.exists(os.path.join(kb_dir, "level1")) else 0
    l2 = len([f for f in os.listdir(os.path.join(kb_dir, "level2")) if f.endswith(".py")]) if os.path.exists(os.path.join(kb_dir, "level2")) else 0
    l3 = len([f for f in os.listdir(os.path.join(kb_dir, "level3")) if f.endswith(".py")]) if os.path.exists(os.path.join(kb_dir, "level3")) else 0
    if l1 >= 100 and l2 >= 100 and l3 >= 50:
        return True, f"Level 1: {l1}, Level 2: {l2}, Level 3: {l3} problems found"
    return False, f"Incomplete dataset: l1={l1}, l2={l2}, l3={l3}"

def check_static_checker():
    from kernelbench.kernel_static_checker import validate_kernel_static
    valid, errors, warnings = validate_kernel_static("import triton\n@triton.jit\ndef foo(): pass", backend="triton")
    if not valid and any("pass" in str(err) for err in errors):
        return True, "Static checker active and correctly flags bypass pattern"
    return False, f"Static checker unexpected response: valid={valid}, errors={errors}"

def check_helpers():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    required = ["doctor.py", "verify_kernel.py"]
    missing = []
    not_executable = []
    for s in required:
        p = os.path.join(scripts_dir, s)
        if not os.path.exists(p):
            missing.append(s)
        elif not os.access(p, os.X_OK):
            not_executable.append(s)
    if missing:
        return False, f"Missing helper scripts: {missing}"
    if not_executable:
        return False, f"Helper scripts missing executable permissions: {not_executable}"
    return True, f"All helper scripts present and executable ({', '.join(required)})"

def check_baseline_cache():
    repo_top = _find_git_repo_root(os.path.dirname(__file__))
    candidates = [
        os.path.join(repo_top, "results/timing/L40S_Modal/baseline_time_torch.json"),
        os.path.join(repo_top, ".turn4_worktree/results/timing/L40S_Modal/baseline_time_torch.json"),
        os.path.join(repo_top, ".turn3_worktree/results/timing/L40S_Modal/baseline_time_torch.json"),
        os.path.join(repo_top, ".turn2_worktree/results/timing/L40S_Modal/baseline_time_torch.json"),
        os.path.abspath("results/timing/L40S_Modal/baseline_time_torch.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            size_kb = os.path.getsize(c) / 1024
            rel = os.path.relpath(c, repo_top)
            return True, f"Found L40S baseline ({size_kb:.1f} KB) at {rel}"
    return False, "L40S Modal baseline timing file not found"

def main():
    print("=" * 60)
    print("KernelBench Environment Doctor")
    print("=" * 60)
    checks = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_dependencies),
        ("Modal Authentication", check_modal_auth),
        ("Dataset Presence", check_datasets),
        ("Static Checker", check_static_checker),
        ("Helper Scripts", check_helpers),
        ("Baseline Timing Cache", check_baseline_cache),
    ]
    all_passed = True
    for name, fn in checks:
        if not check_step(name, fn):
            all_passed = False
    print("=" * 60)
    if all_passed:
        print("[SUCCESS] Environment is fully ready for kernel verification.")
        sys.exit(0)
    else:
        print("[BLOCKED] Environment has failing checks. Resolve them before running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
