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

def check_modal_auth():
    try:
        res = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            return True, f"Active profile: {res.stdout.strip()}"
        return False, f"Modal not authenticated: {res.stderr.strip()}"
    except Exception as e:
        return False, f"Modal CLI check failed: {e}"

def check_datasets():
    repo_top = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
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
