# KernelBench Agent Behavioral Guidelines

## 1. Mandatory Verification Protocol (`/verify-kernelbench`)
When validating, testing, or repairing GPU kernels in KernelBench:
- **Never bypass `/verify-kernelbench`:** Always use `.agents/skills/verify-kernelbench/scripts/verify_kernel.py` and `doctor.py` for evaluating individual kernels. Do not default to raw upstream scripts (`scripts/eval_from_generations.py`) for single-kernel triage or iterative repair verification.
- **Enforce 3-Tier Diagnostic Triage:**
  1. **Tier 1 (Static AST Lint):** Run `verify_kernel.py --lint-only` first. Zero-cost local check for `ModelNew`, disallowed AST constructs (`continue`/`break`), nonexistent `tl.tanh`/`tl.math.tanh` and `tl.pow`, host JIT invocations, duplicate arguments, and unclamped `tl.exp`.
  2. **Tier 2 (Quick Cloud Test):** Run `verify_kernel.py --quick` (1 correctness trial, no perf profiling) to verify basic execution on Modal.
  3. **Tier 3 (Full Verification):** Run `verify_kernel.py --num-correct-trials 5 --num-perf-trials 50 --use-cached-baseline` to establish numerical correctness and true speedup vs PyTorch eager.
- **Harness Integrity:** Run `uv run pytest .agents/skills/verify-kernelbench/tests/test_verify_kernel.py -v` before modifying verification rules.
- **Speedup Integrity:** Never accept `-1.00x` speedups. Always ensure speedup is computed against `results/timing/L40S_Modal/baseline_time_torch.json`.

## 2. Batch Evaluation Guardrails
- **Pre-Triage Requirement:** Never launch batch evaluations (`scripts/eval_from_generations.py`) on unverified candidate kernels en masse. Every candidate kernel must pass at least Tier 1 (`--lint-only`) and Tier 2 (`--quick`) before inclusion in a batch run.
- **Modal Concurrency Caution:** Avoid flooding Modal with large parallel GPU allocations on untriaged sets. Watch for `Future was None` errors which indicate dropped or timed-out tasks.
- **Hardware Boundary Awareness:** On NVIDIA L40S, shared memory is strictly bounded at 100 KB (101,376 bytes) per thread block. Kernels exceeding this limit fail with runtime resource errors. Size block tiles ($\le 64 \times 64$) and `num_stages \le 2` accordingly.
- **Batch Dropout Recovery:** When batch runs omit tasks due to timeouts or dropped containers, never accept `patch()` default failures without triaging individual dropped kernels via `verify_kernel.py`.

## 3. Accurate Reporting Invariant (No Lip-Service)
- Never state in user messages or summaries that a specific harness or skill is being executed (e.g. "via verify-kernelbench") unless the exact CLI tools and scripts from that skill are actively being called.

## 4. Subagent Isolation & Git Workspace Hygiene
- **Never Use `Workspace: 'inherit'` for Modifying Subagents:** Any subagent tasked with creating branches, committing changes, or modifying files must be spawned with `Workspace: 'branch'` or operate inside a dedicated `git worktree`. Modifying the root git branch inside an inherited subagent causes branch collisions and corrupts parent context.
- **Preserve `.agents/` Across Branches:** Custom skills in `.agents/` must remain preserved and accessible. When creating branches off `upstream/main`, ensure `.agents/` customizations are maintained.
