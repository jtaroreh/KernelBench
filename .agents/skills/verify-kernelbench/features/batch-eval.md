# Batch evaluation

Batch evaluation drives multi-kernel evaluations across problem subsets or full levels on Modal cloud GPU workers in parallel without mutual interference.

## Sub-features

- `batch-spawn` dispatches parallel container evaluations via Modal.
- `batch-timeout` enforces per-problem timeouts to prevent hung kernels from stalling the queue.
- `batch-aggregation` collects individual `KernelExecResult` objects into `eval_results.json`.

## How to get to it (user POV)

- Run `uv run python scripts/eval_from_generations.py run_name=<run_name> dataset_src=local level=<level> eval_mode=modal gpu=L40S backend=triton`.

## Driving it with verify-kernelbench

Preconditions:
- Kernels are stored in `runs/<run_name>/level_<level>_problem_<id>_sample_0_kernel.py`.
- Unique `run_name` is chosen per agent to prevent collisions.

- **Launch batch run.** Run `uv run python scripts/eval_from_generations.py run_name=<unique_run> dataset_src=local level=2 problem_ids="[1]" eval_mode=modal gpu=L40S backend=triton timeout=180`.
- **Confirm results file.** Check `runs/<unique_run>/eval_results.json`. File exists and is non-empty.

## Gotchas

- Avoid running two agents pointing to the same `run_name`. Always namespace with agent IDs.
- Watch container memory limits when compiling heavy C++ extensions simultaneously.
- **Pre-Triage Invariant**: Never launch batch evaluations on unverified candidate kernels en masse. Every candidate kernel must pass at least Tier 1 (`--lint-only`) and Tier 2 (`--quick`) before inclusion in a batch run.
- **Modal Concurrency Caution**: Modal imposes account-level limits on concurrent GPU containers and application creation rates. Untriaged batches can trigger dropped tasks (`Future was None`).
- **Modal App Recovery**: When worker capacity transitions cause apps to hang in the queue, run `modal app stop -y <app_id>` to release local process handles and allow clean rescheduling.
- **Batch Dropout Recovery**: When batch runs omit tasks due to timeouts or dropped containers, never accept `patch()` default failures without triaging individual dropped kernels via `verify_kernel.py`.
