# Benchmark scoring

Benchmark scoring computes aggregate metrics including compilation rate, correctness rate, geometric mean speedup, and fast_p scores for varying speedup thresholds.

## Sub-features

- `score-fast-p` calculates the fraction of total problems that are correct and exceed speedup threshold p.
- `score-gmean` computes geometric mean speedup for correct samples against baseline reference timings.
- `score-patch` pads missing problems as failed to ensure honest, uncheatable scores.

## How to get to it (user POV)

- Run `uv run python scripts/benchmark_eval_analysis.py run_name=<run> level=<level> hardware=<hardware> baseline=<baseline>`.

## Driving it with verify-kernelbench

Preconditions:
- `runs/<run_name>/eval_results.json` exists.
- Baseline timing file exists (e.g. `results/timing/L40S_Modal/baseline_time_torch.json`).

- **Compute scores.** Run `uv run python scripts/benchmark_eval_analysis.py run_name=<run> level=2 hardware=L40S_Modal baseline=baseline_time_torch baseline_file=results/timing/L40S_Modal/baseline_time_torch.json eval_results_dir=runs`.
- **Verify table.** Output renders table with Fast_0, Fast_0.5, Fast_0.8, Fast_1, Fast_1.5, Fast_2, and geometric mean speedup.

## Gotchas

- Any problem missing from `eval_results.json` is automatically treated as `compiled=False, correctness=False` by `patch()`.
- Baseline file must have problem filenames matching dataset names.
