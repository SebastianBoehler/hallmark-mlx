# Official Benchmarking

Use this file for public benchmark claims only.

## Reporting Rule

- Official splits only:
  - `dev_public`
  - `test_public`
  - `stress_test`
- Do not present `search64` or `compare32` as benchmark results.

## Canonical Confirmed Result Root

Use:

`artifacts/official_eval_sharded_fast5_confirm`

That is the current confirmed official controller rerun root.

## Current Confirmed Official Rows

- `dev_public`:
  `F1-H 0.9344`, `DR 0.9634`, `TW-F1 0.9523`, `FPR 0.1556`, `ECE 0.0675`
- `test_public`:
  `F1-H 0.9280`, `DR 0.9444`, `TW-F1 0.9379`, `FPR 0.1722`, `ECE 0.0355`
- `stress_test`:
  `F1-H 0.9774`, `DR 0.9559`, `TW-F1 0.9745`, `ECE 0.1601`

## Exact Sharded Reproduction

Run one split with the official controller entrypoint and then merge:

```bash
for offset in 0 140 280 420 560 700 840 980; do
  shard=$(printf "shard_%02d" $((offset / 140)))
  PYTHONPATH=src uv run python scripts/run_official_controller_eval.py \
    --upstream-root /tmp/hallmark-upstream \
    --split dev_public \
    --mode bibtex_first_fallback \
    --output-dir \
      artifacts/official_eval_sharded_fast5_confirm/dev_public_bibtex_first_fallback/shards/${shard} \
    --entry-timeout-seconds 5 \
    --offset ${offset} \
    --limit 140
done

description=$(
  printf %s \
    "hallmark-mlx controller with deterministic finalizer " \
    "(5s per-entry timeout, 8-way sharded official run)"
)

PYTHONPATH=src uv run python scripts/merge_official_eval_shards.py \
  --upstream-root /tmp/hallmark-upstream \
  --split dev_public \
  --shards-root artifacts/official_eval_sharded_fast5_confirm/dev_public_bibtex_first_fallback/shards \
  --output-dir artifacts/official_eval_sharded_fast5_confirm/dev_public_bibtex_first_fallback/merged \
  --method-name hallmark_mlx_bibtex_first_fallback \
  --description "${description}"
```

Notes:

- `test_public` and `stress_test` use the same pattern.
- For smaller splits, later shard offsets will simply produce zero-entry shard dirs. That is expected.

## Refresh Published Artifacts

After confirmed reruns, refresh the summary files and regenerate plots/tables:

```bash
PYTHONPATH=src uv run python scripts/refresh_confirmed_benchmarks.py
```

That syncs the confirmed rows into:

- `artifacts/official_split_suite/summary.json`
- `artifacts/hallmark_baseline_compare/summary.json`

and regenerates:

- `docs/reports/hallmark_official_splits.md`
- `docs/reports/hallmark_submission_readiness.md`
- `docs/reports/hallmark_benchmark_report.md`
- `docs/figures/hallmark_official_splits.png`
- `docs/figures/hallmark_official_vs_bibtexupdater.png`
- `docs/figures/hallmark_submission_leaderboard.png`
- `docs/figures/hallmark_registry_comparison.png`

## Claim Discipline

- If a fresh rerun disagrees with a cached report, refresh the reports first.
- Use the confirmed rows, not the stale cached `official_eval_sharded_fast5` rows, for publication claims.
