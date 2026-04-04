# Evaluation

The repository supports two evaluation modes.

## 1. Local Development Evaluation

`hallmark_mlx.eval.metrics` computes a small local metric set:

- detection rate,
- false positive rate,
- precision on hallucinated entries,
- recall on hallucinated entries,
- F1 on hallucinated entries.

This is for iteration speed, not for claiming benchmark parity.

## 2. HALLMARK-Style Serialization

`hallmark_mlx.eval.hallmark_adapter` formats local outputs into HALLMARK-style JSONL. A prediction row looks like this:

```json
{
  "bibtex_key": "a3f9c2b1d4e76f85",
  "label": "HALLUCINATED",
  "confidence": 0.87,
  "reason": "DOI does not resolve and no candidate record was found.",
  "subtest_results": {
    "doi_resolves": false
  },
  "api_sources_queried": ["crossref", "semantic_scholar"],
  "wall_clock_seconds": 1.2,
  "api_calls": 3
}
```

For real HALLMARK runs, do not synthesize `bibtex_key` values. Use the exact keys provided by the benchmark loader.

## 3. Resumable Official Runs

Full official `dev_public` runs are expensive because the controller may invoke external
verification CLIs per entry. Use the dedicated runner:

```bash
uv run python scripts/run_official_controller_eval.py \
  --upstream-root /tmp/hallmark-upstream \
  --split dev_public \
  --mode bibtex_first_fallback \
  --output-dir artifacts/official_eval/dev_public_bibtex_first_fallback \
  --entry-timeout-seconds 20 \
  --progress-every 10
```

The output directory is resumable and traceable:

- `entries/0000.json`, `entries/0001.json`, ... store one completed trace per official entry
- `progress.json` records completed entries, remaining entries, timeout budget, and run status
- `traces.jsonl`, `predictions.jsonl`, and `result.json` are written once the requested slice completes

Re-running the same command resumes from existing entry checkpoints. Use `--limit N` for a
partial smoke run that still uses the official evaluator on the first `N` official entries.

## Benchmark Hygiene

To avoid overfitting or leakage:

- keep benchmark labels out of retrieval-time evidence,
- avoid title-leak shortcuts,
- separate tuning from final evaluation,
- and treat any dev-set prompt or config iteration as benchmark optimization that must be reported honestly.

## Recommended Comparison Table

The project is designed to compare at least:

- prompting without tools,
- prompting with tools,
- MLX LoRA fine-tuning on traces,
- and Weco-optimized policy-frontier variants.

Every comparison should report:

- dataset version,
- held-out split,
- model and adapter version,
- tools enabled,
- calibration assumptions,
- and contamination controls.

## Current Benchmark Figure

The repository includes a generated comparison figure at
`docs/figures/hallmark_baseline_comparison.png`.

It separates:

- a local apples-to-apples 10-example HALLMARK slice comparison between repository modes and
  HALLMARK-provided local baselines,
- and the official HALLMARK README baseline table on `dev_public` for full-split context.

Do not mix those panels when making headline claims. The local slice is useful for debugging
protocol behavior and tool budgets. The official `dev_public` numbers are the benchmark-wide
reference point.

For repo-native Weco search over the budget-aware frontier, see `docs/weco.md`.
