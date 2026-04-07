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

Full official HALLMARK runs are expensive because the controller may invoke external
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

## 4. One-Command Submission Eval

For benchmark handoff, prefer the wrapper script:

```bash
uv run python scripts/run_submission_eval.py \
  --upstream-root /tmp/hallmark-upstream \
  --split test_public \
  --target controller \
  --output-dir artifacts/submission_eval/test_public_controller
```

That reproduces the actual submission-style controller row with the default 8-way sharding and
official merge/rescoring step.

To benchmark the published HF LoRA adapter instead, switch the target:

```bash
uv run python scripts/run_submission_eval.py \
  --upstream-root /tmp/hallmark-upstream \
  --split test_public \
  --target hf_policy \
  --hf-model-repo sebastianboehler/hallmark-mlx-qwen25-1.5b-lora \
  --output-dir artifacts/submission_eval/test_public_hf_policy
```

The HF-backed path resolves `adapters.safetensors` from the model repo, infers the base model
from the published manifest, and then runs the same official benchmark pipeline locally.

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

The repository includes generated official-split comparison artifacts at:

- `docs/figures/hallmark_official_splits.png`
- `docs/reports/hallmark_official_splits.md`

These use official HALLMARK splits only:

- `dev_public`
- `test_public`
- `stress_test`

Do not mix internal model-selection slices with official benchmark claims. Internal slices are
useful for debugging protocol behavior and tool budgets. Official splits such as `dev_public`,
`test_public`, and `stress_test` are the benchmark-wide reference points.

For repo-native Weco search over the budget-aware frontier, see `docs/weco.md`.
