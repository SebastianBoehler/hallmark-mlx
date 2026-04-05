---
name: hallmark-mlx-repro
description: Reproduce hallmark-mlx training, tracked evals, official HALLMARK benchmark runs, and Hugging Face release bundles. Use when working inside this repo to rerun the kept Qwen 1.5B policy, compare against the larger 3B 4-bit MLX model, refresh official dev_public/test_public/stress_test reports, or package the exact dataset and LoRA adapter artifacts for publication.
---

# Hallmark MLX Repro

## Overview

Use this skill to keep `hallmark-mlx` reproducible. It covers the canonical trainable-policy
configs, tracked internal evals, official benchmark reruns, report refresh, and Hugging Face
release bundles.

## Workflow

1. Pick the target surface.
   - Train or compare policy models: read `references/training-and-tracked-eval.md`.
   - Rerun official HALLMARK splits or refresh benchmark plots/tables: read `references/official-benchmarking.md`.
   - Prepare the dataset and LoRA adapter for Hugging Face release: read `references/hf-release.md`.
2. Prefer the canonical configs and scripts in this repo over ad hoc commands.
3. Keep internal model-selection results separate from official benchmark results.
4. Record material experiment changes, benchmark reruns, and promotion decisions in `.lab-book/`.

## Guardrails

- Treat `search64` and `compare32` as internal Weco/model-selection splits only.
- Treat `artifacts/official_eval_sharded_fast5_confirm` as the canonical confirmed official result root.
- If a fresh official rerun differs from a cached report, refresh the reports before making claims.
- For apples-to-apples model comparisons, reuse the prepared dataset snapshot from the kept 1.5B run instead of rebuilding a new dataset.
- On this Mac, prefer `mlx-community/Qwen2.5-3B-Instruct-4bit` over the raw upstream 3B model.
- Log benchmark-facing changes in a dated `.lab-book/` note before or after refreshing public reports.

## Canonical Targets

- Kept 1.5B policy: `configs/train_qwen_1_5b_kept.yaml`
- Larger Mac comparison: `configs/train_qwen_3b_4bit_compare.yaml`
- Official rerun entrypoints: `scripts/run_official_controller_eval.py`, `scripts/merge_official_eval_shards.py`
- Report refresh: `scripts/refresh_confirmed_benchmarks.py`
- HF release bundle: `scripts/prepare_hf_release.py`
- Experiment notes: `.lab-book/`

## References

- `references/training-and-tracked-eval.md`
- `references/official-benchmarking.md`
- `references/hf-release.md`
