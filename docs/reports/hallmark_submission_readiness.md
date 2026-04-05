# HALLMARK Submission Readiness

## Protocol Check

- split == dev_public: True
- num_entries == 1119: True
- partial == False: True
- coverage == 1.0: True
- source == executed_here: True
- Official row file: `/Users/sebastianboehler/Documents/GitHub/hallmark-mlx/artifacts/official_eval_sharded_fast5_confirm/dev_public_bibtex_first_fallback/merged/row.json`
- Official result file: `/Users/sebastianboehler/Documents/GitHub/hallmark-mlx/artifacts/official_eval_sharded_fast5_confirm/dev_public_bibtex_first_fallback/merged/result.json`
- Runtime disclosure: 5s per-entry timeout, 8-way sharded run, merged and rescored with the upstream official evaluator.

## Leaderboard Snapshot

Compared against the public Agora leaderboard snapshot from 2026-04-02 at https://rpatrik96.github.io/research-agora/benchmarks.html.

| rank | name | type | detection_rate | f1_hallucination | tier_weighted_f1 | false_positive_rate | ece | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hallmark-mlx | HYBRID | 0.963 | 0.934 | 0.952 | 0.156 | 0.068 | executed_here |
| 2 | bibtex-updater | TOOL | 0.946 | 0.908 | 0.936 | 0.179 | 0.297 | research_agora_snapshot |
| 3 | GPT-5.1 | LLM | 0.797 | 0.822 | 0.846 | 0.171 | 0.107 | research_agora_snapshot |
| 4 | Qwen 3 235B | LLM | 0.832 | 0.737 | 0.806 | 0.551 | 0.294 | research_agora_snapshot |
| 5 | DeepSeek R1 | LLM | 0.871 | 0.737 | 0.814 | 0.640 | 0.247 | research_agora_snapshot |
| 6 | Mistral Large | LLM | 0.691 | 0.731 | 0.743 | 0.258 | 0.247 | research_agora_snapshot |
| 7 | DeepSeek V3 | LLM | 0.880 | 0.721 | 0.805 | 0.730 | 0.331 | research_agora_snapshot |
| 8 | Gemini 2.5 Flash | LLM | 0.482 | 0.617 | 0.608 | 0.101 | 0.286 | research_agora_snapshot |
| 9 | DOI-only | TOOL | 0.256 | 0.361 | 0.314 | 0.195 | 0.143 | research_agora_snapshot |
| 10 | HaRC | TOOL | 0.143 | 0.250 | 0.165 | 0.002 | 0.011 | research_agora_snapshot |
| 11 | verify-citations | TOOL | 0.300 | 0.240 | 0.302 | 0.133 | — | research_agora_snapshot |

## Outcome

`hallmark-mlx` ranks #1 by `F1-H` in this snapshot.
- Our row: DR 0.963, F1-H 0.934, TW-F1 0.952, FPR 0.156, ECE 0.068.
- Current public top row before ours: bibtex-updater with DR 0.946, F1-H 0.908, TW-F1 0.936, FPR 0.179, ECE 0.297.

## Submission Caveats

- This is a full `dev_public` result, not a hidden-test result.
- The evaluation logic is upstream-official; the orchestration is local.
- The timeout/sharding setup should be disclosed in any submission notes.

## Reproduction

```bash
for offset in 0 140 280 420 560 700 840 980; do
  shard=$(printf "shard_%02d" $((offset / 140)))
  uv run python scripts/run_official_controller_eval.py \
    --upstream-root /tmp/hallmark-upstream \
    --split dev_public \
    --mode bibtex_first_fallback \
    --output-dir \
      artifacts/official_eval_sharded_fast5_confirm/ \
      dev_public_bibtex_first_fallback/shards/${shard} \
    --entry-timeout-seconds 5 \
    --offset ${offset} \
    --limit 140
done

description=$(
  printf %s     "hallmark-mlx controller with deterministic finalizer "     "(5s per-entry timeout, 8-way sharded official run)"
)

uv run python scripts/merge_official_eval_shards.py \
  --upstream-root /tmp/hallmark-upstream \
  --split dev_public \
  --shards-root artifacts/official_eval_sharded_fast5_confirm/ \
    dev_public_bibtex_first_fallback/shards \
  --output-dir artifacts/official_eval_sharded_fast5_confirm/ \
    dev_public_bibtex_first_fallback/merged \
  --method-name hallmark_mlx_bibtex_first_fallback \
  --description "${description}"
```

## Extra Context

- Full metrics from `result.json`: AUROC 0.893, AUPRC 0.900, MCC 0.825, mean API calls 0.962.
