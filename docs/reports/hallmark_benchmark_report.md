# HALLMARK Benchmark Comparison

## dev_public

| method | source | num_entries | partial | f1_hallucination | detection_rate | tier_weighted_f1 | false_positive_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallmark_mlx_bibtex_first_fallback | executed_here | 1119 | False | 0.934 | 0.963 | 0.952 | 0.156 | Merged 8 shard directories from /Users/sebastianboehler/Documents/GitHub/hallmark-mlx/artifacts/official_eval_sharded_fast5_confirm/dev_public_bibtex_first_fallback/shards. |
| bibtexupdater | upstream_published_result | 1119 | False | 0.908 | 0.946 | 0.936 | 0.179 | Published by upstream repo in bibtexupdater_dev_public.json. |
| llm_tool_augmented | upstream_published_predictions | 1119 | False | 0.846 | 0.818 | 0.856 | 0.144 | Predictions JSONL shipped upstream; metrics recomputed locally. |
| always_hallucinated | executed_here | 1119 | False | 0.723 | 1.000 | 0.842 | 1.000 | Generated 1119 predictions locally. |
| ensemble | upstream_history_partial | 500 | True | 0.610 | 0.500 | 0.544 | 0.016 | Historical upstream run; not a current full dev_public published result. |
| random | executed_here | 1119 | False | 0.523 | 0.485 | 0.575 | 0.481 | Generated 1119 predictions locally. |
| doi_presence_heuristic | executed_here | 1119 | False | 0.492 | 0.466 | 0.576 | 0.556 | Generated 1119 predictions locally. |
| title_oracle | executed_here | 1119 | False | 0.457 | 0.523 | 0.473 | 1.000 | Generated 1119 predictions locally. |
| venue_oracle | executed_here | 1119 | False | 0.399 | 0.250 | 0.396 | 0.000 | Generated 1119 predictions locally. |
| harc | upstream_published_result | 521 | True | 0.268 | 0.155 | 0.188 | 0.000 | Published by upstream repo in harc_dev_public.json. |
| verify_citations | upstream_history_partial | 500 | True | 0.240 | 0.300 | 0.302 | 0.133 | Historical upstream run; not a current full dev_public published result. |
| doi_only | upstream_history_partial | 500 | True | 0.163 | 0.240 | 0.175 | 0.189 | Historical upstream run; not a current full dev_public published result. |
| always_valid | executed_here | 1119 | False | 0.000 | 0.000 | 0.000 | 0.000 | Generated 1119 predictions locally. |
| doi_only_no_prescreening | not_run | 0 | False | - | - | - | - | Available here but not executed in this run. |
| bibtexupdater_no_prescreening | not_run | 0 | False | - | - | - | - | Available here but not executed in this run. |
| harc_no_prescreening | not_run | 0 | False | - | - | - | - | Available here but not executed in this run. |
| verify_citations_no_prescreening | unavailable | 0 | False | - | - | - | - | Missing CLI commands: verify-citations. Install the package that provides them. |
| llm_openai | unavailable | 0 | False | - | - | - | - | Missing packages: openai. Install with: pip install openai |
| llm_anthropic | unavailable | 0 | False | - | - | - | - | Missing packages: anthropic. Install with: pip install anthropic |
| llm_openrouter_deepseek_r1 | unavailable | 0 | False | - | - | - | - | Missing packages: openai. Install with: pip install openai |
| llm_openrouter_deepseek_v3 | unavailable | 0 | False | - | - | - | - | Missing packages: openai. Install with: pip install openai |
| llm_openrouter_qwen | unavailable | 0 | False | - | - | - | - | Missing packages: openai. Install with: pip install openai |
| llm_openrouter_mistral | unavailable | 0 | False | - | - | - | - | Missing packages: openai. Install with: pip install openai |
| llm_openrouter_gemini_flash | unavailable | 0 | False | - | - | - | - | Missing packages: openai. Install with: pip install openai |

## Provenance

- `executed_here`: run locally in this repo during comparison generation.
- `upstream_published_result`: official result JSON shipped by the upstream HALLMARK repo.
- `upstream_published_predictions`: official prediction JSONL shipped upstream and re-evaluated locally.
- `upstream_history_partial`: historical upstream result, usually partial coverage or smaller sample.
- `not_run`: baseline appears available here but was intentionally not executed in this pass.
- `unavailable`: baseline could not be run in this environment.
