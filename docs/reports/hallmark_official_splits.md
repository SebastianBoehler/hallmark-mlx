# HALLMARK Official Split Report

This report contains official HALLMARK split results only. Internal model-selection splits used by Weco are intentionally excluded.

- Hidden test available locally: False

## dev_public

| method | source | num_entries | partial | f1_hallucination | detection_rate | tier_weighted_f1 | false_positive_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallmark-mlx | executed_here | 1119 | False | 0.942 | 0.947 | 0.946 | 0.129 | Merged 8 shard directories from /Users/sebastianboehler/Documents/GitHub/hallmark-mlx/artifacts/official_eval_sharded_fast5/dev_public_bibtex_first_fallback/shards. |
| BibTeX Updater | upstream_published_result | 1119 | False | 0.908 | 0.946 | 0.936 | 0.179 | Published by upstream repo in bibtexupdater_dev_public.json. |
| HaRC | upstream_published_result | 521 | True | 0.268 | 0.155 | 0.188 | 0.000 | Published by upstream repo in harc_dev_public.json. |

## test_public

| method | source | num_entries | partial | f1_hallucination | detection_rate | tier_weighted_f1 | false_positive_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallmark-mlx | executed_here | 831 | False | 0.960 | 0.939 | 0.929 | 0.083 | Merged 8 shard directories from /Users/sebastianboehler/Documents/GitHub/hallmark-mlx/artifacts/official_eval_sharded_fast5/test_public_bibtex_first_fallback/shards. |

## stress_test

| method | source | num_entries | partial | f1_hallucination | detection_rate | tier_weighted_f1 | false_positive_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallmark-mlx | executed_here | 121 | False | 1.000 | 1.000 | 1.000 | - | Merged 8 shard directories from /Users/sebastianboehler/Documents/GitHub/hallmark-mlx/artifacts/official_eval_sharded_fast5/stress_test_bibtex_first_fallback/shards. |

## Notes

- `stress_test` is useful for robustness, but `FPR` is less meaningful there than on mixed official splits.
- `test_hidden` is not available in the local upstream checkout, so no local score is reported here.
