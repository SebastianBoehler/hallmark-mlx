# Hugging Face Release

Use this file when packaging the trainable-policy artifacts for publication.

## Canonical Release Bundle Command

```bash
PYTHONPATH=src uv run python scripts/prepare_hf_release.py
```

Default output:

`artifacts/hf_release/qwen25_1_5b_kept`

## What It Packages

Dataset bundle:

- `dataset/train.jsonl`
- `dataset/valid.jsonl`
- `dataset/reviewed_seed_traces_combined.jsonl`
- `dataset/README.md`
- `dataset/metadata.json`

Model bundle:

- `model/adapters.safetensors`
- `model/adapter_config.json`
- `model/training_manifest.json`
- `model/tracked_eval_search64.json`
- `model/tracked_eval_compare32.json`
- `model/official_dev_public_row.json`
- `model/README.md`
- `model/.gitattributes`

Upload guide:

- `UPLOAD.md`

## Source Artifacts

The default bundle uses the exact kept trainable-policy run:

- Adapter:
  `artifacts/weco/hallmark-qwen-train-frontier-iterative/61460529b25c/adapter`
- Prepared dataset snapshot:
  `artifacts/weco/hallmark-qwen-train-frontier-iterative/61460529b25c/train/prepared_dataset`
- Official controller reference row:
  `artifacts/official_eval_sharded_fast5_confirm/dev_public_bibtex_first_fallback/merged/row.json`

## Upload

Use the generated guide:

`artifacts/hf_release/qwen25_1_5b_kept/UPLOAD.md`

The guide contains `hf repo create` and `hf upload` examples for both the dataset and model repo.

## Pre-Publish Checks

- Set the final public license. The generated cards currently use `license: other`.
- Confirm the README cards still match the exact artifacts you intend to upload.
- Do not swap in a different prepared dataset or adapter without updating the model and dataset cards.
