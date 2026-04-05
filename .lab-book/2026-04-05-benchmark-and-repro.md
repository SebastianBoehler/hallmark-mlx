# Benchmark And Repro Status

- Date: 2026-04-05
- Goal: lock the current reproducible benchmark state, package the kept trainable run, and compare a larger Mac-runnable model

## Official Confirmed Controller Rows

- Root: `artifacts/official_eval_sharded_fast5_confirm`
- `dev_public`: `F1-H 0.9344`, `DR 0.9634`, `TW-F1 0.9523`, `FPR 0.1556`, `ECE 0.0675`
- `test_public`: `F1-H 0.9280`, `DR 0.9444`, `TW-F1 0.9379`, `FPR 0.1722`, `ECE 0.0355`
- `stress_test`: `F1-H 0.9774`, `DR 0.9559`, `TW-F1 0.9745`, `ECE 0.1601`
- Decision: these confirmed rows replace the older stale cached official rows in public-facing docs

## Kept Trainable Policy

- Config: `configs/train_qwen_1_5b_kept.yaml`
- Dataset snapshot:
  `artifacts/weco/hallmark-qwen-train-frontier-iterative/61460529b25c/train/prepared_dataset`
- Internal tracked eval:
  - `compare32`: `F1-H 0.8649`, `FPR 0.3125`, `label_accuracy 0.8438`
  - `search64`: `F1-H 0.9126`, `FPR 0.5000`, `label_accuracy 0.8594`
- Decision: keep as the canonical trainable 1.5B baseline

## Larger Mac Comparison

- Config: `configs/train_qwen_3b_4bit_compare.yaml`
- Model: `mlx-community/Qwen2.5-3B-Instruct-4bit`
- Adapter:
  `artifacts/adapters/qwen25_3b_4bit_compare/adapters.safetensors`
- Internal tracked eval:
  - `compare32`: unchanged vs kept 1.5B
  - `search64`: `F1-H 0.9216`, `FPR 0.4667`, `avg_tool_calls 1.0`
- Decision: viable Mac-side comparison path, but not yet promoted over the kept 1.5B run

## HF Release Bundle

- Root: `artifacts/hf_release/qwen25_1_5b_kept`
- Includes:
  - dataset snapshot and source reviewed traces
  - LoRA adapter and training manifest
  - tracked eval metrics
  - official `dev_public` row
- Decision: ready for Hugging Face upload after final license choice

## Weco Notes

- Guarded Weco search did not beat the kept trainable trial on the held-out internal objective
- Small hand-added batches can move policy behavior materially, which suggests the trainable policy is still in a high-variance regime
- Decision: prefer larger, balanced batches and treat the deterministic controller as the benchmark-facing system of record

## Next

- if a model is promoted, rerun official splits before updating headline claims
- log future training or benchmark changes in a new dated file under `.lab-book/`
