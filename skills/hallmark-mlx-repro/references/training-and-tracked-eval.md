# Training And Tracked Eval

Use this file for trainable-policy reproduction only. `search64` and `compare32` are internal
model-selection splits, not official benchmark results.

## Canonical Dataset Snapshot

- Exact prepared dataset snapshot:
  `artifacts/weco/hallmark-qwen-train-frontier-iterative/61460529b25c/train/prepared_dataset`
- Rows:
  - `train.jsonl`: 75
  - `valid.jsonl`: 6
- Source reviewed traces:
  `examples/reviewed_seed_traces_combined.jsonl`

Reuse that prepared dataset for apples-to-apples model comparisons.

## Kept Qwen 1.5B Run

- Config:
  `configs/train_qwen_1_5b_kept.yaml`
- Train:
  ```bash
  PYTHONPATH=src uv run python -m hallmark_mlx.cli train \
    --config configs/train_qwen_1_5b_kept.yaml
  ```
- Adapter target:
  `artifacts/adapters/qwen25_1_5b_kept`

Tracked eval:

```bash
PYTHONPATH=src uv run python -m hallmark_mlx.cli eval-policy \
  --config configs/train_qwen_1_5b_kept.yaml \
  --input-path data/weco/hallmark_dev_compare32_gold_traces.jsonl \
  --output-path artifacts/confirm_qwen_kept_compare32_metrics.json \
  --predictions-path artifacts/confirm_qwen_kept_compare32_predictions.jsonl \
  --traces-path artifacts/confirm_qwen_kept_compare32_traces.jsonl

PYTHONPATH=src uv run python -m hallmark_mlx.cli eval-policy \
  --config configs/train_qwen_1_5b_kept.yaml \
  --input-path data/weco/hallmark_dev_search64_gold_traces.jsonl \
  --output-path artifacts/confirm_qwen_kept_search64_metrics.json \
  --predictions-path artifacts/confirm_qwen_kept_search64_predictions.jsonl \
  --traces-path artifacts/confirm_qwen_kept_search64_traces.jsonl
```

Current kept numbers:

- `compare32`: `F1-H 0.8649`, `FPR 0.3125`, `label_accuracy 0.8438`
- `search64`: `F1-H 0.9126`, `FPR 0.5000`, `label_accuracy 0.8594`

## Larger Mac Comparison: Qwen 3B 4-bit

- Config:
  `configs/train_qwen_3b_4bit_compare.yaml`
- Train:
  ```bash
  PYTHONPATH=src uv run python -m hallmark_mlx.cli train \
    --config configs/train_qwen_3b_4bit_compare.yaml
  ```
- Adapter target:
  `artifacts/adapters/qwen25_3b_4bit_compare`

Tracked eval:

```bash
PYTHONPATH=src uv run python -m hallmark_mlx.cli eval-policy \
  --config configs/train_qwen_3b_4bit_compare.yaml \
  --input-path data/weco/hallmark_dev_compare32_gold_traces.jsonl \
  --output-path artifacts/eval_qwen25_3b_4bit_compare_compare32_metrics.json \
  --predictions-path artifacts/eval_qwen25_3b_4bit_compare_compare32_predictions.jsonl \
  --traces-path artifacts/eval_qwen25_3b_4bit_compare_compare32_traces.jsonl

PYTHONPATH=src uv run python -m hallmark_mlx.cli eval-policy \
  --config configs/train_qwen_3b_4bit_compare.yaml \
  --input-path data/weco/hallmark_dev_search64_gold_traces.jsonl \
  --output-path artifacts/eval_qwen25_3b_4bit_compare_search64_metrics.json \
  --predictions-path artifacts/eval_qwen25_3b_4bit_compare_search64_predictions.jsonl \
  --traces-path artifacts/eval_qwen25_3b_4bit_compare_search64_traces.jsonl
```

Current 3B 4-bit numbers:

- `compare32`: unchanged vs kept 1.5B
- `search64`: `F1-H 0.9216`, `FPR 0.4667`, `avg_tool_calls 1.0`

## Notes

- The raw upstream `Qwen/Qwen2.5-3B-Instruct` download is much slower here; prefer the MLX 4-bit variant.
- If a new trainable run beats the kept numbers on tracked evals, promote it only after an official rerun.
- Record promoted runs, rejected runs, and the reason in `.lab-book/`.
