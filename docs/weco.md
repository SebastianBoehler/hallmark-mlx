# Weco

`hallmark-mlx` uses Weco to optimize the citation-verification frontier rather than raw score at any cost.

There are two separate Weco loops:

- `weco_targets/hallmark_policy_trial.py`: inference/controller frontier for the deterministic policy.
- `weco_targets/hallmark_qwen_train_trial.py`: short-run Qwen LoRA frontier for the trainable policy.

## What To Optimize

The current scalar objective is `frontier_score`:

- `0.50 * budget_1_f1_hallucinated`
- `0.25 * budget_2_f1_hallucinated`
- `0.15 * budget_4_f1_hallucinated`
- `0.05 * budget_8_f1_hallucinated`
- `0.05 * label_accuracy`

This favors policies that:

- detect hallucinations correctly,
- finish in 1 or 2 tool calls when evidence is strong,
- keep label quality stable,
- and avoid tool-heavy but slow verification loops.

The helper lives in `src/hallmark_mlx/eval/frontier.py`.

## Search Surface

The default controller target is `weco_targets/hallmark_policy_trial.py`.

For the controller target, Weco should optimize:

- `POLICY_MODE`
- `model.max_rollout_rounds`
- `model.temperature`
- `model.force_bibtex_updater_first`
- per-tool `enabled` flags
- per-tool `rows`

For the Qwen training target, Weco should optimize:

- `model.max_tokens`
- `model.max_rollout_rounds`
- `training.num_layers`
- `training.learning_rate`
- `training.num_iterations`
- `training.max_seq_length`
- `training.grad_accumulation_steps`
- per-tool `enabled` flags
- per-tool `rows`

Weco should not optimize:

- the benchmark slice path,
- the tool-call budgets,
- verdict label semantics,
- metric definitions,
- or broad MLX training hyperparameters.

The evaluator fixes the frontier budgets to `(1, 2, 4, 8)` inside the Weco eval scripts so trial edits cannot change the objective itself.

## Files

- editable target: `weco_targets/hallmark_policy_trial.py`
- instructions: `weco_targets/hallmark_policy_trial_instructions.md`
- train target: `weco_targets/hallmark_qwen_train_trial.py`
- train instructions: `weco_targets/hallmark_qwen_train_trial_instructions.md`
- eval entrypoint: `scripts/weco_eval.py`
- train eval entrypoint: `scripts/weco_train_eval.py`
- launcher: `scripts/weco_run.py`
- tracked search split: `data/weco/hallmark_dev_search64_gold_traces.jsonl`
- tracked comparison split: `data/weco/hallmark_dev_compare32_gold_traces.jsonl`
- tracked split manifest: `data/weco/hallmark_dev_search_manifest.json`

## Usage

Install the optional dependency:

```bash
uv sync --extra weco
```

Run a local trial evaluation without Weco mutating anything:

```bash
python scripts/weco_eval.py --source weco_targets/hallmark_policy_trial.py
```

The default trial searches on the 64-example tracked search split. Keep the 32-example
comparison split out of the Weco loop and use it only for post-search evaluation.

Print the exact Weco command that will be used:

```bash
python scripts/weco_run.py --dry-run
```

Launch a search:

```bash
python scripts/weco_run.py --steps 12 --model gpt-5-mini
```

Launch a short Qwen-training search:

```bash
python scripts/weco_run.py \
  --source weco_targets/hallmark_qwen_train_trial.py \
  --instructions weco_targets/hallmark_qwen_train_trial_instructions.md \
  --steps 4 \
  --model gpt-5-mini
```

## Interpreting Results

The frontier matters more than a single scalar headline. Track at least:

- `frontier_score`
- `budget_1_f1_hallucinated`
- `budget_2_f1_hallucinated`
- `budget_4_f1_hallucinated`
- `label_accuracy`
- `avg_tool_calls`
- `completion_rate`

The best outcome for this repository is not “the LLM reasons better in free text.” The best outcome is a controller that chooses the right deterministic tools, stops early when evidence is sufficient, and escalates only when needed.
