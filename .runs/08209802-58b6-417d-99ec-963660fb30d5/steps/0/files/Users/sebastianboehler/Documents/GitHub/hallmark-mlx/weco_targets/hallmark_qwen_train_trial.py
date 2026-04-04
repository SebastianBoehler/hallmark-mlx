"""
Editable Weco search space for short-run Qwen policy training trials.

Weco should keep the structure intact and only adjust values inside TRIAL_OVERRIDES.
"""

BASE_CONFIG_PATH = "configs/train_qwen_1_5b.yaml"
EVAL_SCRIPT_PATH = "scripts/weco_train_eval.py"
EVAL_INPUT_PATH = "data/weco/hallmark_dev_search64_gold_traces.jsonl"
POLICY_MODE = "policy_deterministic"
TRIAL_NAME = "hallmark-qwen-train-frontier"

TRIAL_OVERRIDES = {
    "model": {
        "backend": "mlx",
        "temperature": 0.0,
        "max_tokens": 768,
        "max_rollout_rounds": 4,
        "finalization_mode": "deterministic",
        "force_bibtex_updater_first": True,
    },
    "training": {
        "num_layers": 12,
        "learning_rate": 8e-5,
        "batch_size": 1,
        "num_iterations": 80,
        "val_batches": 4,
        "max_seq_length": 4096,
        "grad_accumulation_steps": 4,
        "save_every": 1000,
        "eval_every": 20,
        "report_every": 20,
        "seed": 42,
    },
    "tools": {
        "crossref": {"enabled": True, "rows": 3},
        "openalex": {"enabled": True, "rows": 3},
        "dblp": {"enabled": True, "rows": 3},
        "acl_anthology": {"enabled": True},
        "arxiv": {"enabled": True},
        "semantic_scholar": {"enabled": True, "rows": 3},
    },
    "eval": {
        "tool_call_budgets": [1, 2, 4, 8],
    },
}
