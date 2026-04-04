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
        # Increase generation capacity so the model can synthesize richer evidence from
        # fuller retrieval results in 1-2 tool calls.
        "max_tokens": 1024,
        # Keep rollout rounds modest to encourage resolving within 1-2 calls,
        # but allow a short follow-up if needed.
        "max_rollout_rounds": 3,
        "finalization_mode": "deterministic",
        "force_bibtex_updater_first": True,
    },
    "training": {
        # Use a compact model (fewer layers) for faster, focused LoRA adaptation,
        # but extend iterations and grad accumulation to improve short-run learning.
        "num_layers": 8,
        "learning_rate": 1.2e-4,
        "batch_size": 1,
        "num_iterations": 140,
        "val_batches": 4,
        "max_seq_length": 6144,
        "grad_accumulation_steps": 8,
        "save_every": 1000,
        "eval_every": 20,
        "report_every": 20,
        "seed": 42,
    },
    "tools": {
        # Increase rows per tool so a single tool call returns richer evidence,
        # encouraging resolution within 1-2 calls while keeping the same toolset.
        "crossref": {"enabled": True, "rows": 5},
        "openalex": {"enabled": True, "rows": 5},
        "dblp": {"enabled": True, "rows": 5},
        "acl_anthology": {"enabled": True},
        "arxiv": {"enabled": True},
        "semantic_scholar": {"enabled": True, "rows": 5},
    },
}
