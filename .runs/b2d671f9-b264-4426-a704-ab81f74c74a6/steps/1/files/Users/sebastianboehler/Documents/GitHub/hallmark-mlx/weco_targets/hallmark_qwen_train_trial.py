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
        # reduce max tokens to speed iterations while keeping enough output capacity
        "max_tokens": 512,
        # prefer fewer rollout rounds to encourage 1-2 tool calls
        "max_rollout_rounds": 3,
        "finalization_mode": "deterministic",
        # keep strong bibtex-first behavior
        "force_bibtex_updater_first": True,
    },
    "training": {
        # explore a smaller fine-tuned head (fewer layers) for fast LoRA adaptation
        "num_layers": 8,
        # slightly higher LR to adapt quickly in a short-run LoRA
        "learning_rate": 1.2e-4,
        "batch_size": 1,
        # more iterations within short-run bounds to improve frontier tuning
        "num_iterations": 120,
        "val_batches": 4,
        # allow longer context for complex bibtex entries
        "max_seq_length": 6144,
        # increase grad accumulation to keep effective batch size while staying memory-friendly
        "grad_accumulation_steps": 8,
        "save_every": 1000,
        "eval_every": 20,
        "report_every": 20,
        # change seed for exploration diversity
        "seed": 7,
    },
    "tools": {
        # keep the same tool set but reduce per-tool rows to encourage concise retrievals
        "crossref": {"enabled": True, "rows": 2},
        "openalex": {"enabled": True, "rows": 2},
        "dblp": {"enabled": True, "rows": 2},
        "acl_anthology": {"enabled": True},
        "arxiv": {"enabled": True},
        "semantic_scholar": {"enabled": True, "rows": 2},
    },
}
