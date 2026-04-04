"""
Editable Weco search space for short-run Qwen policy training trials.

Weco should keep the structure intact and only adjust values inside TRIAL_OVERRIDES.
"""

BASE_CONFIG_PATH = "configs/train_qwen_1_5b.yaml"
EVAL_SCRIPT_PATH = "scripts/weco_train_eval.py"
EVAL_INPUT_PATH = "data/weco/hallmark_dev_search64_gold_traces.jsonl"
POLICY_MODE = "policy_deterministic"
TRIAL_NAME = "hallmark-qwen-train-frontier-focused-rows1"

TRIAL_OVERRIDES = {
    "model": {
        "backend": "mlx",
        # Keep deterministic generation and force BibTeX-first behavior
        "temperature": 0.0,
        "max_tokens": 1024,            # larger single-turn generation capacity to reduce multi-call workflows
        "max_rollout_rounds": 5,      # allow up to 5 rounds but encourage concise retrieval via tool rows=1
        "finalization_mode": "deterministic",
        "force_bibtex_updater_first": True,
    },
    "training": {
        # Lighter base with aggressive LoRA adaptation: fewer layers, higher LR, more iterations and grad accumulation
        "num_layers": 8,
        "learning_rate": 1.8e-4,
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
        # Strongly constrain retrieved rows to force the policy to pick the best 1 (or 2) results
        "crossref": {"enabled": True, "rows": 1},
        "openalex": {"enabled": True, "rows": 1},
        "dblp": {"enabled": True, "rows": 1},
        "acl_anthology": {"enabled": True, "rows": 1},
        "arxiv": {"enabled": True, "rows": 1},
        "semantic_scholar": {"enabled": True, "rows": 1},
    },
}
