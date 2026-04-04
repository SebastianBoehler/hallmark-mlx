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
        "max_tokens": 512,
        "max_rollout_rounds": 3,
        "finalization_mode": "deterministic",
        "force_bibtex_updater_first": True,
    },
    "training": {
        # moderate model capacity to balance rapid adaptation and representational power
        "num_layers": 10,
        # aggressive but in-range LR to speed short-run LoRA learning
        "learning_rate": 1.8e-4,
        "batch_size": 1,
        # short but sufficient iterations for quick iterative search
        "num_iterations": 100,
        "val_batches": 4,
        # expand context to allow reasoning over longer inputs while keeping concise outputs
        "max_seq_length": 5120,
        # larger effective batch via grad accumulation for stable updates
        "grad_accumulation_steps": 8,
        "save_every": 1000,
        "eval_every": 20,
        "report_every": 20,
        # vary seed to explore different optimization trajectories
        "seed": 7,
    },
    "tools": {
        # reduce rows to encourage 1-2 focused tool calls while preserving the same toolset
        "crossref": {"enabled": True, "rows": 2},
        "openalex": {"enabled": True, "rows": 2},
        "dblp": {"enabled": True, "rows": 2},
        "acl_anthology": {"enabled": True},
        "arxiv": {"enabled": True},
        "semantic_scholar": {"enabled": True, "rows": 2},
    },
}
