"""
Editable Weco search space for short-run Qwen policy training trials.

Weco should keep the structure intact and only adjust values inside TRIAL_OVERRIDES.
"""

BASE_CONFIG_PATH = "configs/train_qwen_1_5b.yaml"
EVAL_SCRIPT_PATH = "scripts/weco_train_eval.py"
EVAL_INPUT_PATH = "data/weco/hallmark_dev_search64_gold_traces.jsonl"
POLICY_MODE = "policy_deterministic"
TRIAL_NAME = "hallmark-qwen-train-frontier-iterative"

TRIAL_OVERRIDES = {
    "model": {
        "backend": "mlx",
        # Keep deterministic behavior; concise outputs via low max_tokens
        "temperature": 0.0,
        "max_tokens": 512,
        # Allow more rollout rounds to enable iterative refinement while forcing
        # each generation to be concise and focused on tool usage.
        "max_rollout_rounds": 5,
        "finalization_mode": "deterministic",
        # Preserve strong BibTeX-first behavior
        "force_bibtex_updater_first": True,
    },
    "training": {
        # Mid-high LoRA capacity to allow richer behavior adaptation
        "num_layers": 14,
        "learning_rate": 1.4e-4,
        "batch_size": 1,
        # Keep the run short-but-exploratory within recommended bounds
        "num_iterations": 140,
        "val_batches": 4,
        # Large context to allow multi-turn reasoning across rollouts
        "max_seq_length": 5120,
        # Moderate effective batch size for stability
        "grad_accumulation_steps": 6,
        "save_every": 1000,
        "eval_every": 20,
        "report_every": 20,
        # Change seed to diversify exploration from prior runs
        "seed": 7,
    },
    "tools": {
        # Give stronger retrieval depth to high-quality scholarly sources,
        # while keeping other tools limited to encourage focused calls.
        "crossref": {"enabled": True, "rows": 4},
        "openalex": {"enabled": True, "rows": 2},
        "dblp": {"enabled": True, "rows": 1},
        "acl_anthology": {"enabled": True},
        "arxiv": {"enabled": True, "rows": 2},
        "semantic_scholar": {"enabled": True, "rows": 5},
    },
}
