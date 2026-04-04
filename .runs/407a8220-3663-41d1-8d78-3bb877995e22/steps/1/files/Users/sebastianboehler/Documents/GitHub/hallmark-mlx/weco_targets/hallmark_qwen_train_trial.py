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
        # reduce token budget to encourage concise answers but keep enough context
        "max_tokens": 768,
        # allow one extra rollout round for more careful tool use when needed
        "max_rollout_rounds": 4,
        "finalization_mode": "deterministic",
        # keep strong bibtex-first behavior
        "force_bibtex_updater_first": True,
    },
    "training": {
        # move to slightly larger LoRA capacity
        "num_layers": 12,
        # lower LR for more stable short-run LoRA fine-tuning
        "learning_rate": 8e-5,
        "batch_size": 1,
        # shorten for quick iterative frontier exploration
        "num_iterations": 80,
        # slightly increase validation checks to better estimate frontier after each short run
        "val_batches": 6,
        # reduce sequence length to speed up runs while retaining necessary context
        "max_seq_length": 4096,
        # fewer grad accumulation steps to reduce effective batch size and enable faster updates
        "grad_accumulation_steps": 4,
        # checkpointing cadence appropriate for short runs
        "save_every": 100,
        "eval_every": 10,
        "report_every": 10,
        "seed": 123,
    },
    "tools": {
        # narrow the retrieval set to high-precision sources and reduce rows per tool
        "crossref": {"enabled": True, "rows": 1},
        "openalex": {"enabled": True, "rows": 1},
        # disable noisier / broader sources to reduce hallucinated but irrelevant retrievals
        "dblp": {"enabled": False, "rows": 0},
        "acl_anthology": {"enabled": False},
        "arxiv": {"enabled": False},
        "semantic_scholar": {"enabled": True, "rows": 1},
    },
}
