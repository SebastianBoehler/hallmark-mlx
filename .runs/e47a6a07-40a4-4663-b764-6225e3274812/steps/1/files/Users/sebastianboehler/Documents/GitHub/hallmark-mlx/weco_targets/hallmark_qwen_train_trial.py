/*
Editable Weco search space for short-run Qwen policy training trials.

Weco should keep the structure intact and only adjust values inside TRIAL_OVERRIDES.
*/

BASE_CONFIG_PATH = "configs/train_qwen_1_5b.yaml"
EVAL_SCRIPT_PATH = "scripts/weco_train_eval.py"
EVAL_INPUT_PATH = "data/weco/hallmark_dev_search64_gold_traces.jsonl"
POLICY_MODE = "policy_deterministic"
TRIAL_NAME = "hallmark-qwen-train-frontier"

TRIAL_OVERRIDES = {
    "model": {
        "backend": "mlx",
        # preserve deterministic finalization and BibTeX-first behavior
        "temperature": 0.0,
        # Increase token budget to give the model more context for citation traces
        "max_tokens": 1024,
        # Slightly increase rollout depth to allow one extra retrieval/refinement step
        "max_rollout_rounds": 5,
        "finalization_mode": "deterministic",
        "force_bibtex_updater_first": True,
    },
    "training": {
        # explore a slightly leaner architecture for faster LoRA adaptation
        "num_layers": 10,
        # larger learning rate to speed short-run adaptation
        "learning_rate": 1.5e-4,
        "batch_size": 1,
        # keep the run short but a bit longer than baseline for frontier gains
        "num_iterations": 120,
        # evaluate on a few more batches to better estimate frontier during short runs
        "val_batches": 6,
        "max_seq_length": 6144,
        # simulate larger effective batch with more grad accumulation
        "grad_accumulation_steps": 8,
        # keep frequent reporting/eval cadence for iterative search
        "save_every": 1000,
        "eval_every": 20,
        "report_every": 20,
        "seed": 42,
    },
    "tools": {
        # keep the same toolset but reduce rows to encourage more precise retrievals
        "crossref": {"enabled": True, "rows": 2},
        "openalex": {"enabled": True, "rows": 2},
        "dblp": {"enabled": True, "rows": 2},
        "acl_anthology": {"enabled": True},
        "arxiv": {"enabled": True},
        "semantic_scholar": {"enabled": True, "rows": 2},
    },
}
