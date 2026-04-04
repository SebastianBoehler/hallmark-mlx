``"
Editable Weco search space for short-run Qwen policy training trials.

Weco should keep the structure intact and only adjust values inside TRIAL_OVERRIDES.
"""

BASE_CONFIG_PATH = "configs/train_qwen_1_5b.yaml"
EVAL_SCRIPT_PATH = "scripts/weco_train_eval.py"
EVAL_INPUT_PATH = "data/weco/hallmark_dev_search64_gold_traces.jsonl"
POLICY_MODE = "policy_deterministic"
TRIAL_NAME = "hallmark-qwen-train-frontier-explore-ctx"

TRIAL_OVERRIDES = {
    "model": {
        "backend": "mlx",
        "temperature": 0.0,
        # Increase max_tokens to allow more informative, longer generations for finalization
        "max_tokens": 1024,
        # Encourage concise resolution by limiting rollout rounds to 3 (aim for 1-2 tool calls)
        "max_rollout_rounds": 3,
        # Preserve deterministic finalization as required
        "finalization_mode": "deterministic",
        # Keep BibTeX-first behavior strong for bibtex_entry inputs
        "force_bibtex_updater_first": True,
    },
    "training": {
        # Explore a slightly smaller layer count to test capacity vs. generalization tradeoff
        "num_layers": 10,
        # Slightly higher learning rate for faster adaptation during short LoRA runs
        "learning_rate": 1.2e-4,
        "batch_size": 1,
        # Increase iterations to explore a longer short-run budget
        "num_iterations": 120,
        "val_batches": 4,
        # Increase sequence length to capture more context from the documents
        "max_seq_length": 6144,
        # Increase grad accumulation to keep effective batch dynamics stable
        "grad_accumulation_steps": 8,
        "save_every": 1000,
        "eval_every": 20,
        "report_every": 20,
        # Use a different seed to diversify exploration
        "seed": 7,
    },
    "tools": {
        # Reduce rows per tool to encourage fewer, higher-quality results per call
        "crossref": {"enabled": True, "rows": 2},
        "openalex": {"enabled": True, "rows": 2},
        # Disable less relevant / noisy sources to keep tool calls focused
        "dblp": {"enabled": False, "rows": 0},
        "acl_anthology": {"enabled": False},
        "arxiv": {"enabled": True, "rows": 2},
        "semantic_scholar": {"enabled": True, "rows": 2},
    },
}
