"""
Editable Weco search space for budget-aware citation verification policy runs.

Weco should keep the structure intact and only adjust values inside TRIAL_OVERRIDES
or POLICY_MODE.
"""

BASE_CONFIG_PATH = "configs/train_qwen_1_5b.yaml"
EVAL_INPUT_PATH = "data/weco/hallmark_dev_search64_gold_traces.jsonl"
POLICY_MODE = "bibtex_first_fallback"
TRIAL_NAME = "hallmark-policy-frontier"

TRIAL_OVERRIDES = {
    "model": {
        "temperature": 0.0,
        "max_rollout_rounds": 4,
        "finalization_mode": "deterministic",
        "force_bibtex_updater_first": True,
    },
    "tools": {
        "crossref": {"enabled": True, "rows": 3},
        "openalex": {"enabled": True, "rows": 3},
        "dblp": {"enabled": True, "rows": 3},
        "acl_anthology": {"enabled": True},
        "arxiv": {"enabled": True},
        "semantic_scholar": {"enabled": True, "rows": 3},
    },
}
