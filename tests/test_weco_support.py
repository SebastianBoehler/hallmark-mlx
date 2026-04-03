from pathlib import Path

import yaml

from hallmark_mlx.weco_support import load_trial_spec, materialize_trial_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_trial_spec_reads_repo_relative_paths() -> None:
    spec = load_trial_spec(REPO_ROOT / "weco_targets" / "hallmark_policy_trial.py")

    assert spec.trial_name == "hallmark-policy-frontier"
    assert spec.base_config_path == (REPO_ROOT / "configs" / "train_qwen_1_5b.yaml").resolve()
    assert spec.eval_input_path == (
        REPO_ROOT / "data" / "weco" / "hallmark_dev_search64_gold_traces.jsonl"
    ).resolve()


def test_materialized_trial_config_keeps_absolute_base_paths(tmp_path: Path) -> None:
    spec = load_trial_spec(REPO_ROOT / "weco_targets" / "hallmark_policy_trial.py")

    config, materialized_path = materialize_trial_config(spec, tmp_path)
    payload = yaml.safe_load(materialized_path.read_text(encoding="utf-8"))

    assert Path(payload["paths"]["processed_dir"]).is_absolute()
    assert config.weco.objective_name == "frontier_score"
    assert config.model.force_bibtex_updater_first is True
