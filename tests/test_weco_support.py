from pathlib import Path

import yaml

from hallmark_mlx.weco_support import (
    load_trial_spec,
    materialize_trial_config,
    trial_run_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_trial_spec_reads_repo_relative_paths() -> None:
    spec = load_trial_spec(REPO_ROOT / "weco_targets" / "hallmark_policy_trial.py")

    assert spec.trial_name == "hallmark-policy-frontier"
    assert spec.base_config_path == (REPO_ROOT / "configs" / "train_qwen_1_5b.yaml").resolve()
    assert spec.eval_input_path == (
        REPO_ROOT / "data" / "weco" / "hallmark_dev_search64_gold_traces.jsonl"
    ).resolve()
    assert spec.eval_script_path == (REPO_ROOT / "scripts" / "weco_eval.py").resolve()


def test_materialized_trial_config_keeps_absolute_base_paths(tmp_path: Path) -> None:
    spec = load_trial_spec(REPO_ROOT / "weco_targets" / "hallmark_policy_trial.py")

    config, materialized_path = materialize_trial_config(spec, tmp_path)
    payload = yaml.safe_load(materialized_path.read_text(encoding="utf-8"))

    assert Path(payload["paths"]["processed_dir"]).is_absolute()
    assert config.weco.objective_name == "frontier_score"
    assert config.model.force_bibtex_updater_first is True


def test_load_trial_spec_reads_custom_eval_script() -> None:
    spec = load_trial_spec(REPO_ROOT / "weco_targets" / "hallmark_qwen_train_trial.py")

    assert spec.eval_script_path == (REPO_ROOT / "scripts" / "weco_train_eval.py").resolve()
    assert spec.policy_mode == "policy_deterministic"


def test_trial_run_fingerprint_is_stable(tmp_path: Path) -> None:
    config_path = tmp_path / "trial.yaml"
    config_path.write_text("model:\n  temperature: 0.0\n", encoding="utf-8")

    first = trial_run_fingerprint(config_path)
    second = trial_run_fingerprint(config_path)

    assert first == second
    assert len(first) == 12
