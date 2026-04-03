"""Helpers for repo-native Weco optimization flows."""

from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from hallmark_mlx.config import AppConfig, load_config
from hallmark_mlx.eval.frontier import DEFAULT_FRONTIER_METRIC_ORDER
from hallmark_mlx.eval.policy_modes import build_policy_runner


class WecoSupportError(RuntimeError):
    """Raised when the local Weco integration is misconfigured."""


@dataclass(slots=True)
class WecoTrialSpec:
    """Editable Weco trial specification loaded from a Python source file."""

    source_path: Path
    base_config_path: Path
    eval_input_path: Path
    policy_mode: str
    trial_name: str
    overrides: dict[str, object]


def require_weco_cli() -> str:
    """Return the Weco CLI path or raise a helpful error."""

    executable = shutil.which("weco")
    if executable is None:
        sibling = Path(sys.executable).resolve().parent / "weco"
        if sibling.exists():
            executable = str(sibling)
    if executable is None:
        raise WecoSupportError(
            "Weco CLI not found. Install it first, for example with `uv sync --extra weco`."
        )
    return executable


def load_trial_spec(source_path: str | Path) -> WecoTrialSpec:
    """Load a Weco trial specification from a Python source file."""

    resolved = Path(source_path).resolve()
    spec = importlib.util.spec_from_file_location("hallmark_mlx_weco_trial", resolved)
    if spec is None or spec.loader is None:
        raise WecoSupportError(f"Could not load Weco trial source: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo_root = resolved.parent.parent
    try:
        base_config = repo_root / module.BASE_CONFIG_PATH
        eval_input = repo_root / module.EVAL_INPUT_PATH
        policy_mode = str(module.POLICY_MODE)
        trial_name = str(module.TRIAL_NAME)
        overrides = dict(module.TRIAL_OVERRIDES)
    except AttributeError as exc:
        raise WecoSupportError(f"Missing required field in Weco trial source: {resolved}") from exc
    return WecoTrialSpec(
        source_path=resolved,
        base_config_path=base_config.resolve(),
        eval_input_path=eval_input.resolve(),
        policy_mode=policy_mode,
        trial_name=trial_name,
        overrides=overrides,
    )


def _deep_merge(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _yaml_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_yaml_safe(item) for item in value]
    return value


def materialize_trial_config(spec: WecoTrialSpec, output_dir: str | Path) -> tuple[AppConfig, Path]:
    """Write a concrete YAML config for a Weco trial and load it."""

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(load_config(spec.base_config_path).model_dump_json())
    merged_payload = _yaml_safe(_deep_merge(payload, spec.overrides))
    materialized_path = output_root / f"{spec.trial_name}.yaml"
    materialized_path.write_text(
        yaml.safe_dump(merged_payload, sort_keys=False),
        encoding="utf-8",
    )
    return load_config(materialized_path), materialized_path


def build_weco_runner(config: AppConfig, policy_mode: str):
    """Build the runner used by a Weco trial."""

    return build_policy_runner(config, policy_mode)


def trial_output_dir(config: AppConfig, spec: WecoTrialSpec) -> Path:
    """Return the output directory for one materialized Weco trial."""

    path = config.weco.experiment_dir / spec.trial_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_metric_lines(metrics: dict[str, float]) -> list[str]:
    """Serialize numeric metrics into the plain-text format Weco expects."""

    lines: list[str] = []
    for key in DEFAULT_FRONTIER_METRIC_ORDER:
        value = metrics.get(key)
        if value is None:
            continue
        lines.append(f"{key}: {value:.6f}")
    return lines


def build_weco_eval_command(source_path: str | Path, python_executable: str) -> str:
    """Build the shell command passed to `weco run --eval-command`."""

    resolved_source = Path(source_path).resolve()
    repo_root = resolved_source.parent.parent
    command = [
        python_executable,
        str(repo_root / "scripts" / "weco_eval.py"),
        "--source",
        str(resolved_source),
    ]
    return shlex.join(command)
