"""Configuration models and YAML loading helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from hallmark_mlx.types import FinalizationMode, ModelBackend, TrainingExampleFormat


class PathsConfig(BaseModel):
    """Filesystem layout for data and artifacts."""

    raw_trace_path: Path = Path("data/raw/traces.jsonl")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    predictions_path: Path = Path("artifacts/predictions.jsonl")


class ModelConfig(BaseModel):
    """Policy model configuration."""

    backend: ModelBackend = ModelBackend.MLX
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: Path = Path("artifacts/adapters/latest")
    max_tokens: int = 768
    temperature: float = 0.0
    max_rollout_rounds: int = 6  # needs room for n_tools calls + at least 1 finalization round
    system_prompt_version: str = "v0"
    finalization_mode: FinalizationMode = FinalizationMode.DETERMINISTIC
    force_bibtex_updater_first: bool = True


class ToolServiceConfig(BaseModel):
    """Per-tool enablement and runtime controls."""

    enabled: bool = True
    timeout_seconds: float = 15.0
    rows: int = 5
    command: str | None = None
    update_command: str | None = None  # explicit binary for update_bibtex; avoids fragile string replacement
    email: str | None = None


class ToolsConfig(BaseModel):
    """Verification tool configuration."""

    bibtex_updater: ToolServiceConfig = Field(
        default_factory=lambda: ToolServiceConfig(command="bibtex-check"),
    )
    crossref: ToolServiceConfig = Field(default_factory=ToolServiceConfig)
    openalex: ToolServiceConfig = Field(default_factory=ToolServiceConfig)
    dblp: ToolServiceConfig = Field(default_factory=ToolServiceConfig)
    acl_anthology: ToolServiceConfig = Field(default_factory=ToolServiceConfig)
    arxiv: ToolServiceConfig = Field(default_factory=ToolServiceConfig)
    semantic_scholar: ToolServiceConfig = Field(default_factory=ToolServiceConfig)


class DatasetConfig(BaseModel):
    """Dataset build and split controls."""

    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    group_by: str = "citation_family"
    private_holdout_tag: str = "private_holdout"

    @model_validator(mode="after")
    def validate_ratios(self) -> "DatasetConfig":
        if self.valid_ratio + self.test_ratio >= 1:
            msg = "valid_ratio + test_ratio must be < 1."
            raise ValueError(msg)
        return self


class TrainingConfig(BaseModel):
    """MLX LoRA planning configuration."""

    enabled: bool = True
    output_dir: Path = Path("artifacts/train")
    example_format: TrainingExampleFormat = TrainingExampleFormat.TOOL_TRANSCRIPT_STEPS
    python_executable: str = sys.executable
    fine_tune_type: str = "lora"
    optimizer: str = "adamw"
    mask_prompt: bool = True
    num_layers: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 1
    num_iterations: int = 300
    val_batches: int = 10
    max_seq_length: int = 6144
    grad_accumulation_steps: int = 4
    save_every: int = 50
    eval_every: int = 25
    report_every: int = 10
    test_batches: int = -1
    grad_checkpoint: bool = True
    seed: int = 42


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    split_name: str = "dev_public"
    gold_path: Path = Path("data/hallmark/dev_public.jsonl")
    output_path: Path = Path("artifacts/eval/metrics.json")
    tool_call_budgets: tuple[int, ...] = (1, 2, 4, 8)


class WecoConfig(BaseModel):
    """Optional Weco experiment hook."""

    enabled: bool = False
    command: str = "weco"
    objective_name: str = "frontier_score"
    experiment_dir: Path = Path("artifacts/weco")


class AppConfig(BaseModel):
    """Top-level application config."""

    project_name: str = "hallmark-mlx"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    data: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    weco: WecoConfig = Field(default_factory=WecoConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load an application config from YAML."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(payload)
    base_dir = config_path.parent

    def _resolve(candidate: Path) -> Path:
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()

    config.paths.raw_trace_path = _resolve(config.paths.raw_trace_path)
    config.paths.processed_dir = _resolve(config.paths.processed_dir)
    config.paths.artifacts_dir = _resolve(config.paths.artifacts_dir)
    config.paths.predictions_path = _resolve(config.paths.predictions_path)
    config.model.adapter_path = _resolve(config.model.adapter_path)
    config.training.output_dir = _resolve(config.training.output_dir)
    config.eval.gold_path = _resolve(config.eval.gold_path)
    config.eval.output_path = _resolve(config.eval.output_path)
    config.weco.experiment_dir = _resolve(config.weco.experiment_dir)
    return config
