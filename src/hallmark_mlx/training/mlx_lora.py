"""MLX LoRA planning and execution helpers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys

from hallmark_mlx.config import AppConfig
from hallmark_mlx.training.dataset_loader import prepare_training_dataset
from hallmark_mlx.utils.io import ensure_dir, write_json


@dataclass(slots=True)
class MLXLoRATrainPlan:
    """Filesystem artifacts and command for a training run."""

    dataset_dir: Path
    manifest_path: Path
    command: list[str]
    weco_stub_path: Path | None = None


def build_mlx_command(config: AppConfig, dataset_dir: Path) -> list[str]:
    """Build an MLX LoRA command using the standard `python -m mlx_lm lora` CLI."""

    train = config.training
    return [
        train.python_executable or sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        config.model.base_model,
        "--data",
        str(dataset_dir),
        "--train",
        "--fine-tune-type",
        train.fine_tune_type,
        "--optimizer",
        train.optimizer,
        "--num-layers",
        str(train.num_layers),
        "--batch-size",
        str(train.batch_size),
        "--iters",
        str(train.num_iterations),
        "--val-batches",
        str(train.val_batches),
        "--learning-rate",
        str(train.learning_rate),
        "--steps-per-report",
        str(train.report_every),
        "--steps-per-eval",
        str(train.eval_every),
        "--grad-accumulation-steps",
        str(train.grad_accumulation_steps),
        "--adapter-path",
        str(config.model.adapter_path),
        "--save-every",
        str(train.save_every),
        "--test-batches",
        str(train.test_batches),
        "--max-seq-length",
        str(train.max_seq_length),
        "--seed",
        str(train.seed),
    ] + (["--mask-prompt"] if train.mask_prompt else []) + (
        ["--grad-checkpoint"] if train.grad_checkpoint else []
    )


def build_training_manifest(config: AppConfig, dataset_dir: Path, command: list[str]) -> dict[str, object]:
    """Build a reproducibility manifest for an MLX LoRA run."""

    return {
        "base_model": config.model.base_model,
        "adapter_path": str(config.model.adapter_path),
        "dataset_dir": str(dataset_dir),
        "command": command,
        "training": {
            "example_format": config.training.example_format.value,
            "fine_tune_type": config.training.fine_tune_type,
            "optimizer": config.training.optimizer,
            "mask_prompt": config.training.mask_prompt,
            "num_layers": config.training.num_layers,
            "batch_size": config.training.batch_size,
            "iters": config.training.num_iterations,
            "val_batches": config.training.val_batches,
            "learning_rate": config.training.learning_rate,
            "steps_per_report": config.training.report_every,
            "steps_per_eval": config.training.eval_every,
            "grad_accumulation_steps": config.training.grad_accumulation_steps,
            "save_every": config.training.save_every,
            "test_batches": config.training.test_batches,
            "max_seq_length": config.training.max_seq_length,
            "grad_checkpoint": config.training.grad_checkpoint,
            "seed": config.training.seed,
        },
    }


def plan_training_run(config: AppConfig) -> MLXLoRATrainPlan:
    """Prepare dataset artifacts and a train command."""

    output_root = ensure_dir(config.training.output_dir)
    prepared_dataset_dir = ensure_dir(output_root / "prepared_dataset")
    prepare_training_dataset(
        config.paths.processed_dir,
        prepared_dataset_dir,
        example_format=config.training.example_format,
    )

    ensure_dir(config.model.adapter_path.parent)
    command = build_mlx_command(config, prepared_dataset_dir)
    manifest = build_training_manifest(config, prepared_dataset_dir, command)
    manifest_path = output_root / "mlx_lora_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    weco_stub_path: Path | None = None
    if config.weco.enabled:
        weco_stub_path = output_root / "weco_stub.json"
        write_json(
            weco_stub_path,
            {
                "objective_name": config.weco.objective_name,
                "runner_script": "scripts/weco_run.py",
                "trial_source": "weco_targets/hallmark_policy_trial.py",
                "eval_script": "scripts/weco_eval.py",
                "notes": "Use the repo-native Weco frontier loop instead of mutating MLX train commands directly.",
            },
        )

    return MLXLoRATrainPlan(
        dataset_dir=prepared_dataset_dir,
        manifest_path=manifest_path,
        command=command,
        weco_stub_path=weco_stub_path,
    )


def run_training(config: AppConfig, *, dry_run: bool = False) -> MLXLoRATrainPlan:
    """Plan and optionally execute an MLX LoRA run."""

    plan = plan_training_run(config)
    if dry_run:
        return plan
    if importlib.util.find_spec("mlx_lm") is None:
        msg = (
            "mlx_lm is not installed. Install the optional MLX dependencies before "
            "running `hallmark-mlx train`."
        )
        raise RuntimeError(msg)
    subprocess.run(plan.command, check=True)
    return plan
