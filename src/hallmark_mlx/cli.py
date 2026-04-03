"""CLI entrypoints for hallmark-mlx."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from hallmark_mlx.config import load_config
from hallmark_mlx.data.build_dataset import build_trace_dataset
from hallmark_mlx.data.schemas import VerificationInput
from hallmark_mlx.eval.policy_rollout import evaluate_policy_rollout
from hallmark_mlx.eval.run_eval import run_local_eval
from hallmark_mlx.inference.bootstrap import bootstrap_trace_dataset
from hallmark_mlx.inference.policy_runner import PolicyRunner, load_policy_model
from hallmark_mlx.inference.tool_executor import ToolExecutor
from hallmark_mlx.tools.bibtex_updater import check_bibtex, summarize_result
from hallmark_mlx.training.mlx_lora import run_training
from hallmark_mlx.types import InputType
from hallmark_mlx.utils.io import read_json, write_json
from hallmark_mlx.utils.logging import configure_logging

app = typer.Typer(no_args_is_help=True)


@app.command("build-dataset")
def build_dataset_command(
    config: Path = typer.Option(Path("configs/base.yaml")),
    input_path: Path | None = typer.Option(None),
    output_dir: Path | None = typer.Option(None),
) -> None:
    """Build train, valid, test, and holdout trace datasets."""

    configure_logging()
    app_config = load_config(config)
    summary = build_trace_dataset(
        input_path or app_config.paths.raw_trace_path,
        output_dir or app_config.paths.processed_dir,
        valid_ratio=app_config.data.valid_ratio,
        test_ratio=app_config.data.test_ratio,
        seed=app_config.data.seed,
    )
    typer.echo(json.dumps(asdict(summary), indent=2))


@app.command("train")
def train_command(
    config: Path = typer.Option(Path("configs/train_qwen_1_5b.yaml")),
    dry_run: bool = typer.Option(False, help="Plan the run without launching MLX."),
) -> None:
    """Plan or launch an MLX LoRA training run."""

    configure_logging()
    app_config = load_config(config)
    plan = run_training(app_config, dry_run=dry_run)
    typer.echo(
        json.dumps(
            {
                "dataset_dir": str(plan.dataset_dir),
                "manifest_path": str(plan.manifest_path),
                "command": plan.command,
                "weco_stub_path": str(plan.weco_stub_path) if plan.weco_stub_path else None,
            },
            indent=2,
        ),
    )


@app.command("infer")
def infer_command(
    config: Path = typer.Option(Path("configs/base.yaml")),
    input_file: Path | None = typer.Option(None),
    raw_input: str | None = typer.Option(None),
    input_type: InputType = typer.Option(InputType.RAW_CITATION_STRING),
    output_path: Path | None = typer.Option(None),
) -> None:
    """Run policy inference and tool execution for one input."""

    configure_logging()
    app_config = load_config(config)
    if input_file is not None:
        verification_input = VerificationInput.model_validate(read_json(input_file))
    elif raw_input is not None:
        verification_input = VerificationInput(
            record_id="cli-input",
            input_type=input_type,
            raw_input=raw_input,
        )
    else:
        raise typer.BadParameter("Provide either --input-file or --raw-input.")
    model = load_policy_model(app_config.model)
    runner = PolicyRunner(
        model=model,
        tool_executor=ToolExecutor(app_config.tools),
        finalization_mode=app_config.model.finalization_mode,
    )
    trace = runner.run(verification_input)
    if output_path is not None:
        write_json(output_path, trace.model_dump(exclude_none=True))
    typer.echo(trace.model_dump_json(indent=2, exclude_none=True))


@app.command("bootstrap-traces")
def bootstrap_traces_command(
    config: Path = typer.Option(Path("configs/base.yaml")),
    input_path: Path = typer.Option(...),
    output_path: Path = typer.Option(...),
    limit: int | None = typer.Option(None),
) -> None:
    """Generate seed traces from a JSONL file of verification inputs."""

    configure_logging()
    app_config = load_config(config)
    model = load_policy_model(app_config.model)
    runner = PolicyRunner(
        model=model,
        tool_executor=ToolExecutor(app_config.tools),
        finalization_mode=app_config.model.finalization_mode,
    )
    target_path = bootstrap_trace_dataset(input_path, output_path, runner, limit=limit)
    typer.echo(json.dumps({"output_path": str(target_path), "limit": limit}, indent=2))


@app.command("eval")
def eval_command(
    config: Path = typer.Option(Path("configs/eval_hallmark.yaml")),
    predictions: Path | None = typer.Option(None),
    gold: Path | None = typer.Option(None),
) -> None:
    """Run local evaluation over JSONL predictions."""

    configure_logging()
    app_config = load_config(config)
    metrics = run_local_eval(
        predictions or app_config.paths.predictions_path,
        gold or app_config.eval.gold_path,
        output_path=app_config.eval.output_path,
        tool_budgets=app_config.eval.tool_call_budgets,
    )
    typer.echo(json.dumps(metrics, indent=2))


@app.command("eval-policy")
def eval_policy_command(
    config: Path = typer.Option(Path("configs/train_qwen_1_5b.yaml")),
    input_path: Path = typer.Option(..., help="Gold trace JSONL with held-out inputs and decisions."),
    output_path: Path | None = typer.Option(None, help="Optional metrics JSON path."),
    predictions_path: Path | None = typer.Option(None, help="Optional prediction JSONL path."),
    traces_path: Path | None = typer.Option(None, help="Optional predicted trace JSONL path."),
    limit: int | None = typer.Option(None, help="Optional example limit."),
) -> None:
    """Evaluate whether the model uses tools before finalizing decisions."""

    configure_logging()
    app_config = load_config(config)
    model = load_policy_model(app_config.model)
    runner = PolicyRunner(
        model=model,
        tool_executor=ToolExecutor(app_config.tools),
        finalization_mode=app_config.model.finalization_mode,
    )
    metrics = evaluate_policy_rollout(
        input_path,
        runner,
        output_metrics_path=output_path,
        output_predictions_path=predictions_path,
        output_traces_path=traces_path,
        limit=limit,
        tool_budgets=app_config.eval.tool_call_budgets,
    )
    typer.echo(json.dumps(metrics, indent=2))


@app.command("check-bib")
def check_bib_command(
    bib_path: Path,
    strict: bool = typer.Option(False),
    config: Path = typer.Option(Path("configs/tools.yaml")),
) -> None:
    """Run the BibTeX Updater checker wrapper."""

    configure_logging()
    app_config = load_config(config)
    executable = app_config.tools.bibtex_updater.command or "bibtex-check"
    result = check_bibtex(bib_path, strict=strict, executable=executable)
    typer.echo(json.dumps(summarize_result(result), indent=2))


if __name__ == "__main__":
    app()
