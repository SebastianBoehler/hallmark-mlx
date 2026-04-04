"""Evaluate hallmark-mlx methods against official HALLMARK entries."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from hallmark_mlx.data.schemas import VerificationTrace
from hallmark_mlx.eval.hallmark_adapter import prediction_from_trace
from hallmark_mlx.eval.official_progress import (
    build_error_trace,
    load_completed_traces,
    write_progress_manifest,
    write_trace_checkpoint,
)
from hallmark_mlx.eval.timeouts import run_with_timeout
from hallmark_mlx.eval.upstream_hallmark import _ensure_upstream_path, result_to_row
from hallmark_mlx.eval.weco_dataset import hallmark_entry_to_trace
from hallmark_mlx.inference.policy_runner import PolicyRunner
from hallmark_mlx.utils.io import write_json
from hallmark_mlx.utils.jsonl import write_jsonl

JSONDict = dict[str, Any]


def official_gold_traces(
    entries: list[Any],
    *,
    split_name: str,
) -> list[VerificationTrace]:
    """Convert official benchmark entries to local gold traces."""

    return [
        hallmark_entry_to_trace(
            entry.to_dict() if hasattr(entry, "to_dict") else entry,
            split_name=split_name,
        )
        for entry in entries
    ]


def _upstream_predictions(upstream_root: Path, traces: list[VerificationTrace]) -> list[Any]:
    _ensure_upstream_path(upstream_root)
    from hallmark.dataset.schema import Prediction

    return [Prediction.from_dict(prediction_from_trace(trace)) for trace in traces]


def evaluate_runner_on_entries(
    *,
    upstream_root: str | Path,
    entries: list[Any],
    split_name: str,
    runner: PolicyRunner,
    method_name: str,
    output_dir: str | Path,
    description: str,
    source: str,
    resume: bool = True,
    entry_timeout_seconds: int | None = None,
    progress_every: int = 10,
) -> JSONDict:
    """Run one local policy/controller on official entries and evaluate it."""

    root = Path(upstream_root).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gold_traces = official_gold_traces(entries, split_name=split_name)
    completed = load_completed_traces(output_root) if resume else {}
    predicted_traces: list[VerificationTrace] = []

    write_progress_manifest(
        output_root,
        method_name=method_name,
        split_name=split_name,
        total_entries=len(gold_traces),
        completed_entries=len(completed),
        error_entries=_count_error_traces(completed.values()),
        entry_timeout_seconds=entry_timeout_seconds,
        status="running",
    )

    for index, gold_trace in enumerate(gold_traces):
        cached = completed.get(index)
        if cached is not None:
            predicted_traces.append(cached)
            continue
        trace = _run_official_entry(
            runner,
            gold_trace,
            entry_timeout_seconds=entry_timeout_seconds,
        )
        predicted_traces.append(trace)
        completed[index] = trace
        write_trace_checkpoint(output_root, index, trace)

        if (index + 1) % progress_every == 0 or index == len(gold_traces) - 1:
            write_progress_manifest(
                output_root,
                method_name=method_name,
                split_name=split_name,
                total_entries=len(gold_traces),
                completed_entries=len(completed),
                error_entries=_count_error_traces(completed.values()),
                entry_timeout_seconds=entry_timeout_seconds,
                status="running",
            )

    write_jsonl(
        output_root / "traces.jsonl",
        [trace.model_dump(exclude_none=True) for trace in predicted_traces],
    )

    prediction_rows = [prediction_from_trace(trace) for trace in predicted_traces]
    write_jsonl(output_root / "predictions.jsonl", prediction_rows)

    predictions = _upstream_predictions(root, predicted_traces)
    _ensure_upstream_path(root)
    from hallmark.evaluation.metrics import evaluate

    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name=method_name,
        split_name=split_name,
    )
    write_json(output_root / "result.json", result.to_dict())
    write_progress_manifest(
        output_root,
        method_name=method_name,
        split_name=split_name,
        total_entries=len(gold_traces),
        completed_entries=len(predicted_traces),
        error_entries=_count_error_traces(predicted_traces),
        entry_timeout_seconds=entry_timeout_seconds,
        status="complete",
    )
    return result_to_row(
        result,
        source=source,
        description=description,
        available_here=True,
        status_message="Executed here with official evaluator.",
        notes=f"Artifacts written to {output_root}.",
    )


def _run_official_entry(
    runner: PolicyRunner,
    gold_trace: VerificationTrace,
    *,
    entry_timeout_seconds: int | None,
) -> VerificationTrace:
    started = perf_counter()
    try:
        if entry_timeout_seconds is None:
            predicted = runner.run(gold_trace.input)
        else:
            predicted = run_with_timeout(
                entry_timeout_seconds,
                lambda: runner.run(gold_trace.input),
            )
        wall_clock_seconds = perf_counter() - started
        return _with_entry_metadata(predicted, gold_trace, wall_clock_seconds=wall_clock_seconds)
    except Exception as exc:  # noqa: BLE001
        return _with_entry_metadata(
            build_error_trace(
                gold_trace.input,
                reason=str(exc),
                wall_clock_seconds=perf_counter() - started,
            ),
            gold_trace,
            wall_clock_seconds=perf_counter() - started,
        )


def _with_entry_metadata(
    trace: VerificationTrace,
    gold_trace: VerificationTrace,
    *,
    wall_clock_seconds: float,
) -> VerificationTrace:
    metadata = dict(trace.metadata)
    metadata.setdefault("wall_clock_seconds", wall_clock_seconds)
    metadata.setdefault("benchmark_bibtex_key", gold_trace.input.benchmark_bibtex_key)
    metadata.setdefault("trace_status", "ok")
    return trace.model_copy(update={"metadata": metadata})


def _count_error_traces(traces: list[VerificationTrace] | Any) -> int:
    return sum(1 for trace in traces if trace.metadata.get("trace_status") == "error")
