"""Evaluate rollout behavior for transcript-trained policy models."""

from __future__ import annotations

from pathlib import Path

from hallmark_mlx.data.schemas import VerificationTrace
from hallmark_mlx.eval.hallmark_adapter import hallmark_label_from_verdict
from hallmark_mlx.eval.metrics import DEFAULT_TOOL_BUDGETS, compute_metrics
from hallmark_mlx.inference.policy_runner import PolicyRunner
from hallmark_mlx.training.dataset_loader import load_trace_split
from hallmark_mlx.types import HallmarkLabel
from hallmark_mlx.utils.io import write_json


def _prediction_row(trace: VerificationTrace) -> dict[str, object]:
    label = HallmarkLabel.UNCERTAIN
    confidence = 0.0
    reason = "No final decision emitted."
    if trace.final_decision is not None:
        label = hallmark_label_from_verdict(trace.final_decision.verdict)
        confidence = trace.final_decision.confidence
        reason = trace.final_decision.rationale
    return {
        "bibtex_key": trace.input.benchmark_bibtex_key or trace.trace_id,
        "label": label.value,
        "confidence": confidence,
        "reason": reason,
        "api_calls": len(trace.tool_results),
    }


def _gold_row(trace: VerificationTrace) -> dict[str, object]:
    if trace.final_decision is None:
        raise ValueError(f"Gold trace {trace.trace_id} is missing a final decision.")
    return {
        "bibtex_key": trace.input.benchmark_bibtex_key or trace.trace_id,
        "label": hallmark_label_from_verdict(trace.final_decision.verdict).value,
    }


def evaluate_policy_rollout(
    gold_trace_path: str | Path,
    runner: PolicyRunner,
    *,
    output_metrics_path: str | Path | None = None,
    output_predictions_path: str | Path | None = None,
    output_traces_path: str | Path | None = None,
    limit: int | None = None,
    tool_budgets: tuple[int, ...] = DEFAULT_TOOL_BUDGETS,
) -> dict[str, float]:
    """Run the policy model over gold inputs and summarize behavior metrics."""

    gold_traces = load_trace_split(gold_trace_path)
    if limit is not None:
        gold_traces = gold_traces[:limit]
    predicted_traces = [runner.run(trace.input) for trace in gold_traces]

    gold_rows = [_gold_row(trace) for trace in gold_traces]
    prediction_rows = [_prediction_row(trace) for trace in predicted_traces]
    metrics = compute_metrics(gold_rows, prediction_rows, tool_budgets=tool_budgets)

    completed = sum(trace.final_decision is not None for trace in predicted_traces)
    used_tools = sum(bool(trace.tool_calls) for trace in predicted_traces)
    first_turn_tool_calls = sum(
        int(trace.metadata.get("first_response_tool_call_count", 0) > 0)
        for trace in predicted_traces
    )
    premature_finals = sum(
        bool(trace.metadata.get("first_response_had_final_decision", False))
        for trace in predicted_traces
    )
    finalized_after_tool_results = sum(
        trace.final_decision is not None
        and bool(trace.tool_results)
        and not bool(trace.metadata.get("first_response_had_final_decision", False))
        for trace in predicted_traces
    )
    verdict_matches = sum(
        trace.final_decision is not None
        and gold.final_decision is not None
        and trace.final_decision.verdict == gold.final_decision.verdict
        for gold, trace in zip(gold_traces, predicted_traces, strict=True)
    )
    first_tool_matches = sum(
        bool(trace.tool_calls)
        and bool(gold.tool_calls)
        and trace.tool_calls[0].tool == gold.tool_calls[0].tool
        and trace.tool_calls[0].action == gold.tool_calls[0].action
        for gold, trace in zip(gold_traces, predicted_traces, strict=True)
    )

    total = len(predicted_traces)
    behavior_metrics = {
        "num_examples": float(total),
        "completion_rate": completed / total if total else 0.0,
        "tool_use_rate": used_tools / total if total else 0.0,
        "first_turn_tool_call_rate": first_turn_tool_calls / total if total else 0.0,
        "premature_finalization_rate": premature_finals / total if total else 0.0,
        "finalized_after_tool_rate": finalized_after_tool_results / total if total else 0.0,
        "verdict_accuracy": verdict_matches / total if total else 0.0,
        "first_tool_match_rate": first_tool_matches / total if total else 0.0,
        "avg_tool_calls": (
            sum(len(trace.tool_calls) for trace in predicted_traces) / total if total else 0.0
        ),
    }
    for budget in tool_budgets:
        completed_within_budget = sum(
            trace.final_decision is not None and len(trace.tool_results) <= budget
            for trace in predicted_traces
        )
        verdict_matches_within_budget = sum(
            trace.final_decision is not None
            and len(trace.tool_results) <= budget
            and gold.final_decision is not None
            and trace.final_decision.verdict == gold.final_decision.verdict
            for gold, trace in zip(gold_traces, predicted_traces, strict=True)
        )
        tool_use_within_budget = sum(
            bool(trace.tool_calls) and len(trace.tool_results) <= budget
            for trace in predicted_traces
        )
        exceeded_budget = sum(len(trace.tool_results) > budget for trace in predicted_traces)
        behavior_metrics[f"budget_{budget}_completion_rate"] = (
            completed_within_budget / total if total else 0.0
        )
        behavior_metrics[f"budget_{budget}_verdict_accuracy"] = (
            verdict_matches_within_budget / total if total else 0.0
        )
        behavior_metrics[f"budget_{budget}_tool_use_rate"] = (
            tool_use_within_budget / total if total else 0.0
        )
        behavior_metrics[f"budget_{budget}_exceeded_rate"] = (
            exceeded_budget / total if total else 0.0
        )
    metrics.update(behavior_metrics)

    if output_metrics_path is not None:
        write_json(output_metrics_path, metrics)
    if output_predictions_path is not None:
        from hallmark_mlx.utils.jsonl import write_jsonl

        write_jsonl(output_predictions_path, prediction_rows)
    if output_traces_path is not None:
        from hallmark_mlx.utils.jsonl import write_jsonl

        write_jsonl(
            output_traces_path,
            [trace.model_dump(exclude_none=True) for trace in predicted_traces],
        )
    return metrics
