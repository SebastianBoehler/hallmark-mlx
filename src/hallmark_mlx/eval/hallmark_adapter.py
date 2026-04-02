"""Serialization helpers for HALLMARK-style predictions."""

from __future__ import annotations

import hashlib

from hallmark_mlx.data.schemas import VerificationTrace
from hallmark_mlx.types import HallmarkLabel, VerificationVerdict
from hallmark_mlx.utils.jsonl import write_jsonl


def hallmark_label_from_verdict(verdict: VerificationVerdict) -> HallmarkLabel:
    """Map local verdicts onto HALLMARK labels."""

    if verdict in {VerificationVerdict.VERIFIED, VerificationVerdict.CORRECTED}:
        return HallmarkLabel.VALID
    if verdict in {VerificationVerdict.HALLUCINATED, VerificationVerdict.UNSUPPORTED}:
        return HallmarkLabel.HALLUCINATED
    return HallmarkLabel.UNCERTAIN


def _local_fallback_key(trace: VerificationTrace) -> str:
    payload = f"{trace.trace_id}|{trace.input.raw_input}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def prediction_from_trace(trace: VerificationTrace) -> dict[str, object]:
    """Create one HALLMARK-style prediction row."""

    if trace.final_decision is None:
        raise ValueError("Trace must have a final decision before evaluation.")
    key = trace.input.benchmark_bibtex_key or _local_fallback_key(trace)
    return {
        "bibtex_key": key,
        "label": hallmark_label_from_verdict(trace.final_decision.verdict).value,
        "confidence": trace.final_decision.confidence,
        "reason": trace.final_decision.rationale,
        "subtest_results": trace.final_decision.subtest_results,
        "api_sources_queried": [result.tool.value for result in trace.tool_results],
        "wall_clock_seconds": trace.metadata.get("wall_clock_seconds"),
        "api_calls": len(trace.tool_results),
    }


def write_hallmark_predictions(
    traces: list[VerificationTrace],
    output_path: str,
) -> None:
    """Write trace predictions as JSONL."""

    rows = [prediction_from_trace(trace) for trace in traces]
    write_jsonl(output_path, rows)
