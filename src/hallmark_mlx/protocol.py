"""Shared serialization helpers for the compact tool-use protocol."""

from __future__ import annotations

import json
from typing import Any

from hallmark_mlx.data.schemas import FinalDecision, ToolResultSummary


def render_tagged_json(tag: str, payload: dict[str, Any]) -> str:
    """Render a compact tagged JSON block."""

    return f"<{tag}>\n{json.dumps(payload, indent=2, sort_keys=True)}\n</{tag}>"


def compact_tool_result_payload(result: ToolResultSummary) -> dict[str, Any]:
    """Reduce tool results to the evidence needed for protocol learning."""

    payload: dict[str, Any] = {
        "tool": result.tool.value,
        "action": result.action,
        "ok": result.ok,
        "evidence_strength": result.evidence_strength.value,
        "candidate_count": result.candidate_count,
    }
    if result.matched_identifiers:
        payload["matched_identifiers"] = result.matched_identifiers
    if result.notes:
        payload["notes"] = result.notes
    if result.candidate_summaries:
        top = result.candidate_summaries[0]
        payload["top_candidate"] = {
            "source": top.source,
            "title": top.title,
            "authors": top.authors,
            "year": top.year,
            "venue": top.venue,
            "doi": top.doi,
            "score": top.score,
        }
    return payload


def compact_final_decision_payload(final_decision: FinalDecision) -> dict[str, Any]:
    """Reduce the final decision to a compact phase-1 verdict object."""

    payload: dict[str, Any] = {
        "verdict": final_decision.verdict.value,
        "confidence": final_decision.confidence,
        "rationale": final_decision.rationale,
    }
    if final_decision.abstain_reason:
        payload["abstain_reason"] = final_decision.abstain_reason
    if final_decision.should_update_bibtex:
        payload["should_update_bibtex"] = True
    if final_decision.subtest_results:
        payload["subtest_results"] = final_decision.subtest_results
    return payload
