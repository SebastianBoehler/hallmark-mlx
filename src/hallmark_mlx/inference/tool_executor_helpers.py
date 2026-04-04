"""Shared helpers for normalized tool result construction."""

from __future__ import annotations

from collections.abc import Callable

from hallmark_mlx.data.schemas import CandidateMatch, ToolInvocation, ToolResultSummary
from hallmark_mlx.types import EvidenceStrength


def strength_from_candidates(count: int, top_score: float) -> EvidenceStrength:
    """Map candidate count and score to a coarse evidence bucket."""

    if count == 0:
        return EvidenceStrength.NONE
    if count == 1 and top_score < 0.8:
        return EvidenceStrength.WEAK
    if top_score >= 0.9:
        return EvidenceStrength.STRONG
    return EvidenceStrength.MODERATE


def disabled_result(tool_call: ToolInvocation, notes: str) -> ToolResultSummary:
    """Return a normalized disabled-tool result."""

    return ToolResultSummary(
        tool=tool_call.tool,
        action=tool_call.action,
        ok=False,
        notes=notes,
    )


def error_result(tool_call: ToolInvocation, notes: str) -> ToolResultSummary:
    """Return a normalized tool failure result."""

    return ToolResultSummary(
        tool=tool_call.tool,
        action=tool_call.action,
        ok=False,
        evidence_strength=EvidenceStrength.NONE,
        notes=notes,
        raw_payload={"arguments": tool_call.arguments},
    )


def execute_candidate_search(
    tool_call: ToolInvocation,
    *,
    enabled: bool,
    search_fn: Callable[[str], list[CandidateMatch]],
) -> ToolResultSummary:
    """Run a search-style tool and normalize candidates."""

    if not enabled:
        return disabled_result(tool_call, f"{tool_call.tool.value} is disabled in config.")
    query = tool_call.arguments.get("query")
    if not query:
        raise ValueError(f"{tool_call.tool.value} calls require a `query` argument.")
    try:
        candidates = search_fn(query)
    except Exception as exc:  # noqa: BLE001
        return error_result(tool_call, str(exc))
    top_score = max((candidate.score for candidate in candidates), default=0.0)
    matched_identifiers = {}
    if candidates and candidates[0].doi:
        matched_identifiers["doi"] = candidates[0].doi
    return ToolResultSummary(
        tool=tool_call.tool,
        action=tool_call.action,
        ok=True,
        evidence_strength=strength_from_candidates(len(candidates), top_score),
        candidate_count=len(candidates),
        matched_identifiers=matched_identifiers,
        notes=f"{len(candidates)} candidates returned.",
        candidate_summaries=candidates,
    )


def execute_candidate_lookup(
    tool_call: ToolInvocation,
    *,
    enabled: bool,
    lookup_fn: Callable[[str], list[CandidateMatch]],
    key: str = "doi",
) -> ToolResultSummary:
    """Run an identifier-lookup tool and normalize candidates."""

    if not enabled:
        return disabled_result(tool_call, f"{tool_call.tool.value} is disabled in config.")
    value = tool_call.arguments.get(key)
    if not value:
        raise ValueError(f"{tool_call.tool.value} lookups require a `{key}` argument.")
    try:
        candidates = lookup_fn(value)
    except Exception as exc:  # noqa: BLE001
        return error_result(tool_call, str(exc))
    top_score = max((candidate.score for candidate in candidates), default=0.0)
    matched_identifiers = {}
    if candidates and candidates[0].doi:
        matched_identifiers["doi"] = candidates[0].doi
    return ToolResultSummary(
        tool=tool_call.tool,
        action=tool_call.action,
        ok=True,
        evidence_strength=strength_from_candidates(len(candidates), top_score),
        candidate_count=len(candidates),
        matched_identifiers=matched_identifiers,
        notes=f"{len(candidates)} candidates returned.",
        candidate_summaries=candidates,
    )
