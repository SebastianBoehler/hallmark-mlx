"""Helpers for stopping deterministic policy rollouts early."""

from __future__ import annotations

from hallmark_mlx.data.schemas import (
    CandidateRanking,
    ParsedBibliographicFields,
    ProposedQuery,
    SuspectedIssue,
    ToolInvocation,
    ToolResultSummary,
    VerificationInput,
    VerificationTrace,
)
from hallmark_mlx.inference.finalizer import finalize_trace
from hallmark_mlx.types import VerificationAction, VerificationVerdict


def maybe_finalize_deterministically(
    *,
    verification_input: VerificationInput,
    policy_version: str,
    parsed_fields: ParsedBibliographicFields,
    suspected_issues: list[SuspectedIssue],
    proposed_query: ProposedQuery | None,
    next_action: VerificationAction,
    tool_calls: list[ToolInvocation],
    tool_results: list[ToolResultSummary],
    candidate_ranking: CandidateRanking | None,
    metadata: dict[str, object],
) -> VerificationTrace | None:
    """Return a finished trace when current tool evidence is already decisive."""

    finalized = finalize_trace(
        VerificationTrace(
            policy_version=policy_version,
            input=verification_input,
            parsed_fields=parsed_fields,
            suspected_issues=suspected_issues,
            proposed_query=proposed_query,
            next_action=next_action,
            tool_calls=tool_calls,
            tool_results=tool_results,
            candidate_ranking=candidate_ranking,
            metadata=metadata,
        ),
        force=True,
    )
    decision = finalized.final_decision
    if decision is None or decision.verdict == VerificationVerdict.ABSTAIN:
        return None
    return finalized
