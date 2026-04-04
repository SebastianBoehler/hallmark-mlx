"""Deterministic BibTeX-first controller with selective evidence fallback."""

from __future__ import annotations

from hallmark_mlx.config import ModelConfig
from hallmark_mlx.data.schemas import (
    ParsedBibliographicFields,
    ProposedQuery,
    ToolInvocation,
    ToolResultSummary,
    VerificationInput,
    VerificationTrace,
)
from hallmark_mlx.inference.finalizer import finalize_trace
from hallmark_mlx.inference.warm_start_planner import build_query, build_tool_calls
from hallmark_mlx.inference.warm_start_policy import detect_suspected_issues, parse_input
from hallmark_mlx.tools.acl_anthology import extract_anthology_id
from hallmark_mlx.tools.arxiv import extract_arxiv_id
from hallmark_mlx.types import InputType, ToolName, VerificationAction, VerificationVerdict

_OVERRIDEABLE_BIBTEX_STATUSES = {
    "AUTHOR_MISMATCH",
    "DOI_NOT_FOUND",
    "TITLE_MISMATCH",
    "VENUE_MISMATCH",
    "YEAR_MISMATCH",
}
_STOP_IMMEDIATELY_BIBTEX_STATUSES = {
    "FUTURE_DATE",
    "HALLUCINATED",
    "PARTIAL_MATCH",
}


def _bibtex_check_call(raw_bibtex: str) -> ToolInvocation:
    return ToolInvocation(
        tool=ToolName.BIBTEX_UPDATER,
        action="check_bibtex",
        arguments={"bibtex": raw_bibtex, "strict": True},
    )


def _result_status(result: ToolResultSummary | None) -> str | None:
    if result is None:
        return None
    raw_status = result.raw_payload.get("status")
    return raw_status if isinstance(raw_status, str) and raw_status else None


def _dedupe_tool_calls(tool_calls: list[ToolInvocation]) -> list[ToolInvocation]:
    seen: set[tuple[str, str, tuple[tuple[str, object], ...]]] = set()
    deduped: list[ToolInvocation] = []
    for tool_call in tool_calls:
        fingerprint = (
            tool_call.tool.value,
            tool_call.action,
            tuple(sorted(tool_call.arguments.items())),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(tool_call)
    return deduped


def _needs_arxiv_fallback(fields: ParsedBibliographicFields) -> bool:
    return bool(fields.arxiv_id or extract_arxiv_id(fields.doi) or extract_arxiv_id(fields.url))


def _should_verify_author_completeness(
    verification_input: VerificationInput,
    fields: ParsedBibliographicFields,
    status: str | None,
) -> bool:
    return (
        status == "VERIFIED"
        and verification_input.input_type == InputType.BIBTEX_ENTRY
        and bool(fields.doi)
    )


def _search_based_tool_calls(
    verification_input: VerificationInput,
    fields: ParsedBibliographicFields,
) -> list[ToolInvocation]:
    search_fields = fields if not fields.doi else fields.model_copy(update={"doi": None})
    proposed_query = build_query(search_fields, verification_input)
    if proposed_query is None:
        return []
    return [
        tool_call
        for tool_call in build_tool_calls(verification_input, search_fields, proposed_query)
        if tool_call.tool != ToolName.BIBTEX_UPDATER
    ]


def _fallback_tool_calls(
    verification_input: VerificationInput,
    fields: ParsedBibliographicFields,
    proposed_query: ProposedQuery | None,
) -> list[ToolInvocation]:
    tool_calls: list[ToolInvocation] = []
    anthology_id = (
        extract_anthology_id(fields.doi)
        or extract_anthology_id(fields.url)
        or extract_anthology_id(verification_input.raw_input)
    )
    if anthology_id:
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.ACL_ANTHOLOGY,
                action="resolve_record",
                arguments={"anthology_id": anthology_id},
            )
        )
    if _needs_arxiv_fallback(fields):
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.ARXIV,
                action="resolve_record",
                arguments={
                    "arxiv_id": fields.arxiv_id or extract_arxiv_id(fields.doi) or extract_arxiv_id(fields.url)
                },
            )
        )
    elif fields.doi:
        tool_calls.extend(
            [
                ToolInvocation(
                    tool=ToolName.CROSSREF,
                    action="resolve_doi",
                    arguments={"doi": fields.doi},
                ),
            ]
        )
        if not anthology_id:
            tool_calls.append(
                ToolInvocation(
                    tool=ToolName.OPENALEX,
                    action="resolve_doi",
                    arguments={"doi": fields.doi},
                )
            )
    tool_calls.extend(_search_based_tool_calls(verification_input, fields))
    return _dedupe_tool_calls(tool_calls)


class BibtexFirstFallbackPolicyModel:
    """A deterministic controller for benchmark-style BibTeX verification."""

    def __init__(self, config: ModelConfig) -> None:
        self.max_rollout_rounds = max(config.max_rollout_rounds, 1)
        self.force_bibtex_updater_first = config.force_bibtex_updater_first

    def propose_trace(self, verification_input: VerificationInput) -> VerificationTrace:
        fields = parse_input(verification_input)
        issues = detect_suspected_issues(verification_input, fields)
        proposed_query = build_query(fields, verification_input)
        tool_calls = (
            [_bibtex_check_call(verification_input.raw_input)]
            if self.force_bibtex_updater_first
            else _fallback_tool_calls(verification_input, fields, proposed_query)[:1]
        )
        return VerificationTrace(
            policy_version="bibtex-first-fallback-v0",
            input=verification_input,
            parsed_fields=fields,
            suspected_issues=issues,
            proposed_query=proposed_query,
            next_action=VerificationAction.QUERY_BIBTEX_UPDATER,
            tool_calls=tool_calls,
            metadata={"policy_backend": "bibtex_first_fallback"},
        )

    def _finalize(
        self,
        trace: VerificationTrace,
        *,
        allow_bibtex_override: bool,
    ) -> VerificationTrace:
        if not allow_bibtex_override:
            finalized = finalize_trace(trace, force=True)
            return finalized.model_copy(
                update={"metadata": {**finalized.metadata, "finalization_locked": True}}
            )
        status = _result_status(next((result for result in trace.tool_results if result.tool == ToolName.BIBTEX_UPDATER), None))
        if status not in _OVERRIDEABLE_BIBTEX_STATUSES:
            finalized = finalize_trace(trace, force=True)
            return finalized.model_copy(
                update={"metadata": {**finalized.metadata, "finalization_locked": True}}
            )
        fallback_results = [result for result in trace.tool_results if result.tool != ToolName.BIBTEX_UPDATER]
        if not any(result.ok and result.candidate_summaries for result in fallback_results):
            finalized = finalize_trace(trace, force=True)
            return finalized.model_copy(
                update={"metadata": {**finalized.metadata, "finalization_locked": True}}
            )
        fallback_trace = trace.model_copy(update={"tool_results": fallback_results})
        finalized = finalize_trace(fallback_trace, force=True)
        if finalized.final_decision is None or finalized.final_decision.verdict == VerificationVerdict.ABSTAIN:
            fallback_finalized = finalize_trace(trace, force=True)
            return fallback_finalized.model_copy(
                update={"metadata": {**fallback_finalized.metadata, "finalization_locked": True}}
            )
        return trace.model_copy(
            update={
                "candidate_ranking": finalized.candidate_ranking,
                "final_decision": finalized.final_decision,
                "metadata": {**trace.metadata, "finalization_locked": True},
            }
        )

    def run_with_tools(self, verification_input: VerificationInput, tool_executor) -> VerificationTrace:
        fields = parse_input(verification_input)
        issues = detect_suspected_issues(verification_input, fields)
        proposed_query = build_query(fields, verification_input)
        tool_calls: list[ToolInvocation] = []
        tool_results: list[ToolResultSummary] = []
        fallback_calls = _fallback_tool_calls(verification_input, fields, proposed_query)
        initial_call = (
            _bibtex_check_call(verification_input.raw_input)
            if self.force_bibtex_updater_first
            else (fallback_calls[0] if fallback_calls else _bibtex_check_call(verification_input.raw_input))
        )
        tool_calls.append(initial_call)
        tool_results.append(tool_executor.execute(initial_call))
        first_status = _result_status(tool_results[0])
        trace = VerificationTrace(
            policy_version="bibtex-first-fallback-v0",
            input=verification_input,
            parsed_fields=fields,
            suspected_issues=issues,
            proposed_query=proposed_query,
            next_action=VerificationAction.FINALIZE,
            tool_calls=tool_calls,
            tool_results=tool_results,
            metadata={
                "policy_backend": "bibtex_first_fallback",
                "first_response_tool_call_count": 1,
                "first_response_had_final_decision": False,
            },
        )
        if (
            first_status in _STOP_IMMEDIATELY_BIBTEX_STATUSES
            or not self.force_bibtex_updater_first
        ) and not _should_verify_author_completeness(verification_input, fields, first_status):
            return self._finalize(trace, allow_bibtex_override=False)

        remaining_budget = max(self.max_rollout_rounds - 1, 0)
        for tool_call in fallback_calls[:remaining_budget]:
            tool_calls.append(tool_call)
            tool_results.append(tool_executor.execute(tool_call))
            trace = trace.model_copy(
                update={
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "metadata": {
                        **trace.metadata,
                        "fallback_tool_count": len(tool_calls) - 1,
                    },
                }
            )
            finalized = self._finalize(trace, allow_bibtex_override=True)
            if finalized.final_decision is not None and finalized.final_decision.verdict != VerificationVerdict.ABSTAIN:
                return finalized
        return self._finalize(trace, allow_bibtex_override=True)
