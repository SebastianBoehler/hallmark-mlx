"""Execution layer for verification tools."""

from __future__ import annotations

from hallmark_mlx.config import ToolsConfig
from hallmark_mlx.data.schemas import ToolInvocation, ToolResultSummary
from hallmark_mlx.tools.bibtex_updater import check_bibtex, summarize_result, update_bibtex
from hallmark_mlx.tools.crossref import search_works as search_crossref_works
from hallmark_mlx.tools.openalex import search_works as search_openalex_works
from hallmark_mlx.tools.semantic_scholar import search_papers as search_semantic_scholar_papers
from hallmark_mlx.types import EvidenceStrength, ToolName


def _strength_from_candidates(count: int, top_score: float) -> EvidenceStrength:
    if count == 0:
        return EvidenceStrength.NONE
    if count == 1 and top_score < 0.8:
        return EvidenceStrength.WEAK
    if top_score >= 0.9:
        return EvidenceStrength.STRONG
    return EvidenceStrength.MODERATE


class ToolExecutor:
    """Run normalized tool invocations."""

    def __init__(self, config: ToolsConfig) -> None:
        self.config = config

    def execute_many(self, tool_calls: list[ToolInvocation]) -> list[ToolResultSummary]:
        return [self.execute(tool_call) for tool_call in tool_calls]

    def execute(self, tool_call: ToolInvocation) -> ToolResultSummary:
        if tool_call.tool == ToolName.BIBTEX_UPDATER:
            return self._execute_bibtex_updater(tool_call)
        if tool_call.tool == ToolName.CROSSREF:
            return self._execute_crossref(tool_call)
        if tool_call.tool == ToolName.OPENALEX:
            return self._execute_openalex(tool_call)
        if tool_call.tool == ToolName.SEMANTIC_SCHOLAR:
            return self._execute_semantic_scholar(tool_call)
        raise ValueError(f"Unsupported tool: {tool_call.tool}")

    def _execute_bibtex_updater(self, tool_call: ToolInvocation) -> ToolResultSummary:
        service = self.config.bibtex_updater
        if not service.enabled:
            return ToolResultSummary(
                tool=tool_call.tool,
                action=tool_call.action,
                ok=False,
                notes="BibTeX Updater is disabled in config.",
            )
        source = tool_call.arguments.get("bibtex") or tool_call.arguments.get("path")
        if not source:
            raise ValueError("BibTeX Updater calls require `bibtex` or `path`.")
        executable = service.command or "bibtex-check"
        if tool_call.action == "update_bibtex":
            result = update_bibtex(source, executable=executable.replace("check", "update"))
        else:
            result = check_bibtex(
                source,
                strict=bool(tool_call.arguments.get("strict", False)),
                executable=executable,
            )
        summary = summarize_result(result)
        return ToolResultSummary(
            tool=tool_call.tool,
            action=tool_call.action,
            ok=bool(summary["ok"]),
            evidence_strength=EvidenceStrength.MODERATE if summary["ok"] else EvidenceStrength.NONE,
            candidate_count=int(summary["candidate_count"]),
            matched_identifiers=summary["matched_identifiers"],
            notes=summary["notes"],
            raw_payload=summary,
        )

    def _execute_crossref(self, tool_call: ToolInvocation) -> ToolResultSummary:
        service = self.config.crossref
        return self._execute_candidate_search(
            tool_call,
            enabled=service.enabled,
            search_fn=lambda query: search_crossref_works(
                query,
                rows=int(tool_call.arguments.get("rows", service.rows)),
                timeout=service.timeout_seconds,
                email=service.email,
            ),
        )

    def _execute_openalex(self, tool_call: ToolInvocation) -> ToolResultSummary:
        service = self.config.openalex
        return self._execute_candidate_search(
            tool_call,
            enabled=service.enabled,
            search_fn=lambda query: search_openalex_works(
                query,
                per_page=int(tool_call.arguments.get("rows", service.rows)),
                timeout=service.timeout_seconds,
                email=service.email,
            ),
        )

    def _execute_semantic_scholar(self, tool_call: ToolInvocation) -> ToolResultSummary:
        service = self.config.semantic_scholar
        return self._execute_candidate_search(
            tool_call,
            enabled=service.enabled,
            search_fn=lambda query: search_semantic_scholar_papers(
                query,
                limit=int(tool_call.arguments.get("rows", service.rows)),
                timeout=service.timeout_seconds,
            ),
        )

    def _execute_candidate_search(
        self,
        tool_call: ToolInvocation,
        *,
        enabled: bool,
        search_fn,
    ) -> ToolResultSummary:
        if not enabled:
            return ToolResultSummary(
                tool=tool_call.tool,
                action=tool_call.action,
                ok=False,
                notes=f"{tool_call.tool.value} is disabled in config.",
            )
        query = tool_call.arguments.get("query")
        if not query:
            raise ValueError(f"{tool_call.tool.value} calls require a `query` argument.")
        try:
            candidates = search_fn(query)
        except Exception as exc:  # noqa: BLE001
            return ToolResultSummary(
                tool=tool_call.tool,
                action=tool_call.action,
                ok=False,
                notes=str(exc),
            )
        top_score = max((candidate.score for candidate in candidates), default=0.0)
        matched_identifiers = {}
        if candidates and candidates[0].doi:
            matched_identifiers["doi"] = candidates[0].doi
        return ToolResultSummary(
            tool=tool_call.tool,
            action=tool_call.action,
            ok=True,
            evidence_strength=_strength_from_candidates(len(candidates), top_score),
            candidate_count=len(candidates),
            matched_identifiers=matched_identifiers,
            notes=f"{len(candidates)} candidates returned.",
            candidate_summaries=candidates,
        )
