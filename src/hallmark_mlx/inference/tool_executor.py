"""Execution layer for verification tools."""

from __future__ import annotations

from hallmark_mlx.config import ToolsConfig
from hallmark_mlx.data.schemas import ToolInvocation, ToolResultSummary
from hallmark_mlx.tools.acl_anthology import (
    extract_anthology_id,
)
from hallmark_mlx.tools.acl_anthology import (
    resolve_record as resolve_acl_record,
)
from hallmark_mlx.tools.bibtex_updater import check_bibtex, summarize_result, update_bibtex
from hallmark_mlx.tools.crossref import lookup_work_by_doi as lookup_crossref_work_by_doi
from hallmark_mlx.tools.crossref import search_works as search_crossref_works
from hallmark_mlx.tools.dblp import search_works as search_dblp_works
from hallmark_mlx.tools.openalex import lookup_work_by_doi as lookup_openalex_work_by_doi
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
        try:
            if tool_call.tool == ToolName.BIBTEX_UPDATER:
                return self._execute_bibtex_updater(tool_call)
            if tool_call.tool == ToolName.CROSSREF:
                return self._execute_crossref(tool_call)
            if tool_call.tool == ToolName.OPENALEX:
                return self._execute_openalex(tool_call)
            if tool_call.tool == ToolName.DBLP:
                return self._execute_dblp(tool_call)
            if tool_call.tool == ToolName.ACL_ANTHOLOGY:
                return self._execute_acl_anthology(tool_call)
            if tool_call.tool == ToolName.SEMANTIC_SCHOLAR:
                return self._execute_semantic_scholar(tool_call)
            raise ValueError(f"Unsupported tool: {tool_call.tool}")
        except ValueError as exc:
            return ToolResultSummary(
                tool=tool_call.tool,
                action=tool_call.action,
                ok=False,
                evidence_strength=EvidenceStrength.NONE,
                notes=str(exc),
                raw_payload={"arguments": tool_call.arguments},
            )

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
        check_executable = service.command or "bibtex-check"
        if tool_call.action == "update_bibtex":
            # Prefer the explicit update_command config field; fall back to replacing
            # "check" with "update" in the check command only when the pattern is safe.
            if service.update_command:
                update_executable = service.update_command
            else:
                update_executable = check_executable.replace("bibtex-check", "bibtex-update")
                if update_executable == check_executable:
                    update_executable = check_executable + "-update"
            result = update_bibtex(source, executable=update_executable)
        else:
            result = check_bibtex(
                source,
                strict=bool(tool_call.arguments.get("strict", False)),
                executable=check_executable,
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
        if tool_call.action == "resolve_doi":
            return self._execute_candidate_lookup(
                tool_call,
                enabled=service.enabled,
                lookup_fn=lambda doi: lookup_crossref_work_by_doi(
                    doi,
                    timeout=service.timeout_seconds,
                    email=service.email,
                ),
            )
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
        if tool_call.action == "resolve_doi":
            return self._execute_candidate_lookup(
                tool_call,
                enabled=service.enabled,
                lookup_fn=lambda doi: lookup_openalex_work_by_doi(
                    doi,
                    timeout=service.timeout_seconds,
                    email=service.email,
                ),
            )
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

    def _execute_dblp(self, tool_call: ToolInvocation) -> ToolResultSummary:
        service = self.config.dblp
        return self._execute_candidate_search(
            tool_call,
            enabled=service.enabled,
            search_fn=lambda query: search_dblp_works(
                query,
                rows=int(tool_call.arguments.get("rows", service.rows)),
                timeout=service.timeout_seconds,
            ),
        )

    def _execute_acl_anthology(self, tool_call: ToolInvocation) -> ToolResultSummary:
        service = self.config.acl_anthology
        if not service.enabled:
            return ToolResultSummary(
                tool=tool_call.tool,
                action=tool_call.action,
                ok=False,
                notes="acl_anthology is disabled in config.",
            )
        anthology_id = tool_call.arguments.get("anthology_id")
        doi = tool_call.arguments.get("doi")
        url = tool_call.arguments.get("url")
        if not any((anthology_id, doi, url)):
            raise ValueError("acl_anthology calls require `anthology_id`, `doi`, or `url`.")
        try:
            candidates = resolve_acl_record(
                anthology_id=anthology_id,
                doi=doi,
                url=url,
                timeout=service.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResultSummary(
                tool=tool_call.tool,
                action=tool_call.action,
                ok=False,
                notes=str(exc),
            )
        matched_identifiers: dict[str, str] = {}
        resolved_id = anthology_id or extract_anthology_id(doi) or extract_anthology_id(url)
        if resolved_id:
            matched_identifiers["anthology_id"] = resolved_id
        if candidates and candidates[0].doi:
            matched_identifiers["doi"] = candidates[0].doi
        return ToolResultSummary(
            tool=tool_call.tool,
            action=tool_call.action,
            ok=True,
            evidence_strength=_strength_from_candidates(
                len(candidates),
                1.0 if candidates else 0.0,
            ),
            candidate_count=len(candidates),
            matched_identifiers=matched_identifiers,
            notes=f"{len(candidates)} candidates returned.",
            candidate_summaries=candidates,
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

    def _execute_candidate_lookup(
        self,
        tool_call: ToolInvocation,
        *,
        enabled: bool,
        lookup_fn,
    ) -> ToolResultSummary:
        if not enabled:
            return ToolResultSummary(
                tool=tool_call.tool,
                action=tool_call.action,
                ok=False,
                notes=f"{tool_call.tool.value} is disabled in config.",
            )
        doi = tool_call.arguments.get("doi")
        if not doi:
            raise ValueError(f"{tool_call.tool.value} DOI lookups require a `doi` argument.")
        try:
            candidates = lookup_fn(doi)
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
