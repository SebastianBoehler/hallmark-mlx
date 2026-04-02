"""Schema models for trace-based citation verification."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from hallmark_mlx.types import (
    EvidenceStrength,
    InputType,
    JSONDict,
    SuspectedIssueCode,
    ToolName,
    VerificationAction,
    VerificationVerdict,
)


class VerificationInput(BaseModel):
    """A single verification task presented to the policy."""

    record_id: str
    input_type: InputType
    raw_input: str
    context: str | None = None
    benchmark_bibtex_key: str | None = None
    private_holdout: bool = False
    source_metadata: JSONDict = Field(default_factory=dict)


class ParsedBibliographicFields(BaseModel):
    """Structured parse extracted from the input."""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    bibtex_type: str | None = None
    claim_text: str | None = None
    extras: JSONDict = Field(default_factory=dict)


class SuspectedIssue(BaseModel):
    """A potential verification failure mode detected by the policy."""

    code: SuspectedIssueCode
    rationale: str
    severity: float = Field(default=0.5, ge=0.0, le=1.0)


class ProposedQuery(BaseModel):
    """A search query or retrieval proposal."""

    query: str
    purpose: str
    expected_candidates: int = Field(default=5, ge=1)
    target_tool: ToolName | None = None


class ToolInvocation(BaseModel):
    """A concrete tool call requested by the policy."""

    tool: ToolName
    action: str
    arguments: JSONDict = Field(default_factory=dict)


class CandidateMatch(BaseModel):
    """A retrieved candidate paper or citation entry."""

    source: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str | None = None


class ToolResultSummary(BaseModel):
    """Normalized result of a tool call."""

    tool: ToolName
    action: str
    ok: bool
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    candidate_count: int = 0
    matched_identifiers: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None
    candidate_summaries: list[CandidateMatch] = Field(default_factory=list)
    raw_payload: JSONDict = Field(default_factory=dict)


class CandidateRanking(BaseModel):
    """Ranking of candidates after comparison."""

    candidates: list[CandidateMatch] = Field(default_factory=list)
    preferred_index: int | None = None
    needs_disambiguation: bool = False
    rationale: str | None = None


class FinalDecision(BaseModel):
    """Final verification verdict after tool use."""

    verdict: VerificationVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    abstain_reason: str | None = None
    should_update_bibtex: bool = False
    corrected_bibtex: str | None = None
    subtest_results: dict[str, bool] = Field(default_factory=dict)


class VerificationTrace(BaseModel):
    """End-to-end trace for a single verification decision."""

    policy_version: str = "v0"
    input: VerificationInput
    parsed_fields: ParsedBibliographicFields = Field(default_factory=ParsedBibliographicFields)
    suspected_issues: list[SuspectedIssue] = Field(default_factory=list)
    proposed_query: ProposedQuery | None = None
    next_action: VerificationAction = VerificationAction.PARSE_INPUT
    tool_calls: list[ToolInvocation] = Field(default_factory=list)
    tool_results: list[ToolResultSummary] = Field(default_factory=list)
    candidate_ranking: CandidateRanking | None = None
    final_decision: FinalDecision | None = None
    metadata: JSONDict = Field(default_factory=dict)

    @property
    def trace_id(self) -> str:
        """Stable trace identifier."""

        return self.input.record_id

    def to_training_dict(self) -> dict[str, Any]:
        """Serialize for supervised fine-tuning datasets."""

        return self.model_dump(exclude_none=True)
