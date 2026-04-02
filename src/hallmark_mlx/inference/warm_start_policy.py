"""Deterministic warm-start policy for bootstrapping verification traces."""

from __future__ import annotations

import re

from hallmark_mlx.data.schemas import (
    ParsedBibliographicFields,
    ProposedQuery,
    SuspectedIssue,
    ToolInvocation,
    VerificationInput,
    VerificationTrace,
)
from hallmark_mlx.types import InputType, SuspectedIssueCode, ToolName, VerificationAction

_DOI_RE = re.compile(r"\b(?:https?://doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barxiv[:\s]*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PARENS_RE = re.compile(r"\([^)]*\)")
_WHITESPACE_RE = re.compile(r"\s+")
_BIBTEX_FIELD_RE = re.compile(
    r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\")",
    re.IGNORECASE | re.DOTALL,
)


def _clean_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip())


def _strip_bibtex_wrapping(value: str) -> str:
    return value.replace("\n", " ").replace("{", "").replace("}", "").strip()


def _extract_doi(raw_text: str) -> str | None:
    match = _DOI_RE.search(raw_text)
    return match.group(1).rstrip(".,);]") if match else None


def _extract_arxiv_id(raw_text: str) -> str | None:
    match = _ARXIV_RE.search(raw_text)
    return match.group(1) if match else None


def _extract_year(raw_text: str) -> int | None:
    matches = list(_YEAR_RE.finditer(raw_text))
    return int(matches[-1].group(0)) if matches else None


def _split_authors(raw_authors: str | None) -> list[str]:
    if not raw_authors:
        return []
    return [_clean_text(author) for author in raw_authors.split(" and ") if _clean_text(author)]


def _first_author_surname(authors: list[str]) -> str | None:
    if not authors:
        return None
    parts = [part for part in authors[0].split() if part]
    return parts[-1] if parts else None


def _title_from_citation(raw_text: str, year: int | None, doi: str | None) -> str | None:
    cleaned = raw_text
    if doi:
        cleaned = cleaned.replace(doi, " ")
    if year:
        cleaned = cleaned.replace(str(year), " ")
    cleaned = _PARENS_RE.sub(" ", cleaned)
    segments = [segment.strip(" .;,:-") for segment in cleaned.split(".") if segment.strip(" .;,:-")]
    if len(segments) >= 2:
        return _clean_text(segments[1])
    if segments:
        return _clean_text(segments[0])
    return None


def _segments_from_citation(raw_text: str) -> list[str]:
    return [segment.strip(" .;,:-") for segment in raw_text.split(".") if segment.strip(" .;,:-")]


def _authors_from_citation(raw_text: str) -> list[str]:
    segments = _segments_from_citation(raw_text)
    if not segments:
        return []
    author_segment = segments[0]
    lower_segment = author_segment.lower()
    if "et al" in lower_segment:
        author_segment = author_segment.replace("et al", "").replace("et al.", "")
    parts = [part.strip() for part in re.split(r"\band\b|,|;", author_segment) if part.strip()]
    return [_clean_text(part) for part in parts[:4]]


def _venue_from_citation(raw_text: str, year: int | None) -> str | None:
    segments = _segments_from_citation(raw_text)
    if len(segments) < 3:
        return None
    venue_segment = segments[-1]
    if year:
        venue_segment = venue_segment.replace(str(year), " ")
    venue = _clean_text(venue_segment)
    return venue or None


def parse_bibtex_entry(raw_text: str) -> ParsedBibliographicFields:
    """Parse a simple BibTeX entry into structured fields."""

    entry_type_match = re.search(r"@(\w+)\s*\{", raw_text)
    fields: dict[str, str] = {}
    for key, braced_value, quoted_value in _BIBTEX_FIELD_RE.findall(raw_text):
        fields[key.lower()] = _strip_bibtex_wrapping(braced_value or quoted_value)
    doi = fields.get("doi") or _extract_doi(raw_text)
    arxiv_id = fields.get("eprint") or _extract_arxiv_id(raw_text)
    return ParsedBibliographicFields(
        title=fields.get("title"),
        authors=_split_authors(fields.get("author")),
        year=int(fields["year"]) if fields.get("year", "").isdigit() else _extract_year(raw_text),
        venue=fields.get("journal") or fields.get("booktitle"),
        doi=doi,
        arxiv_id=arxiv_id,
        url=fields.get("url"),
        bibtex_type=entry_type_match.group(1).lower() if entry_type_match else None,
        extras={key: value for key, value in fields.items() if key not in {"title", "author", "year", "journal", "booktitle", "doi", "eprint", "url"}},
    )


def parse_input(verification_input: VerificationInput) -> ParsedBibliographicFields:
    """Parse raw user input into coarse bibliographic structure."""

    raw_text = verification_input.raw_input
    if verification_input.input_type == InputType.BIBTEX_ENTRY:
        return parse_bibtex_entry(raw_text)
    if verification_input.input_type == InputType.CLAIM_FOR_SUPPORTING_REFS:
        return ParsedBibliographicFields(claim_text=_clean_text(raw_text))

    doi = _extract_doi(raw_text)
    arxiv_id = _extract_arxiv_id(raw_text)
    year = _extract_year(raw_text)
    title = _title_from_citation(raw_text, year=year, doi=doi)
    authors = _authors_from_citation(raw_text)
    venue = _venue_from_citation(raw_text, year=year)
    if verification_input.input_type == InputType.PARAGRAPH_WITH_CITATION:
        return ParsedBibliographicFields(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            arxiv_id=arxiv_id,
            claim_text=_clean_text(raw_text),
        )
    return ParsedBibliographicFields(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
    )


def detect_suspected_issues(
    verification_input: VerificationInput,
    fields: ParsedBibliographicFields,
) -> list[SuspectedIssue]:
    """Flag obvious verification risks before tool use."""

    issues: list[SuspectedIssue] = []
    if verification_input.input_type != InputType.CLAIM_FOR_SUPPORTING_REFS and not fields.doi:
        issues.append(
            SuspectedIssue(
                code=SuspectedIssueCode.MISSING_DOI,
                rationale="The input does not contain a DOI and needs external resolution.",
                severity=0.65,
            ),
        )
    if not fields.title and not fields.claim_text:
        issues.append(
            SuspectedIssue(
                code=SuspectedIssueCode.INSUFFICIENT_METADATA,
                rationale="The input lacks a stable title or claim string.",
                severity=0.8,
            ),
        )
    if fields.title and len(fields.title.split()) < 4:
        issues.append(
            SuspectedIssue(
                code=SuspectedIssueCode.TITLE_AMBIGUITY,
                rationale="Short titles tend to produce ambiguous candidate sets.",
                severity=0.55,
            ),
        )
    if fields.arxiv_id and fields.venue:
        issues.append(
            SuspectedIssue(
                code=SuspectedIssueCode.PREPRINT_AS_PUBLISHED,
                rationale="The entry mixes arXiv-style metadata with venue metadata.",
                severity=0.7,
            ),
        )
    return issues


def build_query(fields: ParsedBibliographicFields, verification_input: VerificationInput) -> ProposedQuery | None:
    """Construct a deterministic retrieval query."""

    if fields.doi:
        return ProposedQuery(query=fields.doi, purpose="resolve_doi_record", target_tool=ToolName.CROSSREF)
    if fields.title:
        query_parts = [fields.title]
        surname = _first_author_surname(fields.authors)
        if surname:
            query_parts.append(surname)
        if fields.year:
            query_parts.append(str(fields.year))
        if fields.venue:
            query_parts.append(fields.venue)
        return ProposedQuery(
            query=" ".join(query_parts),
            purpose="resolve_canonical_record",
            target_tool=ToolName.CROSSREF,
        )
    if fields.claim_text:
        return ProposedQuery(
            query=fields.claim_text[:220],
            purpose="find_supporting_reference",
            target_tool=ToolName.SEMANTIC_SCHOLAR,
        )
    if verification_input.context:
        return ProposedQuery(query=verification_input.context[:220], purpose="recover_context")
    return None


def build_tool_calls(
    verification_input: VerificationInput,
    fields: ParsedBibliographicFields,
    proposed_query: ProposedQuery | None,
) -> list[ToolInvocation]:
    """Select a deterministic tool-use chain for bootstrapping traces."""

    tool_calls: list[ToolInvocation] = []
    if verification_input.input_type == InputType.BIBTEX_ENTRY:
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.BIBTEX_UPDATER,
                action="check_bibtex",
                arguments={"bibtex": verification_input.raw_input, "strict": True},
            ),
        )
    if proposed_query is None:
        return tool_calls
    rows = 5
    if proposed_query.target_tool in {ToolName.CROSSREF, None}:
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.CROSSREF,
                action="search_works",
                arguments={"query": proposed_query.query, "rows": rows},
            ),
        )
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.OPENALEX,
                action="search_works",
                arguments={"query": proposed_query.query, "rows": rows},
            ),
        )
    if verification_input.input_type in {
        InputType.CLAIM_FOR_SUPPORTING_REFS,
        InputType.PARAGRAPH_WITH_CITATION,
    } or fields.arxiv_id or not fields.doi:
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.SEMANTIC_SCHOLAR,
                action="search_papers",
                arguments={"query": proposed_query.query, "rows": rows},
            ),
        )
    return tool_calls


class WarmStartPolicyModel:
    """Heuristic policy backend for end-to-end bootstrapping and trace curation."""

    def propose_trace(self, verification_input: VerificationInput) -> VerificationTrace:
        fields = parse_input(verification_input)
        issues = detect_suspected_issues(verification_input, fields)
        proposed_query = build_query(fields, verification_input)
        tool_calls = build_tool_calls(verification_input, fields, proposed_query)
        next_action = VerificationAction.ABSTAIN
        if tool_calls:
            first_tool = tool_calls[0].tool
            action_map = {
                ToolName.BIBTEX_UPDATER: VerificationAction.QUERY_BIBTEX_UPDATER,
                ToolName.CROSSREF: VerificationAction.QUERY_CROSSREF,
                ToolName.OPENALEX: VerificationAction.QUERY_OPENALEX,
                ToolName.SEMANTIC_SCHOLAR: VerificationAction.QUERY_SEMANTIC_SCHOLAR,
            }
            next_action = action_map[first_tool]
        return VerificationTrace(
            policy_version="warm-start-v0",
            input=verification_input,
            parsed_fields=fields,
            suspected_issues=issues,
            proposed_query=proposed_query,
            next_action=next_action,
            tool_calls=tool_calls,
            metadata={
                "policy_backend": "warm_start",
                "bootstrap_ready": True,
            },
        )
