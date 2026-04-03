"""Planning helpers for the deterministic warm-start policy."""

from __future__ import annotations

from hallmark_mlx.data.schemas import (
    ParsedBibliographicFields,
    ProposedQuery,
    ToolInvocation,
    VerificationInput,
)
from hallmark_mlx.tools.acl_anthology import extract_anthology_id
from hallmark_mlx.types import InputType, ToolName, VerificationAction

_ACL_VENUE_TOKENS = {"acl", "naacl", "eacl", "emnlp", "aacl", "tacl", "coling", "conll"}
# Only include unambiguous CS/NLP venue names and technical abbreviations.
# Removed "language", "retrieval", "translation" — too generic and present in many
# non-CS fields (linguistics, history, library science, etc.) which would be
# misrouted to DBLP where they have no coverage.
_CS_NLP_TOKENS = _ACL_VENUE_TOKENS | {
    "neurips",
    "nips",
    "iclr",
    "icml",
    "aaai",
    "ijcai",
    "kdd",
    "sigir",
    "bert",
    "transformer",
    "transformers",
    "arxiv",
}
_ACTION_MAP = {
    ToolName.BIBTEX_UPDATER: VerificationAction.QUERY_BIBTEX_UPDATER,
    ToolName.CROSSREF: VerificationAction.QUERY_CROSSREF,
    ToolName.OPENALEX: VerificationAction.QUERY_OPENALEX,
    ToolName.DBLP: VerificationAction.QUERY_DBLP,
    ToolName.ACL_ANTHOLOGY: VerificationAction.QUERY_ACL_ANTHOLOGY,
    ToolName.SEMANTIC_SCHOLAR: VerificationAction.QUERY_SEMANTIC_SCHOLAR,
}


def _joined_hint_text(
    fields: ParsedBibliographicFields,
    verification_input: VerificationInput,
) -> str:
    parts = [
        verification_input.raw_input,
        fields.title or "",
        fields.venue or "",
        fields.claim_text or "",
    ]
    return " ".join(part.lower() for part in parts if part)


def _looks_like_cs_or_nlp(
    fields: ParsedBibliographicFields,
    verification_input: VerificationInput,
) -> bool:
    hint_text = _joined_hint_text(fields, verification_input)
    return any(token in hint_text for token in _CS_NLP_TOKENS)


def _dblp_query(fields: ParsedBibliographicFields) -> str | None:
    if not fields.title:
        return None
    query_parts = [fields.title]
    if fields.authors:
        query_parts.append(fields.authors[0].split()[-1])
    return " ".join(query_parts)


def build_query(
    fields: ParsedBibliographicFields,
    verification_input: VerificationInput,
) -> ProposedQuery | None:
    """Construct a deterministic retrieval query."""

    if fields.doi:
        target_tool = (
            ToolName.ACL_ANTHOLOGY if extract_anthology_id(fields.doi) else ToolName.CROSSREF
        )
        return ProposedQuery(
            query=fields.doi,
            purpose="resolve_doi_record",
            target_tool=target_tool,
        )
    if fields.title:
        query_parts = [fields.title]
        if fields.authors:
            query_parts.append(fields.authors[0].split()[-1])
        if fields.year:
            query_parts.append(str(fields.year))
        if fields.venue:
            query_parts.append(fields.venue)
        target_tool = ToolName.CROSSREF
        if _looks_like_cs_or_nlp(fields, verification_input):
            target_tool = ToolName.DBLP
        return ProposedQuery(
            query=" ".join(query_parts),
            purpose="resolve_canonical_record",
            target_tool=target_tool,
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
    rows = 5
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

    acl_identifier = (
        extract_anthology_id(fields.doi)
        or extract_anthology_id(fields.url)
        or extract_anthology_id(verification_input.raw_input)
    )
    if acl_identifier:
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.ACL_ANTHOLOGY,
                action="resolve_record",
                arguments={"anthology_id": acl_identifier},
            ),
        )

    if fields.doi:
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.CROSSREF,
                action="resolve_doi",
                arguments={"doi": fields.doi},
            ),
        )
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.OPENALEX,
                action="resolve_doi",
                arguments={"doi": fields.doi},
            ),
        )
        if verification_input.input_type in {
            InputType.CLAIM_FOR_SUPPORTING_REFS,
            InputType.PARAGRAPH_WITH_CITATION,
        }:
            tool_calls.append(
                ToolInvocation(
                    tool=ToolName.SEMANTIC_SCHOLAR,
                    action="search_papers",
                    arguments={"query": proposed_query.query, "rows": rows},
                ),
            )
        return tool_calls

    tool_calls.extend(
        [
            ToolInvocation(
                tool=ToolName.CROSSREF,
                action="search_works",
                arguments={"query": proposed_query.query, "rows": rows},
            ),
            ToolInvocation(
                tool=ToolName.OPENALEX,
                action="search_works",
                arguments={"query": proposed_query.query, "rows": rows},
            ),
        ],
    )
    dblp_query = _dblp_query(fields)
    if dblp_query and _looks_like_cs_or_nlp(fields, verification_input):
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.DBLP,
                action="search_works",
                arguments={"query": dblp_query, "rows": rows},
            ),
        )
    # Add Semantic Scholar when the input explicitly needs supporting references,
    # when an arXiv ID is present (SS has strong arXiv coverage), or when the
    # venue/content looks like CS/NLP. Avoid adding it for every no-DOI input
    # (that would append it to Crossref + OpenAlex + DBLP for all unknown papers
    # and inflate the api_calls budget unnecessarily).
    wants_ss = (
        verification_input.input_type in {
            InputType.CLAIM_FOR_SUPPORTING_REFS,
            InputType.PARAGRAPH_WITH_CITATION,
        }
        or bool(fields.arxiv_id)
        or _looks_like_cs_or_nlp(fields, verification_input)
    )
    if wants_ss:
        tool_calls.append(
            ToolInvocation(
                tool=ToolName.SEMANTIC_SCHOLAR,
                action="search_papers",
                arguments={"query": proposed_query.query, "rows": rows},
            ),
        )
    return tool_calls


def next_action_for_tool_calls(tool_calls: list[ToolInvocation]) -> VerificationAction:
    """Map the first tool call to the corresponding high-level action."""

    if not tool_calls:
        return VerificationAction.ABSTAIN
    return _ACTION_MAP[tool_calls[0].tool]
