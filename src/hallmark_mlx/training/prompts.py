"""Prompt templates and native tool schemas for trace-based policy learning."""

from __future__ import annotations

import json
from typing import Any

from hallmark_mlx.data.schemas import VerificationInput

POLICY_SYSTEM_PROMPT = """You are a citation verification policy model.

Your job is not to guess whether a citation is true from memory.
Your job is to decide how to verify it.

Operational rules:
1. the first assistant turn must be a tool call,
2. never emit a final decision before at least one tool result,
3. after each tool result either emit another tool call or one final assistant JSON object,
4. do not answer from latent memory when verification tools are available,
5. finalize as soon as one strong confirming or disconfirming evidence path exists,
6. do not repeat the same lookup after a strong result unless the previous tool explicitly failed,
7. use a fallback tool only when the prior result was missing, ambiguous, or errored,
8. do not emit free-form chain-of-thought.

Return only structured outputs that follow the transcript protocol.
"""


def build_user_prompt(verification_input: VerificationInput) -> str:
    """Render a user prompt for a verification task."""

    return (
        "Construct a verification trace for the following input.\n"
        "Begin with a tool call and verify before deciding.\n\n"
        f"{json.dumps(verification_input.model_dump(exclude_none=True), indent=2)}"
    )


def build_tool_schemas() -> list[dict[str, Any]]:
    """Return Qwen-native function tool schemas for the verification layer."""

    return [
        {
            "type": "function",
            "function": {
                "name": "bibtex_updater.check_bibtex",
                "description": (
                    "Validate or inspect a BibTeX entry. "
                    "Exactly one of 'bibtex' (inline BibTeX string) or 'path' (file path) must be provided."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bibtex": {"type": "string", "description": "Inline BibTeX entry string."},
                        "path": {"type": "string", "description": "Path to a .bib file."},
                        "strict": {"type": "boolean", "description": "Enable strict validation mode."},
                    },
                    "required": ["bibtex"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bibtex_updater.update_bibtex",
                "description": (
                    "Repair a BibTeX entry using external bibliographic metadata. "
                    "Exactly one of 'bibtex' (inline BibTeX string) or 'path' (file path) must be provided."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bibtex": {"type": "string", "description": "Inline BibTeX entry string."},
                        "path": {"type": "string", "description": "Path to a .bib file."},
                    },
                    "required": ["bibtex"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "crossref.resolve_doi",
                "description": "Resolve one DOI against Crossref.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doi": {"type": "string"},
                    },
                    "required": ["doi"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "crossref.search_works",
                "description": (
                    "Search Crossref works by title, author, venue, "
                    "or DOI-like query text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "rows": {"type": "integer"},
                    },
                    "required": ["query", "rows"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "openalex.resolve_doi",
                "description": "Resolve one DOI against OpenAlex.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doi": {"type": "string"},
                    },
                    "required": ["doi"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "openalex.search_works",
                "description": (
                    "Search OpenAlex works by title, author, venue, "
                    "or DOI-like query text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "rows": {"type": "integer"},
                    },
                    "required": ["query", "rows"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dblp.search_works",
                "description": "Search DBLP for computer science and AI publications.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "rows": {"type": "integer"},
                    },
                    "required": ["query", "rows"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "acl_anthology.resolve_record",
                "description": (
                    "Resolve an ACL Anthology paper deterministically from an Anthology ID, "
                    "ACL DOI, or aclanthology.org URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anthology_id": {"type": "string"},
                        "doi": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "arxiv.resolve_record",
                "description": (
                    "Resolve an arXiv paper deterministically from an arXiv ID, "
                    "10.48550/arXiv DOI, or arxiv.org URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arxiv_id": {"type": "string"},
                        "doi": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "semantic_scholar.search_papers",
                "description": (
                    "Search Semantic Scholar papers when Crossref/OpenAlex "
                    "are weak or missing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "rows": {"type": "integer"},
                    },
                    "required": ["query", "rows"],
                },
            },
        },
    ]


def build_available_tools_prompt() -> str:
    """Describe the current tool palette."""

    return (
        "Available tools: bibtex_updater, crossref, openalex, dblp, acl_anthology, arxiv, "
        "semantic_scholar.\n"
        "Allowed tool actions and required arguments:\n"
        "- bibtex_updater.check_bibtex requires {\"bibtex\": \"...\"} or {\"path\": \"...\"}\n"
        "- bibtex_updater.update_bibtex requires {\"bibtex\": \"...\"} or {\"path\": \"...\"}\n"
        "- crossref.resolve_doi requires {\"doi\": \"...\"}\n"
        "- crossref.search_works requires {\"query\": \"...\", \"rows\": N}\n"
        "- openalex.resolve_doi requires {\"doi\": \"...\"}\n"
        "- openalex.search_works requires {\"query\": \"...\", \"rows\": N}\n"
        "- dblp.search_works requires {\"query\": \"...\", \"rows\": N}\n"
        "- acl_anthology.resolve_record requires one of "
        "{\"anthology_id\": \"...\"}, {\"doi\": \"...\"}, or {\"url\": \"...\"}\n"
        "- arxiv.resolve_record requires one of "
        "{\"arxiv_id\": \"...\"}, {\"doi\": \"...\"}, or {\"url\": \"...\"}\n"
        "- semantic_scholar.search_papers requires {\"query\": \"...\", \"rows\": N}\n"
        "Assistant protocol:\n"
        "- tool requests are emitted as native function calls and rendered by the tokenizer "
        "chat template\n"
        "- tool observations arrive as tool messages containing compact JSON evidence, not "
        "full raw dumps\n"
        "- the final assistant turn must be one compact JSON object with verdict, confidence, "
        "rationale, and optional subtest_results\n"
        "- valid final verdict values are exactly: verified, corrected, hallucinated, "
        "unsupported, abstain\n"
        "- do not emit a final decision before a tool observation\n"
        "- use dblp for CS/ML/NLP venue disambiguation with title plus first-author queries\n"
        "- use acl_anthology.resolve_record when the DOI or URL clearly points to ACL Anthology\n"
        "- use arxiv.resolve_record when the DOI or URL clearly points to arXiv\n"
        "- if one DOI lookup resolves the exact work, finalize instead of searching again\n"
        "- if one DOI lookup proves the cited venue/year/title is materially wrong, finalize "
        "instead of searching again\n"
        "- if a DOI lookup fails but one fallback title+author search gives a strong exact "
        "match, finalize after that fallback"
    )
