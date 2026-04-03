"""ACL Anthology deterministic record lookup wrapper."""

from __future__ import annotations

import re

from hallmark_mlx.data.schemas import CandidateMatch
from hallmark_mlx.tools.http import build_user_agent, fetch_text

_BASE_URL = "https://aclanthology.org"
_DOI_PREFIX = "10.18653/v1/"
_ID_RE = re.compile(
    r"https?://aclanthology\.org/([A-Z0-9][\w.-]+?)(?:\.pdf|\.bib)?/?$",
    re.IGNORECASE,
)


def extract_anthology_id(value: str | None) -> str | None:
    """Extract an ACL Anthology paper identifier from DOI or URL."""

    if not value:
        return None
    stripped = value.strip()
    doi_value = re.sub(r"^https?://(dx\.)?doi\.org/", "", stripped, flags=re.IGNORECASE)
    if doi_value.lower().startswith(_DOI_PREFIX):
        identifier = doi_value[len(_DOI_PREFIX) :]
        return identifier or None
    match = _ID_RE.search(stripped)
    return match.group(1) if match else None


def _extract_bib_field(name: str, text: str) -> str | None:
    pattern = rf"{name}\s*=\s*"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    rest = text[match.end() :]
    if not rest:
        return None
    delimiter = rest[0]
    if delimiter == "{":
        depth = 0
        for index, char in enumerate(rest):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return re.sub(r"\s+", " ", rest[1:index]).strip() or None
        return None
    if delimiter == '"':
        end = rest.find('"', 1)
        if end == -1:
            return None
        return re.sub(r"\s+", " ", rest[1:end]).strip() or None
    return None


def _authors_from_bib(author_text: str | None) -> list[str]:
    if not author_text:
        return []
    return [author.strip() for author in author_text.split(" and ") if author.strip()]


def resolve_record(
    *,
    anthology_id: str | None = None,
    doi: str | None = None,
    url: str | None = None,
    timeout: float = 15.0,
) -> list[CandidateMatch]:
    """Resolve one ACL Anthology record by Anthology ID, DOI, or ACL URL."""

    resolved_id = anthology_id or extract_anthology_id(doi) or extract_anthology_id(url)
    if not resolved_id:
        return []
    bib_url = f"{_BASE_URL}/{resolved_id}.bib"
    bib_text = fetch_text(
        bib_url,
        timeout=timeout,
        user_agent=build_user_agent(),
        accept="text/plain",
    )
    title = _extract_bib_field("title", bib_text)
    if not title:
        return []
    doi_value = _extract_bib_field("doi", bib_text)
    url_value = _extract_bib_field("url", bib_text) or f"{_BASE_URL}/{resolved_id}"
    year_value = _extract_bib_field("year", bib_text)
    return [
        CandidateMatch(
            source="acl_anthology",
            title=title,
            authors=_authors_from_bib(_extract_bib_field("author", bib_text)),
            year=int(year_value) if year_value and year_value.isdigit() else None,
            venue=(
                _extract_bib_field("booktitle", bib_text)
                or _extract_bib_field("journal", bib_text)
            ),
            doi=doi_value,
            url=url_value,
            score=1.0,
            rationale="ACL Anthology authoritative record lookup.",
        ),
    ]
