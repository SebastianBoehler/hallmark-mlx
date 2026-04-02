"""Helpers for contamination-aware citation grouping."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from hallmark_mlx.data.schemas import ParsedBibliographicFields, VerificationTrace

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def normalize_doi(doi: str | None) -> str | None:
    """Normalize DOI strings for grouping."""

    if not doi:
        return None
    cleaned = doi.strip().lower()
    cleaned = cleaned.removeprefix("https://doi.org/")
    cleaned = cleaned.removeprefix("http://doi.org/")
    cleaned = cleaned.removeprefix("doi:")
    return cleaned or None


def normalize_title(title: str | None) -> str | None:
    """Normalize titles before hashing."""

    if not title:
        return None
    lowered = title.lower()
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", lowered).strip() or None


def _author_signature(authors: list[str]) -> str:
    surnames = []
    for author in authors:
        parts = [part for part in author.lower().split() if part]
        if parts:
            surnames.append(parts[-1])
    return "-".join(sorted(surnames[:4]))


def citation_family_id(fields: ParsedBibliographicFields) -> str:
    """Build a citation-family identifier from DOI or normalized metadata."""

    doi = normalize_doi(fields.doi)
    if doi:
        return f"doi:{doi}"
    fingerprint = "|".join(
        filter(
            None,
            [
                normalize_title(fields.title),
                _author_signature(fields.authors),
                str(fields.year) if fields.year else None,
            ],
        ),
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"meta:{digest}"


def trace_family_id(trace: VerificationTrace) -> str:
    """Return the family identifier for a trace."""

    return citation_family_id(trace.parsed_fields)


def detect_family_overlap(
    left: Iterable[VerificationTrace],
    right: Iterable[VerificationTrace],
) -> set[str]:
    """Return family identifiers that appear in both collections."""

    left_ids = {trace_family_id(trace) for trace in left}
    right_ids = {trace_family_id(trace) for trace in right}
    return left_ids & right_ids
