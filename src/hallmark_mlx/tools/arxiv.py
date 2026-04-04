"""arXiv API record lookup wrapper."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from hallmark_mlx.data.schemas import CandidateMatch
from hallmark_mlx.tools.http import build_user_agent, fetch_text

_BASE_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_ID_RE = re.compile(
    r"(?:10\.48550/arxiv\.|arxiv[:/\s]*)(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def extract_arxiv_id(value: str | None) -> str | None:
    """Extract an arXiv identifier from a DOI, URL, or raw arXiv string."""

    if not value:
        return None
    match = _ARXIV_ID_RE.search(value.strip())
    return match.group(1) if match else None


def _text(parent: ET.Element, tag: str) -> str | None:
    node = parent.find(tag, _ATOM_NS)
    if node is None or node.text is None:
        return None
    return re.sub(r"\s+", " ", node.text).strip() or None


def _candidate_from_entry(entry: ET.Element, *, score: float) -> CandidateMatch:
    authors = [
        name.text.strip()
        for name in entry.findall("atom:author/atom:name", _ATOM_NS)
        if name.text and name.text.strip()
    ]
    published = _text(entry, "atom:published")
    year = int(published[:4]) if published and published[:4].isdigit() else None
    return CandidateMatch(
        source="arxiv",
        title=_text(entry, "atom:title"),
        authors=authors,
        year=year,
        venue="arXiv",
        doi=None,
        url=_text(entry, "atom:id"),
        score=score,
        rationale="arXiv API record lookup.",
    )


def resolve_record(
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    url: str | None = None,
    timeout: float = 15.0,
) -> list[CandidateMatch]:
    """Resolve a single arXiv paper by identifier, DOI, or arxiv.org URL."""

    resolved_id = arxiv_id or extract_arxiv_id(doi) or extract_arxiv_id(url)
    if not resolved_id:
        return []
    request_url = f"{_BASE_URL}?{urlencode({'id_list': resolved_id})}"
    payload = fetch_text(
        request_url,
        timeout=timeout,
        user_agent=build_user_agent(),
        accept="application/atom+xml",
    )
    root = ET.fromstring(payload)
    entries = root.findall("atom:entry", _ATOM_NS)
    if not entries:
        return []
    return [_candidate_from_entry(entries[0], score=1.0)]
