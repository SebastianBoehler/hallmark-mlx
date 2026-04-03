"""DBLP publication search wrapper."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from html import unescape
from urllib.parse import urlencode

from hallmark_mlx.data.schemas import CandidateMatch
from hallmark_mlx.tools.http import build_user_agent, fetch_json

_BASE_URL = "https://dblp.org/search/publ/api"
_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    return _TAG_RE.sub("", unescape(value)).strip() or None


def _normalize_tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _score_query_match(query: str, title: str | None) -> float:
    if not title:
        return 0.0
    query_tokens = _normalize_tokens(query)
    title_tokens = _normalize_tokens(title)
    if not title_tokens:
        return 0.0
    token_recall = len(query_tokens & title_tokens) / len(title_tokens)
    sequence = SequenceMatcher(None, query.lower(), title.lower()).ratio()
    return max(token_recall, sequence)


def _authors_from_hit(info: dict) -> list[str]:
    author_payload = (info.get("authors") or {}).get("author")
    if isinstance(author_payload, str):
        return [_clean_text(author_payload)] if _clean_text(author_payload) else []
    if isinstance(author_payload, dict):
        name = _clean_text(author_payload.get("text") or author_payload.get("@name"))
        return [name] if name else []
    if not isinstance(author_payload, list):
        return []
    authors: list[str] = []
    for author in author_payload:
        if isinstance(author, str):
            cleaned = _clean_text(author)
        else:
            cleaned = _clean_text(author.get("text") or author.get("@name"))
        if cleaned:
            authors.append(cleaned)
    return authors


def search_works(
    query: str,
    *,
    rows: int = 5,
    timeout: float = 15.0,
) -> list[CandidateMatch]:
    """Search DBLP publications by query text."""

    url = f"{_BASE_URL}?{urlencode({'q': query, 'h': rows, 'format': 'json'})}"
    payload = fetch_json(url, timeout=timeout, user_agent=build_user_agent())
    hits_payload = payload.get("result", {}).get("hits", {}).get("hit", [])
    if isinstance(hits_payload, dict):
        hits = [hits_payload]
    elif isinstance(hits_payload, list):
        hits = hits_payload
    else:
        hits = []
    candidates: list[CandidateMatch] = []
    for hit in hits:
        info = hit.get("info") or {}
        title = _clean_text(info.get("title"))
        candidates.append(
            CandidateMatch(
                source="dblp",
                title=title,
                authors=_authors_from_hit(info),
                year=int(info["year"]) if str(info.get("year", "")).isdigit() else None,
                venue=_clean_text(info.get("venue")),
                doi=_clean_text(info.get("doi")),
                url=_clean_text(info.get("url") or info.get("ee")),
                score=_score_query_match(query, title),
                rationale="DBLP publication search result.",
            ),
        )
    return candidates
