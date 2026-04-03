"""Semantic Scholar search wrapper."""

from __future__ import annotations

from urllib.parse import urlencode

from hallmark_mlx.data.schemas import CandidateMatch
from hallmark_mlx.tools.http import build_user_agent, fetch_json

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def _title_query_overlap(query: str, title: str | None) -> float:
    """Fraction of title tokens that appear in the query (recall-oriented)."""
    if not title:
        return 0.0
    query_tokens = set(query.lower().split())
    title_tokens = [t for t in title.lower().split() if len(t) > 2]
    if not title_tokens:
        return 0.0
    matches = sum(1 for t in title_tokens if t in query_tokens)
    return matches / len(title_tokens)


def _candidate_score(query: str, title: str | None, has_doi: bool) -> float:
    """Score a candidate by title-query overlap; DOI presence is a small bonus."""
    overlap = _title_query_overlap(query, title)
    base = 0.3 + overlap * 0.6
    if has_doi:
        base = min(base + 0.1, 1.0)
    return round(min(base, 1.0), 4)


def search_papers(
    query: str,
    *,
    limit: int = 5,
    timeout: float = 15.0,
) -> list[CandidateMatch]:
    """Search Semantic Scholar for candidate papers."""

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,venue,externalIds,url",
    }
    url = f"{_BASE_URL}?{urlencode(params)}"
    payload = fetch_json(url, timeout=timeout, user_agent=build_user_agent())
    candidates: list[CandidateMatch] = []
    for item in payload.get("data", []):
        authors = [author.get("name") for author in item.get("authors", []) if author.get("name")]
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")
        title = item.get("title")
        candidates.append(
            CandidateMatch(
                source="semantic_scholar",
                title=title,
                authors=authors,
                year=item.get("year"),
                venue=item.get("venue"),
                doi=doi,
                url=item.get("url"),
                score=_candidate_score(query, title, bool(doi)),
                rationale="Semantic Scholar paper search result.",
            ),
        )
    return candidates
