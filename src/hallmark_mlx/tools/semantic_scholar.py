"""Semantic Scholar search wrapper."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hallmark_mlx.data.schemas import CandidateMatch

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def _fetch_json(url: str, *, timeout: float, user_agent: str) -> dict:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


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
    payload = _fetch_json(url, timeout=timeout, user_agent="hallmark-mlx/0.1.0")
    candidates: list[CandidateMatch] = []
    for item in payload.get("data", []):
        authors = [author.get("name") for author in item.get("authors", []) if author.get("name")]
        external_ids = item.get("externalIds") or {}
        candidates.append(
            CandidateMatch(
                source="semantic_scholar",
                title=item.get("title"),
                authors=authors,
                year=item.get("year"),
                venue=item.get("venue"),
                doi=external_ids.get("DOI"),
                url=item.get("url"),
                score=1.0 if external_ids.get("DOI") else 0.7,
                rationale="Semantic Scholar paper search result.",
            ),
        )
    return candidates
