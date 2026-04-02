"""OpenAlex search wrapper."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hallmark_mlx.data.schemas import CandidateMatch

_BASE_URL = "https://api.openalex.org/works"


def _fetch_json(url: str, *, timeout: float, user_agent: str) -> dict:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def search_works(
    query: str,
    *,
    per_page: int = 5,
    timeout: float = 15.0,
    email: str | None = None,
) -> list[CandidateMatch]:
    """Search OpenAlex for candidate works."""

    params = {"search": query, "per-page": per_page}
    if email:
        params["mailto"] = email
    url = f"{_BASE_URL}?{urlencode(params)}"
    user_agent = f"hallmark-mlx/0.1.0 ({email})" if email else "hallmark-mlx/0.1.0"
    payload = _fetch_json(url, timeout=timeout, user_agent=user_agent)
    candidates: list[CandidateMatch] = []
    for item in payload.get("results", []):
        authors = [
            author.get("author", {}).get("display_name")
            for author in item.get("authorships", [])
            if author.get("author", {}).get("display_name")
        ]
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        candidates.append(
            CandidateMatch(
                source="openalex",
                title=item.get("title"),
                authors=authors,
                year=item.get("publication_year"),
                venue=source.get("display_name"),
                doi=item.get("doi"),
                url=item.get("id"),
                score=min(float(item.get("relevance_score", 0.0) or 0.0) / 100.0, 1.0),
                rationale="OpenAlex work search result.",
            ),
        )
    return candidates
