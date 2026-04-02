"""Crossref search wrapper."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hallmark_mlx.data.schemas import CandidateMatch

_BASE_URL = "https://api.crossref.org/works"


def _fetch_json(url: str, *, timeout: float, user_agent: str) -> dict:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _extract_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "issued"):
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def search_works(
    query: str,
    *,
    rows: int = 5,
    timeout: float = 15.0,
    email: str | None = None,
) -> list[CandidateMatch]:
    """Search Crossref for bibliographic candidates."""

    params = {"query.bibliographic": query, "rows": rows}
    if email:
        params["mailto"] = email
    url = f"{_BASE_URL}?{urlencode(params)}"
    user_agent = f"hallmark-mlx/0.1.0 ({email})" if email else "hallmark-mlx/0.1.0"
    payload = _fetch_json(url, timeout=timeout, user_agent=user_agent)
    items = payload.get("message", {}).get("items", [])
    candidates: list[CandidateMatch] = []
    for item in items:
        authors = []
        for author in item.get("author", []):
            name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
            if name:
                authors.append(name)
        raw_score = float(item.get("score", 0.0) or 0.0)
        candidates.append(
            CandidateMatch(
                source="crossref",
                title=(item.get("title") or [None])[0],
                authors=authors,
                year=_extract_year(item),
                venue=(item.get("container-title") or [None])[0],
                doi=item.get("DOI"),
                url=item.get("URL"),
                score=min(raw_score / 100.0, 1.0),
                rationale="Crossref bibliographic search result.",
            ),
        )
    return candidates
