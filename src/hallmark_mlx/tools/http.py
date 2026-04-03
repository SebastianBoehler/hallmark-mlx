"""Shared HTTP helpers for scholarly metadata tools."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


def build_user_agent(email: str | None = None) -> str:
    """Return a polite user agent string for public scholarly APIs."""

    return f"hallmark-mlx/0.1.0 ({email})" if email else "hallmark-mlx/0.1.0"


def fetch_json(url: str, *, timeout: float, user_agent: str) -> dict:
    """Fetch and decode a JSON response."""

    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_text(
    url: str,
    *,
    timeout: float,
    user_agent: str,
    accept: str = "text/plain",
) -> str:
    """Fetch and decode a text response."""

    request = Request(url, headers={"User-Agent": user_agent, "Accept": accept})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")
