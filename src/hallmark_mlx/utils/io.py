"""Basic filesystem and JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON object to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def read_text(path: str | Path) -> str:
    """Read UTF-8 text from disk."""

    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> Path:
    """Write UTF-8 text to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
