"""JSONL reader and writer helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""

    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write JSON objects to a JSONL file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return target
