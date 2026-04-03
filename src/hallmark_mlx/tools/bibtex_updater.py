"""BibTeX Updater subprocess wrappers."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hallmark_mlx.types import JSONDict
from hallmark_mlx.utils.io import read_json, read_text

_STATUS_LINE_RE = re.compile(r"^INFO:\s{2,}([A-Z_]+):\s+(\d+)\s*$")


@dataclass(slots=True)
class BibtexUpdaterResult:
    """Normalized BibTeX Updater subprocess result."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    report: JSONDict | None = None
    output_text: str | None = None


def _safe_read_report(report_path: Path | None) -> JSONDict | None:
    if report_path is None or not report_path.exists():
        return None
    try:
        return read_json(report_path)
    except json.JSONDecodeError:
        return None


def _materialize_bibtex_source(bibtex: str | Path) -> tuple[Path, bool]:
    if isinstance(bibtex, Path):
        return bibtex, False
    raw = bibtex.strip()
    if raw.startswith("@") or "\n" in raw:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".bib",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(bibtex)
            return Path(handle.name), True
    return Path(bibtex), False


def check_bibtex(
    bibtex: str | Path,
    *,
    report_path: Path | None = None,
    strict: bool = False,
    executable: str = "bibtex-check",
) -> BibtexUpdaterResult:
    """Run `bibtex-check` on a BibTeX file or entry string."""

    input_path, cleanup_input = _materialize_bibtex_source(bibtex)
    command = [executable, str(input_path)]
    if report_path is not None:
        command.extend(["--report", str(report_path)])
    if strict:
        command.append("--strict")
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    report = _safe_read_report(report_path)
    if cleanup_input:
        input_path.unlink(missing_ok=True)
    return BibtexUpdaterResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        report=report,
    )


def update_bibtex(
    bibtex: str | Path,
    *,
    output_path: Path | None = None,
    executable: str = "bibtex-update",
) -> BibtexUpdaterResult:
    """Run `bibtex-update` on a BibTeX file or entry string."""

    input_path, cleanup_input = _materialize_bibtex_source(bibtex)
    cleanup_output = False
    target_output = output_path
    if target_output is None:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".bib",
            delete=False,
            encoding="utf-8",
        ) as handle:
            target_output = Path(handle.name)
            cleanup_output = True
    command = [executable, str(input_path), "-o", str(target_output)]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    output_text = read_text(target_output) if target_output.exists() else None
    if cleanup_input:
        input_path.unlink(missing_ok=True)
    if cleanup_output:
        target_output.unlink(missing_ok=True)
    return BibtexUpdaterResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_text=output_text,
    )


def extract_status_counts(output: str) -> dict[str, int]:
    """Parse BibTeX Updater summary status counts from stderr/stdout text."""

    counts: dict[str, int] = {}
    for line in output.splitlines():
        match = _STATUS_LINE_RE.match(line.strip())
        if match is None:
            continue
        name, raw_count = match.groups()
        counts[name] = counts.get(name, 0) + int(raw_count)
    return counts


def primary_status(output: str) -> str | None:
    """Return the dominant BibTeX Updater status label, if present."""

    counts = extract_status_counts(output)
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def summarize_result(result: BibtexUpdaterResult) -> JSONDict:
    """Produce a compact summary that is stable across CLI versions."""

    report = result.report or {}
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    raw_output = (result.stderr or result.stdout).strip()
    status_counts = extract_status_counts(raw_output)
    return {
        "ok": result.returncode == 0,
        "issues_detected": int(summary.get("issues_detected", 0)),
        "candidate_count": int(summary.get("checked_entries", 0)),
        "matched_identifiers": {},
        "notes": raw_output or "bibtex_updater completed",
        "status": primary_status(raw_output),
        "status_counts": status_counts,
        "report": report,
    }
