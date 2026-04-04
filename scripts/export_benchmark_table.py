#!/usr/bin/env python3
"""Export publication-style benchmark tables from the comparison summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from hallmark_mlx.utils.io import read_json, write_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = ROOT / "artifacts" / "hallmark_baseline_compare" / "summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "reports"

DEV_PUBLIC_COLUMNS = (
    "method",
    "source",
    "num_entries",
    "partial",
    "f1_hallucination",
    "detection_rate",
    "tier_weighted_f1",
    "false_positive_rate",
    "notes",
)
COMPARE32_COLUMNS = (
    "method",
    "f1_hallucination",
    "label_accuracy",
    "tool_use_rate",
    "completion_rate",
    "mean_api_calls",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _sort_key(row: dict[str, Any]) -> float:
    value = row.get("f1_hallucination")
    return float(value) if value is not None else -1.0


def _dev_public_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary["dev_public_rows"]:
        rows.append(
            {
                "method": row["name"],
                "source": row["source"],
                "num_entries": row["num_entries"],
                "partial": row["partial"],
                "f1_hallucination": row["f1_hallucination"],
                "detection_rate": row["detection_rate"],
                "tier_weighted_f1": row["tier_weighted_f1"],
                "false_positive_rate": row["false_positive_rate"],
                "notes": row["notes"] or row["status_message"],
            }
        )
    return sorted(rows, key=_sort_key, reverse=True)


def _compare32_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary["compare32_rows"]:
        rows.append(
            {
                "method": row["name"],
                "f1_hallucination": row["f1_hallucination"],
                "label_accuracy": row["label_accuracy"],
                "tool_use_rate": row["tool_use_rate"],
                "completion_rate": row["completion_rate"],
                "mean_api_calls": row["mean_api_calls"],
            }
        )
    return sorted(rows, key=_sort_key, reverse=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _latex_table(rows: list[dict[str, Any]], columns: tuple[str, ...], label: str) -> str:
    col_spec = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [_fmt(row.get(column)).replace("_", "\\_") for column in columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", f"% {label}"])
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    summary = read_json(args.summary_path)
    dev_public_rows = _dev_public_rows(summary)
    compare32_rows = _compare32_rows(summary)

    _write_csv(
        output_dir / "hallmark_dev_public_comparison.csv",
        dev_public_rows,
        DEV_PUBLIC_COLUMNS,
    )
    _write_csv(output_dir / "hallmark_compare32_methods.csv", compare32_rows, COMPARE32_COLUMNS)

    report_lines = [
        "# HALLMARK Benchmark Comparison",
        "",
        "## dev_public",
        "",
        _markdown_table(dev_public_rows, DEV_PUBLIC_COLUMNS),
        "",
        "## compare32",
        "",
        _markdown_table(compare32_rows, COMPARE32_COLUMNS),
        "",
        "## Provenance",
        "",
        "- `executed_here`: run locally in this repo during comparison generation.",
        (
            "- `upstream_published_result`: official result JSON shipped by the "
            "upstream HALLMARK repo."
        ),
        (
            "- `upstream_published_predictions`: official prediction JSONL shipped "
            "upstream and re-evaluated locally."
        ),
        (
            "- `upstream_history_partial`: historical upstream result, usually "
            "partial coverage or smaller sample."
        ),
        (
            "- `not_run`: baseline appears available here but was intentionally "
            "not executed in this pass."
        ),
        "- `unavailable`: baseline could not be run in this environment.",
    ]
    write_text(output_dir / "hallmark_benchmark_report.md", "\n".join(report_lines) + "\n")
    write_text(
        output_dir / "hallmark_dev_public_comparison.tex",
        _latex_table(dev_public_rows, DEV_PUBLIC_COLUMNS, "dev_public"),
    )
    write_text(
        output_dir / "hallmark_compare32_methods.tex",
        _latex_table(compare32_rows, COMPARE32_COLUMNS, "compare32"),
    )

    print(output_dir / "hallmark_benchmark_report.md")


if __name__ == "__main__":
    main()
