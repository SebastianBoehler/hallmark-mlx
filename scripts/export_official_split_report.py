#!/usr/bin/env python3
"""Export a public-facing report and figure for official HALLMARK splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tueplots import bundles, figsizes

from hallmark_mlx.utils.io import read_json, write_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = ROOT / "artifacts" / "official_split_suite" / "summary.json"
OUTPUT_REPORT = ROOT / "docs" / "reports" / "hallmark_official_splits.md"
OUTPUT_PNG = ROOT / "docs" / "figures" / "hallmark_official_splits.png"
OUTPUT_PDF = ROOT / "docs" / "figures" / "hallmark_official_splits.pdf"
SPLITS = ("dev_public", "test_public", "stress_test")
ALIAS = {
    "hallmark_mlx_bibtex_first_fallback": "hallmark-mlx",
    "bibtexupdater": "BibTeX Updater",
    "harc": "HaRC",
    "doi_only": "DOI only",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    return parser


def _label(name: str) -> str:
    return ALIAS.get(name, name)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(rows: list[dict[str, Any]]) -> str:
    columns = (
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
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        values = [
            _label(str(row["name"])),
            row["source"],
            row["num_entries"],
            row["partial"],
            row.get("f1_hallucination"),
            row.get("detection_rate"),
            row.get("tier_weighted_f1"),
            row.get("false_positive_rate"),
            row.get("notes") or row.get("status_message"),
        ]
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    return "\n".join(lines)


def _plot(summary: dict[str, Any]) -> None:
    plt.rcParams.update(bundles.icml2024(column="full", nrows=3, ncols=1, usetex=False))
    plt.rcParams.update(figsizes.icml2024_full(nrows=3, ncols=1))
    figure, axes = plt.subplots(3, 1)
    figure.set_size_inches(8.0, 12.0)
    metric_specs = (
        ("f1_hallucination", "F1-H", False),
        ("detection_rate", "DR", False),
        ("false_positive_rate", "FPR", True),
    )
    colors = {
        "hallmark_mlx_bibtex_first_fallback": "#2ca02c",
        "bibtexupdater": "#1f77b4",
        "harc": "#ff7f0e",
        "doi_only": "#7f7f7f",
    }
    for ax, (metric, title, invert) in zip(axes, metric_specs, strict=True):
        split_names = []
        centers = []
        x = 0
        for split in SPLITS:
            rows = summary["splits"].get(split, [])
            method_rows = [row for row in rows if row.get(metric) is not None]
            for row in method_rows:
                ax.bar(
                    x,
                    float(row[metric]),
                    color=colors.get(str(row["name"]), "#999999"),
                    width=0.8,
                )
                ax.text(
                    x,
                    float(row[metric]),
                    f"{float(row[metric]):.3f}",
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize=7,
                )
                x += 1
            if method_rows:
                centers.append(x - (len(method_rows) + 1) / 2)
                split_names.append(split)
            x += 1
        ax.set_title(title)
        ax.set_xticks(centers)
        ax.set_xticklabels(split_names)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        if invert:
            ax.invert_yaxis()
    figure.tight_layout(h_pad=1.8)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    summary = read_json(Path(args.summary_path).resolve())
    _plot(summary)

    lines = [
        "# HALLMARK Official Split Report",
        "",
        (
            "This report contains official HALLMARK split results only. Internal model-selection "
            "splits used by Weco are intentionally excluded."
        ),
        "",
        f"- Hidden test available locally: {summary.get('hidden_test_available', False)}",
    ]
    for split in SPLITS:
        rows = list(summary["splits"].get(split, []))
        if not rows:
            continue
        lines.extend(["", f"## {split}", "", _table(rows)])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                "- `stress_test` is useful for robustness, but `FPR` is less meaningful there "
                "than on mixed official splits."
            ),
            (
                "- `test_hidden` is not available in the local upstream checkout, so no local "
                "score is reported here."
            ),
        ]
    )
    write_text(OUTPUT_REPORT, "\n".join(lines) + "\n")
    print(OUTPUT_REPORT)
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
