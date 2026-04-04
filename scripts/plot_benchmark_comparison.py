#!/usr/bin/env python3
"""Plot HALLMARK baseline comparisons with tueplots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles, figsizes

from hallmark_mlx.utils.io import read_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = ROOT / "artifacts" / "hallmark_baseline_compare" / "summary.json"
OUTPUT_DIR = ROOT / "docs" / "figures"
OUTPUT_BASENAME = "hallmark_registry_comparison"

ALIAS = {
    "doi_presence_heuristic": "DOI presence",
    "doi_only": "DOI only",
    "doi_only_no_prescreening": "DOI only\n(no pre)",
    "bibtexupdater": "BibTeX Updater",
    "bibtexupdater_no_prescreening": "BibTeX Updater\n(no pre)",
    "harc": "HaRC",
    "harc_no_prescreening": "HaRC\n(no pre)",
    "verify_citations": "verify-citations",
    "verify_citations_no_prescreening": "verify-citations\n(no pre)",
    "llm_tool_augmented": "LLM tool\naugmented",
    "title_oracle": "Title oracle",
    "always_hallucinated": "Always hall.",
    "always_valid": "Always valid",
    "venue_oracle": "Venue oracle",
    "hallmark_mlx_bibtex_first_fallback": "hallmark-mlx\ncontroller",
}

STATUS_TO_X = {
    "unavailable": 0,
    "not_run": 1,
    "upstream_published_result": 2,
    "upstream_published_predictions": 2,
    "upstream_history_partial": 2,
    "executed_here": 3,
}
STATUS_LABELS = {0: "Unavailable", 1: "Not run", 2: "Published/history", 3: "Executed here"}
STATUS_COLORS = {0: "#d62728", 1: "#7f7f7f", 2: "#ff7f0e", 3: "#2ca02c"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    return parser


def _label(name: str) -> str:
    return ALIAS.get(name, name.replace("_", "\n"))


def _plot_dev_public(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    ranked = sorted(
        [row for row in rows if row.get("f1_hallucination") is not None],
        key=lambda row: float(row["f1_hallucination"]),
        reverse=True,
    )
    labels = [_label(str(row["name"])) for row in ranked]
    y = np.arange(len(ranked))
    height = 0.22
    offsets = np.array([-height, 0.0, height])
    metric_specs = (
        ("f1_hallucination", "F1", "#1f77b4"),
        ("detection_rate", "Detection", "#2ca02c"),
        ("tier_weighted_f1", "Tier-w. F1", "#9467bd"),
    )
    for offset, (key, legend, color) in zip(offsets, metric_specs, strict=True):
        values = [float(row.get(key) or 0.0) for row in ranked]
        bars = ax.barh(y + offset, values, height=height, color=color, label=legend)
        for bar, row in zip(bars, ranked, strict=True):
            if bool(row.get("partial")):
                bar.set_hatch("//")
                bar.set_edgecolor("#444444")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("Official dev_public Performance")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False, ncol=3)


def _plot_registry_status(
    ax: plt.Axes,
    registry_rows: list[dict[str, object]],
    dev_public_rows: list[dict[str, object]],
) -> None:
    status_by_name = {str(row["name"]): str(row["source"]) for row in dev_public_rows}
    y = np.arange(len(registry_rows))
    labels = [_label(str(row["name"])) for row in registry_rows]
    for index, registry_row in enumerate(registry_rows):
        x = STATUS_TO_X.get(status_by_name.get(str(registry_row["name"]), "unavailable"), 0)
        ax.scatter(x, index, s=55, color=STATUS_COLORS[x])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xticks(list(STATUS_LABELS))
    ax.set_xticklabels([STATUS_LABELS[idx] for idx in STATUS_LABELS])
    ax.set_xlim(-0.5, 3.5)
    ax.set_title("Registry Availability")
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def main() -> None:
    args = build_parser().parse_args()
    summary = read_json(args.summary_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(bundles.icml2024(column="full", nrows=2, ncols=1, usetex=False))
    plt.rcParams.update(figsizes.icml2024_full(nrows=2, ncols=1))
    figure, axes = plt.subplots(2, 1)
    figure.set_size_inches(8.0, 9.0)

    _plot_dev_public(axes[0], list(summary["dev_public_rows"]))
    _plot_registry_status(
        axes[1],
        list(summary["registry_rows"]),
        list(summary["dev_public_rows"]),
    )

    figure.tight_layout(h_pad=2.0)

    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
