#!/usr/bin/env python3
"""Generate a benchmark comparison figure with tueplots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles, figsizes

ROOT = Path(__file__).resolve().parents[1]
LOCAL_BASELINES_PATH = (
    ROOT / "artifacts/hallmark_benchmark_slice10_qwen15_round8_native/comparison_local_metrics.json"
)
TOOL_ONLY_PATH = (
    ROOT / "artifacts/hallmark_benchmark_slice10_policy_modes_round9/tool_only/metrics.json"
)
POLICY_DETERMINISTIC_PATH = (
    ROOT / "artifacts/hallmark_benchmark_slice10_policy_deterministic_rerun_v2/metrics.json"
)
OUTPUT_DIR = ROOT / "docs/figures"
OUTPUT_BASENAME = "hallmark_baseline_comparison"

OFFICIAL_DEV_PUBLIC_BASELINES = {
    "HaRC": {
        "detection_rate": 0.155,
        "f1_hallucinated": 0.268,
        "tier_weighted_f1": 0.188,
        "false_positive_rate": 0.000,
        "ece": 0.361,
    },
    "bibtex-updater": {
        "detection_rate": 0.124,
        "f1_hallucinated": 0.220,
        "tier_weighted_f1": 0.131,
        "false_positive_rate": 0.000,
        "ece": 0.018,
    },
}


def _read_json(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _local_slice_metrics() -> dict[str, dict[str, float]]:
    local = _read_json(LOCAL_BASELINES_PATH)
    tool_only = _read_json(TOOL_ONLY_PATH)
    policy = _read_json(POLICY_DETERMINISTIC_PATH)
    return {
        "DOI-only": {
            "f1_hallucinated": local["doi_only"]["f1_hallucinated"],
            "detection_rate": local["doi_only"]["detection_rate"],
            "label_accuracy": local["doi_only"]["label_accuracy"],
            "avg_tool_calls": 1.0,
            "budget_1_resolved_rate": local["doi_only"]["budget_1_resolved_rate"],
        },
        "BibTeX Updater": {
            "f1_hallucinated": local["bibtexupdater"]["f1_hallucinated"],
            "detection_rate": local["bibtexupdater"]["detection_rate"],
            "label_accuracy": local["bibtexupdater"]["label_accuracy"],
            "avg_tool_calls": 4.0,
            "budget_1_resolved_rate": local["bibtexupdater"]["budget_1_resolved_rate"],
        },
        "Tool-only": {
            "f1_hallucinated": tool_only["f1_hallucinated"],
            "detection_rate": tool_only["detection_rate"],
            "label_accuracy": tool_only["label_accuracy"],
            "avg_tool_calls": tool_only["avg_tool_calls"],
            "budget_1_resolved_rate": tool_only["budget_1_resolved_rate"],
        },
        "Policy +\nDeterministic": {
            "f1_hallucinated": policy["f1_hallucinated"],
            "detection_rate": policy["detection_rate"],
            "label_accuracy": policy["label_accuracy"],
            "avg_tool_calls": policy["avg_tool_calls"],
            "budget_1_resolved_rate": policy["budget_1_resolved_rate"],
        },
    }


def _grouped_bar(
    ax: plt.Axes,
    methods: list[str],
    metrics: dict[str, list[float]],
    title: str,
) -> None:
    x = np.arange(len(methods))
    width = 0.22
    offsets = np.linspace(-width, width, num=len(metrics))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for index, (label, values) in enumerate(metrics.items()):
        ax.bar(x + offsets[index], values, width=width, label=label, color=colors[index])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _plot_local_slice(ax_quality: plt.Axes, ax_efficiency: plt.Axes) -> None:
    local = _local_slice_metrics()
    methods = list(local)
    _grouped_bar(
        ax_quality,
        methods,
        {
            "F1": [local[name]["f1_hallucinated"] for name in methods],
            "Detection": [local[name]["detection_rate"] for name in methods],
            "Accuracy": [local[name]["label_accuracy"] for name in methods],
        },
        title="Local HALLMARK Slice (10 examples)",
    )
    ax_quality.set_ylabel("Score")
    ax_quality.legend(loc="lower right", frameon=False)

    x = np.arange(len(methods))
    width = 0.35
    ax_efficiency.bar(
        x - (width / 2),
        [local[name]["avg_tool_calls"] for name in methods],
        width=width,
        color="#9467bd",
        label="Avg tool calls",
    )
    ax_efficiency.bar(
        x + (width / 2),
        [local[name]["budget_1_resolved_rate"] for name in methods],
        width=width,
        color="#8c564b",
        label="Resolved within 1 call",
    )
    ax_efficiency.set_xticks(x)
    ax_efficiency.set_xticklabels(methods)
    ax_efficiency.set_title("Local Efficiency")
    ax_efficiency.set_ylabel("Calls / Rate")
    ax_efficiency.grid(axis="y", linestyle=":", alpha=0.4)
    ax_efficiency.legend(loc="upper right", frameon=False)


def _plot_official_context(ax_quality: plt.Axes, ax_risk: plt.Axes) -> None:
    methods = list(OFFICIAL_DEV_PUBLIC_BASELINES)
    _grouped_bar(
        ax_quality,
        methods,
        {
            "F1": [OFFICIAL_DEV_PUBLIC_BASELINES[name]["f1_hallucinated"] for name in methods],
            "Detection": [
                OFFICIAL_DEV_PUBLIC_BASELINES[name]["detection_rate"] for name in methods
            ],
            "Tier-w. F1": [
                OFFICIAL_DEV_PUBLIC_BASELINES[name]["tier_weighted_f1"] for name in methods
            ],
        },
        title="Official HALLMARK dev_public (1,119 entries)",
    )
    ax_quality.legend(loc="upper right", frameon=False)

    x = np.arange(len(methods))
    width = 0.35
    ax_risk.bar(
        x - (width / 2),
        [OFFICIAL_DEV_PUBLIC_BASELINES[name]["false_positive_rate"] for name in methods],
        width=width,
        color="#d62728",
        label="FPR",
    )
    ax_risk.bar(
        x + (width / 2),
        [OFFICIAL_DEV_PUBLIC_BASELINES[name]["ece"] for name in methods],
        width=width,
        color="#7f7f7f",
        label="ECE",
    )
    ax_risk.set_xticks(x)
    ax_risk.set_xticklabels(methods)
    ax_risk.set_title("Official dev_public Risk / Calibration")
    ax_risk.set_ylabel("Lower is better")
    ax_risk.grid(axis="y", linestyle=":", alpha=0.4)
    ax_risk.legend(loc="upper right", frameon=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style = bundles.icml2024(column="full", nrows=2, ncols=2, usetex=False)
    plt.rcParams.update(style)
    plt.rcParams.update(figsizes.icml2024_full(nrows=2, ncols=2))
    plt.rcParams["figure.constrained_layout.use"] = False
    figure, axes = plt.subplots(2, 2)

    _plot_local_slice(axes[0, 0], axes[0, 1])
    _plot_official_context(axes[1, 0], axes[1, 1])

    figure.suptitle("HALLMARK Baselines vs hallmark-mlx", fontsize=11, y=0.98)
    figure.subplots_adjust(top=0.85, hspace=0.42, wspace=0.18)
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
