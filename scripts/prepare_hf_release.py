#!/usr/bin/env python3
"""Build HF-ready dataset and adapter bundles for hallmark-mlx."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallmark_mlx.release.hf_bundle import HFBundleSpec, prepare_hf_bundle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    ROOT / "artifacts" / "weco" / "hallmark-qwen-train-frontier-iterative" / "61460529b25c"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "hf_release" / "qwen25_1_5b_kept"),
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_RUN_ROOT / "train" / "prepared_dataset"),
    )
    parser.add_argument(
        "--adapter-dir",
        default=str(DEFAULT_RUN_ROOT / "adapter"),
    )
    parser.add_argument(
        "--train-manifest",
        default=str(DEFAULT_RUN_ROOT / "train" / "mlx_lora_manifest.json"),
    )
    parser.add_argument(
        "--search-summary",
        default=str(DEFAULT_RUN_ROOT / "hallmark_dev_search64_gold_traces" / "summary.json"),
    )
    parser.add_argument(
        "--compare-summary",
        default=str(DEFAULT_RUN_ROOT / "hallmark_dev_compare32_gold_traces" / "summary.json"),
    )
    parser.add_argument(
        "--official-dev-row",
        default=str(
            ROOT
            / "artifacts"
            / "official_eval_sharded_fast5_confirm"
            / "dev_public_bibtex_first_fallback"
            / "merged"
            / "row.json"
        ),
    )
    parser.add_argument(
        "--source-traces",
        default=str(ROOT / "examples" / "reviewed_seed_traces_combined.jsonl"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = prepare_hf_bundle(
        HFBundleSpec(
            output_dir=Path(args.output_dir).resolve(),
            dataset_dir=Path(args.dataset_dir).resolve(),
            adapter_dir=Path(args.adapter_dir).resolve(),
            train_manifest_path=Path(args.train_manifest).resolve(),
            compare32_summary_path=Path(args.compare_summary).resolve(),
            search64_summary_path=Path(args.search_summary).resolve(),
            official_dev_row_path=Path(args.official_dev_row).resolve(),
            source_traces_path=Path(args.source_traces).resolve(),
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
