#!/usr/bin/env python3
"""Build tracked Weco search and comparison splits from official HALLMARK data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallmark_mlx.eval.weco_dataset import build_weco_splits
from hallmark_mlx.utils.io import write_json
from hallmark_mlx.utils.jsonl import write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, help="Official HALLMARK JSONL split path.")
    parser.add_argument("--output-dir", default="data/weco")
    parser.add_argument("--source-split", default="dev_public")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--source-repo-url", default="https://github.com/rpatrik96/hallmark")
    parser.add_argument("--source-relpath", default="data/v1.0/dev_public.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    search_traces, compare_traces, manifest = build_weco_splits(
        entries,
        split_name=args.source_split,
    )
    manifest["source_repo_url"] = args.source_repo_url
    manifest["source_relpath"] = args.source_relpath
    manifest["source_commit"] = args.source_commit

    search_path = output_dir / "hallmark_dev_search64_gold_traces.jsonl"
    compare_path = output_dir / "hallmark_dev_compare32_gold_traces.jsonl"
    manifest_path = output_dir / "hallmark_dev_search_manifest.json"

    write_jsonl(search_path, [trace.model_dump(exclude_none=True) for trace in search_traces])
    write_jsonl(compare_path, [trace.model_dump(exclude_none=True) for trace in compare_traces])
    write_json(manifest_path, manifest)

    print(f"search_path: {search_path}")
    print(f"compare_path: {compare_path}")
    print(f"manifest_path: {manifest_path}")
    print(f"search_count: {len(search_traces)}")
    print(f"compare_count: {len(compare_traces)}")


if __name__ == "__main__":
    main()
