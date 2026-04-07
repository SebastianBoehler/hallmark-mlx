#!/usr/bin/env python3
"""Run a submission-style HALLMARK evaluation from local or HF-backed settings."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

from hallmark_mlx.config import load_config
from hallmark_mlx.eval.hf_adapter import download_hf_adapter
from hallmark_mlx.eval.official_compare import evaluate_runner_on_entries
from hallmark_mlx.eval.official_merge import merge_official_eval_shards
from hallmark_mlx.eval.policy_modes import build_policy_runner
from hallmark_mlx.eval.upstream_hallmark import load_entries
from hallmark_mlx.types import ModelBackend
from hallmark_mlx.utils.io import write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_MODEL_REPO = "sebastianboehler/hallmark-mlx-qwen25-1.5b-lora"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--split", default="test_public")
    parser.add_argument("--target", choices=("controller", "hf_policy"), default="controller")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--entry-timeout-seconds", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hf-model-repo", default=DEFAULT_HF_MODEL_REPO)
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def _shard_plan(num_entries: int, shards: int) -> list[tuple[int, int, int]]:
    chunk = ceil(num_entries / shards)
    plan: list[tuple[int, int, int]] = []
    for shard_index in range(shards):
        offset = shard_index * chunk
        if offset >= num_entries:
            break
        plan.append((shard_index, offset, min(chunk, num_entries - offset)))
    return plan


def _target_description(target: str, hf_repo: str | None) -> str:
    if target == "controller":
        return "hallmark-mlx controller with deterministic finalizer"
    return f"hallmark-mlx HF policy eval ({hf_repo})"


def _target_method_name(target: str) -> str:
    if target == "controller":
        return "hallmark_mlx_bibtex_first_fallback"
    return "hallmark_mlx_policy_deterministic_hf"


def _target_mode(target: str, explicit_mode: str | None) -> str:
    if explicit_mode:
        return explicit_mode
    if target == "controller":
        return "bibtex_first_fallback"
    return "policy_deterministic"


def _prepare_config(args: argparse.Namespace) -> tuple[object, dict[str, object]]:
    config = load_config(Path(args.config))
    metadata: dict[str, object] = {"target": args.target}
    if args.target != "hf_policy":
        return config, metadata

    bundle = download_hf_adapter(
        args.hf_model_repo,
        revision=args.hf_revision,
        cache_dir=None if args.hf_cache_dir is None else Path(args.hf_cache_dir),
        force_download=args.force_download,
    )
    config.model.backend = ModelBackend.MLX
    config.model.base_model = bundle.base_model
    config.model.adapter_path = bundle.local_dir
    metadata["hf_model_repo"] = bundle.repo_id
    metadata["hf_model_revision"] = args.hf_revision
    metadata["resolved_base_model"] = bundle.base_model
    metadata["resolved_adapter_dir"] = str(bundle.local_dir)
    return config, metadata


def _run_sharded_eval(
    *,
    config,
    upstream_root: Path,
    split: str,
    mode: str,
    output_dir: Path,
    entry_timeout_seconds: int,
    progress_every: int,
    shards: int,
    offset: int,
    limit: int | None,
    method_name: str,
    description: str,
) -> dict[str, object]:
    entries = list(load_entries(upstream_root, split=split))
    entries = entries[offset:]
    if limit is not None:
        entries = entries[:limit]
    plan = _shard_plan(len(entries), shards)
    shards_root = output_dir / "shards"
    for shard_index, offset, limit in plan:
        shard_entries = entries[offset : offset + limit]
        shard_dir = shards_root / f"shard_{shard_index:02d}"
        runner = build_policy_runner(config, mode)
        evaluate_runner_on_entries(
            upstream_root=upstream_root,
            entries=shard_entries,
            split_name=split,
            runner=runner,
            method_name=method_name,
            output_dir=shard_dir,
            description=description,
            source="executed_here",
            resume=True,
            entry_timeout_seconds=entry_timeout_seconds,
            progress_every=progress_every,
        )
    return merge_official_eval_shards(
        upstream_root=upstream_root,
        split=split,
        shards_root=shards_root,
        output_dir=output_dir / "merged",
        method_name=method_name,
        description=description,
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config, metadata = _prepare_config(args)
    mode = _target_mode(args.target, args.mode)
    method_name = _target_method_name(args.target)
    description = _target_description(args.target, args.hf_model_repo)
    row = _run_sharded_eval(
        config=config,
        upstream_root=Path(args.upstream_root).resolve(),
        split=args.split,
        mode=mode,
        output_dir=output_dir,
        entry_timeout_seconds=args.entry_timeout_seconds,
        progress_every=args.progress_every,
        shards=args.shards,
        offset=args.offset,
        limit=args.limit,
        method_name=method_name,
        description=description,
    )
    write_json(
        output_dir / "run_metadata.json",
        {
            "config": str(Path(args.config).resolve()),
            "upstream_root": str(Path(args.upstream_root).resolve()),
            "split": args.split,
            "target": args.target,
            "mode": mode,
            "method_name": method_name,
            "description": description,
            "entry_timeout_seconds": args.entry_timeout_seconds,
            "progress_every": args.progress_every,
            "shards": args.shards,
            "offset": args.offset,
            "limit": args.limit,
            **metadata,
        },
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
