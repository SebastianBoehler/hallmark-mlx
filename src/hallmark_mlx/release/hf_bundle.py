"""Prepare HF-ready dataset and adapter bundles from local artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from hallmark_mlx.utils.io import ensure_dir, read_json, write_json, write_text


@dataclass(slots=True)
class HFBundleSpec:
    """Paths and metadata for a Hugging Face release bundle."""

    output_dir: Path
    dataset_dir: Path
    adapter_dir: Path
    train_manifest_path: Path
    compare32_summary_path: Path
    search64_summary_path: Path
    official_dev_row_path: Path
    source_traces_path: Path | None = None
    license_path: Path | None = None
    dataset_name: str = "hallmark-mlx-reviewed-policy-traces"
    model_name: str = "hallmark-mlx-qwen25-1.5b-lora"


def _count_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _dataset_card(spec: HFBundleSpec, train_rows: int, valid_rows: int) -> str:
    source_note = ""
    if spec.source_traces_path is not None:
        source_note = (
            f"- Source reviewed traces: `{spec.source_traces_path.name}` with "
            f"{_count_rows(spec.source_traces_path)} full traces.\n"
        )
    return "\n".join(
        [
            "---",
            f"pretty_name: {spec.dataset_name}",
            "license: mit",
            "language:",
            "- en",
            "task_categories:",
            "- text-generation",
            "size_categories:",
            "- n<1K",
            "---",
            "",
            f"# {spec.dataset_name}",
            "",
            "Reviewed citation-verification training traces for `hallmark-mlx`.",
            "",
            "## Contents",
            "",
            f"- `train.jsonl`: {train_rows} supervised examples",
            f"- `valid.jsonl`: {valid_rows} supervised examples",
            source_note.rstrip(),
            "",
            "## Format",
            "",
            "Each row is a prepared supervised training example for MLX LoRA fine-tuning.",
            "The format is the exact snapshot used by the kept Qwen 1.5B run.",
            "",
            "## Upload Note",
            "",
            "Review the dataset card metadata before publishing publicly.",
            "",
        ]
    ).replace("\n\n\n", "\n\n")


def _model_card(
    spec: HFBundleSpec,
    train_manifest: dict[str, object],
    search_summary: dict[str, object],
    compare_summary: dict[str, object],
    official_row: dict[str, object],
) -> str:
    search_metrics = search_summary["metrics"]
    compare_metrics = compare_summary["metrics"]
    return "\n".join(
        [
            "---",
            f"base_model: {train_manifest['base_model']}",
            "license: mit",
            "library_name: mlx-lm",
            "tags:",
            "- lora",
            "- mlx",
            "- citation-verification",
            "- qwen2.5",
            "---",
            "",
            f"# {spec.model_name}",
            "",
            "LoRA adapter for `hallmark-mlx` citation-verification policy training.",
            "",
            "## Base Model",
            "",
            f"- `{train_manifest['base_model']}`",
            "",
            "## Training Snapshot",
            "",
            f"- Iterations: {train_manifest['training']['iters']}",
            f"- Learning rate: {train_manifest['training']['learning_rate']}",
            f"- Num layers: {train_manifest['training']['num_layers']}",
            f"- Max sequence length: {train_manifest['training']['max_seq_length']}",
            f"- Seed: {train_manifest['training']['seed']}",
            "",
            "## Tracked Eval",
            "",
            (
                f"- `search64`: frontier {search_metrics['frontier_score']:.4f}, "
                f"F1-H {search_metrics['budget_1_f1_hallucinated']:.4f}, "
                f"label accuracy {search_metrics['label_accuracy']:.4f}"
            ),
            (
                f"- `compare32`: frontier {compare_metrics['frontier_score']:.4f}, "
                f"F1-H {compare_metrics['budget_1_f1_hallucinated']:.4f}, "
                f"label accuracy {compare_metrics['label_accuracy']:.4f}"
            ),
            "",
            "## Official Controller Reference",
            "",
            (
                f"- `dev_public`: F1-H {official_row['f1_hallucination']:.4f}, "
                f"DR {official_row['detection_rate']:.4f}, "
                f"TW-F1 {official_row['tier_weighted_f1']:.4f}, "
                f"FPR {official_row['false_positive_rate']:.4f}"
            ),
            "",
            "## Files",
            "",
            "- `adapters.safetensors`",
            "- `adapter_config.json`",
            "- `training_manifest.json`",
            "- `tracked_eval_search64.json`",
            "- `tracked_eval_compare32.json`",
            "- `official_dev_public_row.json`",
            "",
            "## Upload Note",
            "",
            "Review the model card metadata before publishing publicly.",
            "",
        ]
    )


def prepare_hf_bundle(spec: HFBundleSpec) -> dict[str, str]:
    """Copy the exact kept run artifacts into HF-ready dataset/model folders."""

    dataset_root = ensure_dir(spec.output_dir / "dataset")
    model_root = ensure_dir(spec.output_dir / "model")

    train_path = spec.dataset_dir / "train.jsonl"
    valid_path = spec.dataset_dir / "valid.jsonl"
    _copy_file(train_path, dataset_root / "train.jsonl")
    _copy_file(valid_path, dataset_root / "valid.jsonl")
    if spec.source_traces_path is not None and spec.source_traces_path.exists():
        _copy_file(spec.source_traces_path, dataset_root / spec.source_traces_path.name)
    if spec.license_path is not None and spec.license_path.exists():
        _copy_file(spec.license_path, dataset_root / "LICENSE")
        _copy_file(spec.license_path, model_root / "LICENSE")

    train_manifest = read_json(spec.train_manifest_path)
    search_summary = read_json(spec.search64_summary_path)
    compare_summary = read_json(spec.compare32_summary_path)
    official_row = read_json(spec.official_dev_row_path)

    write_text(
        dataset_root / "README.md",
        _dataset_card(spec, _count_rows(train_path), _count_rows(valid_path)),
    )
    write_json(
        dataset_root / "metadata.json",
        {
            "dataset_name": spec.dataset_name,
            "train_rows": _count_rows(train_path),
            "valid_rows": _count_rows(valid_path),
            "source_traces_path": str(spec.source_traces_path) if spec.source_traces_path else None,
            "prepared_dataset_dir": str(spec.dataset_dir),
        },
    )

    _copy_file(spec.adapter_dir / "adapters.safetensors", model_root / "adapters.safetensors")
    _copy_file(spec.adapter_dir / "adapter_config.json", model_root / "adapter_config.json")
    write_text(
        model_root / ".gitattributes",
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
    )
    write_json(model_root / "training_manifest.json", train_manifest)
    write_json(model_root / "tracked_eval_search64.json", search_summary)
    write_json(model_root / "tracked_eval_compare32.json", compare_summary)
    write_json(model_root / "official_dev_public_row.json", official_row)
    write_text(
        model_root / "README.md",
        _model_card(spec, train_manifest, search_summary, compare_summary, official_row),
    )

    write_text(
        spec.output_dir / "UPLOAD.md",
        "\n".join(
            [
                "# Hugging Face Upload",
                "",
                "Upload the generated folders with the `hf` CLI:",
                "",
                "```bash",
                "hf repo create <username>/" + spec.dataset_name + " --repo-type dataset",
                f"hf upload <username>/{spec.dataset_name} {dataset_root} . --repo-type dataset",
                "hf repo create <username>/" + spec.model_name + " --repo-type model",
                f"hf upload <username>/{spec.model_name} {model_root} . --repo-type model",
                "```",
                "",
                "Edit the generated README cards if you want to change the public metadata.",
                "",
            ]
        ),
    )

    return {
        "dataset_dir": str(dataset_root),
        "model_dir": str(model_root),
        "upload_guide": str(spec.output_dir / "UPLOAD.md"),
    }
