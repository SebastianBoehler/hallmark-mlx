"""Resolve Hugging Face adapter bundles for local evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hallmark_mlx.utils.io import read_json


@dataclass(frozen=True)
class HFAdapterBundle:
    """Local snapshot metadata for one HF adapter repo."""

    repo_id: str
    local_dir: Path
    base_model: str


def _read_base_model(local_dir: Path) -> str:
    manifest_path = local_dir / "training_manifest.json"
    if manifest_path.exists():
        payload = read_json(manifest_path)
        base_model = str(payload.get("base_model") or "").strip()
        if base_model:
            return base_model
    adapter_config_path = local_dir / "adapter_config.json"
    if adapter_config_path.exists():
        payload = read_json(adapter_config_path)
        base_model = str(payload.get("model") or "").strip()
        if base_model:
            return base_model
    msg = f"Could not infer base_model from HF adapter bundle at {local_dir}"
    raise ValueError(msg)


def download_hf_adapter(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> HFAdapterBundle:
    """Download a lightweight HF adapter snapshot and infer its base model."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        msg = (
            "huggingface_hub is required for HF adapter evaluation. "
            "Install with `uv sync --extra mlx`."
        )
        raise RuntimeError(msg) from exc

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        allow_patterns=[
            "adapters.safetensors",
            "adapter_config.json",
            "training_manifest.json",
            "README.md",
            "LICENSE",
        ],
        force_download=force_download,
    )
    local_dir = Path(snapshot_path).resolve()
    return HFAdapterBundle(
        repo_id=repo_id,
        local_dir=local_dir,
        base_model=_read_base_model(local_dir),
    )
