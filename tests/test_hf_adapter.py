from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from hallmark_mlx.eval.hf_adapter import download_hf_adapter
from hallmark_mlx.utils.io import write_json


def test_download_hf_adapter_reads_base_model_from_training_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    write_json(snapshot_dir / "training_manifest.json", {"base_model": "Qwen/Test"})

    def fake_snapshot_download(**_: object) -> str:
        return str(snapshot_dir)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    bundle = download_hf_adapter("demo/repo")

    assert bundle.repo_id == "demo/repo"
    assert bundle.local_dir == snapshot_dir.resolve()
    assert bundle.base_model == "Qwen/Test"
