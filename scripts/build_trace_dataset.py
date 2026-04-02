#!/usr/bin/env python3
"""Convenience wrapper for dataset construction."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    source_path = str(repo_root / "src")
    python_executable = repo_root / ".venv" / "bin" / "python"
    env["PYTHONPATH"] = (
        f"{source_path}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else source_path
    )
    subprocess.run(
        [
            str(python_executable if python_executable.exists() else Path(sys.executable)),
            "-m",
            "hallmark_mlx.cli",
            "build-dataset",
            *sys.argv[1:],
        ],
        check=True,
        env=env,
    )
