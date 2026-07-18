from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def create_experiment_id(name: str, seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha1(f"{name}-{seed}-{stamp}".encode()).hexdigest()[:8]
    return f"{name}-{stamp}-{digest}".replace(" ", "-")


def deterministic_key(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def base_metadata(experiment_id: str, start_time: datetime, cwd: Path) -> dict[str, Any]:
    try:
        import flwr

        flower_version = flwr.__version__
    except Exception:
        flower_version = "not-importable"
    return {
        "experiment_id": experiment_id,
        "start_time": start_time.isoformat(),
        "python_version": sys.version,
        "flower_version": flower_version,
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(cwd),
        "status": "running",
    }
