"""
src/utils/reproducibility.py
=============================
Seed management, hardware/software snapshot for experiment reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Sets: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all devices),
    and optionally makes CUDA deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NOTE: full CUDA determinism may reduce performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def hardware_info() -> dict:
    """Collect hardware and environment info for reproducibility snapshot."""
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    else:
        info["gpu_count"] = 0
        info["gpu_names"] = []

    return info


def save_reproducibility_snapshot(
    out_dir: str | Path,
    config: dict,
    seed: int,
) -> None:
    """Save a reproducibility snapshot alongside an experiment.

    Creates:
        {out_dir}/repro_snapshot.json   — hardware + software + config + seed
        {out_dir}/config_hash.txt       — SHA256 of config JSON (quick equality check)

    Parameters
    ----------
    out_dir : path
        Experiment output directory.
    config : dict
        Serializable experiment configuration.
    seed : int
        Random seed used.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "seed": seed,
        "hardware": hardware_info(),
        "config": config,
    }

    snapshot_path = out_dir / "repro_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)

    config_json = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    with open(out_dir / "config_hash.txt", "w") as f:
        f.write(config_hash + "\n")
