#!/usr/bin/env python
"""
scripts/run_experiment.py
==========================
Train a PINN for the 3D NS problem on T^3.

Usage
-----
    python scripts/run_experiment.py --config configs/experiment_01_baseline.yaml
    python scripts/run_experiment.py --config configs/experiment_01_baseline.yaml --seed 123
    python scripts/run_experiment.py --config configs/experiment_01_baseline.yaml --resume path/to/ckpt.pt
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.models.factory import make_model
from src.training.trainer import Trainer
from src.utils.initial_conditions import make_ic
from src.utils.reproducibility import seed_everything


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train NS-PINN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu | cuda | cuda:0). Auto-detected if None.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Seed override
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    seed = cfg.get("experiment", {}).get("seed", 42)
    seed_everything(seed)

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    exp_name = cfg.get("experiment", {}).get("name", "unnamed")
    log_dir = Path(cfg.get("logging", {}).get("log_dir", f"results/{exp_name}"))
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== NS-PINN Training: {exp_name} ===")
    print(f"  seed={seed}  device={device}")

    # Model
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["t_scale"] = cfg.get("domain", {}).get("T", 1.0)
    model = make_model(model_cfg)
    print(f"  Model: {model_cfg.get('type')}  params={model.n_params:,}")

    # IC
    ic_cfg = dict(cfg.get("initial_condition", {}))
    ic_type = ic_cfg.pop("type", "taylor_green")
    ic_cfg["nu"] = cfg.get("physics", {}).get("nu", 0.1)
    ic = make_ic(ic_type, **ic_cfg)
    print(f"  IC: {ic.name}  Re≈{ic.Re_char:.2f}")

    # Trainer
    trainer = Trainer(model=model, config=cfg, out_dir=log_dir, device=device)

    # Resume
    if args.resume:
        ep = trainer.load_checkpoint(args.resume)
        print(f"  Resumed from epoch {ep}")

    # Train
    history = trainer.train(ic)

    print(f"\n✓ Training complete. {len(history)} log entries.")
    print(f"  Best loss: {trainer._best_loss:.4e}")
    print(f"  Results in: {log_dir}")


if __name__ == "__main__":
    main()
