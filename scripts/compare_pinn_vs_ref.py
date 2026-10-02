#!/usr/bin/env python
"""
scripts/compare_pinn_vs_ref.py
================================
Quantitative comparison of PINN predictions against the reference solver.

Usage
-----
    python scripts/compare_pinn_vs_ref.py \\
        --config configs/experiment_01_baseline.yaml \\
        --checkpoint results/baseline_tgv_low_re/checkpoints/ckpt_best.pt \\
        --reference results/baseline_tgv_low_re/reference_solver.h5

Outputs:
    - comparison_metrics.csv
    - comparison_report.json
    - Matplotlib figure panels (velocity_error.png, energy_trace.png, etc.)
    - PASS / FAIL for Milestone 8 gate
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from src.models.factory import make_model
from src.solvers.solver_io import load_snapshots
from src.evaluation.metrics import compute_snapshot_metrics, milestone8_pass
from src.utils.reproducibility import seed_everything


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare PINN vs reference solver")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to PINN checkpoint (.pt)")
    parser.add_argument("--reference", required=True, help="Path to reference solver HDF5")
    parser.add_argument("--out", default=None, help="Output directory (default: from config)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("experiment", {}).get("seed", 42)
    seed_everything(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = cfg.get("experiment", {}).get("name", "unnamed")
    log_dir = Path(args.out or cfg.get("logging", {}).get("log_dir", f"results/{exp_name}"))
    log_dir.mkdir(parents=True, exist_ok=True)

    nu = cfg.get("physics", {}).get("nu", 0.1)
    L = cfg.get("domain", {}).get("L", 2 * math.pi)

    print(f"=== PINN vs Reference Comparison: {exp_name} ===")

    # Load model
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["t_scale"] = cfg.get("domain", {}).get("T", 1.0)
    model = make_model(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded model: {model_cfg.get('type')}  params={model.n_params:,}")

    # Load reference snapshots
    times, snapshots = load_snapshots(args.reference)
    print(f"  Loaded {len(snapshots)} reference snapshots")

    # Compute metrics at each snapshot
    all_metrics = []
    for snap in snapshots:
        m = compute_snapshot_metrics(model, snap, device=device, L=L, nu=nu)
        all_metrics.append(m)
        print(
            f"  t={m.t:.3f}  rel_L2={m.rel_l2_uvw:.3e}  "
            f"div_Linf={m.div_linf:.3e}  E_rel={m.rel_E_err:.3e}"
        )

    # Save CSV
    csv_path = log_dir / "comparison_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].as_dict().keys()))
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m.as_dict())
    print(f"\n  Metrics saved to {csv_path}")

    # Milestone 8 gate
    gate = milestone8_pass(all_metrics)
    status = "PASSED ✓" if gate["passed"] else "FAILED ✗"
    print(f"\n=== Milestone 8 Gate: {status} ===")
    for k, v in gate.items():
        if k != "passed":
            print(f"    {k}: {v}")

    # Save report
    report = {"experiment": exp_name, "milestone_8": gate,
               "snapshots": [m.as_dict() for m in all_metrics]}
    with open(log_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Plots
    ts = [m.t for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"PINN vs Reference — {exp_name}", fontsize=13)

    axes[0].semilogy(ts, [m.rel_l2_uvw for m in all_metrics], "b-o", ms=4, label="rel L2 uvw")
    axes[0].axhline(0.05, color="r", linestyle="--", label="gate (5%)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("Relative L2 error")
    axes[0].set_title("Velocity Error"); axes[0].legend()

    axes[1].semilogy(ts, [m.div_linf for m in all_metrics], "g-s", ms=4, label="div Linf")
    axes[1].axhline(1e-3, color="r", linestyle="--", label="gate (1e-3)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("|div u|_inf")
    axes[1].set_title("Divergence Error"); axes[1].legend()

    axes[2].plot(ts, [m.E_pred for m in all_metrics], "b-o", ms=4, label="PINN")
    axes[2].plot(ts, [m.E_ref  for m in all_metrics], "k--s", ms=4, label="Reference")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("E(t)")
    axes[2].set_title("Kinetic Energy"); axes[2].legend()

    plt.tight_layout()
    fig_path = log_dir / "comparison_panels.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved to {fig_path}")

    sys.exit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
