#!/usr/bin/env python
"""
scripts/run_reference_solver.py
================================
Run the pseudo-spectral Navier–Stokes reference solver for a given
experiment config and save snapshots to HDF5.

Usage
-----
    python scripts/run_reference_solver.py --config configs/experiment_01_baseline.yaml
"""
import argparse
import math
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.utils.initial_conditions import make_ic
from src.utils.domain import cell_centred_grid
from src.utils.reproducibility import seed_everything
from src.solvers.spectral_solver import SpectralNSSolver
from src.solvers.solver_io import save_snapshots


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run pseudo-spectral NS reference solver")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument("--out", default=None, help="Override output HDF5 path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Seed
    seed = cfg.get("experiment", {}).get("seed", 42)
    seed_everything(seed)

    # Experiment name
    exp_name = cfg.get("experiment", {}).get("name", "unnamed")
    log_dir = Path(cfg.get("logging", {}).get("log_dir", f"results/{exp_name}"))
    log_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out or str(log_dir / "reference_solver.h5")

    # Physics
    nu = cfg.get("physics", {}).get("nu", 0.1)
    L = cfg.get("domain", {}).get("L", 2 * math.pi)
    T = cfg.get("domain", {}).get("T", 1.0)

    # Reference solver settings
    rsol_cfg = cfg.get("reference_solver", {})
    N = rsol_cfg.get("N", 64)
    dt = rsol_cfg.get("dt", 0.001)
    dealias = rsol_cfg.get("dealias", True)
    snapshot_times = rsol_cfg.get("snapshot_times", [])

    print(f"=== Reference Solver: {exp_name} ===")
    print(f"  N={N}, nu={nu}, dt={dt}, T={T}, dealias={dealias}")
    print(f"  Output: {out_path}")

    # Build IC
    ic_cfg = dict(cfg.get("initial_condition", {}))
    ic_type = ic_cfg.pop("type", "taylor_green")
    ic_cfg["nu"] = nu
    ic = make_ic(ic_type, **ic_cfg)
    print(f"  IC: {ic.name}  Re≈{ic.Re_char:.2f}")

    # Initialise solver
    solver = SpectralNSSolver(N=N, nu=nu, dt=dt, dealias=dealias, L=L)
    X, Y, Z = cell_centred_grid(N, L=L)
    u0 = ic.u0(X, Y, Z)
    v0 = ic.v0(X, Y, Z)
    w0 = ic.w0(X, Y, Z)
    solver.set_initial_condition(u0, v0, w0)

    E0 = solver.kinetic_energy()
    div_l2, div_linf = solver.divergence_error()
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Initial divergence: L2={div_l2:.2e}  Linf={div_linf:.2e}")

    # Run
    snapshots = solver.run_to(T, snapshot_times=snapshot_times, verbose=True)

    # Save
    metadata = {
        "experiment_name": exp_name,
        "N": N, "nu": nu, "dt": dt, "T": T, "seed": seed,
        "ic_type": ic_type, "Re_char": ic.Re_char,
    }
    save_snapshots(snapshots, out_path, metadata=metadata)
    print(f"\n✓ Saved {len(snapshots)} snapshots to {out_path}")
    print(f"  Final energy: {snapshots[-1]['E']:.6f}  (initial: {E0:.6f})")


if __name__ == "__main__":
    main()
