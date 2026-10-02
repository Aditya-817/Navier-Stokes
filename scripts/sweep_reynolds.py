#!/usr/bin/env python
"""
scripts/sweep_reynolds.py
===========================
Batch Reynolds-number scaling sweep.

Runs the reference solver for a range of ν values, collects:
  - Energy decay rate E(t) / E(0)
  - Enstrophy peak time and magnitude
  - Palinstrophy maximum

Saves a summary CSV and a comparative figure.

Usage
-----
    python scripts/sweep_reynolds.py --nus 0.1 0.05 0.02 0.01 0.005 --T 1.0 --N 64
    python scripts/sweep_reynolds.py  # uses defaults
"""
import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.utils.initial_conditions import TaylorGreenVortex3D
from src.utils.domain import cell_centred_grid, wavenumber_grid
from src.solvers.spectral_solver import SpectralNSSolver
from src.solvers.solver_io import save_snapshots
from src.utils.reproducibility import seed_everything


def _enstrophy(u_hat, v_hat, w_hat, Kx, Ky, Kz, N):
    """Enstrophy Ω = ½ ∫ |ω|² dV from spectral velocity."""
    # Vorticity in spectral space
    wx_hat = 1j * Ky * w_hat - 1j * Kz * v_hat
    wy_hat = 1j * Kz * u_hat - 1j * Kx * w_hat
    wz_hat = 1j * Kx * v_hat - 1j * Ky * u_hat
    # Parseval: ∫|ω|² = (1/N^3) Σ |ω̂|²
    return float(0.5 * np.sum(np.abs(wx_hat)**2 + np.abs(wy_hat)**2 + np.abs(wz_hat)**2) / N**3)


def _palinstrophy(wx_hat, wy_hat, wz_hat, Kx, Ky, Kz, N):
    """Palinstrophy P = ½ ∫ |∇ω|² dV (spectral)."""
    K2 = Kx**2 + Ky**2 + Kz**2
    return float(0.5 * np.sum(K2 * (np.abs(wx_hat)**2 + np.abs(wy_hat)**2 + np.abs(wz_hat)**2)) / N**3)


def run_sweep(nus, T, N, dt_factor, out_dir, seed=42):
    seed_everything(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L = 2 * math.pi
    X, Y, Z = cell_centred_grid(N, L=L)
    Kx, Ky, Kz = wavenumber_grid(N)

    summary_rows = []
    # Store full time traces for comparative plots
    all_traces: dict[float, dict] = {}  # nu -> {t, E, ens, pal}

    for nu in nus:
        Re = 1.0 / (nu * 1)  # Re based on V0=1, k=1 length scale
        dt = dt_factor * nu   # scale dt with nu for stability
        dt = max(dt, 1e-4)    # floor

        print(f"\n─── ν={nu:.4f}  Re≈{Re:.1f}  dt={dt:.5f} ───")
        ic = TaylorGreenVortex3D(V0=1.0, k=1, nu=nu)
        solver = SpectralNSSolver(N=N, nu=nu, dt=dt, dealias=True, L=L)
        solver.set_initial_condition(ic.u0(X, Y, Z), ic.v0(X, Y, Z), ic.w0(X, Y, Z))

        E0 = solver.kinetic_energy()
        times = np.arange(0, T + dt / 2, max(dt * 10, T / 50))
        snapshot_times = list(times)

        snaps = solver.run_to(T, snapshot_times=snapshot_times, verbose=False)

        # Compute diagnostics at each snapshot
        t_arr   = [s["t"] for s in snaps]
        E_arr   = [s["E"] for s in snaps]
        ens_arr = []
        pal_arr = []

        for s in snaps:
            u_h = np.fft.fftn(s["u"])
            v_h = np.fft.fftn(s["v"])
            w_h = np.fft.fftn(s["w"])
            ens = _enstrophy(u_h, v_h, w_h, Kx, Ky, Kz, N)
            wx_h = 1j * Ky * w_h - 1j * Kz * v_h
            wy_h = 1j * Kz * u_h - 1j * Kx * w_h
            wz_h = 1j * Kx * v_h - 1j * Ky * u_h
            pal = _palinstrophy(wx_h, wy_h, wz_h, Kx, Ky, Kz, N)
            ens_arr.append(ens)
            pal_arr.append(pal)

        t_arr_np = np.array(t_arr)
        E_arr_np = np.array(E_arr)
        ens_np   = np.array(ens_arr)
        pal_np   = np.array(pal_arr)

        # Summary statistics
        i_peak_ens = int(np.argmax(ens_np))
        i_peak_pal = int(np.argmax(pal_np))
        row = {
            "nu": nu,
            "Re": Re,
            "E0": E0,
            "E_final": float(E_arr_np[-1]),
            "E_decay_ratio": float(E_arr_np[-1] / (E0 + 1e-30)),
            "enstrophy_peak": float(ens_np[i_peak_ens]),
            "t_enstrophy_peak": float(t_arr_np[i_peak_ens]),
            "palinstrophy_peak": float(pal_np[i_peak_pal]),
            "t_palinstrophy_peak": float(t_arr_np[i_peak_pal]),
        }
        summary_rows.append(row)
        print(f"  E(0)={E0:.4f}  E(T)={row['E_final']:.4f}  "
              f"Ens_peak={row['enstrophy_peak']:.4f} at t={row['t_enstrophy_peak']:.3f}")

        # Store full trace for comparative plot
        all_traces[nu] = {
            "t": t_arr_np, "E": E_arr_np, "E0": E0,
            "ens": ens_np, "pal": pal_np, "Re": Re,
        }

        # Save snapshots
        h5_path = out_dir / f"reference_nu{nu:.4f}.h5"
        save_snapshots(snaps, h5_path, metadata={"nu": nu, "Re": Re, "N": N})

        # Per-nu figure
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"TGV Diagnostics — ν={nu:.4f}, Re≈{Re:.1f}, N={N}", fontsize=12)
        axes[0].plot(t_arr_np, E_arr_np / E0, "b-")
        axes[0].set_xlabel("t"); axes[0].set_ylabel("E(t)/E(0)")
        axes[0].set_title("Kinetic Energy")
        axes[1].plot(t_arr_np, ens_np, "r-")
        axes[1].set_xlabel("t"); axes[1].set_ylabel("Enstrophy")
        axes[1].set_title("Enstrophy")
        axes[2].plot(t_arr_np, pal_np, "g-")
        axes[2].set_xlabel("t"); axes[2].set_ylabel("Palinstrophy")
        axes[2].set_title("Palinstrophy")
        plt.tight_layout()
        plt.savefig(out_dir / f"diagnostics_nu{nu:.4f}.png", dpi=120)
        plt.close()

    # Save CSV summary
    csv_path = out_dir / "sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n✓ Summary saved to {csv_path}")

    # Comparative plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Reynolds Number Sweep — TGV N={N}", fontsize=13)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(nus)))

    for i, nu in enumerate(nus):
        tr = all_traces[nu]
        Re = tr["Re"]
        lbl = f"ν={nu:.4f} (Re≈{Re:.0f})"
        axes[0].plot(tr["t"], tr["E"] / tr["E0"], color=cmap[i], lw=1.8, label=lbl)
        axes[1].plot(tr["t"], tr["ens"], color=cmap[i], lw=1.8, label=lbl)
        axes[2].plot(tr["t"], tr["pal"], color=cmap[i], lw=1.8, label=lbl)

    axes[0].set_xlabel("t"); axes[0].set_ylabel("E(t)/E(0)")
    axes[0].set_title("Energy Decay"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("Enstrophy")
    axes[1].set_title("Enstrophy"); axes[1].legend(fontsize=7)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("Palinstrophy")
    axes[2].set_title("Palinstrophy"); axes[2].legend(fontsize=7)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "energy_sweep.png", dpi=150)
    plt.close()
    print(f"✓ Comparative figure saved to {out_dir / 'energy_sweep.png'}")


def main():
    parser = argparse.ArgumentParser(description="Reynolds sweep for TGV benchmark")
    parser.add_argument("--nus", nargs="+", type=float,
                        default=[0.1, 0.05, 0.02, 0.01, 0.005],
                        help="Kinematic viscosity values to sweep")
    parser.add_argument("--T", type=float, default=1.0, help="Final time")
    parser.add_argument("--N", type=int, default=64, help="Grid resolution")
    parser.add_argument("--dt-factor", type=float, default=0.1,
                        help="dt = dt_factor * nu (auto-scaled)")
    parser.add_argument("--out", default="results/reynolds_sweep",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_sweep(
        nus=args.nus,
        T=args.T,
        N=args.N,
        dt_factor=args.dt_factor,
        out_dir=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
