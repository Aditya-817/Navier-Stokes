#!/usr/bin/env python
"""
scripts/analyze_results.py
============================
Post-processing CLI: load HDF5 snapshots and produce spectral analysis
and vortex diagnostic reports.

Usage
-----
    python scripts/analyze_results.py --h5 results/ref_solution.h5 \\
        --nu 0.1 --out results/analysis/ --plot

    python scripts/analyze_results.py --h5 results/ref.h5 --nu 0.05 \\
        --times 0.0 0.5 1.0 --out results/analysis/

Output files
------------
  analysis/spectral_<t>.json  — spectral_report at each time
  analysis/vortex_<t>.json    — vortex_diagnostics_report at each time
  analysis/energy_spectrum.png — E(k) vs k for all times
  analysis/enstrophy_time.png  — Ω(t) and P(t) time series
  analysis/summary.json        — aggregated metrics
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.analysis.spectral_analysis import spectral_report, kolmogorov_exponent
from src.analysis.vortex_diagnostics import vortex_diagnostics_report
from src.solvers.solver_io import load_snapshots as _load_snapshots_raw


def load_snapshots(h5_path: str) -> list[dict]:
    """Load all snapshots from an HDF5 file produced by solver_io.py."""
    _, snaps = _load_snapshots_raw(h5_path)
    return snaps


def run_analysis(
    h5_path: str,
    nu: float,
    out_dir: str,
    times: list[float] | None = None,
    plot: bool = True,
    L: float = 2 * math.pi,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading snapshots from {h5_path} ...")
    snaps = load_snapshots(h5_path)
    print(f"  Found {len(snaps)} snapshots")

    if times is not None:
        # Filter to requested times (nearest match)
        snap_times = np.array([s["t"] for s in snaps])
        selected = []
        for t_req in times:
            idx = int(np.argmin(np.abs(snap_times - t_req)))
            selected.append(snaps[idx])
        snaps = selected

    spectral_reports = []
    vortex_reports = []

    for snap in snaps:
        t = snap["t"]
        u, v, w = snap["u"], snap["v"], snap["w"]
        print(f"  Analysing t={t:.4f} ...")

        s_rep = spectral_report(u, v, w, nu=nu, L=L, t=t)
        v_rep = vortex_diagnostics_report(u, v, w, nu=nu, L=L, t=t)
        spectral_reports.append(s_rep)
        vortex_reports.append(v_rep)

        # Save individual JSON reports
        with open(out_dir / f"spectral_t{t:.4f}.json", "w") as f:
            json.dump({k: v if not isinstance(v, list) else v
                       for k, v in s_rep.items()}, f, indent=2)
        with open(out_dir / f"vortex_t{t:.4f}.json", "w") as f:
            json.dump(v_rep, f, indent=2)

    # Summary
    summary = {
        "h5_file": h5_path,
        "nu": nu,
        "n_snapshots": len(snaps),
        "E_initial": spectral_reports[0]["E_total"] if spectral_reports else None,
        "E_final": spectral_reports[-1]["E_total"] if spectral_reports else None,
        "epsilon_initial": spectral_reports[0]["epsilon"] if spectral_reports else None,
        "alpha_slopes": [r["alpha"] for r in spectral_reports],
        "enstrophy_time": [(r["t"], r["enstrophy"]) for r in vortex_reports],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary → {out_dir / 'summary.json'}")

    if not plot or not spectral_reports:
        return

    # ------------------------------------------------------------------ #
    # Figure 1: Energy spectra E(k) for all times
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Spectral Analysis — ν={nu}", fontsize=13)

    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(spectral_reports)))
    for rep, color in zip(spectral_reports, cmap):
        k = np.array(rep["k_bins"])
        E = np.array(rep["E_k"])
        ax.loglog(k, E, color=color, lw=1.2, alpha=0.8, label=f"t={rep['t']:.2f}")

    # Kolmogorov -5/3 reference line
    k_ref = np.array([2.0, max(k[-1] / 2, 5.0)])
    E_ref = spectral_reports[0]["E_k"][1] * (k_ref / 2.0) ** (-5.0 / 3.0)
    ax.loglog(k_ref, E_ref, "k--", lw=1, alpha=0.5, label="k⁻⁵/³")

    ax.set_xlabel("Wavenumber k"); ax.set_ylabel("E(k)")
    ax.set_title("Energy Spectrum")
    ax.legend(fontsize=7, ncol=2)

    # Figure 2: E(t), Ω(t), P(t) time series
    ax2 = axes[1]
    t_arr = np.array([r["t"] for r in spectral_reports])
    E_arr = np.array([r["E_total"] for r in spectral_reports])
    ens_arr = np.array([r["enstrophy"] for r in vortex_reports])
    pal_arr = np.array([r["palinstrophy"] for r in vortex_reports])

    ax2.plot(t_arr, E_arr / (E_arr[0] + 1e-30), "b-", label="E(t)/E₀")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t_arr, ens_arr, "r--", label="Enstrophy Ω")
    ax2_twin.plot(t_arr, pal_arr, "g:", label="Palinstrophy P")
    ax2.set_xlabel("t"); ax2.set_ylabel("E(t)/E₀", color="b")
    ax2_twin.set_ylabel("Ω, P", color="r")
    ax2.set_title("Time Evolution")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "energy_spectrum.png", dpi=150)
    plt.close()
    print(f"✓ Plots → {out_dir / 'energy_spectrum.png'}")


def main():
    parser = argparse.ArgumentParser(description="Post-process NS snapshots: spectra + vortex diagnostics")
    parser.add_argument("--h5", required=True, help="Path to HDF5 snapshot file")
    parser.add_argument("--nu", type=float, required=True, help="Kinematic viscosity")
    parser.add_argument("--out", default="results/analysis", help="Output directory")
    parser.add_argument("--times", nargs="*", type=float, default=None,
                        help="Subset of times to analyse (default: all)")
    parser.add_argument("--plot", action="store_true", help="Generate figures")
    parser.add_argument("--L", type=float, default=2 * math.pi, help="Domain length")
    args = parser.parse_args()

    run_analysis(
        h5_path=args.h5,
        nu=args.nu,
        out_dir=args.out,
        times=args.times,
        plot=args.plot,
        L=args.L,
    )


if __name__ == "__main__":
    main()
