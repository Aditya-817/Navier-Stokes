#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/demo_results.py
========================
End-to-end demonstration: run the pseudo-spectral solver on the
Taylor-Green Vortex IC, compute spectral/vortex diagnostics, and
produce a multi-panel results figure.

Run:
    python scripts/demo_results.py
"""
import math
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from src.utils.initial_conditions import TaylorGreenVortex3D
from src.utils.domain import cell_centred_grid
from src.solvers.spectral_solver import SpectralNSSolver
from src.analysis.spectral_analysis import (
    energy_spectrum_3d, dissipation_spectrum, kolmogorov_exponent,
)
from src.analysis.vortex_diagnostics import (
    vorticity_field, q_criterion, enstrophy, palinstrophy,
)

TWO_PI = 2.0 * math.pi
OUT = Path("results/demo")
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Solver parameters
# ─────────────────────────────────────────────────────────────────────────────
N  = 32       # 32^3 = fast demo; use 64 for publication quality
nu = 0.1
dt = 2e-3     # CFL safe for TGV k=1, nu=0.1: dt < 1/(k^2 * Re)
T  = 1.0
snap_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print("=" * 62)
print("  NS3D-PINN Framework  --  Reference Solver Demo")
print("=" * 62)
print(f"  Grid : {N}^3      nu={nu}     dt={dt}     T={T}")
print(f"  Re   : {TWO_PI/nu:.1f}  (characteristic)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Initialise IC + solver, run to T
# ─────────────────────────────────────────────────────────────────────────────
ic     = TaylorGreenVortex3D(nu=nu, V0=1.0, k=1)
X, Y, Z = cell_centred_grid(N, L=TWO_PI)
solver   = SpectralNSSolver(N=N, nu=nu, dt=dt, dealias=True, L=TWO_PI)
solver.set_initial_condition(ic.u0(X, Y, Z), ic.v0(X, Y, Z), ic.w0(X, Y, Z))

E0      = solver.kinetic_energy()
div0, _ = solver.divergence_error()
print(f"  Initial energy     : {E0:.6f}")
print(f"  Initial div (L2)   : {div0:.2e}  (Leray-projected)")
print(f"\n  Running RK4  t=0 -> T={T} ...")

snapshots = solver.run_to(T, snapshot_times=snap_times, verbose=False)
print(f"  Done -- {len(snapshots)} snapshots.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute diagnostics at each snapshot
# ─────────────────────────────────────────────────────────────────────────────
times_list, energies_list = [], []
enst_list, pal_list = [], []
spectra_k_list, spectra_E_list = [], []
diss_k_list, diss_D_list = [], []
qfrac_list = []

for snap in snapshots:
    u, v, w = snap["u"], snap["v"], snap["w"]
    k_bins,   E_k = energy_spectrum_3d(u, v, w)
    k_bins_d, D_k = dissipation_spectrum(u, v, w, nu=nu)

    times_list.append(snap["t"])
    energies_list.append(float(np.sum(E_k)))
    enst_list.append(enstrophy(u, v, w))
    pal_list.append(palinstrophy(u, v, w))
    spectra_k_list.append(k_bins)
    spectra_E_list.append(E_k)
    diss_k_list.append(k_bins_d)
    diss_D_list.append(D_k)
    Q = q_criterion(u, v, w)
    qfrac_list.append(float((Q > 0).mean()))

times        = np.array(times_list)
energies     = np.array(energies_list)
enstrophies  = np.array(enst_list)
palinstrs    = np.array(pal_list)
qfrac        = np.array(qfrac_list)
eps_arr      = np.array([float(D.sum()) for D in diss_D_list])

slope_fit = kolmogorov_exponent(spectra_k_list[-1], spectra_E_list[-1],
                                k_min=3.0, k_max=15.0)
alpha = slope_fit["alpha"]
r2    = slope_fit["r_squared"]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Print results table
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  RESULTS SUMMARY")
print("=" * 62)
print(f"  E(t=0)         = {energies[0]:.6f}")
print(f"  E(t=1)         = {energies[-1]:.6f}")
print(f"  dE/E0          = {(energies[0]-energies[-1])/energies[0]*100:.2f}%  (viscous decay)")
print(f"  Enstrophy(t=0) = {enstrophies[0]:.4f}")
print(f"  Enstrophy peak = {enstrophies.max():.4f}  at t={times[enstrophies.argmax()]:.2f}")
print(f"  Diss. peak     = {eps_arr.max():.4f}")
print(f"  Kolmogorov a   = {alpha:.4f}  (theory: -5/3 = {-5/3:.4f})")
print(f"  R^2 of fit     = {r2:.4f}")
print(f"  Q>0 @ t=0      = {qfrac[0]*100:.1f}%  (vortex core volume)")
print(f"  Q>0 @ t=1      = {qfrac[-1]*100:.1f}%")
print()
print(f"  {'t':>5}  {'E(t)':>9}  {'Omega':>9}  {'eps':>9}  {'Q>0%':>7}")
print("  " + "-" * 46)
for i, t in enumerate(times):
    print(f"  {t:5.2f}  {energies[i]:9.5f}  {enstrophies[i]:9.4f}"
          f"  {eps_arr[i]:9.4f}  {qfrac[i]*100:7.1f}")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 9-panel figure
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "figure.dpi": 150,
})

DARK  = "#0d1117"
PANEL = "#161b22"
BLUE  = "#58a6ff"
GOLD  = "#f0c040"
GREEN = "#3fb950"
RED   = "#f85149"
VIOL  = "#bc8cff"
GRID  = "#21262d"

def style(ax, title=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    ax.grid(True, color=GRID, lw=0.5, ls="--", alpha=0.6)
    if title: ax.set_title(title, pad=8)

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
                         left=0.08, right=0.97, top=0.93, bottom=0.07)

n_snaps   = len(snapshots)
cmap_spec = plt.cm.plasma(np.linspace(0.2, 0.95, n_snaps))

# ── A: Energy Spectrum ────────────────────────────────────────────────────────
ax_A = fig.add_subplot(gs[0, 0])
for i, (k, E) in enumerate(zip(spectra_k_list, spectra_E_list)):
    bold = i in (0, n_snaps//2, n_snaps-1)
    ax_A.loglog(k, E, color=cmap_spec[i], lw=2.0 if bold else 0.8,
                alpha=1.0 if bold else 0.4)
k_ref = np.array([3., 20.])
E_ref = spectra_E_list[0][2] * (k_ref / 3.) ** (-5./3.)
ax_A.loglog(k_ref, E_ref, "w--", lw=1.3, alpha=0.7, label="k^{-5/3}")
ax_A.set_xlabel("Wavenumber k"); ax_A.set_ylabel("E(k)")
ax_A.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor="white", fontsize=9)
style(ax_A, "Energy Spectrum E(k)")
sm = plt.cm.ScalarMappable(cmap="plasma",
     norm=plt.Normalize(vmin=times.min(), vmax=times.max()))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax_A, pad=0.02, fraction=0.046)
cb.set_label("t", color="#c9d1d9")
cb.ax.yaxis.set_tick_params(color="#8b949e")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

# ── B: Dissipation Spectrum ───────────────────────────────────────────────────
ax_B = fig.add_subplot(gs[0, 1])
for i, (k, D) in enumerate(zip(diss_k_list, diss_D_list)):
    bold = i in (0, n_snaps//2, n_snaps-1)
    ax_B.semilogx(k, D, color=cmap_spec[i], lw=2.0 if bold else 0.8,
                  alpha=1.0 if bold else 0.4)
ax_B.set_xlabel("Wavenumber k"); ax_B.set_ylabel("D(k) = 2*nu*k^2*E(k)")
style(ax_B, "Dissipation Spectrum D(k)")

# ── C: Kolmogorov slope fit ───────────────────────────────────────────────────
ax_C = fig.add_subplot(gs[0, 2])
k_fit = slope_fit.get("k_fit", np.array([]))
E_fit = slope_fit.get("E_fit", np.array([]))
ax_C.loglog(spectra_k_list[-1], spectra_E_list[-1], color=BLUE, lw=1.8,
            label="E(k) at t=1")
if len(k_fit):
    ax_C.loglog(k_fit, E_fit, color=RED, lw=2, ls="--",
                label=f"Fit: a={alpha:.3f}  R2={r2:.3f}")
ax_C.axvline(3,  color=GRID, lw=1, ls=":")
ax_C.axvline(15, color=GRID, lw=1, ls=":", label="Fit range [3,15]")
ax_C.set_xlabel("k"); ax_C.set_ylabel("E(k)")
ax_C.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor="white", fontsize=8)
style(ax_C, f"Kolmogorov Slope  a={alpha:.3f}")

# ── D: E(t) + eps(t) ─────────────────────────────────────────────────────────
ax_D = fig.add_subplot(gs[1, 0])
ax_D.plot(times, energies / energies[0], color=BLUE, lw=2, label="E(t)/E0")
ax_D2 = ax_D.twinx()
ax_D2.set_facecolor(PANEL)
ax_D2.plot(times, eps_arr, color=GOLD, lw=2, ls="--", label="eps(t)")
ax_D2.tick_params(colors="#8b949e")
ax_D2.yaxis.label.set_color("#c9d1d9")
ax_D2.set_ylabel("Dissipation rate", color="#c9d1d9")
ax_D.set_xlabel("t"); ax_D.set_ylabel("E(t) / E(0)")
lines = ax_D.get_lines() + ax_D2.get_lines()
ax_D.legend(lines, [l.get_label() for l in lines],
            facecolor=PANEL, edgecolor="#30363d", labelcolor="white", fontsize=8)
style(ax_D, "Energy & Dissipation Rate")

# ── E: Enstrophy + Palinstrophy ───────────────────────────────────────────────
ax_E = fig.add_subplot(gs[1, 1])
ax_E.plot(times, enstrophies, color=GREEN,  lw=2, label="Enstrophy")
ax_E.plot(times, palinstrs,   color=VIOL, lw=2, ls="--", label="Palinstrophy")
ax_E.set_xlabel("t"); ax_E.set_ylabel("Value")
ax_E.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor="white", fontsize=9)
style(ax_E, "Enstrophy & Palinstrophy")

# ── F: Q>0 vortex fraction ───────────────────────────────────────────────────
ax_F = fig.add_subplot(gs[1, 2])
ax_F.plot(times, qfrac * 100, color=RED, lw=2, marker="o",
          ms=4, mfc=DARK, mec=RED)
ax_F.fill_between(times, qfrac * 100, alpha=0.15, color=RED)
ax_F.set_xlabel("t"); ax_F.set_ylabel("Volume fraction (%)")
style(ax_F, "Q>0 Vortex Core Fraction")

# ── G: u-velocity slice at t=0 ───────────────────────────────────────────────
ax_G = fig.add_subplot(gs[2, 0])
u0 = snapshots[0]["u"]
im = ax_G.imshow(u0[:, :, N//2], origin="lower",
                  extent=[0, TWO_PI, 0, TWO_PI], cmap="RdBu_r",
                  norm=TwoSlopeNorm(vcenter=0))
cb2 = plt.colorbar(im, ax=ax_G, fraction=0.046, pad=0.02)
cb2.ax.yaxis.set_tick_params(color="#8b949e")
plt.setp(cb2.ax.yaxis.get_ticklabels(), color="#8b949e")
ax_G.set_xlabel("x"); ax_G.set_ylabel("y")
style(ax_G, "u-velocity  z=pi, t=0")

# ── H: Q-criterion at t=1 ─────────────────────────────────────────────────────
ax_H = fig.add_subplot(gs[2, 1])
u1, v1, w1 = snapshots[-1]["u"], snapshots[-1]["v"], snapshots[-1]["w"]
Q_final = q_criterion(u1, v1, w1)
Q_slice = Q_final[:, :, N//2]
vq = float(np.percentile(np.abs(Q_slice), 98))
im2 = ax_H.imshow(Q_slice, origin="lower",
                   extent=[0, TWO_PI, 0, TWO_PI], cmap="seismic",
                   vmin=-vq, vmax=vq)
cb3 = plt.colorbar(im2, ax=ax_H, fraction=0.046, pad=0.02)
cb3.ax.yaxis.set_tick_params(color="#8b949e")
plt.setp(cb3.ax.yaxis.get_ticklabels(), color="#8b949e")
ax_H.set_xlabel("x"); ax_H.set_ylabel("y")
style(ax_H, "Q-Criterion  z=pi, t=1")

# ── I: Vorticity magnitude at t=1 ────────────────────────────────────────────
ax_I = fig.add_subplot(gs[2, 2])
ox, oy, oz = vorticity_field(u1, v1, w1)
om_mag = np.sqrt(ox**2 + oy**2 + oz**2)
im3 = ax_I.imshow(om_mag[:, :, N//2], origin="lower",
                   extent=[0, TWO_PI, 0, TWO_PI], cmap="inferno")
cb4 = plt.colorbar(im3, ax=ax_I, fraction=0.046, pad=0.02)
cb4.ax.yaxis.set_tick_params(color="#8b949e")
plt.setp(cb4.ax.yaxis.get_ticklabels(), color="#8b949e")
ax_I.set_xlabel("x"); ax_I.set_ylabel("y")
style(ax_I, "|omega| Vorticity  z=pi, t=1")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    f"3D Incompressible Navier-Stokes on T3  |  TGV IC  |  "
    f"N={N}^3  |  nu={nu}  |  Re~{TWO_PI/nu:.0f}",
    color="#e6edf3", fontsize=13, fontweight="bold", y=0.97,
)

out_fig = OUT / "ns3d_pinn_results.png"
fig.savefig(out_fig, facecolor=DARK, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nFigure saved: {out_fig.resolve()}")
print(f"All output  : {OUT.resolve()}/")
