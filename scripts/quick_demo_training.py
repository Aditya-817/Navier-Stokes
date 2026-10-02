#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/quick_demo_training.py
================================
Quick PINN training demo — runs a small model for a few epochs to showcase
the entire pipeline: model → PDE residuals → training → checkpoint.

This script is designed to run in ~2 minutes on CPU and produces:
  - Training loss curve figure
  - A saved checkpoint
  - Console output showing loss convergence

Usage:
    python scripts/quick_demo_training.py
"""
import math
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.fourier_resmlp import FourierResMLP
from src.utils.initial_conditions import TaylorGreenVortex3D
from src.utils.reproducibility import seed_everything
from src.sampling.samplers import uniform_sample, ic_sample
from src.training.loss import compute_loss

# ─── Configuration ────────────────────────────────────────────────────────────
SEED = 42
NU = 0.1
T = 1.0
L = 2 * math.pi
N_COLLOC = 4000        # small for speed
N_IC = 1000
N_EPOCHS = 2000        # ~2 min on CPU
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("results/demo")
OUT.mkdir(parents=True, exist_ok=True)

LOSS_WEIGHTS = {"pde": 1.0, "ic": 10.0, "div": 1.0, "gauge": 1.0}

# ─── Setup ────────────────────────────────────────────────────────────────────
seed_everything(SEED)

print("=" * 62)
print("  NS3D-PINN Framework  —  Quick Training Demo")
print("=" * 62)
print(f"  Device   : {DEVICE}")
print(f"  Epochs   : {N_EPOCHS}")
print(f"  PDE pts  : {N_COLLOC}")
print(f"  IC pts   : {N_IC}")

# ─── Model ────────────────────────────────────────────────────────────────────
model = FourierResMLP(
    fourier_modes=4,        # small for fast demo
    hidden_width=128,       # narrow
    n_layers=2,             # shallow
    activation="tanh",
    use_layernorm=True,
    t_scale=T,
    dtype="float64",
).to(DEVICE)

print(f"  Model    : FourierResMLP  ({model.n_params:,} params)")
print(f"  ν = {NU},  Re ≈ {1.0/NU:.0f}")
print()

# ─── Initial Condition ────────────────────────────────────────────────────────
ic = TaylorGreenVortex3D(V0=1.0, k=1, nu=NU)

# ─── Optimizer ────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

# ─── Sample collocation points ────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def sample_points():
    pde_np = uniform_sample(N_COLLOC, T, L, rng=rng)
    pde_t = torch.tensor(pde_np, device=DEVICE, dtype=torch.float64)

    ic_np = ic_sample(N_IC, L, rng=rng)
    ic_t = torch.tensor(ic_np, device=DEVICE, dtype=torch.float64)

    x_np, y_np, z_np = ic_np[:, 0], ic_np[:, 1], ic_np[:, 2]
    u0 = torch.tensor(ic.u0(x_np, y_np, z_np), device=DEVICE, dtype=torch.float64)
    v0 = torch.tensor(ic.v0(x_np, y_np, z_np), device=DEVICE, dtype=torch.float64)
    w0 = torch.tensor(ic.w0(x_np, y_np, z_np), device=DEVICE, dtype=torch.float64)

    return pde_t, ic_t, u0, v0, w0

pde_pts, ic_pts, u0, v0, w0 = sample_points()

# ─── Training Loop ────────────────────────────────────────────────────────────
history = {"epoch": [], "L_total": [], "L_PDE": [], "L_IC": [], "L_div": [], "lr": []}

print(f"  {'Epoch':>6}  {'L_total':>10}  {'L_PDE':>10}  {'L_IC':>10}  {'L_div':>10}  {'LR':>10}")
print("  " + "─" * 60)

t_start = time.time()
model.train()

for epoch in range(N_EPOCHS):
    # Resample every 500 epochs
    if epoch > 0 and epoch % 500 == 0:
        pde_pts, ic_pts, u0, v0, w0 = sample_points()

    optimizer.zero_grad()
    lc = compute_loss(model, pde_pts, ic_pts, (u0, v0, w0),
                      nu=NU, weights=LOSS_WEIGHTS, device=DEVICE)
    lc.L_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        history["epoch"].append(epoch)
        history["L_total"].append(float(lc.L_total))
        history["L_PDE"].append(float(lc.L_PDE))
        history["L_IC"].append(float(lc.L_IC))
        history["L_div"].append(float(lc.L_div))
        history["lr"].append(scheduler.get_last_lr()[0])

        print(f"  {epoch:6d}  {float(lc.L_total):10.3e}  {float(lc.L_PDE):10.3e}  "
              f"{float(lc.L_IC):10.3e}  {float(lc.L_div):10.3e}  {scheduler.get_last_lr()[0]:10.3e}")

elapsed = time.time() - t_start
print(f"\n  Training complete in {elapsed:.1f}s")
print(f"  Final loss: {history['L_total'][-1]:.3e}")

# ─── Save checkpoint ──────────────────────────────────────────────────────────
ckpt_path = OUT / "quick_demo_checkpoint.pt"
torch.save({
    "model_state": model.state_dict(),
    "epoch": N_EPOCHS,
    "config": model.config_dict(),
    "final_loss": history["L_total"][-1],
}, ckpt_path)
print(f"  Checkpoint saved: {ckpt_path}")

# ─── Training curve figure ────────────────────────────────────────────────────
DARK = "#0d1117"
PANEL = "#161b22"
BLUE = "#58a6ff"
GOLD = "#f0c040"
GREEN = "#3fb950"
RED = "#f85149"
GRID = "#21262d"

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor(DARK)
fig.suptitle("PINN Training Demo — TGV, Re≈10, 2000 epochs",
             color="#e6edf3", fontsize=13, fontweight="bold")

for ax in axes:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    ax.grid(True, color=GRID, lw=0.5, ls="--", alpha=0.6)

epochs = history["epoch"]

# Panel 1: Total loss
axes[0].semilogy(epochs, history["L_total"], color=BLUE, lw=2, label="L_total")
axes[0].semilogy(epochs, history["L_PDE"], color=RED, lw=1.5, alpha=0.7, label="L_PDE")
axes[0].semilogy(epochs, history["L_div"], color=GREEN, lw=1.5, alpha=0.7, label="L_div")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].legend(facecolor=PANEL, edgecolor="#30363d", labelcolor="white", fontsize=8)

# Panel 2: IC loss
axes[1].semilogy(epochs, history["L_IC"], color=GOLD, lw=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("L_IC")
axes[1].set_title("Initial Condition Loss")

# Panel 3: Learning rate
axes[2].semilogy(epochs, history["lr"], color="#bc8cff", lw=2)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("Cosine LR Schedule")

plt.tight_layout()
fig_path = OUT / "training_demo.png"
fig.savefig(fig_path, facecolor=DARK, dpi=150, bbox_inches="tight")
plt.close()

print(f"  Training curve: {fig_path}")
print("=" * 62)
