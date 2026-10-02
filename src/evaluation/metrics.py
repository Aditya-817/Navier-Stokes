"""
src/evaluation/metrics.py
==========================
Quantitative metrics for comparing PINN predictions against reference solutions.

Metrics computed per snapshot:
  - Relative L2 and Linf velocity error
  - Relative L2 pressure error (after mean subtraction)
  - Divergence error: L2 and Linf of ∇·u_PINN
  - Per-component PDE residual MSE
  - Vorticity L2 and Linf
  - Kinetic energy
  - Enstrophy
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class SnapshotMetrics:
    """All metrics for a single time snapshot."""
    t: float

    # Velocity errors vs. reference
    rel_l2_u: float
    rel_l2_v: float
    rel_l2_w: float
    rel_l2_uvw: float     # combined: ||u_pred - u_ref||_2 / ||u_ref||_2
    rel_linf_uvw: float

    # Pressure errors
    rel_l2_p: float

    # Divergence (PINN)
    div_l2: float
    div_linf: float

    # Vorticity
    vort_l2_pred: float
    vort_l2_ref: float

    # Energy and enstrophy
    E_pred: float
    E_ref: float
    rel_E_err: float

    def as_dict(self) -> dict[str, float]:
        return {
            "t": self.t,
            "rel_l2_u": self.rel_l2_u,
            "rel_l2_v": self.rel_l2_v,
            "rel_l2_w": self.rel_l2_w,
            "rel_l2_uvw": self.rel_l2_uvw,
            "rel_linf_uvw": self.rel_linf_uvw,
            "rel_l2_p": self.rel_l2_p,
            "div_l2": self.div_l2,
            "div_linf": self.div_linf,
            "vort_l2_pred": self.vort_l2_pred,
            "vort_l2_ref": self.vort_l2_ref,
            "E_pred": self.E_pred,
            "E_ref": self.E_ref,
            "rel_E_err": self.rel_E_err,
        }


def compute_snapshot_metrics(
    model,
    snapshot: dict,
    device: str | torch.device,
    L: float = 2 * np.pi,
    nu: float = 0.1,
) -> SnapshotMetrics:
    """Compute metrics at a single snapshot time.

    Parameters
    ----------
    model : PINNModel
        Trained PINN model.
    snapshot : dict
        Reference solver snapshot with keys 't', 'u', 'v', 'w', 'p'.
        Arrays of shape (N, N, N).
    device : str or device
    L : float
        Domain side length.
    nu : float
        Kinematic viscosity.
    """
    t_val = snapshot["t"]
    u_ref = snapshot["u"]
    v_ref = snapshot["v"]
    w_ref = snapshot["w"]
    p_ref = snapshot["p"]
    N = u_ref.shape[0]

    # Build evaluation grid (cell-centred)
    x1d = (np.arange(N) + 0.5) * L / N
    Xg, Yg, Zg = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    t_arr = np.full((N**3,), t_val)
    pts_np = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel(), t_arr], axis=-1)

    dtype = model.dtype()
    pts_t = torch.tensor(pts_np, device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        out = model(pts_t).cpu().numpy()  # (N^3, 4)

    u_pred = out[:, 0].reshape(N, N, N)
    v_pred = out[:, 1].reshape(N, N, N)
    w_pred = out[:, 2].reshape(N, N, N)
    p_pred = out[:, 3].reshape(N, N, N)

    # ---- Velocity errors ----
    def rel_l2(pred, ref):
        denom = np.sqrt(np.mean(ref**2)) + 1e-12
        return float(np.sqrt(np.mean((pred - ref)**2)) / denom)

    def rel_linf(pred, ref):
        denom = np.max(np.abs(ref)) + 1e-12
        return float(np.max(np.abs(pred - ref)) / denom)

    rl2_u = rel_l2(u_pred, u_ref)
    rl2_v = rel_l2(v_pred, v_ref)
    rl2_w = rel_l2(w_pred, w_ref)

    # Combined relative L2
    uvw_pred = np.stack([u_pred, v_pred, w_pred], axis=-1)
    uvw_ref  = np.stack([u_ref, v_ref, w_ref], axis=-1)
    rl2_uvw  = float(np.sqrt(np.mean((uvw_pred - uvw_ref)**2))
                     / (np.sqrt(np.mean(uvw_ref**2)) + 1e-12))
    rlinf_uvw = float(np.max(np.abs(uvw_pred - uvw_ref))
                      / (np.max(np.abs(uvw_ref)) + 1e-12))

    # ---- Pressure error (remove mean) ----
    p_pred_zm = p_pred - p_pred.mean()
    p_ref_zm  = p_ref  - p_ref.mean()
    rl2_p = rel_l2(p_pred_zm, p_ref_zm)

    # ---- Divergence of PINN prediction ----
    dx = L / N
    du_dx = (np.roll(u_pred, -1, 0) - np.roll(u_pred, 1, 0)) / (2 * dx)
    dv_dy = (np.roll(v_pred, -1, 1) - np.roll(v_pred, 1, 1)) / (2 * dx)
    dw_dz = (np.roll(w_pred, -1, 2) - np.roll(w_pred, 1, 2)) / (2 * dx)
    div = du_dx + dv_dy + dw_dz
    div_l2   = float(np.sqrt(np.mean(div**2)))
    div_linf = float(np.max(np.abs(div)))

    # ---- Vorticity ----
    from src.physics.vorticity import vorticity_spectral, energy_spectrum
    from src.utils.domain import wavenumber_grid

    def _vort_l2(uu, vv, ww):
        kx, ky, kz = wavenumber_grid(N)
        u_hat = np.fft.fftn(uu)
        v_hat = np.fft.fftn(vv)
        w_hat = np.fft.fftn(ww)
        ox, oy, oz = vorticity_spectral(u_hat, v_hat, w_hat, kx, ky, kz)
        ox_p = np.real(np.fft.ifftn(ox))
        oy_p = np.real(np.fft.ifftn(oy))
        oz_p = np.real(np.fft.ifftn(oz))
        return float(np.sqrt(np.mean(ox_p**2 + oy_p**2 + oz_p**2)))

    vl2_pred = _vort_l2(u_pred, v_pred, w_pred)
    vl2_ref  = _vort_l2(u_ref,  v_ref,  w_ref)

    # ---- Energy ----
    from src.physics.vorticity import kinetic_energy_grid
    E_pred = kinetic_energy_grid(u_pred, v_pred, w_pred, L)
    E_ref  = kinetic_energy_grid(u_ref,  v_ref,  w_ref, L)
    rel_E  = abs(E_pred - E_ref) / (abs(E_ref) + 1e-12)

    return SnapshotMetrics(
        t=t_val,
        rel_l2_u=rl2_u, rel_l2_v=rl2_v, rel_l2_w=rl2_w,
        rel_l2_uvw=rl2_uvw, rel_linf_uvw=rlinf_uvw,
        rel_l2_p=rl2_p,
        div_l2=div_l2, div_linf=div_linf,
        vort_l2_pred=vl2_pred, vort_l2_ref=vl2_ref,
        E_pred=E_pred, E_ref=E_ref, rel_E_err=float(rel_E),
    )


def milestone8_pass(metrics: list[SnapshotMetrics]) -> dict:
    """Check whether the Milestone 8 quantitative gate is cleared.

    Success criteria:
      - max(rel_l2_uvw) < 0.05       (5% relative L2 velocity error)
      - max(div_linf)   < 1e-3       (Linf divergence)
      - max(rel_E_err)  < 0.02       (2% energy error)

    Returns
    -------
    dict with 'passed' (bool) and per-criterion results.
    """
    max_rel_l2 = max(m.rel_l2_uvw for m in metrics)
    max_div    = max(m.div_linf for m in metrics)
    max_E_err  = max(m.rel_E_err for m in metrics)

    return {
        "passed": (max_rel_l2 < 0.05 and max_div < 1e-3 and max_E_err < 0.02),
        "max_rel_l2_uvw": max_rel_l2,
        "threshold_rel_l2_uvw": 0.05,
        "max_div_linf": max_div,
        "threshold_div_linf": 1e-3,
        "max_rel_E_err": max_E_err,
        "threshold_rel_E_err": 0.02,
        "n_snapshots": len(metrics),
    }
