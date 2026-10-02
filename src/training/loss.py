"""
src/training/loss.py
=====================
Modular loss components for the NS-PINN.

Loss = λ_PDE * L_PDE  +  λ_IC * L_IC  +  λ_div * L_div  +  λ_gauge * L_gauge

Each component is computed independently and logged separately.
The weighting is handled by LossWeights (see loss_weights.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.physics.ns_residual import NSResiduals, ns_residuals
from src.physics.derivatives import split_xyzt


@dataclass
class LossComponents:
    """All individual loss scalars for one training step."""
    L_PDE: Tensor       # mean momentum residual MSE
    L_div: Tensor       # mean continuity MSE
    L_IC: Tensor        # initial condition MSE
    L_gauge: Tensor     # pressure gauge constraint
    L_total: Tensor     # weighted sum
    ns_residuals: NSResiduals  # raw per-component residuals

    def as_dict(self) -> dict[str, float]:
        return {
            "L_PDE": float(self.L_PDE),
            "L_div": float(self.L_div),
            "L_IC": float(self.L_IC),
            "L_gauge": float(self.L_gauge),
            "L_total": float(self.L_total),
            "R_u_mse": float(self.ns_residuals.R_u.pow(2).mean()),
            "R_v_mse": float(self.ns_residuals.R_v.pow(2).mean()),
            "R_w_mse": float(self.ns_residuals.R_w.pow(2).mean()),
        }


def compute_loss(
    model,
    pde_pts: Tensor,
    ic_pts: Tensor,
    ic_uvw: tuple[Tensor, Tensor, Tensor],
    nu: float,
    weights: dict[str, float],
    device: str | torch.device = "cpu",
) -> LossComponents:
    """Compute all NS-PINN loss components.

    Parameters
    ----------
    model : PINNModel
        The neural network.
    pde_pts : Tensor, shape (N_pde, 4)
        Collocation points for PDE residual (x,y,z,t) ∈ T^3 × (0,T].
    ic_pts : Tensor, shape (N_ic, 4)
        Collocation points for IC (x,y,z, t=0).
    ic_uvw : (u0, v0, w0) each Tensor shape (N_ic,)
        Ground-truth velocity at t=0.
    nu : float
        Kinematic viscosity.
    weights : dict with keys 'pde', 'ic', 'div', 'gauge'
        Loss term weights.
    device : str or torch.device

    Returns
    -------
    LossComponents
    """
    # ------------------------------------------------------------------ #
    # PDE loss: NS residuals at interior collocation points
    # ------------------------------------------------------------------ #
    x, y, z, t = split_xyzt(pde_pts)
    xyzt = torch.stack([x, y, z, t], dim=1)  # (N, 4)

    out = model(xyzt)                  # (N, 4)
    u = out[:, 0]                      # (N,)
    v = out[:, 1]
    w = out[:, 2]
    p = out[:, 3]

    res = ns_residuals(u, v, w, p, x, y, z, t, nu=nu)

    L_PDE = res.momentum_mse()
    L_div = res.continuity_mse()

    # ------------------------------------------------------------------ #
    # IC loss: |u_pred(x,0) - u0(x)|^2
    # ------------------------------------------------------------------ #
    x_ic, y_ic, z_ic, t_ic = split_xyzt(ic_pts)
    xyzt_ic = torch.stack([x_ic, y_ic, z_ic, t_ic], dim=1)  # (N_ic, 4)

    out_ic = model(xyzt_ic)       # (N_ic, 4)
    u_pred_ic = out_ic[:, 0]
    v_pred_ic = out_ic[:, 1]
    w_pred_ic = out_ic[:, 2]

    u0, v0, w0 = ic_uvw
    L_IC = (
        (u_pred_ic - u0).pow(2).mean()
        + (v_pred_ic - v0).pow(2).mean()
        + (w_pred_ic - w0).pow(2).mean()
    ) / 3.0

    # ------------------------------------------------------------------ #
    # Pressure gauge: mean(p) = 0 on PDE collocation points
    # (pressure is defined only up to an additive constant)
    # ------------------------------------------------------------------ #
    L_gauge = p.mean().pow(2)

    # ------------------------------------------------------------------ #
    # Weighted total
    # ------------------------------------------------------------------ #
    L_total = (
        weights.get("pde", 1.0) * L_PDE
        + weights.get("ic", 10.0) * L_IC
        + weights.get("div", 1.0) * L_div
        + weights.get("gauge", 1.0) * L_gauge
    )

    return LossComponents(
        L_PDE=L_PDE,
        L_div=L_div,
        L_IC=L_IC,
        L_gauge=L_gauge,
        L_total=L_total,
        ns_residuals=res,
    )
