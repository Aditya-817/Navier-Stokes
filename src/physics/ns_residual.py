"""
src/physics/ns_residual.py
===========================
3D incompressible Navier–Stokes residuals on T^3.

The residuals are computed via automatic differentiation and are
completely independent of the neural network architecture.

Equations (nondimensional)
--------------------------
Momentum:
    R_u = u_t + u u_x + v u_y + w u_z + p_x - ν Δu
    R_v = v_t + u v_x + v v_y + w v_z + p_y - ν Δv
    R_w = w_t + u w_x + v w_y + w w_z + p_z - ν Δw

Continuity:
    R_c = u_x + v_y + w_z

All spatial derivatives are with respect to (x, y, z).
All time derivatives are with respect to t.

Usage
-----
    from src.physics.ns_residual import ns_residuals
    from src.physics.derivatives import split_xyzt

    pts = torch.rand(N, 4, requires_grad=False)  # (x,y,z,t)
    x, y, z, t = split_xyzt(pts)
    xyzt = torch.cat([x, y, z, t], dim=1)        # reconnected for model

    u, v, w, p = model(xyzt).unbind(dim=1)       # each (N,)
    residuals = ns_residuals(u, v, w, p, x, y, z, t, nu=0.1)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.physics.derivatives import (
    partial,
    partial2,
    laplacian_scalar,
)


@dataclass
class NSResiduals:
    """Container for all four Navier–Stokes residual tensors.

    Attributes
    ----------
    R_u : Tensor, shape (N,)  — x-momentum residual
    R_v : Tensor, shape (N,)  — y-momentum residual
    R_w : Tensor, shape (N,)  — z-momentum residual
    R_c : Tensor, shape (N,)  — continuity (divergence) residual
    """

    R_u: Tensor
    R_v: Tensor
    R_w: Tensor
    R_c: Tensor

    def momentum_mse(self) -> Tensor:
        """Mean squared momentum residual (sum of all three components)."""
        return (
            self.R_u.pow(2).mean()
            + self.R_v.pow(2).mean()
            + self.R_w.pow(2).mean()
        ) / 3.0

    def continuity_mse(self) -> Tensor:
        """Mean squared divergence residual."""
        return self.R_c.pow(2).mean()

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "R_u": self.R_u,
            "R_v": self.R_v,
            "R_w": self.R_w,
            "R_c": self.R_c,
        }

    def mse_dict(self) -> dict[str, Tensor]:
        return {k: v.pow(2).mean() for k, v in self.as_dict().items()}


def ns_residuals(
    u: Tensor,
    v: Tensor,
    w: Tensor,
    p: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    t: Tensor,
    nu: float,
) -> NSResiduals:
    """Compute all four NS residuals at a set of collocation points.

    Parameters
    ----------
    u, v, w : Tensor, shape (N, 1) or (N,)
        Velocity components predicted by the model.
    p : Tensor, shape (N, 1) or (N,)
        Pressure predicted by the model.
    x, y, z, t : Tensor, shape (N, 1), requires_grad=True
        Coordinate tensors (must be leaf variables with grad enabled).
    nu : float
        Kinematic viscosity ν.

    Returns
    -------
    NSResiduals dataclass with tensors R_u, R_v, R_w, R_c.

    Notes
    -----
    All returned tensors retain the computation graph (create_graph=True)
    so that second-order optimizers can compute meta-gradients through
    the residuals.
    """
    # Flatten to 1D if necessary
    u = u.reshape(-1)
    v = v.reshape(-1)
    w = w.reshape(-1)
    p = p.reshape(-1)

    # ------------------------------------------------------------------
    # First-order spatial partials  (used in both momentum and continuity)
    # ------------------------------------------------------------------
    u_x = partial(u, x)
    u_y = partial(u, y)
    u_z = partial(u, z)
    u_t = partial(u, t)

    v_x = partial(v, x)
    v_y = partial(v, y)
    v_z = partial(v, z)
    v_t = partial(v, t)

    w_x = partial(w, x)
    w_y = partial(w, y)
    w_z = partial(w, z)
    w_t = partial(w, t)

    p_x = partial(p, x)
    p_y = partial(p, y)
    p_z = partial(p, z)

    # ------------------------------------------------------------------
    # Laplacians (viscous terms)
    # Each Laplacian requires 3 second-order AD passes
    # ------------------------------------------------------------------
    lap_u = laplacian_scalar(u, [x, y, z])
    lap_v = laplacian_scalar(v, [x, y, z])
    lap_w = laplacian_scalar(w, [x, y, z])

    # ------------------------------------------------------------------
    # Momentum residuals
    # R_i = ∂ᵢu/∂t + (u·∇)u_i + ∂ᵢp - ν Δu_i
    # ------------------------------------------------------------------
    u_sq = u.squeeze(-1)
    v_sq = v.squeeze(-1)
    w_sq = w.squeeze(-1)

    R_u = u_t + u_sq * u_x + v_sq * u_y + w_sq * u_z + p_x - nu * lap_u
    R_v = v_t + u_sq * v_x + v_sq * v_y + w_sq * v_z + p_y - nu * lap_v
    R_w = w_t + u_sq * w_x + v_sq * w_y + w_sq * w_z + p_z - nu * lap_w

    # ------------------------------------------------------------------
    # Continuity
    # ------------------------------------------------------------------
    R_c = u_x + v_y + w_z

    return NSResiduals(R_u=R_u, R_v=R_v, R_w=R_w, R_c=R_c)
