"""
src/models/divfree_net.py
==========================
Divergence-free PINN architecture via the curl-of-potential ansatz.

Motivation
-----------
For the incompressible NS equation, enforcing ∇·u = 0 by construction
eliminates the continuity loss term entirely and guarantees the divergence
constraint is satisfied to machine precision regardless of training.

Construction
------------
Given a neural network potential A(x,y,z,t) = [Aₓ, Aᵧ, A_z], define:

    u = ∇ × A

This velocity field is automatically divergence-free since ∇·(∇×A) = 0
for any smooth A.

We parameterise A using a FourierResMLP (structural periodicity preserved).
The output has 3+1 = 4 components: [Aₓ, Aᵧ, A_z, p] where p is pressure.
The velocity (u, v, w) = curl(A) is computed by AD at evaluation time.

Trade-offs
----------
+ ∇·u = 0 exactly by construction (no divergence loss needed)
+ Constraint satisfaction is independent of the network capacity
- Each evaluation requires 3 additional AD passes (to compute the curl)
- The mapping from A to u is non-injective; different A fields can give
  the same u (gauge freedom of the vector potential)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import PINNModel
from src.models.fourier_resmlp import FourierResMLP
from src.physics.derivatives import partial


class DivFreeNet(PINNModel):
    """Curl-of-potential divergence-free PINN for 3D periodic NS.

    The network predicts a vector potential A = (Ax, Ay, Az) and pressure p.
    The velocity is u = ∇×A, computed via AD.

    Parameters
    ----------
    fourier_modes : int — max wavenumber K for structural embedding
    hidden_width : int
    n_layers : int
    activation : str
    dtype : str
    L : float — domain length (default 2π)
    t_scale : float
    """

    def __init__(
        self,
        fourier_modes: int = 8,
        hidden_width: int = 256,
        n_layers: int = 3,
        activation: str = "tanh",
        dtype: str = "float64",
        L: float = 2 * math.pi,
        t_scale: float = 1.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.fourier_modes = fourier_modes
        self.hidden_width = hidden_width
        self.n_layers = n_layers
        self._dtype = torch.float64 if dtype == "float64" else torch.float32
        self.L = L
        self.t_scale = t_scale

        # Potential network: predicts (Ax, Ay, Az, p) — 4 outputs
        self._potential_net = FourierResMLP(
            fourier_modes=fourier_modes,
            hidden_width=hidden_width,
            n_layers=n_layers,
            activation=activation,
            use_layernorm=use_layernorm,
            dtype=dtype,
        )

    def forward_potential(self, pts: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict the vector potential and pressure.

        Returns Ax, Ay, Az, p each of shape (N,).
        """
        out = self._potential_net(pts)  # (N, 4)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward(self, pts: Tensor) -> Tensor:
        """Compute divergence-free velocity (u, v, w) via curl(A), plus pressure p.

        Parameters
        ----------
        pts : Tensor, shape (N, 4) — [x, y, z, t]

        Returns
        -------
        Tensor, shape (N, 4) — [u, v, w, p]
        """
        # Need grad for AD curl computation
        x = pts[:, 0].detach().requires_grad_(True)
        y = pts[:, 1].detach().requires_grad_(True)
        z = pts[:, 2].detach().requires_grad_(True)
        t = pts[:, 3].detach().requires_grad_(True)

        xyzt = torch.stack([x, y, z, t], dim=1)
        Ax, Ay, Az, p = self.forward_potential(xyzt)

        # u = ∇×A = (∂Az/∂y - ∂Ay/∂z, ∂Ax/∂z - ∂Az/∂x, ∂Ay/∂x - ∂Ax/∂y)
        dAz_dy = partial(Az, y, create_graph=True)
        dAy_dz = partial(Ay, z, create_graph=True)
        dAx_dz = partial(Ax, z, create_graph=True)
        dAz_dx = partial(Az, x, create_graph=True)
        dAy_dx = partial(Ay, x, create_graph=True)
        dAx_dy = partial(Ax, y, create_graph=True)

        u = dAz_dy - dAy_dz
        v = dAx_dz - dAz_dx
        w = dAy_dx - dAx_dy

        return torch.stack([u, v, w, p], dim=1)

    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return {
            "type": "divfree_net",
            "fourier_modes": self.fourier_modes,
            "hidden_width": self.hidden_width,
            "n_layers": self.n_layers,
        }
