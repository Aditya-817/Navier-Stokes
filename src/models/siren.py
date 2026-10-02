"""
src/models/siren.py
====================
SIREN — Sinusoidal Representation Network (Sitzmann et al. 2020).

Key properties
--------------
- Uses sin(ω₀ · (Wx + b)) activations throughout.
- First layer is initialised with U[-1/d, 1/d] and subsequent layers
  with U[-√(6/d), √(6/d)] (preserving the distribution through sine).
- High-frequency representation: suitable for fine-scale PDE solutions.
- For periodic boundary conditions on T^3 we use the same structural
  Fourier embedding as FourierResMLP — SIREN handles the hidden layers
  while periodicity is imposed at the input level.

Reference
---------
Sitzmann et al. (2020) "Implicit neural representations with periodic
activation functions."  NeurIPS 2020.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import PINNModel
from src.utils.domain import fourier_mode_grid


class SineLayer(nn.Module):
    """Single SIREN hidden layer: sin(ω₀ * (Wx + b))."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        d = self.linear.in_features
        with torch.no_grad():
            if self.is_first:
                # U[-1/d, 1/d]
                self.linear.weight.uniform_(-1.0 / d, 1.0 / d)
            else:
                # U[-sqrt(6/d), sqrt(6/d)]
                bound = math.sqrt(6.0 / d)
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(PINNModel):
    """SIREN for 3D periodic NS.

    Uses the Fourier structural embedding for 2π-periodicity at the input
    and SIREN layers for the hidden representation.

    Parameters
    ----------
    fourier_modes : int
        Max wavenumber K for the structural embedding.
    hidden_width : int
    n_layers : int — number of hidden SIREN layers (default 4)
    omega_0 : float — frequency multiplier (default 30.0)
    dtype : str — 'float32' or 'float64'
    L : float — domain length (default 2π)
    """

    def __init__(
        self,
        fourier_modes: int = 8,
        hidden_width: int = 256,
        n_layers: int = 4,
        omega_0: float = 30.0,
        dtype: str = "float64",
        L: float = 2 * math.pi,
        t_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.fourier_modes = fourier_modes
        self.hidden_width = hidden_width
        self.n_layers = n_layers
        self.omega_0 = omega_0
        self._dtype = torch.float64 if dtype == "float64" else torch.float32
        self.L = L
        self.t_scale = t_scale

        # Structural Fourier embedding (same as FourierResMLP)
        modes = fourier_mode_grid(fourier_modes)  # (M, 3)
        self.register_buffer("modes", torch.tensor(modes, dtype=self._dtype))
        M = modes.shape[0]  # number of spatial modes
        # Input dim: 2*M spatial features (sin+cos) + 1 temporal
        input_dim = 2 * M + 1

        # SIREN layers
        layers = [SineLayer(input_dim, hidden_width, omega_0=omega_0, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SineLayer(hidden_width, hidden_width, omega_0=omega_0))
        self.net = nn.Sequential(*layers)

        # Output head: linear (no sine at the last layer per Sitzmann et al.)
        self.out = nn.Linear(hidden_width, 4)
        self._init_output()
        self.to(self._dtype)

    def _init_output(self) -> None:
        d = self.out.in_features
        bound = math.sqrt(6.0 / d)
        with torch.no_grad():
            self.out.weight.uniform_(-bound, bound)
            self.out.bias.zero_()

    def forward(self, pts: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        pts : Tensor, shape (N, 4) — [x, y, z, t] in [0, L]^3 × [0, T]

        Returns
        -------
        Tensor, shape (N, 4) — [u, v, w, p]
        """
        x = pts[:, :3]           # (N, 3)
        t = pts[:, 3:4]          # (N, 1)

        # Structural Fourier embedding — guarantees 2π-periodicity
        # phases: (N, M)  = x @ modes^T * 2π/L
        phases = (2.0 * math.pi / self.L) * (x @ self.modes.T)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # (N, 2M)

        # Normalise t to [0, 1]
        t_norm = t / self.t_scale  # (N, 1)

        feat = torch.cat([emb, t_norm], dim=-1)  # (N, 2M+1)
        h = self.net(feat)
        return self.out(h)

    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return {
            "type": "siren",
            "fourier_modes": self.fourier_modes,
            "hidden_width": self.hidden_width,
            "n_layers": self.n_layers,
            "omega_0": self.omega_0,
        }
