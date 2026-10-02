"""
src/sampling/adaptive.py
==========================
Residual-based adaptive collocation point resampling for PINNs.

Strategy
--------
After every ``resample_interval`` epochs the trainer calls
``AdaptiveSampler.resample(model, ...)``.  This:

1. Evaluates the full NS residual on a large "candidate pool".
2. Builds a probability distribution proportional to the residual
   magnitude (RAR — Residual-based Adaptive Refinement).
3. Draws ``n_pde`` new collocation points, blending with a fixed
   uniform fraction so dense-residual regions don't dominate entirely.

References
----------
- Lu et al. (2021) "DeepXDE: A deep learning library for solving
  differential equations."  SIAM Review 63(1):208–228.
- Wight & Zhao (2020) "Solving Allen–Cahn and Cahn–Hilliard equations
  using the adaptive physics-informed neural networks."
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from src.physics.derivatives import split_xyzt
from src.physics.ns_residual import ns_residuals


class AdaptiveSampler:
    """Residual-Adaptive Refinement (RAR) sampler for 3D periodic NS.

    Parameters
    ----------
    L : float
        Domain side length (default 2π).
    T : float
        Temporal horizon.
    n_pde : int
        Number of collocation points to maintain after resampling.
    n_candidate : int
        Size of the candidate pool evaluated each resampling step.
        Larger = better coverage but more compute.
    uniform_fraction : float in [0, 1]
        Fraction of n_pde drawn uniformly (vs. residual-weighted).
        Keeps sampling from not over-focusing on a single region.
    power : float
        Exponent applied to the residual magnitude before normalising
        into a probability.  Higher → sharper focus on large residuals.
    nu : float
        Kinematic viscosity (used to evaluate NS residuals).
    device : str
    dtype : torch.dtype
    """

    def __init__(
        self,
        L: float = 2 * math.pi,
        T: float = 1.0,
        n_pde: int = 20_000,
        n_candidate: int = 100_000,
        uniform_fraction: float = 0.5,
        power: float = 1.0,
        nu: float = 0.1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.L = L
        self.T = T
        self.n_pde = n_pde
        self.n_candidate = n_candidate
        self.uniform_fraction = float(uniform_fraction)
        self.power = float(power)
        self.nu = nu
        self.device = device
        self.dtype = dtype

        # Current collocation pool — initialise uniformly
        self._pts: Tensor = self._uniform_sample(n_pde)
        self._history: list[float] = []  # mean residual at each resample step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def pts(self) -> Tensor:
        """Current collocation points, shape (n_pde, 4)."""
        return self._pts

    def resample(self, model: torch.nn.Module) -> Tensor:
        """Generate new collocation points and update the internal pool.

        Parameters
        ----------
        model : trained PINN (must accept (N, 4) tensor → (N, 4) tensor)

        Returns
        -------
        pts : Tensor, shape (n_pde, 4)  — the new pool (also stored in self.pts)
        """
        candidate = self._uniform_sample(self.n_candidate)
        weights = self._residual_weights(model, candidate)

        mean_res = float(weights.mean())
        self._history.append(mean_res)

        n_uniform = int(self.n_pde * self.uniform_fraction)
        n_adaptive = self.n_pde - n_uniform

        # Uniform fraction
        idx_u = torch.randperm(self.n_candidate, device=self.device)[:n_uniform]
        pts_u = candidate[idx_u]

        # Residual-weighted fraction
        probs = (weights ** self.power).to(torch.float64)
        probs = probs + 1e-10  # uniform floor to prevent all-zero distribution
        probs = probs / (probs.sum() + 1e-30)
        idx_a = torch.multinomial(probs, num_samples=n_adaptive, replacement=False)
        pts_a = candidate[idx_a]

        self._pts = torch.cat([pts_u, pts_a], dim=0).detach()
        return self._pts

    def uniform_reset(self) -> Tensor:
        """Reset the pool to a pure uniform sample (useful at epoch 0)."""
        self._pts = self._uniform_sample(self.n_pde)
        return self._pts

    @property
    def history(self) -> list[float]:
        """Mean residual magnitude at each resampling step."""
        return self._history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _uniform_sample(self, n: int) -> Tensor:
        """Sample n points uniformly on [0, L]^3 × [0, T]."""
        pts = torch.rand(n, 4, dtype=self.dtype, device=self.device)
        pts[:, :3] *= self.L
        pts[:, 3] *= self.T
        return pts

    def _residual_weights(self, model: torch.nn.Module, pts: Tensor) -> Tensor:
        """Evaluate total NS residual magnitude at candidate points.

        Returns a (n_candidate,) tensor of non-negative weights.
        NOTE: we do NOT use torch.no_grad() here because AD is needed
        to compute the NS residuals.
        """
        pts_req = pts.clone()

        # Batch in chunks to limit memory
        chunk = 10_000
        all_weights = []

        for i in range(0, len(pts), chunk):
            p = pts_req[i : i + chunk].to(self.device)
            # Enable grad only for this chunk
            x = p[:, 0].detach().requires_grad_(True)
            y = p[:, 1].detach().requires_grad_(True)
            z = p[:, 2].detach().requires_grad_(True)
            t = p[:, 3].detach().requires_grad_(True)

            xyzt = torch.stack([x, y, z, t], dim=1)
            out = model(xyzt)
            u, v, w, pres = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

            res = ns_residuals(u, v, w, pres, x, y, z, t, nu=self.nu)
            # Total pointwise residual magnitude (momentum + continuity)
            total = (res.R_u**2 + res.R_v**2 + res.R_w**2 + res.R_c**2).detach()
            all_weights.append(total)

        return torch.cat(all_weights, dim=0)
