"""
src/models/multiscale_mlp.py
==============================
Multi-scale Fourier MLP (Cai et al. 2021 / Wang et al. 2021).

The idea is to embed the input at multiple spatial scales simultaneously,
allowing the network to represent both large-scale structures and fine-scale
features without requiring a very large number of Fourier modes at a single
scale.

Architecture
------------
Given K scale levels σ₁ < σ₂ < ... < σ_K:

  1. For each scale σᵢ, compute a Fourier embedding at that frequency.
  2. Pass each through a small "sub-MLP" (or share a common backbone).
  3. Aggregate (sum or concatenate + project).

Here we use the concatenation-then-project approach for flexibility.
Periodicity is preserved by using only integer wavenumbers in the embedding.

Reference
---------
Wang et al. (2021) "On the eigenvector bias of Fourier feature networks."
Computer Methods in Applied Mechanics and Engineering 384.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import PINNModel
from src.utils.domain import fourier_mode_grid


class ScaleBranch(nn.Module):
    """One frequency-scale branch: Fourier embedding + small MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        n_hidden: int,
        activation: str,
    ) -> None:
        super().__init__()
        act_cls = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}.get(
            activation, nn.Tanh
        )
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_width), act_cls()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_width, hidden_width), act_cls()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiscaleMLP(PINNModel):
    """Multi-scale Fourier MLP for 3D periodic NS.

    Parameters
    ----------
    fourier_scales : list of int — wavenumber limits for each scale branch.
        e.g. [2, 4, 8] means three branches with K=2, 4, 8 respectively.
    hidden_width : int — width of each branch and the fusion network
    n_layers_branch : int — depth of each scale branch
    n_layers_fusion : int — depth of the fusion MLP
    activation : str
    dtype : str
    L : float
    t_scale : float
    """

    def __init__(
        self,
        fourier_scales: list[int] | None = None,
        hidden_width: int = 128,
        n_layers_branch: int = 2,
        n_layers_fusion: int = 2,
        activation: str = "tanh",
        dtype: str = "float64",
        L: float = 2 * math.pi,
        t_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if fourier_scales is None:
            fourier_scales = [2, 4, 8]
        self.fourier_scales = fourier_scales
        self.hidden_width = hidden_width
        self.n_layers_branch = n_layers_branch
        self.n_layers_fusion = n_layers_fusion
        self._dtype = torch.float64 if dtype == "float64" else torch.float32
        self.L = L
        self.t_scale = t_scale

        # Build mode grids for each scale
        self._scale_modes: list[Tensor] = []
        scale_dims = []
        for K in fourier_scales:
            modes = fourier_mode_grid(K)  # (M_k, 3)
            t_modes = torch.tensor(modes, dtype=self._dtype)
            self.register_buffer(f"modes_K{K}", t_modes)
            self._scale_modes.append(t_modes)
            M_k = modes.shape[0]
            scale_dims.append(2 * M_k + 1)  # sin+cos + time

        # Scale branches
        self.branches = nn.ModuleList([
            ScaleBranch(dim, hidden_width, n_layers_branch, activation)
            for dim in scale_dims
        ])

        # Fusion MLP: concatenated branch outputs → 4 outputs
        fusion_in = hidden_width * len(fourier_scales)
        act_cls = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}.get(
            activation, nn.Tanh
        )
        fusion_layers: list[nn.Module] = [nn.Linear(fusion_in, hidden_width), act_cls()]
        for _ in range(n_layers_fusion - 1):
            fusion_layers += [nn.Linear(hidden_width, hidden_width), act_cls()]
        fusion_layers.append(nn.Linear(hidden_width, 4))
        self.fusion = nn.Sequential(*fusion_layers)

        self.to(self._dtype)

    def _get_mode_buffer(self, K: int) -> Tensor:
        return getattr(self, f"modes_K{K}")

    def forward(self, pts: Tensor) -> Tensor:
        x = pts[:, :3]   # (N, 3)
        t = pts[:, 3:4]  # (N, 1)
        t_norm = t / self.t_scale

        branch_outputs = []
        for K, branch in zip(self.fourier_scales, self.branches):
            modes = self._get_mode_buffer(K)  # (M_k, 3)
            phases = (2.0 * math.pi / self.L) * (x @ modes.T)
            emb = torch.cat([torch.sin(phases), torch.cos(phases), t_norm], dim=-1)
            branch_outputs.append(branch(emb))

        fused = torch.cat(branch_outputs, dim=-1)
        return self.fusion(fused)

    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return {
            "type": "multiscale_mlp",
            "fourier_scales": self.fourier_scales,
            "hidden_width": self.hidden_width,
            "n_layers_branch": self.n_layers_branch,
            "n_layers_fusion": self.n_layers_fusion,
        }
