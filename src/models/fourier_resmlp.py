"""
src/models/fourier_resmlp.py
=============================
Fourier-feature residual MLP for 3D Navier–Stokes on T^3.

Architecture
------------
Input (x, y, z, t) ∈ R^4
  → Fourier embedding layer (exactly 2π-periodic in x, y, z)
  → Linear projection to hidden width W
  → N × ResBlock(Linear → Norm → Activation → Linear → skip)
  → Linear head → (u, v, w, p)

Periodicity guarantee
---------------------
The Fourier embedding uses only integer wavenumbers k ∈ Z^3 for the
spatial coordinates (x, y, z).  The embedding is:

    γ(x,y,z,t) = [sin(k·r) for k in modes]
               ⊕ [cos(k·r) for k in modes]
               ⊕ [t / t_scale]          ← time is NOT embedded periodically

where r = (x, y, z) and k ∈ Z^3 with 0 < |k|_inf ≤ K.

Since sin and cos of integer multiples of the coordinates are exactly
2π-periodic, any linear combination — and hence any MLP output that
depends only on this embedding — is exactly 2π-periodic in x, y, z.

This is a structural guarantee, not a penalty term.  Boundary condition
losses for periodicity are therefore unnecessary in x, y, z.

Time is passed as a scaled linear feature; periodicity in time is not
assumed.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import PINNModel
from src.utils.domain import fourier_mode_grid

# Supported activation functions
_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "sin": None,  # handled separately as a lambda
}


class SineActivation(nn.Module):
    """sin activation, elementwise."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


def _build_activation(name: str) -> nn.Module:
    if name == "sin":
        return SineActivation()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# Fourier embedding layer
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """Exactly 2π-periodic Fourier embedding for (x, y, z).

    Time is appended as a scaled scalar feature.

    Output dimension: 2 * M + 1
    where M = number of non-zero spatial modes with |k|_inf ≤ K.

    Parameters
    ----------
    K : int
        Max per-component integer wavenumber for spatial embedding.
    t_scale : float
        Divisor for the time feature (set to T, the time horizon).
    trainable_freqs : bool
        If True, the Fourier frequencies are treated as learnable parameters
        (breaks exact periodicity guarantee — use with care).
    """

    def __init__(
        self,
        K: int = 8,
        t_scale: float = 1.0,
        trainable_freqs: bool = False,
    ) -> None:
        super().__init__()
        self.K = K
        self.t_scale = t_scale
        self.trainable_freqs = trainable_freqs

        modes = fourier_mode_grid(K)  # shape (M, 3)
        self.M = len(modes)  # number of spatial modes

        freqs = torch.tensor(modes, dtype=torch.float64)  # (M, 3)
        if trainable_freqs:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 2 * self.M + 1  # sin + cos + time

    def forward(self, pts: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pts : Tensor, shape (N, 4)  — columns [x, y, z, t]

        Returns
        -------
        Tensor, shape (N, 2*M + 1)
        """
        xyz = pts[:, :3]  # (N, 3)
        t = pts[:, 3:4]   # (N, 1)

        # (N, M) = (N, 3) @ (3, M)
        phase = xyz @ self.freqs.T  # (N, M)

        embedding = torch.cat([
            torch.sin(phase),   # (N, M)
            torch.cos(phase),   # (N, M)
            t / self.t_scale,   # (N, 1)
        ], dim=1)  # (N, 2M+1)

        return embedding


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Single residual block: Linear → Norm → Activation → Linear → skip.

    The skip connection is added at the output, not before the second linear.
    If input and output widths differ, a 1×1 projection is used.

    Parameters
    ----------
    width : int
        Hidden dimension (in and out).
    activation : str
        Activation function name.
    use_layernorm : bool
        Apply LayerNorm before the activation.
    """

    def __init__(
        self,
        width: int,
        activation: str = "tanh",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width) if use_layernorm else nn.Identity()
        self.act = _build_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.linear1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.linear2(out)
        return out + identity


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FourierResMLP(PINNModel):
    """Fourier-feature residual MLP for 3D NS on T^3.

    Parameters
    ----------
    fourier_modes : int
        Max per-component wavenumber K for the spatial Fourier embedding.
    hidden_width : int
        Width of the residual MLP.
    n_layers : int
        Number of residual blocks.
    activation : str
        Activation in residual blocks ('tanh', 'gelu', 'silu', 'sin', 'relu').
    use_layernorm : bool
        Whether to use LayerNorm inside residual blocks.
    skip_period : int
        Residual connection every N blocks (1 = every block).
    t_scale : float
        Time normalization scale (set to T, the temporal horizon).
    output_repr : str
        'uvwp' — direct output of (u, v, w, p).
        Future: 'vorticity', 'vector_potential'.
    dtype : str
        'float32' or 'float64'.
    """

    def __init__(
        self,
        fourier_modes: int = 8,
        hidden_width: int = 256,
        n_layers: int = 3,
        activation: str = "tanh",
        use_layernorm: bool = True,
        skip_period: int = 1,
        t_scale: float = 1.0,
        output_repr: Literal["uvwp"] = "uvwp",
        dtype: str = "float64",
        trainable_freqs: bool = False,
    ) -> None:
        super().__init__()

        from src.utils.precision import resolve_dtype
        _dtype = resolve_dtype(dtype)
        torch.set_default_dtype(_dtype)

        self._cfg = dict(
            fourier_modes=fourier_modes,
            hidden_width=hidden_width,
            n_layers=n_layers,
            activation=activation,
            use_layernorm=use_layernorm,
            skip_period=skip_period,
            t_scale=t_scale,
            output_repr=output_repr,
            dtype=dtype,
            trainable_freqs=trainable_freqs,
        )

        # Fourier embedding
        self.embedding = FourierEmbedding(
            K=fourier_modes,
            t_scale=t_scale,
            trainable_freqs=trainable_freqs,
        )
        emb_dim = self.embedding.out_dim

        # Input projection: embedding → hidden width
        self.input_proj = nn.Linear(emb_dim, hidden_width)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_width, activation=activation, use_layernorm=use_layernorm)
            for _ in range(n_layers)
        ])
        self.skip_period = skip_period

        # Output head: hidden → (u, v, w, p)
        self.output_head = nn.Linear(hidden_width, 4)

        # Initialise weights
        self._init_weights()

        # Cast all parameters to target dtype
        self.to(_dtype)

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pts: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pts : Tensor, shape (N, 4) — [x, y, z, t]

        Returns
        -------
        Tensor, shape (N, 4) — [u, v, w, p]
        """
        h = self.embedding(pts)        # (N, 2M+1)
        h = self.input_proj(h)         # (N, W)

        for i, block in enumerate(self.blocks):
            h = block(h)               # (N, W) + skip inside block

        out = self.output_head(h)      # (N, 4)
        return out

    def config_dict(self) -> dict:
        return dict(self._cfg)

    def predict_uvwp(
        self,
        pts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convenience: return (u, v, w, p) as separate tensors."""
        out = self.forward(pts)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


# ---------------------------------------------------------------------------
# Standard MLP (baseline, no Fourier embedding)
# ---------------------------------------------------------------------------

class StandardMLP(PINNModel):
    """Standard fully-connected MLP without Fourier embedding.

    Included for ablation comparisons against FourierResMLP.
    Periodicity is NOT structurally enforced; must rely on sampling
    and/or boundary penalty terms.

    Parameters
    ----------
    hidden_width : int
    n_layers : int
    activation : str
    use_layernorm : bool
    t_scale : float
    dtype : str
    """

    def __init__(
        self,
        hidden_width: int = 256,
        n_layers: int = 4,
        activation: str = "tanh",
        use_layernorm: bool = False,
        t_scale: float = 1.0,
        dtype: str = "float64",
    ) -> None:
        super().__init__()

        from src.utils.precision import resolve_dtype
        _dtype = resolve_dtype(dtype)

        self._cfg = dict(
            hidden_width=hidden_width,
            n_layers=n_layers,
            activation=activation,
            use_layernorm=use_layernorm,
            t_scale=t_scale,
            dtype=dtype,
        )

        layers = [nn.Linear(4, hidden_width)]
        for _ in range(n_layers - 1):
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_width))
            layers.append(_build_activation(activation))
            layers.append(nn.Linear(hidden_width, hidden_width))

        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_width))
        layers.append(_build_activation(activation))
        layers.append(nn.Linear(hidden_width, 4))

        self.net = nn.Sequential(*layers)
        self.t_scale = t_scale

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.to(_dtype)

    def forward(self, pts: Tensor) -> Tensor:
        # Normalise time
        pts_scaled = pts.clone()
        pts_scaled[:, 3] = pts[:, 3] / self.t_scale
        return self.net(pts_scaled)

    def config_dict(self) -> dict:
        return dict(self._cfg)
