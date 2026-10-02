"""
src/models/base.py
==================
Abstract base class for all PINN models in the NS-PINN framework.

Every model:
  - Accepts a (N, 4) tensor of (x, y, z, t) collocation points.
  - Returns a (N, 4) tensor of (u, v, w, p) predictions.
  - Reports its parameter count.
  - Supports dtype and device specification.
  - Exposes a config dict for logging/reproducibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class PINNModel(nn.Module, ABC):
    """Abstract base for all PINN architectures."""

    @abstractmethod
    def forward(self, pts: Tensor) -> Tensor:
        """Map collocation points to field predictions.

        Parameters
        ----------
        pts : Tensor, shape (N, 4)
            Columns: [x, y, z, t]

        Returns
        -------
        Tensor, shape (N, 4)
            Columns: [u, v, w, p]
        """
        ...

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def config_dict(self) -> dict:
        """Return a serializable dict of architecture hyperparameters."""
        ...

    def dtype(self) -> torch.dtype:
        """Infer dtype from first parameter."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32
