"""
src/training/curriculum.py
============================
Time-window (causal) curriculum scheduler for NS-PINNs.

Motivation
----------
For time-dependent problems the PINN loss naturally mixes early and late
time residuals.  When the model hasn't learned early dynamics yet, large
late-time gradients pollute the early-time gradient signal.

The causal / time-window scheduler addresses this by:

  1. **Sequential window training**: divide [0, T] into K overlapping
     windows and train them sequentially, using the previous window's
     solution as a "soft IC" for the next.

  2. **Causal weighting** (Wang et al. 2022): inside each window apply
     per-point exponential weights that downweight future residuals when
     the model hasn't fit the past.

References
----------
- Wang et al. (2022) "Respecting causality for training physics-informed
  neural networks."  CMAME 421.
- Wight & Zhao (2020) "Solving Allen–Cahn and Cahn–Hilliard equations
  using the adaptive physics-informed neural networks."
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Causal weighting
# ---------------------------------------------------------------------------

def causal_weights(
    t: Tensor,
    t_max: float,
    eps: float = 1.0,
) -> Tensor:
    """Compute causal loss weights for collocation times.

    w(t) = exp(-eps * (t / t_max))

    Points at t≈0 receive weight ≈ 1; points at t≈t_max receive weight
    exp(-eps) ≈ 0.37 (eps=1) or ≈ 0.007 (eps=5).

    Parameters
    ----------
    t : Tensor, shape (N,) — collocation times in [0, t_max]
    t_max : float — maximum time for this window
    eps : float — decay rate. Larger → sharper causal decay.

    Returns
    -------
    w : Tensor, shape (N,), values in (0, 1].
    """
    return torch.exp(-eps * t / t_max)


def apply_causal_weights(residuals: list[Tensor], t: Tensor, t_max: float,
                          eps: float = 1.0) -> Tensor:
    """Compute causal-weighted MSE for a list of residual tensors.

    Parameters
    ----------
    residuals : list of (N,) tensors — individual PDE residual components
    t : Tensor (N,) — time coordinates of collocation points
    t_max : float
    eps : float

    Returns
    -------
    weighted_mse : scalar Tensor
    """
    w = causal_weights(t, t_max=t_max, eps=eps)
    total = sum(r**2 for r in residuals)
    return (w * total).mean()


# ---------------------------------------------------------------------------
# Time window schedule
# ---------------------------------------------------------------------------

@dataclass
class TimeWindow:
    """One time window in the curriculum."""
    t_start: float
    t_end: float
    index: int
    n_epochs: int


class TimeWindowScheduler:
    """Sequential time-window curriculum for NS-PINN training.

    Splits [0, T] into ``n_windows`` overlapping windows, training each
    for ``epochs_per_window`` gradient steps before advancing.  An overlap
    fraction ``overlap`` means adjacent windows share that fraction of
    their total length so the model has a smooth transition.

    Parameters
    ----------
    T : float — total simulation time
    n_windows : int — number of windows
    epochs_per_window : int — gradient steps per window
    overlap : float in [0, 1) — fractional overlap between consecutive windows
    """

    def __init__(
        self,
        T: float = 1.0,
        n_windows: int = 5,
        epochs_per_window: int = 10_000,
        overlap: float = 0.2,
    ) -> None:
        self.T = T
        self.n_windows = n_windows
        self.epochs_per_window = epochs_per_window
        self.overlap = float(overlap)
        self._windows = self._build_windows()
        self._current_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def windows(self) -> list[TimeWindow]:
        return self._windows

    @property
    def current_window(self) -> TimeWindow:
        return self._windows[self._current_idx]

    @property
    def n_windows_total(self) -> int:
        return len(self._windows)

    @property
    def total_epochs(self) -> int:
        return sum(w.n_epochs for w in self._windows)

    def advance(self) -> bool:
        """Advance to the next window.  Returns True if there are more windows."""
        if self._current_idx < len(self._windows) - 1:
            self._current_idx += 1
            return True
        return False

    def reset(self) -> None:
        self._current_idx = 0

    def epoch_to_window(self, epoch: int) -> TimeWindow:
        """Return the window that owns the given global epoch."""
        cumulative = 0
        for w in self._windows:
            cumulative += w.n_epochs
            if epoch < cumulative:
                return w
        return self._windows[-1]

    def sample_pts_in_window(
        self,
        n: int,
        window: TimeWindow | None = None,
        L: float = 2 * math.pi,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ) -> Tensor:
        """Sample n points uniformly in [0,L]^3 × [t_start, t_end].

        Parameters
        ----------
        n : int — number of points
        window : TimeWindow or None (uses current window)
        L : float — domain side length
        """
        if window is None:
            window = self.current_window
        pts = torch.rand(n, 4, dtype=dtype, device=device)
        pts[:, :3] *= L
        pts[:, 3] = window.t_start + pts[:, 3] * (window.t_end - window.t_start)
        return pts

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_windows(self) -> list[TimeWindow]:
        dt = self.T / self.n_windows
        overlap_dt = dt * self.overlap
        windows = []
        for i in range(self.n_windows):
            t_start = max(0.0, i * dt - (overlap_dt if i > 0 else 0.0))
            t_end = min(self.T, (i + 1) * dt)
            windows.append(TimeWindow(
                t_start=t_start,
                t_end=t_end,
                index=i,
                n_epochs=self.epochs_per_window,
            ))
        return windows

    def __repr__(self) -> str:
        return (
            f"TimeWindowScheduler(T={self.T}, n_windows={self.n_windows}, "
            f"epochs_per_window={self.epochs_per_window}, "
            f"current={self._current_idx})"
        )
