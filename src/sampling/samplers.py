"""
src/sampling/uniform.py + lhs.py + sobol.py
=============================================
Collocation point samplers for T^3 × [0, T].

All samplers return np.ndarray of shape (N, 4) with columns [x, y, z, t]
and values in [0, 2π]^3 × [0, T].
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc


TWO_PI = 2.0 * np.pi


def uniform_sample(
    N: int,
    T: float,
    L: float = TWO_PI,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform random sampling on [0,L]^3 × [0,T].

    Parameters
    ----------
    N : int
        Number of collocation points.
    T : float
        Temporal horizon.
    L : float
        Spatial period (default 2π).
    rng : np.random.Generator, optional
        For reproducibility. If None, a fresh generator is used.

    Returns
    -------
    pts : np.ndarray, shape (N, 4), dtype float64
        Columns: [x, y, z, t]
    """
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.uniform(0.0, 1.0, (N, 4))
    pts[:, :3] *= L
    pts[:, 3] *= T
    return pts.astype(np.float64)


def lhs_sample(
    N: int,
    T: float,
    L: float = TWO_PI,
    seed: int = 0,
) -> np.ndarray:
    """Latin Hypercube Sampling on [0,L]^3 × [0,T].

    Uses scipy.stats.qmc.LatinHypercube for stratified sampling.
    Better space-filling than uniform random for moderate N.
    """
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    unit_pts = sampler.random(N)  # (N, 4) in [0, 1]^4
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([L, L, L, T])
    return qmc.scale(unit_pts, lower, upper).astype(np.float64)


def sobol_sample(
    N: int,
    T: float,
    L: float = TWO_PI,
    seed: int = 0,
) -> np.ndarray:
    """Sobol quasi-random sequence on [0,L]^3 × [0,T].

    Provides better uniformity than LHS for larger N.
    N is rounded up to the next power of 2 for Sobol sequences.
    """
    # Sobol requires power-of-2 sample sizes
    N_actual = 2 ** int(np.ceil(np.log2(max(N, 2))))
    sampler = qmc.Sobol(d=4, seed=seed)
    unit_pts = sampler.random(N_actual)[:N]  # take first N
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([L, L, L, T])
    return qmc.scale(unit_pts, lower, upper).astype(np.float64)


def ic_sample(
    N: int,
    L: float = TWO_PI,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform random sample on [0,L]^3 × {t=0} for IC loss.

    Returns
    -------
    pts : np.ndarray, shape (N, 4), last column = 0.
    """
    if rng is None:
        rng = np.random.default_rng()
    xyz = rng.uniform(0.0, L, (N, 3))
    t_col = np.zeros((N, 1))
    return np.concatenate([xyz, t_col], axis=1).astype(np.float64)
