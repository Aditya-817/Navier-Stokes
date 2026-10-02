"""
src/utils/domain.py
====================
Periodic torus T^3 = [0, 2pi]^3 domain utilities.

Provides:
  - Grid generation (cell-centred and vertex-centred).
  - Wavenumber index arrays for spectral work.
  - Periodicity-check helpers for scalar and vector fields.
  - Fourier-mode index sets for Fourier-embedding construction.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TWO_PI: float = 2.0 * np.pi  # period of T^3


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def cell_centred_grid(
    N: int | Sequence[int],
    L: float = TWO_PI,
    dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell-centred coordinates on [0, L)^3.

    For a grid with N points per dimension the i-th cell centre is at
    x_i = (i + 0.5) * L / N,   i = 0, ..., N-1.

    Parameters
    ----------
    N : int or (Nx, Ny, Nz)
        Number of grid points per spatial dimension.
    L : float
        Side length of the periodic box (default 2π).
    dtype : numpy dtype
        Floating-point precision.

    Returns
    -------
    X, Y, Z : np.ndarray of shape (Nx, Ny, Nz)
        Meshgrid arrays of cell-centre coordinates.
    """
    if isinstance(N, int):
        Nx = Ny = Nz = N
    else:
        Nx, Ny, Nz = N

    x = (np.arange(Nx, dtype=dtype) + 0.5) * L / Nx
    y = (np.arange(Ny, dtype=dtype) + 0.5) * L / Ny
    z = (np.arange(Nz, dtype=dtype) + 0.5) * L / Nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


def vertex_grid(
    N: int | Sequence[int],
    L: float = TWO_PI,
    dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vertex-centred coordinates on [0, L]^3 (N+1 points per dim).

    Includes both 0 and L endpoints; useful for periodicity verification.
    """
    if isinstance(N, int):
        Nx = Ny = Nz = N
    else:
        Nx, Ny, Nz = N

    x = np.linspace(0.0, L, Nx + 1, dtype=dtype)
    y = np.linspace(0.0, L, Ny + 1, dtype=dtype)
    z = np.linspace(0.0, L, Nz + 1, dtype=dtype)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


# ---------------------------------------------------------------------------
# Wavenumber arrays
# ---------------------------------------------------------------------------

def wavenumbers_1d(N: int) -> np.ndarray:
    """Return the standard Fourier wavenumbers for N points.

    For N even: k = [0, 1, ..., N/2-1, -N/2, ..., -1]
    (matches numpy.fft.fftfreq * N).

    Returns
    -------
    k : np.ndarray of shape (N,), dtype int
    """
    return np.fft.fftfreq(N, d=1.0 / N).astype(int)


def wavenumber_grid(
    N: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 3D wavenumber index grids (Kx, Ky, Kz).

    Parameters
    ----------
    N : int or (Nx, Ny, Nz)

    Returns
    -------
    Kx, Ky, Kz : np.ndarray of shape (Nx, Ny, Nz), dtype int
    """
    if isinstance(N, int):
        Nx = Ny = Nz = N
    else:
        Nx, Ny, Nz = N

    kx = wavenumbers_1d(Nx)
    ky = wavenumbers_1d(Ny)
    kz = wavenumbers_1d(Nz)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing="ij")
    return Kx, Ky, Kz


def fourier_mode_grid(K: int) -> np.ndarray:
    """Return all integer wavenumber vectors k ∈ Z^3 with max(|k|) ≤ K.

    These are used to construct the Fourier-feature embedding matrix.

    Parameters
    ----------
    K : int
        Maximum per-component wavenumber magnitude.

    Returns
    -------
    modes : np.ndarray of shape (M, 3)
        Each row is a wavenumber triplet (kx, ky, kz).
        The zero mode (0,0,0) is excluded (constant feature, not useful).
    """
    k = np.arange(-K, K + 1, dtype=np.float64)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    modes = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=-1)
    # Remove zero mode
    nonzero = np.any(modes != 0, axis=1)
    return modes[nonzero]


# ---------------------------------------------------------------------------
# Periodicity checks
# ---------------------------------------------------------------------------

def check_periodicity_numpy(
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    N: int = 32,
    L: float = TWO_PI,
    atol: float = 1e-10,
) -> dict[str, float]:
    """Verify that f is L-periodic in each spatial direction.

    Tests f(0, y, z) ≈ f(L, y, z) and similarly for y, z using a random
    interior slice (to avoid coincidental cancellations at grid boundaries).

    Parameters
    ----------
    f : callable
        f(X, Y, Z) -> array broadcastable to (N, N, N).
    N : int
        Number of interior test points per dimension.
    L : float
        Period length.
    atol : float
        Absolute tolerance for the max difference.

    Returns
    -------
    dict with keys 'max_err_x', 'max_err_y', 'max_err_z', 'passed'.
    """
    rng = np.random.default_rng(seed=0)
    s = N  # sample size for interior slices

    # Random interior coordinates for the two "free" dimensions
    y_rand = rng.uniform(0, L, (s, s))
    z_rand = rng.uniform(0, L, (s, s))
    x_rand = rng.uniform(0, L, (s, s))

    # --- x-periodicity ---
    f0_x = f(np.zeros((s, s)), y_rand, z_rand)
    fL_x = f(np.full((s, s), L), y_rand, z_rand)
    err_x = float(np.max(np.abs(f0_x - fL_x)))

    # --- y-periodicity ---
    f0_y = f(x_rand, np.zeros((s, s)), z_rand)
    fL_y = f(x_rand, np.full((s, s), L), z_rand)
    err_y = float(np.max(np.abs(f0_y - fL_y)))

    # --- z-periodicity ---
    f0_z = f(x_rand, y_rand, np.zeros((s, s)))
    fL_z = f(x_rand, y_rand, np.full((s, s), L))
    err_z = float(np.max(np.abs(f0_z - fL_z)))

    passed = (err_x < atol) and (err_y < atol) and (err_z < atol)
    return {
        "max_err_x": err_x,
        "max_err_y": err_y,
        "max_err_z": err_z,
        "passed": passed,
    }


def check_periodicity_torch(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    component_idx: int,
    N: int = 64,
    L: float = TWO_PI,
    t_val: float = 0.5,
    atol: float = 1e-6,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    """Verify that a PINN output is L-periodic in x, y, z.

    Parameters
    ----------
    model_fn : callable
        Accepts a tensor of shape (N_pts, 4) — columns (x, y, z, t) —
        and returns a tensor of shape (N_pts, n_outputs).
    component_idx : int
        Which output component to test (0=u, 1=v, 2=w, 3=p).
    N : int
        Number of random test points per boundary direction.
    L : float
        Period.
    t_val : float
        Fixed time value for the test.
    atol : float
        Absolute tolerance.
    device, dtype : torch settings.

    Returns
    -------
    dict with keys 'max_err_x', 'max_err_y', 'max_err_z', 'passed'.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    ki = component_idx

    def _rand(n: int) -> torch.Tensor:
        return torch.rand(n, generator=rng, device=device, dtype=dtype) * L

    t = torch.full((N,), t_val, device=device, dtype=dtype)
    y_r = _rand(N)
    z_r = _rand(N)
    x_r = _rand(N)

    def _eval(pts: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model_fn(pts)[:, ki]

    # x-periodicity
    p0x = torch.stack([torch.zeros(N, device=device, dtype=dtype), y_r, z_r, t], dim=1)
    pLx = torch.stack([torch.full((N,), L, device=device, dtype=dtype), y_r, z_r, t], dim=1)
    err_x = float((_eval(pLx) - _eval(p0x)).abs().max())

    # y-periodicity
    p0y = torch.stack([x_r, torch.zeros(N, device=device, dtype=dtype), z_r, t], dim=1)
    pLy = torch.stack([x_r, torch.full((N,), L, device=device, dtype=dtype), z_r, t], dim=1)
    err_y = float((_eval(pLy) - _eval(p0y)).abs().max())

    # z-periodicity
    p0z = torch.stack([x_r, y_r, torch.zeros(N, device=device, dtype=dtype), t], dim=1)
    pLz = torch.stack([x_r, y_r, torch.full((N,), L, device=device, dtype=dtype), t], dim=1)
    err_z = float((_eval(pLz) - _eval(p0z)).abs().max())

    passed = (err_x < atol) and (err_y < atol) and (err_z < atol)
    return {
        "max_err_x": err_x,
        "max_err_y": err_y,
        "max_err_z": err_z,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Collocation point helpers
# ---------------------------------------------------------------------------

def make_xyzt_tensor(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    t: float | np.ndarray,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Flatten grid arrays and a time value into a (N_pts, 4) tensor.

    Parameters
    ----------
    X, Y, Z : np.ndarray, same shape
    t : scalar or array broadcastable to X
    """
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    if np.isscalar(t):
        t_flat = np.full_like(x_flat, t)
    else:
        t_flat = np.broadcast_to(t, X.shape).ravel()

    pts = np.stack([x_flat, y_flat, z_flat, t_flat], axis=-1)
    return torch.tensor(pts, device=device, dtype=dtype)
