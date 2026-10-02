"""
src/physics/vorticity.py
=========================
Vorticity, enstrophy, and kinetic energy utilities.

All functions accept PyTorch tensors and return tensors.
Grid integrals use the trapezoidal rule on structured grids.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.physics.derivatives import curl3d, partial


# ---------------------------------------------------------------------------
# Vorticity via AD
# ---------------------------------------------------------------------------

def vorticity_ad(
    u: Tensor,
    v: Tensor,
    w: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute vorticity ω = ∇×u via automatic differentiation.

    Parameters
    ----------
    u, v, w : Tensor, shape (N,)   — velocity components
    x, y, z : Tensor, shape (N,1), requires_grad=True

    Returns
    -------
    (ωx, ωy, ωz) : each Tensor of shape (N,)
    """
    u = u.squeeze(-1)
    v = v.squeeze(-1)
    w = w.squeeze(-1)
    omegas = curl3d(
        [u, v, w], [x, y, z],
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return omegas[0].squeeze(-1), omegas[1].squeeze(-1), omegas[2].squeeze(-1)


# ---------------------------------------------------------------------------
# Grid-based energy / enstrophy (for structured evaluation grids)
# ---------------------------------------------------------------------------

def kinetic_energy_grid(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    L: float = 2.0 * np.pi,
) -> float:
    """Compute kinetic energy E = (1/2) ∫_{T^3} |u|² dx on a uniform grid.

    Uses the trapezoidal rule (or equivalently Parseval for periodic fields
    on a cell-centred grid: mean of |u|²/2 × L³).

    Parameters
    ----------
    u, v, w : np.ndarray of shape (Nx, Ny, Nz)
        Velocity components on a uniform cell-centred grid.
    L : float
        Side length of the domain.

    Returns
    -------
    E : float   (dimensional if u is dimensional)
    """
    return 0.5 * float(np.mean(u**2 + v**2 + w**2)) * (L**3)


def enstrophy_grid(
    omega_x: np.ndarray,
    omega_y: np.ndarray,
    omega_z: np.ndarray,
    L: float = 2.0 * np.pi,
) -> float:
    """Compute enstrophy Ɛ = (1/2) ∫_{T^3} |ω|² dx.

    Parameters
    ----------
    omega_x, omega_y, omega_z : np.ndarray of shape (Nx, Ny, Nz)
    L : float

    Returns
    -------
    Ɛ : float
    """
    return 0.5 * float(np.mean(omega_x**2 + omega_y**2 + omega_z**2)) * (L**3)


def vorticity_spectral(
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    w_hat: np.ndarray,
    Kx: np.ndarray,
    Ky: np.ndarray,
    Kz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vorticity in Fourier space: ω̂ = ik × û.

    Parameters
    ----------
    u_hat, v_hat, w_hat : complex np.ndarray of shape (Nx, Ny, Nz)
        Fourier coefficients of velocity.
    Kx, Ky, Kz : np.ndarray of shape (Nx, Ny, Nz)
        Wavenumber index grids (integers).

    Returns
    -------
    (ox_hat, oy_hat, oz_hat) : complex np.ndarray
        Fourier coefficients of vorticity components.
    """
    # ω̂_x = i*ky*ŵ - i*kz*v̂
    # ω̂_y = i*kz*û - i*kx*ŵ
    # ω̂_z = i*kx*v̂ - i*ky*û
    ikx = 1j * Kx
    iky = 1j * Ky
    ikz = 1j * Kz

    ox_hat = iky * w_hat - ikz * v_hat
    oy_hat = ikz * u_hat - ikx * w_hat
    oz_hat = ikx * v_hat - iky * u_hat

    return ox_hat, oy_hat, oz_hat


def energy_spectrum(
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    w_hat: np.ndarray,
    Kx: np.ndarray,
    Ky: np.ndarray,
    Kz: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spherically-averaged kinetic energy spectrum E(k).

    Parameters
    ----------
    u_hat, v_hat, w_hat : complex np.ndarray, shape (N, N, N)
        Fourier coefficients (output of np.fft.fftn).
    Kx, Ky, Kz : int np.ndarray, shape (N, N, N)
    N : int
        Grid size per dimension.

    Returns
    -------
    k_bins : np.ndarray of int wavenumber magnitudes
    E_k    : np.ndarray of energy in each shell
    """
    k_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2).astype(int)
    k_max = int(np.max(k_mag))

    # Parseval factor: energy in physical space = (1/N^3) Σ |û|^2
    factor = 1.0 / (N**3) ** 2
    energy_density = 0.5 * factor * (
        np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2
    )

    k_bins = np.arange(0, k_max + 1)
    E_k = np.zeros(k_max + 1)
    for k in k_bins:
        mask = k_mag == k
        E_k[k] = energy_density[mask].sum()

    return k_bins, E_k
