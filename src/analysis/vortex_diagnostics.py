"""
src/analysis/vortex_diagnostics.py
=====================================
Vorticity field diagnostics: Q-criterion, enstrophy budget, palinstrophy,
and vorticity alignment statistics.

All functions operate on physical-space arrays of shape (N, N, N) or their
Fourier transforms.

Key diagnostics
---------------
- vorticity_field  : ω = ∇×u (spectral, exact)
- q_criterion      : Q = ½(|Ω|² - |S|²) where Ω = antisymmetric part of ∇u,
                     S = symmetric part.  Q > 0 marks vortex cores.
- enstrophy_budget : dΩ/dt = production - viscous dissipation
- helicity_spectrum: H(k) = u·ω spectrum (topological measure)
- vorticity_alignment: cos angle between ω and eigenvectors of S

References
----------
- Hunt et al. (1988) — Q-criterion for vortex identification.
- Moffatt (1969) — Helicity in fluid dynamics.
- Tsinober (2001) — "An Informal Introduction to Turbulence."
"""
from __future__ import annotations

import math

import numpy as np


def vorticity_field(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    L: float = 2 * math.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vorticity ω = ∇×u using spectral differentiation.

    Returns
    -------
    omega_x, omega_y, omega_z : each (N, N, N)
    """
    N = u.shape[0]
    k1d = np.fft.fftfreq(N, d=1.0 / N)
    Kx, Ky, Kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")

    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)

    # ω_x = ∂w/∂y - ∂v/∂z
    wx_hat = 1j * Ky * w_hat - 1j * Kz * v_hat
    # ω_y = ∂u/∂z - ∂w/∂x
    wy_hat = 1j * Kz * u_hat - 1j * Kx * w_hat
    # ω_z = ∂v/∂x - ∂u/∂y
    wz_hat = 1j * Kx * v_hat - 1j * Ky * u_hat

    omega_x = np.real(np.fft.ifftn(wx_hat))
    omega_y = np.real(np.fft.ifftn(wy_hat))
    omega_z = np.real(np.fft.ifftn(wz_hat))

    return omega_x, omega_y, omega_z


def enstrophy(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    L: float = 2 * math.pi,
) -> float:
    """Volume-averaged enstrophy Ω = ½⟨|ω|²⟩."""
    ox, oy, oz = vorticity_field(u, v, w, L=L)
    return float(0.5 * np.mean(ox**2 + oy**2 + oz**2))


def palinstrophy(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    L: float = 2 * math.pi,
) -> float:
    """Volume-averaged palinstrophy P = ½⟨|∇ω|²⟩ (spectral)."""
    N = u.shape[0]
    k1d = np.fft.fftfreq(N, d=1.0 / N)
    Kx, Ky, Kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    K2 = Kx**2 + Ky**2 + Kz**2

    ox, oy, oz = vorticity_field(u, v, w, L=L)
    ox_hat = np.fft.fftn(ox)
    oy_hat = np.fft.fftn(oy)
    oz_hat = np.fft.fftn(oz)

    return float(0.5 * np.sum(K2 * (
        np.abs(ox_hat)**2 + np.abs(oy_hat)**2 + np.abs(oz_hat)**2
    )) / N**3)


def q_criterion(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    L: float = 2 * math.pi,
) -> np.ndarray:
    """Compute the Q-criterion field Q(x) = ½(|Ω|² - |S|²).

    Q > 0 indicates regions dominated by rotation (vortex cores).
    Q < 0 indicates regions dominated by strain.

    Velocity gradients are computed spectrally.

    Returns
    -------
    Q : ndarray, shape (N, N, N)
    """
    N = u.shape[0]
    k1d = np.fft.fftfreq(N, d=1.0 / N)
    Kx, Ky, Kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")

    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)

    def spectral_grad(f_hat, K):
        return np.real(np.fft.ifftn(1j * K * f_hat))

    # Velocity gradient tensor components ∂u_i/∂x_j
    du_dx = spectral_grad(u_hat, Kx); du_dy = spectral_grad(u_hat, Ky); du_dz = spectral_grad(u_hat, Kz)
    dv_dx = spectral_grad(v_hat, Kx); dv_dy = spectral_grad(v_hat, Ky); dv_dz = spectral_grad(v_hat, Kz)
    dw_dx = spectral_grad(w_hat, Kx); dw_dy = spectral_grad(w_hat, Ky); dw_dz = spectral_grad(w_hat, Kz)

    # Antisymmetric part Ω_ij = ½(∂u_i/∂x_j - ∂u_j/∂x_i)
    # |Ω|² = 2 Σ_{i<j} Ω_ij²
    omega_12 = 0.5 * (du_dy - dv_dx)
    omega_13 = 0.5 * (du_dz - dw_dx)
    omega_23 = 0.5 * (dv_dz - dw_dy)
    omega_sq = 2.0 * (omega_12**2 + omega_13**2 + omega_23**2)

    # Symmetric part S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)
    # |S|² = Σ_i S_ii² + 2 Σ_{i<j} S_ij²
    S_11 = du_dx; S_22 = dv_dy; S_33 = dw_dz
    S_12 = 0.5 * (du_dy + dv_dx)
    S_13 = 0.5 * (du_dz + dw_dx)
    S_23 = 0.5 * (dv_dz + dw_dy)
    S_sq = S_11**2 + S_22**2 + S_33**2 + 2.0 * (S_12**2 + S_13**2 + S_23**2)

    return 0.5 * (omega_sq - S_sq)


def helicity_spectrum(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    L: float = 2 * math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Helicity spectrum H(k) = Σ_{|k|∈shell} Re[û·ω̂*].

    Helicity measures the linkage between velocity and vorticity.
    For mirror-symmetric flows H(k) = 0; non-zero H indicates chirality.

    Returns
    -------
    k_bins, H_k : (K,) arrays
    """
    N = u.shape[0]
    k1d = np.fft.fftfreq(N, d=1.0 / N)
    Kx, Ky, Kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)

    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)

    # Vorticity in Fourier space
    wx_hat = 1j * Ky * w_hat - 1j * Kz * v_hat
    wy_hat = 1j * Kz * u_hat - 1j * Kx * w_hat
    wz_hat = 1j * Kx * v_hat - 1j * Ky * u_hat

    # Helicity density: Re(û · ω̂*) / N^6
    H_density = np.real(
        u_hat * wx_hat.conj() + v_hat * wy_hat.conj() + w_hat * wz_hat.conj()
    ) / N**6

    k_max = N // 2
    k_bins = np.arange(1, k_max + 1, dtype=float)
    H_k = np.zeros_like(k_bins)
    for i, k in enumerate(k_bins):
        mask = (K_mag >= k - 0.5) & (K_mag < k + 0.5)
        H_k[i] = H_density[mask].sum()

    return k_bins, H_k


def vortex_diagnostics_report(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    nu: float, L: float = 2 * math.pi, t: float = 0.0,
) -> dict:
    """Full vortex diagnostics report for one snapshot.

    Returns
    -------
    dict with: t, enstrophy, palinstrophy, Q_max, Q_min, Q_mean,
               Q_vortex_fraction, total_helicity
    """
    ens = enstrophy(u, v, w, L=L)
    pal = palinstrophy(u, v, w, L=L)
    Q = q_criterion(u, v, w, L=L)
    k_bins, H_k = helicity_spectrum(u, v, w, L=L)

    return {
        "t": t,
        "enstrophy": ens,
        "palinstrophy": pal,
        "Q_max": float(Q.max()),
        "Q_min": float(Q.min()),
        "Q_mean": float(Q.mean()),
        "Q_vortex_fraction": float((Q > 0).mean()),  # fraction of domain in vortex cores
        "total_helicity": float(H_k.sum()),
    }
