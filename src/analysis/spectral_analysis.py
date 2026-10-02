"""
src/analysis/spectral_analysis.py
====================================
Energy spectra, Kolmogorov scaling, and spectral diagnostics for 3D NS
solutions on the periodic torus T^3.

All functions operate on physical-space velocity arrays of shape (N, N, N)
and return spectral quantities as NumPy arrays.

Key functions
-------------
- energy_spectrum_3d : E(k) vs. wavenumber shell k
- kolmogorov_scale   : η = (ν³/ε)^(1/4), ε from enstrophy
- kolmogorov_exponent: linear fit of log E vs log k in inertial range
- dissipation_spectrum: D(k) = 2ν k² E(k)
- cumulative_dissipation: ∫₀^k D(k') dk' / ε_total

References
----------
- Pope, S.B. (2000) "Turbulent Flows," Cambridge.
- Kolmogorov (1941) — k^{-5/3} inertial range scaling.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Core: 3D energy spectrum E(k)
# ---------------------------------------------------------------------------

def energy_spectrum_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    L: float = 2 * math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 1D energy spectrum E(k) from 3D velocity fields.

    Parameters
    ----------
    u, v, w : ndarray, shape (N, N, N) — velocity components
    L : float — domain side length

    Returns
    -------
    k_bins : ndarray, shape (K,) — integer wavenumber shells k = 1, 2, ..., N//2
    E_k    : ndarray, shape (K,) — energy in each shell
             E(k) = Σ_{|k|∈[k-0.5, k+0.5)} ½|û(k)|²/N^6
    """
    N = u.shape[0]
    assert u.shape == v.shape == w.shape == (N, N, N)

    # FFT and normalise
    u_hat = np.fft.fftn(u) / N**3
    v_hat = np.fft.fftn(v) / N**3
    w_hat = np.fft.fftn(w) / N**3

    # Wavenumber arrays
    k1d = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers
    Kx, Ky, Kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)

    # Energy density: ½|û|² per mode
    E_density = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)

    # Bin into integer shells
    k_max = N // 2
    k_bins = np.arange(1, k_max + 1, dtype=float)
    E_k = np.zeros_like(k_bins)

    for i, k in enumerate(k_bins):
        mask = (K_mag >= k - 0.5) & (K_mag < k + 0.5)
        E_k[i] = E_density[mask].sum()

    return k_bins, E_k


# ---------------------------------------------------------------------------
# Dissipation and Kolmogorov scale
# ---------------------------------------------------------------------------

def dissipation_rate(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    nu: float,
    L: float = 2 * math.pi,
) -> float:
    """Estimate dissipation rate ε = 2ν Σ_k k² E(k).

    Parameters
    ----------
    u, v, w : (N,N,N) velocity fields
    nu : float — kinematic viscosity
    L : float — domain side length

    Returns
    -------
    epsilon : float — volume-averaged dissipation rate
    """
    k_bins, E_k = energy_spectrum_3d(u, v, w, L=L)
    return float(2.0 * nu * np.sum(k_bins**2 * E_k))


def kolmogorov_scale(nu: float, epsilon: float) -> float:
    """Kolmogorov microscale η = (ν³/ε)^(1/4).

    Parameters
    ----------
    nu : float — kinematic viscosity
    epsilon : float — dissipation rate

    Returns
    -------
    eta : float — Kolmogorov microscale
    """
    if epsilon <= 0:
        return float("inf")
    return float((nu**3 / epsilon) ** 0.25)


def dissipation_spectrum(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    nu: float,
    L: float = 2 * math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Dissipation spectrum D(k) = 2ν k² E(k).

    Returns
    -------
    k_bins, D_k : (K,) arrays
    """
    k_bins, E_k = energy_spectrum_3d(u, v, w, L=L)
    D_k = 2.0 * nu * k_bins**2 * E_k
    return k_bins, D_k


# ---------------------------------------------------------------------------
# Spectral scaling exponent
# ---------------------------------------------------------------------------

def kolmogorov_exponent(
    k_bins: np.ndarray,
    E_k: np.ndarray,
    k_min: float = 3.0,
    k_max: float | None = None,
) -> dict:
    """Fit log E(k) = α log k + const in the inertial range [k_min, k_max].

    Returns
    -------
    dict with:
        'alpha'     : float — fitted slope (expect ≈ -5/3 for Kolmogorov)
        'intercept' : float
        'r_squared' : float — coefficient of determination
        'k_fit'     : ndarray — wavenumbers used in fit
        'E_fit'     : ndarray — fitted E values
    """
    if k_max is None:
        k_max = k_bins.max() / 2.0

    mask = (k_bins >= k_min) & (k_bins <= k_max) & (E_k > 0)
    if mask.sum() < 3:
        return {"alpha": float("nan"), "intercept": float("nan"),
                "r_squared": float("nan"), "k_fit": np.array([]), "E_fit": np.array([])}

    log_k = np.log(k_bins[mask])
    log_E = np.log(E_k[mask])

    # Linear fit: log_E = alpha * log_k + intercept
    A = np.column_stack([log_k, np.ones_like(log_k)])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_E, rcond=None)
    alpha, intercept = coeffs

    # R²
    log_E_pred = alpha * log_k + intercept
    ss_res = np.sum((log_E - log_E_pred)**2)
    ss_tot = np.sum((log_E - log_E.mean())**2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)

    return {
        "alpha": float(alpha),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "k_fit": k_bins[mask],
        "E_fit": np.exp(log_E_pred),
    }


# ---------------------------------------------------------------------------
# Enstrophy spectrum
# ---------------------------------------------------------------------------

def enstrophy_spectrum_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    L: float = 2 * math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Enstrophy spectrum Ω(k) = k² E(k) (spectral shell binning).

    Returns
    -------
    k_bins, Omega_k
    """
    k_bins, E_k = energy_spectrum_3d(u, v, w, L=L)
    return k_bins, k_bins**2 * E_k


# ---------------------------------------------------------------------------
# Full spectral report
# ---------------------------------------------------------------------------

def spectral_report(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    nu: float,
    L: float = 2 * math.pi,
    t: float = 0.0,
) -> dict:
    """Compute all spectral diagnostics for one snapshot.

    Returns
    -------
    dict with keys:
        t, E_total, epsilon, eta (Kolmogorov scale), alpha (spectral slope),
        r_squared, k_bins, E_k, D_k
    """
    k_bins, E_k = energy_spectrum_3d(u, v, w, L=L)
    k_bins_d, D_k = dissipation_spectrum(u, v, w, nu=nu, L=L)
    epsilon = float(D_k.sum())
    eta = kolmogorov_scale(nu, epsilon)
    slope_fit = kolmogorov_exponent(k_bins, E_k)
    E_total = float(E_k.sum())

    return {
        "t": t,
        "E_total": E_total,
        "epsilon": epsilon,
        "eta": eta,
        "alpha": slope_fit["alpha"],
        "r_squared": slope_fit["r_squared"],
        "k_bins": k_bins.tolist(),
        "E_k": E_k.tolist(),
        "D_k": D_k.tolist(),
    }
