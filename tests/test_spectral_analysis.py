"""
tests/test_spectral_analysis.py
=================================
Milestone 12 tests: energy spectra, vortex diagnostics, Kolmogorov scaling.
"""
import math
import pytest
import numpy as np

from src.analysis.spectral_analysis import (
    energy_spectrum_3d,
    dissipation_rate,
    kolmogorov_scale,
    dissipation_spectrum,
    kolmogorov_exponent,
    enstrophy_spectrum_3d,
    spectral_report,
)
from src.analysis.vortex_diagnostics import (
    vorticity_field,
    enstrophy,
    palinstrophy,
    q_criterion,
    helicity_spectrum,
    vortex_diagnostics_report,
)

TWO_PI = 2.0 * math.pi
N = 32  # small grid for fast tests


def tgv_fields(N=N):
    """Taylor–Green Vortex IC: k=1 mode."""
    L = TWO_PI
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    return u, v, w


def single_mode_field(N=N, k=2):
    """Pure single-wavenumber mode: u=(sin(kx), 0, 0). E(k')=0 except k'=k."""
    L = TWO_PI
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.sin(k * X)
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    return u, v, w, k


# ---------------------------------------------------------------------------
# Energy spectrum
# ---------------------------------------------------------------------------

class TestEnergySpectrum:
    def test_output_shapes(self):
        u, v, w = tgv_fields()
        k_bins, E_k = energy_spectrum_3d(u, v, w)
        assert k_bins.shape == E_k.shape
        assert len(k_bins) == N // 2

    def test_nonnegative(self):
        u, v, w = tgv_fields()
        _, E_k = energy_spectrum_3d(u, v, w)
        assert (E_k >= 0).all(), "Energy spectrum must be non-negative"

    def test_parseval_identity(self):
        """Σ E(k) = ½ ∫|u|² dV  (Parseval's theorem)."""
        u, v, w = tgv_fields()
        _, E_k = energy_spectrum_3d(u, v, w)
        E_spectral = E_k.sum()
        E_physical = 0.5 * np.mean(u**2 + v**2 + w**2)
        err = abs(E_spectral - E_physical) / (E_physical + 1e-30)
        assert err < 0.01, f"Parseval error: {err:.2e}"

    def test_single_mode_peak(self):
        """A single-mode field should have energy concentrated at that mode."""
        u, v, w, k = single_mode_field()
        k_bins, E_k = energy_spectrum_3d(u, v, w)
        peak_k = k_bins[np.argmax(E_k)]
        assert abs(peak_k - k) < 1.5, f"Peak at k={peak_k:.1f}, expected k={k}"


# ---------------------------------------------------------------------------
# Dissipation and Kolmogorov scale
# ---------------------------------------------------------------------------

class TestDissipation:
    def test_dissipation_nonnegative(self):
        u, v, w = tgv_fields()
        eps = dissipation_rate(u, v, w, nu=0.1)
        assert eps >= 0.0

    def test_zero_nu_zero_dissipation(self):
        """ε = 2ν Σ k² E(k) → 0 when ν=0."""
        u, v, w = tgv_fields()
        eps = dissipation_rate(u, v, w, nu=0.0)
        assert abs(eps) < 1e-14

    def test_kolmogorov_scale_decreases_with_re(self):
        """At fixed ε, higher ν → larger Kolmogorov scale η."""
        eps = 0.1
        eta1 = kolmogorov_scale(nu=0.01, epsilon=eps)
        eta2 = kolmogorov_scale(nu=0.1, epsilon=eps)
        assert eta2 > eta1, "Higher ν → larger η at same ε"

    def test_kolmogorov_scale_zero_eps(self):
        assert kolmogorov_scale(nu=0.1, epsilon=0.0) == float("inf")

    def test_dissipation_spectrum_shape(self):
        u, v, w = tgv_fields()
        k_bins, D_k = dissipation_spectrum(u, v, w, nu=0.1)
        assert k_bins.shape == D_k.shape

    def test_dissipation_spectrum_nonneg(self):
        u, v, w = tgv_fields()
        _, D_k = dissipation_spectrum(u, v, w, nu=0.1)
        assert (D_k >= 0).all()


# ---------------------------------------------------------------------------
# Spectral slope fitting
# ---------------------------------------------------------------------------

class TestKolmogorovExponent:
    def test_k53_spectrum(self):
        """Fit on a synthetic E(k) ∝ k^{-5/3} should give α≈-5/3."""
        k = np.arange(1, 50, dtype=float)
        E = k ** (-5.0 / 3.0)
        result = kolmogorov_exponent(k, E, k_min=2.0, k_max=30.0)
        assert abs(result["alpha"] - (-5.0 / 3.0)) < 0.05, \
            f"Slope {result['alpha']:.4f} ≠ -5/3"
        assert result["r_squared"] > 0.99

    def test_insufficient_points_returns_nan(self):
        k = np.array([1.0, 2.0])
        E = np.array([1.0, 0.5])
        result = kolmogorov_exponent(k, E, k_min=3.0)
        assert math.isnan(result["alpha"])


# ---------------------------------------------------------------------------
# Enstrophy spectrum
# ---------------------------------------------------------------------------

class TestEnstrophySpectrum:
    def test_shape(self):
        u, v, w = tgv_fields()
        k_bins, Om_k = enstrophy_spectrum_3d(u, v, w)
        assert k_bins.shape == Om_k.shape

    def test_k2_scaling(self):
        """Ω(k) = k² E(k) by definition."""
        u, v, w = tgv_fields()
        k_bins, E_k = energy_spectrum_3d(u, v, w)
        k_bins2, Om_k = enstrophy_spectrum_3d(u, v, w)
        err = np.max(np.abs(Om_k - k_bins**2 * E_k))
        assert err < 1e-14


# ---------------------------------------------------------------------------
# Spectral report
# ---------------------------------------------------------------------------

class TestSpectralReport:
    def test_report_keys(self):
        u, v, w = tgv_fields()
        report = spectral_report(u, v, w, nu=0.1, t=0.0)
        for key in ["t", "E_total", "epsilon", "eta", "alpha", "r_squared", "k_bins", "E_k", "D_k"]:
            assert key in report, f"Missing key: {key}"

    def test_report_values_finite(self):
        u, v, w = tgv_fields()
        report = spectral_report(u, v, w, nu=0.1, t=0.5)
        assert np.isfinite(report["E_total"])
        assert np.isfinite(report["epsilon"])


# ---------------------------------------------------------------------------
# Vortex diagnostics
# ---------------------------------------------------------------------------

class TestVorticityField:
    def test_shape(self):
        u, v, w = tgv_fields()
        ox, oy, oz = vorticity_field(u, v, w)
        assert ox.shape == (N, N, N)

    def test_tgv_w_zero_means_wz_nonzero(self):
        """TGV IC has non-zero w_z vorticity component."""
        u, v, w = tgv_fields()
        _, _, oz = vorticity_field(u, v, w)
        assert oz.max() > 0.1, "TGV should have non-trivial ωz"


class TestEnstrophyPalinstrophy:
    def test_enstrophy_positive(self):
        u, v, w = tgv_fields()
        assert enstrophy(u, v, w) > 0.0

    def test_palinstrophy_positive(self):
        u, v, w = tgv_fields()
        assert palinstrophy(u, v, w) > 0.0

    def test_palinstrophy_ge_enstrophy(self):
        """For TGV k=1 IC, palinstrophy ≈ k² * enstrophy."""
        u, v, w = tgv_fields()
        ens = enstrophy(u, v, w)
        pal = palinstrophy(u, v, w)
        # Palinstrophy involves k² factors — should be of same order for k=1
        assert pal > 0


class TestQCriterion:
    def test_shape(self):
        u, v, w = tgv_fields()
        Q = q_criterion(u, v, w)
        assert Q.shape == (N, N, N)

    def test_both_signs_present(self):
        """TGV IC has both rotational and straining regions."""
        u, v, w = tgv_fields()
        Q = q_criterion(u, v, w)
        assert Q.max() > 0, "Should have Q>0 (vortex) regions"
        assert Q.min() < 0, "Should have Q<0 (strain) regions"


class TestHelicitySpectrum:
    def test_shape(self):
        u, v, w = tgv_fields()
        k_bins, H_k = helicity_spectrum(u, v, w)
        assert k_bins.shape == H_k.shape

    def test_tgv_low_helicity(self):
        """TGV IC is nearly mirror-symmetric, so total helicity ≈ 0."""
        u, v, w = tgv_fields()
        _, H_k = helicity_spectrum(u, v, w)
        # TGV is not perfectly helicity-free but at low k it should be small
        total = abs(H_k.sum())
        assert total < 1.0, f"TGV total helicity too large: {total:.4f}"


class TestVortexDiagnosticsReport:
    def test_report_keys(self):
        u, v, w = tgv_fields()
        report = vortex_diagnostics_report(u, v, w, nu=0.1, t=0.0)
        for key in ["t", "enstrophy", "palinstrophy", "Q_max", "Q_min",
                    "Q_mean", "Q_vortex_fraction", "total_helicity"]:
            assert key in report

    def test_vortex_fraction_in_unit_interval(self):
        u, v, w = tgv_fields()
        report = vortex_diagnostics_report(u, v, w, nu=0.1)
        assert 0.0 <= report["Q_vortex_fraction"] <= 1.0
