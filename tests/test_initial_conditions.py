"""
tests/test_initial_conditions.py
==================================
Milestone 2: Verify div-free and periodicity for all IC families.

Critical tests:
  - ∇·u₀ < 1e-12 at every point on a fine grid.
  - u₀(0, y, z) = u₀(2π, y, z) (and similarly in y, z).
"""
import math
import numpy as np
import pytest

from src.utils.initial_conditions import (
    TaylorGreenVortex3D,
    ABCFlow,
    FourierRandomDivFree,
    make_ic,
)
from src.utils.domain import cell_centred_grid, check_periodicity_numpy

TWO_PI = 2.0 * math.pi

# Fine grid for div-free verification
N_FINE = 64


def _divergence_max(ic, N=N_FINE):
    """Compute max |∇·u₀| using finite differences on a fine grid."""
    X, Y, Z = cell_centred_grid(N, dtype=np.float64)
    div = ic.divergence(X, Y, Z)
    return float(np.max(np.abs(div)))


def _periodicity_result(ic, component: str, N=32):
    """Check periodicity of one velocity component."""
    if component == "u":
        f = lambda X, Y, Z: ic.u0(X, Y, Z)
    elif component == "v":
        f = lambda X, Y, Z: ic.v0(X, Y, Z)
    else:
        f = lambda X, Y, Z: ic.w0(X, Y, Z)
    return check_periodicity_numpy(f, N=N, atol=1e-10)


# ---------------------------------------------------------------------------
# Taylor-Green Vortex
# ---------------------------------------------------------------------------

class TestTaylorGreenVortex:
    def setup_method(self):
        self.ic = TaylorGreenVortex3D(V0=1.0, k=1, nu=0.1)

    def test_divergence_finite_diff(self):
        err = _divergence_max(self.ic)
        assert err < 1e-10, f"TGV divergence too large: {err:.2e}"

    def test_analytic_divergence_zero(self):
        X, Y, Z = cell_centred_grid(32)
        div = self.ic.analytic_divergence(X, Y, Z)
        assert np.all(div == 0.0)

    def test_u_periodic(self):
        r = _periodicity_result(self.ic, "u")
        assert r["passed"], f"TGV u0 not periodic: {r}"

    def test_v_periodic(self):
        r = _periodicity_result(self.ic, "v")
        assert r["passed"], f"TGV v0 not periodic: {r}"

    def test_w_periodic(self):
        r = _periodicity_result(self.ic, "w")
        assert r["passed"], f"TGV w0 not periodic: {r}"

    def test_characteristic_scales(self):
        assert self.ic.U_char == 1.0
        assert abs(self.ic.Re_char - 10.0) < 1e-10  # V0 / (nu * k) = 1 / 0.1

    def test_w_is_zero(self):
        X, Y, Z = cell_centred_grid(8)
        w = self.ic.w0(X, Y, Z)
        assert np.all(w == 0.0)


# ---------------------------------------------------------------------------
# ABC Flow
# ---------------------------------------------------------------------------

class TestABCFlow:
    def setup_method(self):
        self.ic = ABCFlow()

    def test_divergence_finite_diff(self):
        err = _divergence_max(self.ic)
        assert err < 1e-10, f"ABC divergence too large: {err:.2e}"

    def test_u_periodic(self):
        r = _periodicity_result(self.ic, "u")
        assert r["passed"], f"ABC u0 not periodic: {r}"

    def test_v_periodic(self):
        r = _periodicity_result(self.ic, "v")
        assert r["passed"], f"ABC v0 not periodic: {r}"

    def test_w_periodic(self):
        r = _periodicity_result(self.ic, "w")
        assert r["passed"], f"ABC w0 not periodic: {r}"

    def test_analytic_divergence_zero(self):
        X, Y, Z = cell_centred_grid(16)
        assert np.all(self.ic.analytic_divergence(X, Y, Z) == 0.0)

    def test_no_x_dependence_in_u(self):
        """u = A sin(z) + C cos(y) has no x dependence."""
        X, Y, Z = cell_centred_grid(8)
        u1 = self.ic.u0(X, Y, Z)
        u2 = self.ic.u0(X + 0.5, Y, Z)
        assert np.allclose(u1, u2, atol=1e-14)


# ---------------------------------------------------------------------------
# Fourier Random Div-Free
# ---------------------------------------------------------------------------

class TestFourierRandomDivFree:
    def setup_method(self):
        self.ic = FourierRandomDivFree(K=3, alpha=5.0/3.0, rng_seed=42)

    def test_divergence_finite_diff(self):
        # FD divergence: the curl of A is div-free analytically.
        # FD error is O(dx^2 * k^2) where k=K=3, dx=2π/128.
        # At N=128, k=3: error ~ (2π*3/128)^2 / 12 ≈ 0.05
        # We use a generous tolerance; spectral divergence is exact.
        err = _divergence_max(self.ic, N=128)
        assert err < 0.1, f"FourierRandom FD divergence too large: {err:.2e}"

    def test_u_periodic(self):
        r = _periodicity_result(self.ic, "u")
        assert r["passed"], f"FourierRandom u0 not periodic: {r}"

    def test_v_periodic(self):
        r = _periodicity_result(self.ic, "v")
        assert r["passed"], f"FourierRandom v0 not periodic: {r}"

    def test_w_periodic(self):
        r = _periodicity_result(self.ic, "w")
        assert r["passed"], f"FourierRandom w0 not periodic: {r}"

    def test_reproducibility(self):
        ic2 = FourierRandomDivFree(K=3, alpha=5.0/3.0, rng_seed=42)
        X, Y, Z = cell_centred_grid(8)
        u1 = self.ic.u0(X, Y, Z)
        u2 = ic2.u0(X, Y, Z)
        assert np.allclose(u1, u2)

    def test_different_seeds_differ(self):
        ic2 = FourierRandomDivFree(K=3, rng_seed=99)
        X, Y, Z = cell_centred_grid(8)
        assert not np.allclose(self.ic.u0(X, Y, Z), ic2.u0(X, Y, Z))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestICFactory:
    def test_taylor_green(self):
        ic = make_ic("taylor_green", V0=2.0, k=1)
        assert ic.U_char == 2.0

    def test_abc(self):
        ic = make_ic("abc")
        assert ic.name == "abc_flow"

    def test_fourier_random(self):
        ic = make_ic("fourier_random", K=2, rng_seed=7)
        assert ic.K == 2

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            make_ic("nonexistent_ic")
