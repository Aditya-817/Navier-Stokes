"""
tests/test_derivatives.py
===========================
Milestone 3: AD derivative engine correctness tests.

Tests verify each operator against known analytic functions to
machine (or near-machine) precision in both float32 and float64.

All coordinates from split_xyzt are flat tensors of shape (N,).

Tolerances
----------
  float64: absolute error < 1e-9
  float32: absolute error < 1e-4  (reduced precision expected)
"""
import math
import pytest
import torch
import numpy as np

from src.physics.derivatives import (
    partial,
    partial2,
    laplacian_scalar,
    divergence,
    curl3d,
    jacobian_row,
    split_xyzt,
)

DTYPES = [
    pytest.param(torch.float64, id="f64"),
    pytest.param(torch.float32, id="f32"),
]

TOL = {torch.float64: 1e-9, torch.float32: 1e-4}

TWO_PI = 2.0 * math.pi
N = 200  # number of random test points


def random_pts(dtype):
    torch.manual_seed(0)
    pts = torch.rand(N, 4, dtype=dtype) * TWO_PI
    return pts


def _make_vars(pts):
    """Return flat (N,) leaf tensors for x, y, z, t."""
    return split_xyzt(pts)


# ---------------------------------------------------------------------------
# ∂f/∂x
# ---------------------------------------------------------------------------

class TestPartial:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_d_sinx_dx(self, dtype):
        """∂ sin(x) / ∂x = cos(x)"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        f = torch.sin(x)
        df = partial(f, x)
        expected = torch.cos(x).detach()
        err = (df.detach() - expected).abs().max().item()
        assert err < TOL[dtype], f"dtype={dtype} err={err:.2e}"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_d_xsquared_dx(self, dtype):
        """∂ x² / ∂x = 2x"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        f = x ** 2
        df = partial(f, x)
        expected = (2.0 * x).detach()
        err = (df.detach() - expected).abs().max().item()
        assert err < TOL[dtype]

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_partial_wrt_t(self, dtype):
        """∂ (t²) / ∂t = 2t"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        f = t ** 2
        df = partial(f, t)
        expected = (2.0 * t).detach()
        err = (df.detach() - expected).abs().max().item()
        assert err < TOL[dtype]


# ---------------------------------------------------------------------------
# ∂²f/∂x²
# ---------------------------------------------------------------------------

class TestPartial2:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_d2_sinx(self, dtype):
        """∂² sin(x) / ∂x² = -sin(x)"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        f = torch.sin(x)
        d2f = partial2(f, x)
        expected = (-torch.sin(x)).detach()
        err = (d2f.detach() - expected).abs().max().item()
        assert err < TOL[dtype] * 10  # two passes — slightly higher tol

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_d2_polynomial(self, dtype):
        """∂² (x³) / ∂x² = 6x"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        f = x ** 3
        d2f = partial2(f, x)
        expected = (6.0 * x).detach()
        err = (d2f.detach() - expected).abs().max().item()
        assert err < TOL[dtype] * 10


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

class TestLaplacian:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_laplacian_sin3(self, dtype):
        """Δ [sin(x)sin(y)sin(z)] = -3 sin(x)sin(y)sin(z)"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        f = torch.sin(x) * torch.sin(y) * torch.sin(z)
        lap = laplacian_scalar(f, [x, y, z])
        expected = (-3.0 * f).detach()
        err = (lap.detach() - expected).abs().max().item()
        tol = TOL[dtype] * 100  # three second-order passes
        assert err < tol, f"Laplacian err={err:.2e} tol={tol:.2e}"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_laplacian_quadratic(self, dtype):
        """Δ (x² + y² + z²) = 6"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        f = x**2 + y**2 + z**2
        lap = laplacian_scalar(f, [x, y, z])
        expected = torch.full_like(f.detach(), 6.0)
        err = (lap.detach() - expected).abs().max().item()
        assert err < TOL[dtype] * 100


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

class TestDivergence:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_div_sinusoidal(self, dtype):
        """∇·(sin(x), sin(y), sin(z)) = cos(x) + cos(y) + cos(z)"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        Fx = torch.sin(x)
        Fy = torch.sin(y)
        Fz = torch.sin(z)
        div = divergence([Fx, Fy, Fz], [x, y, z])
        expected = (torch.cos(x) + torch.cos(y) + torch.cos(z)).detach()
        err = (div.detach() - expected).abs().max().item()
        assert err < TOL[dtype], f"div err={err:.2e}"


# ---------------------------------------------------------------------------
# Curl
# ---------------------------------------------------------------------------

class TestCurl:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_curl_zero_for_gradient(self, dtype):
        """∇×(∇f) = 0 for any scalar f (curl of gradient = 0)."""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        f = torch.sin(x) * torch.cos(y) * z
        Fx = partial(f, x)
        Fy = partial(f, y)
        Fz = partial(f, z)
        wx, wy, wz = curl3d([Fx, Fy, Fz], [x, y, z])
        tol = TOL[dtype] * 1000  # three separate second-order terms
        assert wx.abs().max().item() < tol, f"curl_x: {wx.abs().max().item():.2e}"
        assert wy.abs().max().item() < tol, f"curl_y: {wy.abs().max().item():.2e}"
        assert wz.abs().max().item() < tol, f"curl_z: {wz.abs().max().item():.2e}"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_curl_known(self, dtype):
        """∇×(sin(y), sin(z), sin(x)) = (-cos(z), -cos(x), -cos(y))"""
        pts = random_pts(dtype)
        x, y, z, t = _make_vars(pts)
        Fx = torch.sin(y)   # depends on y leaf
        Fy = torch.sin(z)   # depends on z leaf
        Fz = torch.sin(x)   # depends on x leaf
        wx, wy, wz = curl3d([Fx, Fy, Fz], [x, y, z])
        # dFz/dy = 0, dFy/dz = cos(z)  → wx = -cos(z)
        # dFx/dz = 0, dFz/dx = cos(x)  → wy = -cos(x)
        # dFy/dx = 0, dFx/dy = cos(y)  → wz = -cos(y)
        err_x = (wx.detach() - (-torch.cos(z).detach())).abs().max().item()
        err_y = (wy.detach() - (-torch.cos(x).detach())).abs().max().item()
        err_z = (wz.detach() - (-torch.cos(y).detach())).abs().max().item()
        assert err_x < TOL[dtype], f"curl_x err: {err_x:.2e}"
        assert err_y < TOL[dtype], f"curl_y err: {err_y:.2e}"
        assert err_z < TOL[dtype], f"curl_z err: {err_z:.2e}"


# ---------------------------------------------------------------------------
# split_xyzt
# ---------------------------------------------------------------------------

class TestSplitXYZT:
    def test_shapes(self):
        pts = torch.rand(50, 4)
        x, y, z, t = split_xyzt(pts)
        assert x.shape == (50,)   # flat (N,) tensors
        assert x.requires_grad

    def test_values(self):
        pts = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        x, y, z, t = split_xyzt(pts)
        assert float(x[0].detach()) == pytest.approx(1.0)
        assert float(t[0].detach()) == pytest.approx(4.0)
