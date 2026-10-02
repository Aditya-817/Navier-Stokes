"""
tests/test_ns_residual.py
==========================
Milestone 4: Critical test — NS residuals evaluated on exact analytic
solutions must be zero (to floating-point precision).

All coordinate variables are flat (N,) tensors from split_xyzt.
Fields are built as differentiable functions of these leaf tensors.
"""
import math
import pytest
import torch
import numpy as np

from src.physics.ns_residual import ns_residuals, NSResiduals
from src.physics.derivatives import split_xyzt

TWO_PI = 2.0 * math.pi
N = 100  # test points

ATOL_F64 = 1e-8   # residual abs tolerance on float64 exact solutions


def random_spacetime_pts(N=N, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    pts = torch.rand(N, 4, dtype=dtype) * TWO_PI
    return pts


# ---------------------------------------------------------------------------
# Continuity on ABC flow (R_c should be zero for any div-free field)
# ---------------------------------------------------------------------------

class TestContinuityOnABC:
    """R_c should be zero for any div-free field."""

    def test_abc_continuity_f64(self):
        """ABC flow is analytically div-free: R_c should vanish."""
        pts = random_spacetime_pts(dtype=torch.float64)
        x, y, z, t = split_xyzt(pts)  # flat (N,) leaves

        A = math.sqrt(6.0)
        B = math.sqrt(2.0)
        C = math.sqrt(3.0)

        # Each component depends on leaf tensors through differentiable ops
        u = A * torch.sin(z) + C * torch.cos(y)
        v = B * torch.sin(x) + A * torch.cos(z)
        w = C * torch.sin(y) + B * torch.cos(x)
        p = u * 0.0  # graph-connected zero

        res = ns_residuals(u, v, w, p, x, y, z, t, nu=0.0)
        R_c = res.R_c
        err = R_c.abs().max().item()
        assert err < ATOL_F64, f"ABC continuity residual: {err:.2e}"


# ---------------------------------------------------------------------------
# Manufactured solution: verify NS residuals match analytic form
# ---------------------------------------------------------------------------

class TestManufacturedSolutionNS:
    """
    Manufactured solution:
        u = sin(x - t),  v = 0,  w = 0,  p = 0

    R_u = u_t + u*u_x + p_x - ν*Δu
         = -cos(x-t) + sin(x-t)*cos(x-t) + ν*sin(x-t)
    R_v = R_w = 0
    R_c = u_x = cos(x-t)    (NOT div-free — intentional for residual testing)
    """

    def test_known_residual_f64(self):
        pts = random_spacetime_pts(N=200, dtype=torch.float64)
        x, y, z, t = split_xyzt(pts)

        nu = 0.05
        arg = x - t
        u = torch.sin(arg)
        v = u * 0.0   # graph-connected zero
        w = u * 0.0   # graph-connected zero
        p = u * 0.0   # graph-connected zero

        res = ns_residuals(u, v, w, p, x, y, z, t, nu=nu)

        # Analytic residuals
        R_u_expected = (-torch.cos(arg) + torch.sin(arg) * torch.cos(arg)
                        + nu * torch.sin(arg)).detach()
        R_c_expected = torch.cos(arg).detach()

        err_Ru = (res.R_u.detach() - R_u_expected).abs().max().item()
        err_Rv = res.R_v.detach().abs().max().item()
        err_Rw = res.R_w.detach().abs().max().item()
        err_Rc = (res.R_c.detach() - R_c_expected).abs().max().item()

        assert err_Ru < 1e-8, f"R_u err: {err_Ru:.2e}"
        assert err_Rv < 1e-10, f"R_v err: {err_Rv:.2e}"
        assert err_Rw < 1e-10, f"R_w err: {err_Rw:.2e}"
        assert err_Rc < 1e-8, f"R_c err: {err_Rc:.2e}"


# ---------------------------------------------------------------------------
# TGV IC as exact NS solution — verify R_c = 0
# ---------------------------------------------------------------------------

class TestTGVContinuity:
    def test_tgv_r_c_is_zero(self):
        """TGV IC is div-free, so R_c = u_x + v_y + w_z = 0."""
        pts = random_spacetime_pts(N=200, dtype=torch.float64)
        x, y, z, t = split_xyzt(pts)

        k = 1
        u = torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
        v = -torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
        w = u * 0.0   # TGV w=0, but graph-connected
        p = u * 0.0   # graph-connected zero pressure

        res = ns_residuals(u, v, w, p, x, y, z, t, nu=0.1)
        err = res.R_c.abs().max().item()
        assert err < 1e-9, f"TGV R_c: {err:.2e}"


# ---------------------------------------------------------------------------
# NSResiduals helpers
# ---------------------------------------------------------------------------

class TestNSResidualsHelpers:
    def test_momentum_mse_positive(self):
        pts = random_spacetime_pts(N=50)
        x, y, z, t = split_xyzt(pts)
        u = torch.sin(x); v = torch.cos(x); w = u * 0.0; p = u * 0.0
        res = ns_residuals(u, v, w, p, x, y, z, t, nu=0.1)
        mse = res.momentum_mse()
        assert mse.item() >= 0.0

    def test_as_dict_keys(self):
        pts = random_spacetime_pts(N=10)
        x, y, z, t = split_xyzt(pts)
        # All fields must be connected to graph
        u = x * 0.0 + y * 0.0 + z * 0.0 + t * 0.0
        v = u.clone()
        w = u.clone()
        p = u.clone()
        res = ns_residuals(u, v, w, p, x, y, z, t, nu=0.0)
        d = res.mse_dict()
        assert set(d.keys()) == {"R_u", "R_v", "R_w", "R_c"}
