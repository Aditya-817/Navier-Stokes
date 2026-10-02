"""
tests/test_domain.py
=====================
Milestone 1: Unit tests for periodic domain utilities.
"""
import math
import numpy as np
import pytest

from src.utils.domain import (
    cell_centred_grid,
    vertex_grid,
    wavenumbers_1d,
    wavenumber_grid,
    fourier_mode_grid,
    check_periodicity_numpy,
    make_xyzt_tensor,
)

TWO_PI = 2.0 * math.pi


class TestCellCentredGrid:
    def test_shape(self):
        X, Y, Z = cell_centred_grid(16)
        assert X.shape == (16, 16, 16)
        assert Y.shape == (16, 16, 16)
        assert Z.shape == (16, 16, 16)

    def test_range(self):
        N = 16
        X, Y, Z = cell_centred_grid(N)
        assert X.min() > 0.0
        assert X.max() < TWO_PI
        # Cell centres: (i+0.5)*2π/N
        expected_first = 0.5 * TWO_PI / N
        assert abs(X[0, 0, 0] - expected_first) < 1e-14

    def test_asymmetric_N(self):
        X, Y, Z = cell_centred_grid((8, 16, 32))
        assert X.shape == (8, 16, 32)


class TestVertexGrid:
    def test_includes_endpoints(self):
        N = 8
        X, Y, Z = vertex_grid(N)
        assert X.shape == (N + 1, N + 1, N + 1)
        assert abs(X[0, 0, 0]) < 1e-14
        assert abs(X[-1, 0, 0] - TWO_PI) < 1e-14


class TestWavenumbers:
    def test_1d_length(self):
        k = wavenumbers_1d(16)
        assert len(k) == 16

    def test_1d_range(self):
        N = 8
        k = wavenumbers_1d(N)
        assert k.min() == -N // 2
        assert k.max() == N // 2 - 1

    def test_3d_shape(self):
        Kx, Ky, Kz = wavenumber_grid(8)
        assert Kx.shape == (8, 8, 8)


class TestFourierModeGrid:
    def test_no_zero_mode(self):
        modes = fourier_mode_grid(2)
        # No row should be all zeros
        is_zero = np.all(modes == 0, axis=1)
        assert not np.any(is_zero)

    def test_count(self):
        K = 2
        modes = fourier_mode_grid(K)
        # Total modes = (2K+1)^3 - 1 (excluding zero mode)
        expected = (2 * K + 1) ** 3 - 1
        assert len(modes) == expected

    def test_max_component(self):
        K = 3
        modes = fourier_mode_grid(K)
        assert np.max(np.abs(modes)) == K


class TestPeriodicityCheck:
    def test_sin_passes(self):
        # sin(x) is 2π-periodic
        result = check_periodicity_numpy(
            lambda X, Y, Z: np.sin(X) * np.cos(Y),
            N=32, atol=1e-10
        )
        assert result["passed"], f"sin*cos should be periodic: {result}"
        assert result["max_err_x"] < 1e-10

    def test_nonperiodic_fails(self):
        # x is NOT 2π-periodic on [0,2π]
        result = check_periodicity_numpy(
            lambda X, Y, Z: X,
            N=16, atol=1e-10
        )
        assert not result["passed"]
        assert result["max_err_x"] > 0.1

    def test_constant_passes(self):
        result = check_periodicity_numpy(
            lambda X, Y, Z: np.ones_like(X),
            N=16, atol=1e-12
        )
        assert result["passed"]


class TestMakeXYZTTensor:
    def test_shape_and_values(self):
        import torch
        X, Y, Z = cell_centred_grid(4)
        pts = make_xyzt_tensor(X, Y, Z, t=0.5)
        assert pts.shape == (4 ** 3, 4)
        assert torch.all(pts[:, 3] == 0.5)

    def test_dtype(self):
        import torch
        X, Y, Z = cell_centred_grid(4)
        pts = make_xyzt_tensor(X, Y, Z, t=0.0, dtype=torch.float64)
        assert pts.dtype == torch.float64
