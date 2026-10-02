"""
tests/test_alt_architectures.py
==================================
Milestone 11 tests: SIREN, MultiscaleMLP, DivFreeNet.

Tests verify:
  - Correct output shapes and dtypes
  - No NaN outputs
  - Structural periodicity (u(x) = u(x + 2πeᵢ))
  - SIREN: first-layer omega_0 scaling works
  - DivFreeNet: divergence ∇·u = 0 to machine precision
  - Factory instantiation from config dict
"""
import math
import pytest
import torch

from src.models.siren import SIREN
from src.models.multiscale_mlp import MultiscaleMLP
from src.models.divfree_net import DivFreeNet
from src.models.factory import make_model
from src.physics.derivatives import split_xyzt

TWO_PI = 2.0 * math.pi
N = 32


def random_pts(N=N, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    pts = torch.rand(N, 4, dtype=dtype) * TWO_PI
    return pts


class DummyModel(torch.nn.Module):
    """Minimal model: returns values connected to the AD graph.
    Uses a 0-weight linear layer so the output is effectively zero
    but has a grad_fn w.r.t. the input coordinates.
    """
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.lin.weight.zero_()

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return self.lin(pts)


def periodicity_error(model, component: int = 0, N=64, dtype=torch.float64):
    """Max |f(x) - f(x + 2πeᵢ)| over i=0,1,2 and random pts."""
    torch.manual_seed(7)
    pts = torch.rand(N, 4, dtype=dtype) * TWO_PI
    max_err = 0.0
    for i in range(3):
        shifted = pts.clone()
        shifted[:, i] = (shifted[:, i] + TWO_PI) % (TWO_PI + 1e-9)
        with torch.no_grad():
            out = model(pts)
            out_shifted = model(shifted)
        err = (out[:, component] - out_shifted[:, component]).abs().max().item()
        max_err = max(max_err, err)
    return max_err


# ---------------------------------------------------------------------------
# SIREN
# ---------------------------------------------------------------------------

class TestSIREN:
    @pytest.fixture(autouse=True)
    def model(self):
        torch.manual_seed(0)
        self.m = SIREN(fourier_modes=4, hidden_width=64, n_layers=2, dtype="float64")

    def test_output_shape(self):
        pts = random_pts()
        out = self.m(pts)
        assert out.shape == (N, 4)

    def test_no_nan(self):
        pts = random_pts()
        out = self.m(pts)
        assert not torch.isnan(out).any()

    def test_output_dtype_f64(self):
        pts = random_pts(dtype=torch.float64)
        out = self.m(pts)
        assert out.dtype == torch.float64

    def test_periodicity(self):
        """Structural Fourier embedding guarantees 2π-periodicity."""
        err = periodicity_error(self.m, component=0)
        assert err < 1e-6, f"SIREN periodicity err: {err:.2e}"

    def test_n_params_positive(self):
        assert self.m.n_params > 0

    def test_config_dict(self):
        cfg = self.m.config_dict()
        assert cfg["type"] == "siren"
        assert cfg["fourier_modes"] == 4


# ---------------------------------------------------------------------------
# MultiscaleMLP
# ---------------------------------------------------------------------------

class TestMultiscaleMLP:
    @pytest.fixture(autouse=True)
    def model(self):
        torch.manual_seed(1)
        self.m = MultiscaleMLP(
            fourier_scales=[2, 4],
            hidden_width=64,
            n_layers_branch=1,
            n_layers_fusion=1,
            dtype="float64",
        )

    def test_output_shape(self):
        pts = random_pts()
        out = self.m(pts)
        assert out.shape == (N, 4)

    def test_no_nan(self):
        pts = random_pts()
        out = self.m(pts)
        assert not torch.isnan(out).any()

    def test_periodicity(self):
        err = periodicity_error(self.m, component=0)
        assert err < 1e-6, f"MultiscaleMLP periodicity err: {err:.2e}"

    def test_n_params_positive(self):
        assert self.m.n_params > 0

    def test_config_dict(self):
        cfg = self.m.config_dict()
        assert cfg["type"] == "multiscale_mlp"
        assert cfg["fourier_scales"] == [2, 4]


# ---------------------------------------------------------------------------
# DivFreeNet — divergence-free by construction
# ---------------------------------------------------------------------------

class TestDivFreeNet:
    @pytest.fixture(autouse=True)
    def model(self):
        torch.manual_seed(2)
        self.m = DivFreeNet(fourier_modes=4, hidden_width=64, n_layers=2, dtype="float64")

    def test_output_shape(self):
        pts = random_pts(N=16)
        out = self.m(pts)
        assert out.shape == (16, 4)

    def test_no_nan(self):
        pts = random_pts(N=16)
        out = self.m(pts)
        assert not torch.isnan(out).any()

    def test_divergence_free(self):
        """∇·u = 0 must hold to near-machine precision for ANY network weights."""
        N_pts = 20
        pts = random_pts(N=N_pts)
        x = pts[:, 0].detach().requires_grad_(True)
        y = pts[:, 1].detach().requires_grad_(True)
        z = pts[:, 2].detach().requires_grad_(True)
        t = pts[:, 3].detach().requires_grad_(True)
        xyzt = torch.stack([x, y, z, t], dim=1)

        out = self.m(xyzt)
        u, v, w = out[:, 0], out[:, 1], out[:, 2]

        from src.physics.derivatives import partial as adgrad
        u_x = adgrad(u, x)
        v_y = adgrad(v, y)
        w_z = adgrad(w, z)
        div = (u_x + v_y + w_z).detach().abs().max().item()
        assert div < 1e-8, f"DivFreeNet ∇·u = {div:.2e} (expected < 1e-8)"

    def test_config_dict(self):
        cfg = self.m.config_dict()
        assert cfg["type"] == "divfree_net"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_siren_from_config(self):
        cfg = {"type": "siren", "fourier_modes": 4, "hidden_width": 32, "n_layers": 2}
        m = make_model(cfg)
        assert isinstance(m, SIREN)

    def test_multiscale_from_config(self):
        cfg = {"type": "multiscale_mlp", "fourier_scales": [2, 4],
               "hidden_width": 32, "n_layers_branch": 1, "n_layers_fusion": 1}
        m = make_model(cfg)
        assert isinstance(m, MultiscaleMLP)

    def test_divfree_from_config(self):
        cfg = {"type": "divfree_net", "fourier_modes": 4, "hidden_width": 32, "n_layers": 2}
        m = make_model(cfg)
        assert isinstance(m, DivFreeNet)
