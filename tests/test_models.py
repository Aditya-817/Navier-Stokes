"""
tests/test_models.py
=====================
Milestone 5: Model shape, dtype, and periodicity tests.

Critical test: every output component of FourierResMLP must be
exactly 2π-periodic in x, y, z for any t.
"""
import math
import pytest
import torch
import numpy as np

from src.models.fourier_resmlp import FourierResMLP, StandardMLP
from src.models.factory import make_model
from src.utils.domain import check_periodicity_torch

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# FourierResMLP — shape and dtype
# ---------------------------------------------------------------------------

class TestFourierResMLP:
    def setup_method(self):
        self.model = FourierResMLP(
            fourier_modes=4,
            hidden_width=64,
            n_layers=2,
            activation="tanh",
            use_layernorm=True,
            dtype="float64",
        )

    def test_output_shape(self):
        pts = torch.rand(100, 4, dtype=torch.float64) * TWO_PI
        out = self.model(pts)
        assert out.shape == (100, 4)

    def test_output_dtype(self):
        pts = torch.rand(50, 4, dtype=torch.float64)
        out = self.model(pts)
        assert out.dtype == torch.float64

    def test_n_params_positive(self):
        assert self.model.n_params > 0

    def test_config_dict_roundtrip(self):
        cfg = self.model.config_dict()
        assert cfg["fourier_modes"] == 4
        assert cfg["hidden_width"] == 64
        assert cfg["n_layers"] == 2

    def test_no_nan_output(self):
        pts = torch.rand(200, 4, dtype=torch.float64) * TWO_PI
        out = self.model(pts)
        assert not torch.isnan(out).any()

    # ------------------------------------------------------------------
    # Critical: structural periodicity in x, y, z
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("component", [0, 1, 2, 3], ids=["u", "v", "w", "p"])
    def test_periodicity_component(self, component):
        """Each output must be exactly 2π-periodic in x, y, z."""
        self.model.eval()
        result = check_periodicity_torch(
            model_fn=self.model,
            component_idx=component,
            N=128,
            L=TWO_PI,
            t_val=0.5,
            atol=1e-6,   # float64 model: machine precision expected
            dtype=torch.float64,
        )
        assert result["passed"], (
            f"Component {component} not periodic: "
            f"x_err={result['max_err_x']:.2e} "
            f"y_err={result['max_err_y']:.2e} "
            f"z_err={result['max_err_z']:.2e}"
        )


# ---------------------------------------------------------------------------
# FourierResMLP with float32
# ---------------------------------------------------------------------------

class TestFourierResMLPFloat32:
    def test_float32_runs(self):
        model = FourierResMLP(
            fourier_modes=3, hidden_width=32, n_layers=1, dtype="float32"
        )
        pts = torch.rand(50, 4, dtype=torch.float32) * TWO_PI
        out = model(pts)
        assert out.shape == (50, 4)
        assert out.dtype == torch.float32

    def test_periodicity_f32(self):
        model = FourierResMLP(
            fourier_modes=3, hidden_width=32, n_layers=1, dtype="float32"
        )
        model.eval()
        result = check_periodicity_torch(
            model_fn=model,
            component_idx=0,
            N=64,
            L=TWO_PI,
            t_val=0.3,
            atol=1e-5,
            dtype=torch.float32,
        )
        assert result["passed"], f"float32 periodicity: {result}"


# ---------------------------------------------------------------------------
# StandardMLP
# ---------------------------------------------------------------------------

class TestStandardMLP:
    def test_output_shape(self):
        model = StandardMLP(hidden_width=64, n_layers=2, dtype="float64")
        pts = torch.rand(30, 4, dtype=torch.float64)
        out = model(pts)
        assert out.shape == (30, 4)

    def test_no_nan(self):
        model = StandardMLP(hidden_width=32, n_layers=2, dtype="float64")
        pts = torch.rand(100, 4, dtype=torch.float64) * TWO_PI
        assert not torch.isnan(model(pts)).any()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_fourier_resmlp_from_config(self):
        cfg = {
            "type": "fourier_resmlp",
            "fourier_modes": 4,
            "hidden_width": 32,
            "n_layers": 1,
            "dtype": "float64",
        }
        model = make_model(cfg)
        assert isinstance(model, FourierResMLP)

    def test_standard_mlp_from_config(self):
        cfg = {"type": "standard_mlp", "hidden_width": 32, "n_layers": 2, "dtype": "float64"}
        model = make_model(cfg)
        assert isinstance(model, StandardMLP)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            make_model({"type": "nonexistent_arch"})
