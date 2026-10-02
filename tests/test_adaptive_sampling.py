"""
tests/test_adaptive_sampling.py
=================================
Milestone 9 tests: adaptive sampling and curriculum training.
"""
import math
import pytest
import torch

from src.sampling.adaptive import AdaptiveSampler
from src.training.curriculum import (
    TimeWindowScheduler,
    causal_weights,
    apply_causal_weights,
)

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Causal weights
# ---------------------------------------------------------------------------

class TestCausalWeights:
    def test_monotone_decay(self):
        """Weights must decrease with time (causal)."""
        t = torch.linspace(0, 1.0, 100)
        w = causal_weights(t, t_max=1.0, eps=2.0)
        assert (w[1:] <= w[:-1]).all(), "Causal weights must be monotone decreasing"

    def test_boundary_values(self):
        """w(0) ≈ 1, w(T) ≈ exp(-eps)."""
        t = torch.tensor([0.0, 1.0])
        w = causal_weights(t, t_max=1.0, eps=1.0)
        assert abs(float(w[0]) - 1.0) < 1e-6
        assert abs(float(w[1]) - math.exp(-1.0)) < 1e-6

    def test_larger_eps_faster_decay(self):
        """Higher eps → faster causal decay → smaller weights at t=T."""
        t = torch.tensor([1.0])
        w1 = causal_weights(t, t_max=1.0, eps=1.0)
        w2 = causal_weights(t, t_max=1.0, eps=3.0)
        assert float(w2) < float(w1), "Higher eps must give smaller weight at t=T"

    def test_apply_causal_weights_returns_scalar(self):
        N = 50
        torch.manual_seed(0)
        t = torch.rand(N) * 1.0
        r1 = torch.randn(N)
        r2 = torch.randn(N)
        r3 = torch.randn(N)
        loss = apply_causal_weights([r1, r2, r3], t, t_max=1.0, eps=2.0)
        assert loss.ndim == 0, "apply_causal_weights must return a scalar"
        assert float(loss) >= 0.0


# ---------------------------------------------------------------------------
# Time-window scheduler
# ---------------------------------------------------------------------------

class TestTimeWindowScheduler:
    def test_window_count(self):
        sched = TimeWindowScheduler(T=1.0, n_windows=5, epochs_per_window=1000)
        assert len(sched.windows) == 5

    def test_windows_cover_domain(self):
        sched = TimeWindowScheduler(T=1.0, n_windows=4, epochs_per_window=100)
        assert sched.windows[0].t_start == pytest.approx(0.0, abs=1e-8)
        assert sched.windows[-1].t_end == pytest.approx(1.0, abs=1e-8)

    def test_advance_moves_window(self):
        sched = TimeWindowScheduler(T=1.0, n_windows=3, epochs_per_window=100)
        assert sched.current_window.index == 0
        sched.advance()
        assert sched.current_window.index == 1

    def test_advance_returns_false_at_end(self):
        sched = TimeWindowScheduler(T=1.0, n_windows=2, epochs_per_window=100)
        sched.advance()
        result = sched.advance()
        assert result is False

    def test_total_epochs(self):
        sched = TimeWindowScheduler(T=1.0, n_windows=5, epochs_per_window=2000)
        assert sched.total_epochs == 10000

    def test_sample_pts_in_window_shape(self):
        sched = TimeWindowScheduler(T=1.0, n_windows=3, epochs_per_window=100)
        pts = sched.sample_pts_in_window(n=200)
        assert pts.shape == (200, 4)

    def test_sample_pts_time_in_range(self):
        sched = TimeWindowScheduler(T=2.0, n_windows=4, epochs_per_window=100)
        w = sched.current_window
        pts = sched.sample_pts_in_window(n=500, window=w)
        t = pts[:, 3]
        assert float(t.min()) >= w.t_start - 1e-6
        assert float(t.max()) <= w.t_end + 1e-6

    def test_sample_spatial_in_domain(self):
        L = TWO_PI
        sched = TimeWindowScheduler(T=1.0, n_windows=2, epochs_per_window=100)
        pts = sched.sample_pts_in_window(n=300, L=L)
        xyz = pts[:, :3]
        assert float(xyz.min()) >= 0.0
        assert float(xyz.max()) <= L + 1e-6


# ---------------------------------------------------------------------------
# Adaptive sampler (lightweight: we don't need a real trained model)
# ---------------------------------------------------------------------------

class DummyModel(torch.nn.Module):
    """Minimal model: zero-weight linear so output has grad_fn."""
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.lin.weight.zero_()

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return self.lin(pts)  # effectively zero but graph-connected


class TestAdaptiveSampler:
    def test_initial_pts_shape(self):
        sampler = AdaptiveSampler(n_pde=100, n_candidate=500, nu=0.1)
        assert sampler.pts.shape == (100, 4)

    def test_pts_spatial_in_domain(self):
        sampler = AdaptiveSampler(n_pde=200, n_candidate=1000, L=TWO_PI)
        xyz = sampler.pts[:, :3]
        assert float(xyz.min()) >= 0.0 - 1e-8
        assert float(xyz.max()) <= TWO_PI + 1e-8

    def test_pts_temporal_in_domain(self):
        T = 0.5
        sampler = AdaptiveSampler(n_pde=200, n_candidate=500, T=T)
        t = sampler.pts[:, 3]
        assert float(t.min()) >= 0.0 - 1e-8
        assert float(t.max()) <= T + 1e-8

    def test_resample_shape_preserved(self):
        """After resampling, pool size must be exactly n_pde."""
        model = DummyModel()
        sampler = AdaptiveSampler(n_pde=80, n_candidate=400, nu=0.1)
        new_pts = sampler.resample(model)
        assert new_pts.shape == (80, 4)

    def test_history_records_step(self):
        model = DummyModel()
        sampler = AdaptiveSampler(n_pde=60, n_candidate=300, nu=0.1)
        assert len(sampler.history) == 0
        sampler.resample(model)
        assert len(sampler.history) == 1

    def test_uniform_reset_shape(self):
        sampler = AdaptiveSampler(n_pde=150, n_candidate=500)
        pts = sampler.uniform_reset()
        assert pts.shape == (150, 4)
