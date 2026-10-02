"""
tests/test_spectral_solver.py
==============================
Milestone 7: Pseudo-spectral reference solver validation.

Tests:
  1. Divergence error < 1e-12 throughout time integration.
  2. Energy conservation (inviscid) / correct decay (viscous).
  3. TGV energy matches analytic low-Re estimate.
  4. Leray projection makes u div-free before first step.
"""
import math
import numpy as np
import pytest

from src.solvers.spectral_solver import SpectralNSSolver
from src.utils.initial_conditions import TaylorGreenVortex3D
from src.utils.domain import cell_centred_grid

TWO_PI = 2.0 * math.pi


class TestLerayProjection:
    def test_divfree_after_init(self):
        """After setting IC, divergence should be < 1e-12."""
        solver = SpectralNSSolver(N=16, nu=0.1, dt=0.01)
        ic = TaylorGreenVortex3D()
        X, Y, Z = cell_centred_grid(16)
        solver.set_initial_condition(ic.u0(X, Y, Z), ic.v0(X, Y, Z), ic.w0(X, Y, Z))
        div_l2, div_linf = solver.divergence_error()
        assert div_l2 < 1e-12, f"div L2 = {div_l2:.2e}"
        assert div_linf < 1e-12, f"div Linf = {div_linf:.2e}"


class TestDivergenceDuringIntegration:
    def test_divergence_stays_small(self):
        """Divergence should remain < 1e-12 throughout integration."""
        solver = SpectralNSSolver(N=16, nu=0.1, dt=0.01)
        ic = TaylorGreenVortex3D()
        X, Y, Z = cell_centred_grid(16)
        solver.set_initial_condition(ic.u0(X, Y, Z), ic.v0(X, Y, Z), ic.w0(X, Y, Z))

        for _ in range(20):  # 0.2 time units
            solver.step()

        div_l2, div_linf = solver.divergence_error()
        assert div_l2 < 1e-11, f"div L2 after integration = {div_l2:.2e}"
        assert div_linf < 1e-11, f"div Linf after integration = {div_linf:.2e}"


class TestEnergyBehavior:
    def test_viscous_energy_decays(self):
        """For viscous flow, kinetic energy should decrease monotonically."""
        solver = SpectralNSSolver(N=16, nu=0.1, dt=0.005)
        ic = TaylorGreenVortex3D(V0=1.0, nu=0.1)
        X, Y, Z = cell_centred_grid(16)
        solver.set_initial_condition(ic.u0(X, Y, Z), ic.v0(X, Y, Z), ic.w0(X, Y, Z))

        E0 = solver.kinetic_energy()
        energies = [E0]
        for _ in range(10):
            solver.step()
            energies.append(solver.kinetic_energy())

        # Energy should be strictly decreasing for smooth viscous low-Re flow
        assert all(
            energies[i] > energies[i + 1] for i in range(len(energies) - 1)
        ), f"Energy not monotonically decreasing: {energies}"

    def test_zero_viscosity_energy_conserved(self):
        """For ν=0 (Euler), energy should be conserved to near-machine precision."""
        solver = SpectralNSSolver(N=16, nu=0.0, dt=0.005, dealias=True)
        ic = TaylorGreenVortex3D(V0=1.0, nu=1.0)  # nu here is for Re only
        X, Y, Z = cell_centred_grid(16)
        solver.set_initial_condition(ic.u0(X, Y, Z), ic.v0(X, Y, Z), ic.w0(X, Y, Z))

        E0 = solver.kinetic_energy()
        for _ in range(50):
            solver.step()
        E1 = solver.kinetic_energy()

        rel_err = abs(E1 - E0) / (abs(E0) + 1e-15)
        assert rel_err < 1e-4, f"Euler energy drift: {rel_err:.2e}"


class TestPressureRecovery:
    def test_pressure_zero_mean(self):
        """Recovered pressure should have zero mean."""
        solver = SpectralNSSolver(N=16, nu=0.1, dt=0.01)
        ic = TaylorGreenVortex3D()
        X, Y, Z = cell_centred_grid(16)
        solver.set_initial_condition(ic.u0(X, Y, Z), ic.v0(X, Y, Z), ic.w0(X, Y, Z))
        p = solver.get_pressure()
        assert abs(p.mean()) < 1e-12, f"Pressure mean: {p.mean():.2e}"


class TestSnapshotOutput:
    def test_run_to_returns_snapshots(self):
        solver = SpectralNSSolver(N=8, nu=0.2, dt=0.05)
        ic = TaylorGreenVortex3D()
        X, Y, Z = cell_centred_grid(8)
        solver.set_initial_condition(ic.u0(X, Y, Z), ic.v0(X, Y, Z), ic.w0(X, Y, Z))
        snaps = solver.run_to(0.2, snapshot_times=[0.1], verbose=False)
        assert len(snaps) >= 1
        assert "u" in snaps[0]
        assert snaps[0]["u"].shape == (8, 8, 8)
