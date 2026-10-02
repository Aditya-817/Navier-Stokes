"""Pseudo-spectral reference solver for 3D NS on T³."""

from src.solvers.spectral_solver import SpectralNSSolver
from src.solvers.solver_io import save_snapshots, load_snapshots

__all__ = ["SpectralNSSolver", "save_snapshots", "load_snapshots"]
