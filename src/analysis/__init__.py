"""Spectral diagnostics and vortex analysis for 3D NS solutions."""

from src.analysis.spectral_analysis import (
    energy_spectrum_3d, dissipation_rate, kolmogorov_scale,
    kolmogorov_exponent, spectral_report,
)
from src.analysis.vortex_diagnostics import (
    vorticity_field, enstrophy, palinstrophy, q_criterion,
    helicity_spectrum, vortex_diagnostics_report,
)

__all__ = [
    "energy_spectrum_3d", "dissipation_rate", "kolmogorov_scale",
    "kolmogorov_exponent", "spectral_report",
    "vorticity_field", "enstrophy", "palinstrophy", "q_criterion",
    "helicity_spectrum", "vortex_diagnostics_report",
]
