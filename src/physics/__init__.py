"""PDE residuals and automatic differentiation engine."""

from src.physics.derivatives import partial, partial2, laplacian_scalar, divergence, curl3d, split_xyzt
from src.physics.ns_residual import ns_residuals, NSResiduals

__all__ = [
    "partial", "partial2", "laplacian_scalar", "divergence", "curl3d", "split_xyzt",
    "ns_residuals", "NSResiduals",
]
