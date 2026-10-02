"""Neural network architectures for NS-PINNs."""

from src.models.base import PINNModel
from src.models.factory import make_model
from src.models.fourier_resmlp import FourierResMLP, StandardMLP

__all__ = ["PINNModel", "make_model", "FourierResMLP", "StandardMLP"]
