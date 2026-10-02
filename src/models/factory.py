"""
src/models/factory.py
======================
Model factory: construct a PINNModel from a config dict or YAML section.
"""
from __future__ import annotations

from src.models.base import PINNModel
from src.models.fourier_resmlp import FourierResMLP, StandardMLP
from src.models.siren import SIREN
from src.models.multiscale_mlp import MultiscaleMLP
from src.models.divfree_net import DivFreeNet

_MODEL_REGISTRY: dict[str, type] = {
    "fourier_resmlp": FourierResMLP,
    "standard_mlp": StandardMLP,
    "siren": SIREN,
    "multiscale_mlp": MultiscaleMLP,
    "divfree_net": DivFreeNet,
}


def make_model(model_cfg: dict) -> PINNModel:
    """Construct a PINNModel from a config dict.

    The config dict must contain a 'type' key identifying the model class.
    All other keys are passed as kwargs to the constructor.

    Parameters
    ----------
    model_cfg : dict
        Must include 'type'. Example::

            {
                "type": "fourier_resmlp",
                "fourier_modes": 8,
                "hidden_width": 256,
                "n_layers": 3,
                "activation": "tanh",
                "dtype": "float64",
            }

    Returns
    -------
    PINNModel instance.
    """
    cfg = dict(model_cfg)
    model_type = cfg.pop("type")
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[model_type](**cfg)
