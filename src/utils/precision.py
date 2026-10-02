"""
src/utils/precision.py
======================
Precision context manager and type aliases.

Usage
-----
    with use_precision("float64"):
        # all new tensors default to float64
        ...

    dtype = resolve_dtype("float32")  # -> torch.float32
"""
from __future__ import annotations

import contextlib
import torch

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Convert a string dtype name to a torch.dtype."""
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Choose from {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


@contextlib.contextmanager
def use_precision(dtype: str | torch.dtype):
    """Context manager that sets torch default floating-point dtype.

    Parameters
    ----------
    dtype : str or torch.dtype
        e.g. 'float64' or torch.float64
    """
    if isinstance(dtype, str):
        dtype = resolve_dtype(dtype)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield dtype
    finally:
        torch.set_default_dtype(old)
