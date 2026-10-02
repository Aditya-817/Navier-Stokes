"""
src/solvers/solver_io.py
=========================
HDF5 I/O for pseudo-spectral solver snapshots.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def save_snapshots(
    snapshots: list[dict],
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save a list of solver snapshots to an HDF5 file.

    Parameters
    ----------
    snapshots : list of dicts with keys 't', 'u', 'v', 'w', 'p', 'E'
    path : output .h5 file path
    metadata : optional dict of scalar attributes to store in the file root
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v

        times = np.array([s["t"] for s in snapshots])
        f.create_dataset("t", data=times)
        f.create_dataset("E", data=np.array([s["E"] for s in snapshots]))

        N = snapshots[0]["u"].shape[0]
        n_snaps = len(snapshots)

        for field in ("u", "v", "w", "p"):
            arr = np.stack([s[field] for s in snapshots], axis=0)  # (n_snaps, N, N, N)
            f.create_dataset(field, data=arr, compression="gzip", compression_opts=4)


def load_snapshots(path: str | Path) -> tuple[np.ndarray, list[dict]]:
    """Load snapshots from an HDF5 file.

    Returns
    -------
    times : np.ndarray, shape (n_snaps,)
    snapshots : list of dicts with keys 't', 'u', 'v', 'w', 'p', 'E'
    """
    path = Path(path)
    snapshots = []
    with h5py.File(path, "r") as f:
        times = f["t"][:]
        energies = f["E"][:]
        u_arr = f["u"][:]
        v_arr = f["v"][:]
        w_arr = f["w"][:]
        p_arr = f["p"][:]

    for i, t in enumerate(times):
        snapshots.append({
            "t": float(t),
            "u": u_arr[i],
            "v": v_arr[i],
            "w": w_arr[i],
            "p": p_arr[i],
            "E": float(energies[i]),
        })

    return times, snapshots
