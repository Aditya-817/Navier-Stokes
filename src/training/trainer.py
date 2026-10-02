"""
src/training/trainer.py
========================
Main PINN training loop with checkpointing, logging, and reproducibility.

Features
--------
- AdamW optimizer with cosine LR schedule.
- Gradient clipping.
- Per-step loss component logging (CSV + TensorBoard).
- Checkpoint save/resume.
- Resample collocation points every N steps (configurable).
- Early stopping on plateau (configurable).
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.training.loss import compute_loss, LossComponents
from src.utils.reproducibility import save_reproducibility_snapshot


class Trainer:
    """PINN trainer for 3D NS on T^3.

    Parameters
    ----------
    model : nn.Module (PINNModel)
    config : dict
        Full experiment config (for logging).
    out_dir : str or Path
        Directory for checkpoints and logs.
    device : str
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        out_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.out_dir = Path(out_dir)
        self.device = device

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "checkpoints").mkdir(exist_ok=True)

        # Training hypers
        tr = config.get("training", {})
        self.max_epochs: int = tr.get("max_epochs", 50000)
        self.lr: float = tr.get("lr", 1e-3)
        self.lr_min: float = tr.get("lr_min", 1e-6)
        self.grad_clip: float = tr.get("grad_clip", 1.0)
        self.checkpoint_interval: int = tr.get("checkpoint_interval", 2000)
        self.log_interval: int = tr.get("log_interval", 100)
        self.resample_interval: int = tr.get("resample_interval", 1000)
        self.n_collocation: int = tr.get("n_collocation", 20000)
        self.n_ic_points: int = tr.get("n_ic_points", 5000)
        self.loss_weights: dict = tr.get("loss_weights", {
            "pde": 1.0, "ic": 10.0, "div": 1.0, "gauge": 1.0
        })

        # Physics
        ph = config.get("physics", {})
        self.nu: float = ph.get("nu", 0.1)

        # Domain
        dom = config.get("domain", {})
        self.L: float = dom.get("L", 2 * math.pi)
        self.T: float = dom.get("T", 1.0)

        # Optimizer & scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=self.lr_min
        )

        # Logging
        self._log_path = self.out_dir / "training_log.csv"
        self._history: list[dict] = []
        self._best_loss = float("inf")
        self._step = 0

        # TensorBoard (optional)
        self._tb_writer = None
        log_cfg = config.get("logging", {})
        if log_cfg.get("backend", "tensorboard") == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(
                    log_dir=str(self.out_dir / "tensorboard")
                )
            except ImportError:
                pass

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_pde_pts(self, rng: np.random.Generator) -> Tensor:
        from src.sampling.samplers import uniform_sample
        pts = uniform_sample(self.n_collocation, self.T, self.L, rng=rng)
        return torch.tensor(pts, device=self.device, dtype=self.model.dtype())

    def _sample_ic_pts(
        self,
        rng: np.random.Generator,
        ic,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample IC points and evaluate ground-truth velocities."""
        from src.sampling.samplers import ic_sample
        pts_np = ic_sample(self.n_ic_points, self.L, rng=rng)
        x_np, y_np, z_np = pts_np[:, 0], pts_np[:, 1], pts_np[:, 2]

        u0_np = ic.u0(x_np, y_np, z_np)
        v0_np = ic.v0(x_np, y_np, z_np)
        w0_np = ic.w0(x_np, y_np, z_np)

        pts_t = torch.tensor(pts_np, device=self.device, dtype=self.model.dtype())
        u0 = torch.tensor(u0_np, device=self.device, dtype=self.model.dtype())
        v0 = torch.tensor(v0_np, device=self.device, dtype=self.model.dtype())
        w0 = torch.tensor(w0_np, device=self.device, dtype=self.model.dtype())
        return pts_t, u0, v0, w0

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, ic) -> list[dict]:
        """Run the training loop.

        Parameters
        ----------
        ic : DivFreeIC
            Initial condition object providing u0, v0, w0.

        Returns
        -------
        history : list of dicts, one per logged step.
        """
        # Save reproducibility snapshot
        save_reproducibility_snapshot(
            self.out_dir,
            self.config,
            seed=self.config.get("experiment", {}).get("seed", 42),
        )

        rng = np.random.default_rng(
            seed=self.config.get("experiment", {}).get("seed", 42)
        )

        # Initial sample
        pde_pts = self._sample_pde_pts(rng)
        ic_pts, u0, v0, w0 = self._sample_ic_pts(rng, ic)

        self.model.train()
        t_start = time.time()

        pbar = tqdm(range(self.max_epochs), desc="Training", dynamic_ncols=True)
        for epoch in pbar:
            self._step = epoch

            # Resample collocation points periodically
            if epoch > 0 and epoch % self.resample_interval == 0:
                pde_pts = self._sample_pde_pts(rng)
                ic_pts, u0, v0, w0 = self._sample_ic_pts(rng, ic)

            # Forward + loss
            self.optimizer.zero_grad()
            lc: LossComponents = compute_loss(
                self.model,
                pde_pts,
                ic_pts,
                (u0, v0, w0),
                nu=self.nu,
                weights=self.loss_weights,
                device=self.device,
            )

            lc.L_total.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            # Logging
            if epoch % self.log_interval == 0:
                step_log = {
                    "epoch": epoch,
                    "time_s": time.time() - t_start,
                    "lr": self.scheduler.get_last_lr()[0],
                    **lc.as_dict(),
                }
                self._history.append(step_log)
                self._write_log_row(step_log)

                if self._tb_writer is not None:
                    for k, v in lc.as_dict().items():
                        self._tb_writer.add_scalar(k, v, epoch)

                pbar.set_postfix({
                    "L_tot": f"{float(lc.L_total):.3e}",
                    "L_PDE": f"{float(lc.L_PDE):.3e}",
                    "L_div": f"{float(lc.L_div):.3e}",
                })

                # Track best
                if float(lc.L_total) < self._best_loss:
                    self._best_loss = float(lc.L_total)
                    self._save_checkpoint("best")

            # Periodic checkpoint
            if epoch > 0 and epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(f"epoch_{epoch:07d}")

        # Final checkpoint
        self._save_checkpoint("final")
        if self._tb_writer is not None:
            self._tb_writer.close()

        return self._history

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        ckpt = {
            "epoch": self._step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_loss": self._best_loss,
            "config": self.config,
        }
        path = self.out_dir / "checkpoints" / f"ckpt_{tag}.pt"
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint; returns the epoch."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self._best_loss = ckpt["best_loss"]
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _write_log_row(self, row: dict) -> None:
        write_header = not self._log_path.exists()
        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
