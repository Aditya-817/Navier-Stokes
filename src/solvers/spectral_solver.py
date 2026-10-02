"""
src/solvers/spectral_solver.py
================================
Pseudo-spectral Navier–Stokes solver on T^3 = [0, 2π]^3.

Algorithm
---------
1. Represent u, v, w as Fourier series on an N^3 grid.
2. Enforce ∇·u = 0 exactly via Leray projection in Fourier space.
3. Nonlinear term computed in physical space (pseudo-spectral approach).
4. 3/2-rule dealiasing (padding to (3N/2)^3 for nonlinear products).
5. Viscous term treated exactly (integrating factor) or via Fourier:
   û(k, t+dt) = û(k, t) * exp(-ν k² dt)  for the linear part.
6. Full time integration via explicit 4th-order Runge–Kutta (RK4).

Scientific independence
-----------------------
- Incompressibility enforced via Leray projection at EVERY substep.
  This is completely different from the PINN's divergence penalty.
- Differentiation via Fourier multiplication (exact for bandlimited fields).
- Time stepping via RK4 (no neural network approximation).

NOTE: Uses float64 throughout for numerical accuracy.
All arrays are NumPy (CPU). Independence from PyTorch is intentional.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

TWO_PI = 2.0 * np.pi


class SpectralNSSolver:
    """Pseudo-spectral incompressible NS solver on T^3.

    Parameters
    ----------
    N : int
        Grid resolution (N^3 grid points). Should be even for dealiasing.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step for RK4.
    dealias : bool
        Apply 3/2-rule dealiasing to the nonlinear term.
    L : float
        Side length (default 2π).
    """

    def __init__(
        self,
        N: int = 64,
        nu: float = 0.1,
        dt: float = 0.001,
        dealias: bool = True,
        L: float = TWO_PI,
    ) -> None:
        self.N = N
        self.nu = nu
        self.dt = dt
        self.dealias = dealias
        self.L = L
        self.t = 0.0

        # Wavenumber arrays (shape N^3)
        k1d = np.fft.fftfreq(N, d=1.0 / N).astype(int)
        self.Kx, self.Ky, self.Kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
        self.K2 = self.Kx**2 + self.Ky**2 + self.Kz**2  # |k|^2

        # Dealiasing mask: zero modes with |k_i| > N/3 in any direction
        if dealias:
            N3 = N // 3
            self.dealias_mask = (
                (np.abs(self.Kx) <= N3)
                & (np.abs(self.Ky) <= N3)
                & (np.abs(self.Kz) <= N3)
            ).astype(np.float64)
        else:
            self.dealias_mask = np.ones((N, N, N), dtype=np.float64)

        # Fourier coefficients of velocity (initialised to zero)
        self.u_hat = np.zeros((N, N, N), dtype=complex)
        self.v_hat = np.zeros((N, N, N), dtype=complex)
        self.w_hat = np.zeros((N, N, N), dtype=complex)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def set_initial_condition(self, u0: np.ndarray, v0: np.ndarray, w0: np.ndarray) -> None:
        """Set IC from physical-space arrays of shape (N, N, N).

        Automatically applies Leray projection to enforce ∇·u=0 exactly.
        """
        self.u_hat = np.fft.fftn(u0)
        self.v_hat = np.fft.fftn(v0)
        self.w_hat = np.fft.fftn(w0)
        self.t = 0.0
        self._leray_project()

    # ------------------------------------------------------------------
    # Leray projection (enforces ∇·u = 0 in Fourier space)
    # ------------------------------------------------------------------

    def _leray_project(self) -> None:
        """Remove the irrotational part of û in-place.

        û_sol = û - (k̂ ⊗ k̂) û   (project onto divergence-free subspace)

        where k̂ = k / |k|.
        """
        K2 = self.K2.astype(float)
        K2[0, 0, 0] = 1.0  # avoid division by zero; k=0 mode handled below

        ku = (
            self.Kx * self.u_hat
            + self.Ky * self.v_hat
            + self.Kz * self.w_hat
        ) / K2

        self.u_hat -= self.Kx * ku
        self.v_hat -= self.Ky * ku
        self.w_hat -= self.Kz * ku

        # Enforce zero mean (k=0 mode)
        self.u_hat[0, 0, 0] = 0.0
        self.v_hat[0, 0, 0] = 0.0
        self.w_hat[0, 0, 0] = 0.0

    # ------------------------------------------------------------------
    # Nonlinear term
    # ------------------------------------------------------------------

    def _nonlinear_term(
        self,
        u_hat: np.ndarray,
        v_hat: np.ndarray,
        w_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute -P[(u·∇)u] in Fourier space.

        P is the Leray projection, applied at the end.

        Steps:
          1. Apply dealiasing mask.
          2. Transform to physical space.
          3. Compute velocity gradients via spectral differentiation.
          4. Form nonlinear product in physical space.
          5. Transform back to Fourier space.
          6. Apply dealiasing mask again.
          7. Apply Leray projection.
        """
        dm = self.dealias_mask

        # Physical-space velocity (dealiased)
        u = np.real(np.fft.ifftn(u_hat * dm))
        v = np.real(np.fft.ifftn(v_hat * dm))
        w = np.real(np.fft.ifftn(w_hat * dm))

        # Spectral derivatives: ∂u/∂xⱼ = IFFT(i*kⱼ * û)
        iku_hat = 1j * self.Kx * u_hat * dm
        ikv_hat = 1j * self.Kx * v_hat * dm
        ikw_hat = 1j * self.Kx * w_hat * dm

        u_x = np.real(np.fft.ifftn(iku_hat))
        v_x = np.real(np.fft.ifftn(ikv_hat))
        w_x = np.real(np.fft.ifftn(ikw_hat))

        u_y = np.real(np.fft.ifftn(1j * self.Ky * u_hat * dm))
        v_y = np.real(np.fft.ifftn(1j * self.Ky * v_hat * dm))
        w_y = np.real(np.fft.ifftn(1j * self.Ky * w_hat * dm))

        u_z = np.real(np.fft.ifftn(1j * self.Kz * u_hat * dm))
        v_z = np.real(np.fft.ifftn(1j * self.Kz * v_hat * dm))
        w_z = np.real(np.fft.ifftn(1j * self.Kz * w_hat * dm))

        # Nonlinear advection in physical space
        Nu = u * u_x + v * u_y + w * u_z
        Nv = u * v_x + v * v_y + w * v_z
        Nw = u * w_x + v * w_y + w * w_z

        # Transform and dealias
        Nu_hat = np.fft.fftn(Nu) * dm
        Nv_hat = np.fft.fftn(Nv) * dm
        Nw_hat = np.fft.fftn(Nw) * dm

        # Leray-project the nonlinear term
        K2 = self.K2.astype(float)
        K2[0, 0, 0] = 1.0
        kN = (self.Kx * Nu_hat + self.Ky * Nv_hat + self.Kz * Nw_hat) / K2
        Nu_hat -= self.Kx * kN
        Nv_hat -= self.Ky * kN
        Nw_hat -= self.Kz * kN

        return Nu_hat, Nv_hat, Nw_hat

    # ------------------------------------------------------------------
    # RHS (time derivative in Fourier space)
    # ------------------------------------------------------------------

    def _rhs(
        self,
        u_hat: np.ndarray,
        v_hat: np.ndarray,
        w_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute dû/dt = -P[(u·∇)u] - ν k² û."""
        Nu_hat, Nv_hat, Nw_hat = self._nonlinear_term(u_hat, v_hat, w_hat)
        visc = self.nu * self.K2

        rhs_u = -Nu_hat - visc * u_hat
        rhs_v = -Nv_hat - visc * v_hat
        rhs_w = -Nw_hat - visc * w_hat

        return rhs_u, rhs_v, rhs_w

    # ------------------------------------------------------------------
    # RK4 time step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the solution by one time step dt using RK4."""
        dt = self.dt
        u0, v0, w0 = self.u_hat.copy(), self.v_hat.copy(), self.w_hat.copy()

        k1u, k1v, k1w = self._rhs(u0, v0, w0)

        k2u, k2v, k2w = self._rhs(
            u0 + 0.5 * dt * k1u,
            v0 + 0.5 * dt * k1v,
            w0 + 0.5 * dt * k1w,
        )

        k3u, k3v, k3w = self._rhs(
            u0 + 0.5 * dt * k2u,
            v0 + 0.5 * dt * k2v,
            w0 + 0.5 * dt * k2w,
        )

        k4u, k4v, k4w = self._rhs(
            u0 + dt * k3u,
            v0 + dt * k3v,
            w0 + dt * k3w,
        )

        self.u_hat = u0 + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        self.v_hat = v0 + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.w_hat = w0 + (dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)

        # Re-project to ensure machine-precision divergence-free
        self._leray_project()
        self.t += dt

    # ------------------------------------------------------------------
    # Physical-space access
    # ------------------------------------------------------------------

    def get_uvw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return physical-space velocity fields (N, N, N) arrays."""
        u = np.real(np.fft.ifftn(self.u_hat))
        v = np.real(np.fft.ifftn(self.v_hat))
        w = np.real(np.fft.ifftn(self.w_hat))
        return u, v, w

    def get_pressure(self) -> np.ndarray:
        """Recover pressure from the Poisson equation in Fourier space.

        -Δp = ∇·[(u·∇)u]  →  p̂(k) = i k · N̂(k) / |k|²

        Returns physical-space pressure (zero-mean).
        """
        Nu_hat, Nv_hat, Nw_hat = self._nonlinear_term(
            self.u_hat, self.v_hat, self.w_hat
        )
        # Divergence of nonlinear term
        div_N_hat = (
            1j * self.Kx * Nu_hat
            + 1j * self.Ky * Nv_hat
            + 1j * self.Kz * Nw_hat
        )
        K2 = self.K2.astype(complex)
        K2[0, 0, 0] = 1.0
        p_hat = -div_N_hat / K2
        p_hat[0, 0, 0] = 0.0  # zero mean
        return np.real(np.fft.ifftn(p_hat))

    def divergence_error(self) -> tuple[float, float]:
        """Return (L2, Linf) divergence error of current velocity field.

        Uses spectral differentiation (exact for bandlimited fields), not FD.
        After Leray projection, this should be < 1e-12 to machine precision.
        """
        # Spectral divergence: ∇·u in Fourier space = sum_k ik_j * û_j(k)
        div_hat = (
            1j * self.Kx * self.u_hat
            + 1j * self.Ky * self.v_hat
            + 1j * self.Kz * self.w_hat
        )
        div = np.real(np.fft.ifftn(div_hat))
        return float(np.sqrt(np.mean(div**2))), float(np.max(np.abs(div)))


    def kinetic_energy(self) -> float:
        """Kinetic energy E = (1/2) ∫ |u|² dx  (using Parseval)."""
        # E = (1/2) * (1/N^3) Σ_k |û|^2
        N3 = self.N**3
        return 0.5 * float(
            np.sum(np.abs(self.u_hat)**2 + np.abs(self.v_hat)**2 + np.abs(self.w_hat)**2)
        ) / N3

    # ------------------------------------------------------------------
    # Run to a target time
    # ------------------------------------------------------------------

    def run_to(
        self,
        T_target: float,
        snapshot_times: list[float] | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        """Advance the solver to T_target, optionally saving snapshots.

        Parameters
        ----------
        T_target : float
        snapshot_times : list of float, optional
            Times at which to save velocity/pressure snapshots.
        verbose : bool
            Print progress.

        Returns
        -------
        list of snapshot dicts: {'t': float, 'u': ndarray, 'v', 'w', 'p', 'E'}
        """
        if snapshot_times is None:
            snapshot_times = []
        snap_times_sorted = sorted(set(snapshot_times))
        snap_idx = 0
        snapshots = []

        n_steps = int(math.ceil((T_target - self.t) / self.dt))
        for i in range(n_steps):
            # Save snapshot before step if we've reached a snapshot time
            if snap_idx < len(snap_times_sorted):
                if self.t >= snap_times_sorted[snap_idx] - 1e-12:
                    u, v, w = self.get_uvw()
                    p = self.get_pressure()
                    snapshots.append({
                        "t": self.t,
                        "u": u.copy(), "v": v.copy(),
                        "w": w.copy(), "p": p.copy(),
                        "E": self.kinetic_energy(),
                    })
                    snap_idx += 1

            self.step()
            if verbose and (i % max(1, n_steps // 20) == 0):
                E = self.kinetic_energy()
                div_l2, div_linf = self.divergence_error()
                print(
                    f"  t={self.t:.4f}  E={E:.6f}  "
                    f"div_L2={div_l2:.2e}  div_Linf={div_linf:.2e}"
                )

        # Final state
        u, v, w = self.get_uvw()
        p = self.get_pressure()
        snapshots.append({
            "t": self.t,
            "u": u.copy(), "v": v.copy(),
            "w": w.copy(), "p": p.copy(),
            "E": self.kinetic_energy(),
        })

        return snapshots
