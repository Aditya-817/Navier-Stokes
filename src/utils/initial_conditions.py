"""
src/utils/initial_conditions.py
================================
Analytic divergence-free, L-periodic initial conditions on T^3.

Design
------
Every IC is a dataclass that exposes:
  - u0(X, Y, Z) -> np.ndarray   (x-velocity)
  - v0(X, Y, Z) -> np.ndarray   (y-velocity)
  - w0(X, Y, Z) -> np.ndarray   (z-velocity)
  - divergence(X, Y, Z) -> np.ndarray  (should be ~0 to machine precision)
  - name: str
  - description: str
  - Re_char: float               (characteristic Reynolds number)
  - L_char: float                (characteristic length scale)
  - U_char: float                (characteristic velocity scale)

Three benchmark families are provided:
  1. TaylorGreenVortex3D   — classical TGV IC (analytic for t=0, DNS benchmark)
  2. ABCFlow               — Arnold–Beltrami–Childress flow (exact NS steady state)
  3. FourierRandomDivFree  — vector-potential construction with random Fourier modes

All ICs satisfy ∇·u = 0 analytically (verified in tests to < 1e-12 on fine grids).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DivFreeIC(ABC):
    """Abstract base class for divergence-free initial conditions."""

    name: str
    description: str
    Re_char: float
    L_char: float
    U_char: float
    nu: float  # kinematic viscosity used to define Re

    @abstractmethod
    def u0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def v0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def w0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray: ...

    def velocity(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (u0, v0, w0) on grid (X, Y, Z)."""
        return self.u0(X, Y, Z), self.v0(X, Y, Z), self.w0(X, Y, Z)

    def divergence(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Numerical divergence via finite differences on the provided grid.

        Uses second-order central differences with periodic wrap-around.
        The spacing is inferred from X assuming a regular grid.

        NOTE: For analytic verification the grid spacing should be fine
        (N >= 64) and the return values should be < 1e-12 for analytic ICs.
        """
        # Grid spacings (assumes uniform)
        dx = float(X[1, 0, 0] - X[0, 0, 0]) if X.ndim == 3 else (TWO_PI / X.size)
        dy = float(Y[0, 1, 0] - Y[0, 0, 0]) if Y.ndim == 3 else dx
        dz = float(Z[0, 0, 1] - Z[0, 0, 0]) if Z.ndim == 3 else dx

        u = self.u0(X, Y, Z)
        v = self.v0(X, Y, Z)
        w = self.w0(X, Y, Z)

        du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
        dv_dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
        dw_dz = (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2 * dz)

        return du_dx + dv_dy + dw_dz

    def analytic_divergence(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        """Analytic divergence (should be identically zero for all ICs here).

        Override in subclasses if an analytic expression is known.
        Default returns array of zeros with same shape as X.
        """
        return np.zeros_like(X)


# ---------------------------------------------------------------------------
# 1. Taylor–Green Vortex (3D)
# ---------------------------------------------------------------------------

@dataclass
class TaylorGreenVortex3D(DivFreeIC):
    """3D Taylor–Green vortex initial condition.

    Definition
    ----------
    u₀(x,y,z) =  V₀ sin(kx) cos(ky) cos(kz)
    v₀(x,y,z) = -V₀ cos(kx) sin(ky) cos(kz)
    w₀(x,y,z) =  0

    where k = 1 (wavenumber, so one full period fits in [0, 2π]).

    Properties
    ----------
    - Exactly divergence-free: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0 analytically.
    - Periodic on [0, 2π]³.
    - Well-known DNS benchmark; reference data available in literature.
    - At t=0, the pressure for the incompressible NS equations is:
        p₀(x,y,z) = (V₀²/4)(cos(2kx) + cos(2ky))(cos(2kz) + 2) / 16
      (this expression is for the standard TGV pressure at t=0).
    - Develops vortex stretching and eventually turbulent breakdown.

    References
    ----------
    Taylor & Green (1937); Brachet et al. (1983); van Rees et al. (2011).
    """

    name: str = "taylor_green_3d"
    description: str = (
        "3D Taylor–Green vortex IC. Analytic at t=0; well-known DNS benchmark. "
        "Develops vortex stretching and turbulent breakdown."
    )
    V0: float = 1.0        # velocity amplitude
    k: int = 1             # wavenumber (integer for periodicity on [0,2π])
    nu: float = 0.1        # kinematic viscosity
    L_char: float = 1.0    # 1/k
    U_char: float = field(init=False)
    Re_char: float = field(init=False)

    def __post_init__(self) -> None:
        self.U_char = self.V0
        self.L_char = 1.0 / self.k
        self.Re_char = self.V0 / (self.nu * self.k)

    def u0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return self.V0 * np.sin(self.k * X) * np.cos(self.k * Y) * np.cos(self.k * Z)

    def v0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return -self.V0 * np.cos(self.k * X) * np.sin(self.k * Y) * np.cos(self.k * Z)

    def w0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.zeros_like(X)

    def analytic_divergence(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        # du/dx = V0*k*cos(kx)*cos(ky)*cos(kz)
        # dv/dy = -V0*k*cos(kx)*cos(ky)*cos(kz)    <- exactly cancels
        # dw/dz = 0
        return np.zeros_like(X)

    def p0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Analytic initial pressure for TGV (from incompressibility + NS at t=0)."""
        k = self.k
        V0 = self.V0
        return (V0**2 / 16.0) * (
            np.cos(2 * k * X) + np.cos(2 * k * Y)
        ) * (np.cos(2 * k * Z) + 2.0)


# ---------------------------------------------------------------------------
# 2. Arnold–Beltrami–Childress (ABC) Flow
# ---------------------------------------------------------------------------

@dataclass
class ABCFlow(DivFreeIC):
    """Arnold–Beltrami–Childress flow.

    Definition
    ----------
    u₀(x,y,z) = A sin(z) + C cos(y)
    v₀(x,y,z) = B sin(x) + A cos(z)
    w₀(x,y,z) = C sin(y) + B cos(x)

    Properties
    ----------
    - Divergence-free: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0 identically.
    - Periodic on [0, 2π]³ (wavenumber 1 in each direction).
    - Beltrami flow: ω = curl(u) = u  (for A=B=C=1 after normalisation).
      This means it is an exact steady Euler solution (not NS with ν>0).
    - Exhibits Lagrangian chaos even at the IC level.
    - Standard coefficients: A = √6, B = √2, C = √3 (for chaotic parameter).

    References
    ----------
    Arnold (1965); Childress (1970); Dombre et al. (1986).
    """

    name: str = "abc_flow"
    description: str = (
        "Arnold–Beltrami–Childress flow. Exact steady Euler solution. "
        "Lagrangian chaos; Beltrami property curl(u)=u."
    )
    A: float = np.sqrt(6.0)
    B: float = np.sqrt(2.0)
    C: float = np.sqrt(3.0)
    nu: float = 0.1
    L_char: float = 1.0
    U_char: float = field(init=False)
    Re_char: float = field(init=False)

    def __post_init__(self) -> None:
        self.U_char = max(abs(self.A), abs(self.B), abs(self.C))
        self.Re_char = self.U_char * self.L_char / self.nu

    def u0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return self.A * np.sin(Z) + self.C * np.cos(Y)

    def v0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return self.B * np.sin(X) + self.A * np.cos(Z)

    def w0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return self.C * np.sin(Y) + self.B * np.cos(X)

    def analytic_divergence(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        # du/dx = 0  (u has no x-dependence)
        # dv/dy = 0  (v has no y-dependence)
        # dw/dz = 0  (w has no z-dependence)
        return np.zeros_like(X)


# ---------------------------------------------------------------------------
# 3. Fourier-Random Divergence-Free Flow (vector-potential construction)
# ---------------------------------------------------------------------------

@dataclass
class FourierRandomDivFree(DivFreeIC):
    """Random divergence-free velocity field constructed via vector potential.

    Construction
    ------------
    Choose vector potential A(x) as a sum of Fourier modes:
        A_i(x) = Σ_{k} c_{ik} exp(i k·x) + c.c.
    Then u = ∇×A is automatically divergence-free.

    For a real-valued result we use:
        A_j(x) = Σ_{k,|k|≤K} a_{jk} sin(k·x) + b_{jk} cos(k·x)

    The coefficients are drawn from N(0,1) and optionally shaped by a
    power-law energy spectrum E(k) ∝ k^{-alpha}.

    Properties
    ----------
    - Exactly divergence-free by construction (curl of any field).
    - Periodic on [0, 2π]³.
    - Tunable energy spectrum via alpha.
    - Reproducible via rng_seed.
    """

    name: str = "fourier_random_div_free"
    description: str = (
        "Random div-free IC via vector-potential construction. "
        "Tunable energy spectrum E(k) ∝ k^{-alpha}."
    )
    K: int = 4                # max wavenumber in each direction
    alpha: float = 5.0 / 3.0  # energy spectrum exponent (Kolmogorov: 5/3)
    amplitude: float = 1.0
    rng_seed: int = 42
    nu: float = 0.1
    L_char: float = 1.0
    U_char: float = field(init=False)
    Re_char: float = field(init=False)

    # Internal: coefficient arrays (set in __post_init__)
    _a: np.ndarray = field(default=None, repr=False)
    _b: np.ndarray = field(default=None, repr=False)
    _modes: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.rng_seed)

        # Build wavenumber list for |k|_inf ≤ K, excluding zero mode
        k_range = np.arange(-self.K, self.K + 1)
        kx, ky, kz = np.meshgrid(k_range, k_range, k_range, indexing="ij")
        modes = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=-1)
        nonzero = np.any(modes != 0, axis=1)
        self._modes = modes[nonzero].astype(np.float64)  # shape (M, 3)

        M = len(self._modes)
        k_norms = np.linalg.norm(self._modes, axis=1)  # shape (M,)

        # Spectral energy shaping: weight ∝ k^{-(alpha+2)/2} for E(k)∝k^{-alpha}
        spectral_weight = k_norms ** (-(self.alpha + 2.0) / 2.0)
        spectral_weight /= spectral_weight.max()

        # Coefficients for vector potential A = (A1, A2, A3) ∈ R^3
        # a_{jk}: coefficient of sin(k·x) for component j
        # b_{jk}: coefficient of cos(k·x) for component j
        raw_a = rng.standard_normal((3, M)) * spectral_weight[None, :]
        raw_b = rng.standard_normal((3, M)) * spectral_weight[None, :]
        self._a = raw_a * self.amplitude
        self._b = raw_b * self.amplitude

        # Characteristic velocity (rough estimate from amplitude)
        self.U_char = self.amplitude
        self.Re_char = self.U_char * self.L_char / self.nu

    def _eval_A(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate vector potential A = (A1, A2, A3) on grid."""
        shape = X.shape
        A1 = np.zeros(shape)
        A2 = np.zeros(shape)
        A3 = np.zeros(shape)

        for m_idx in range(len(self._modes)):
            kx, ky, kz = self._modes[m_idx]
            phase = kx * X + ky * Y + kz * Z
            sin_phase = np.sin(phase)
            cos_phase = np.cos(phase)

            A1 += self._a[0, m_idx] * sin_phase + self._b[0, m_idx] * cos_phase
            A2 += self._a[1, m_idx] * sin_phase + self._b[1, m_idx] * cos_phase
            A3 += self._a[2, m_idx] * sin_phase + self._b[2, m_idx] * cos_phase

        return A1, A2, A3

    def _eval_curl_A(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute u = ∇×A analytically using Fourier differentiation.

        u_i = ε_{ijk} ∂A_k/∂x_j
        u = (∂A3/∂y - ∂A2/∂z,
             ∂A1/∂z - ∂A3/∂x,
             ∂A2/∂x - ∂A1/∂y)
        """
        shape = X.shape
        u = np.zeros(shape)
        v = np.zeros(shape)
        w = np.zeros(shape)

        for m_idx in range(len(self._modes)):
            kx, ky, kz = self._modes[m_idx]
            a1, b1 = self._a[0, m_idx], self._b[0, m_idx]
            a2, b2 = self._a[1, m_idx], self._b[1, m_idx]
            a3, b3 = self._a[2, m_idx], self._b[2, m_idx]

            phase = kx * X + ky * Y + kz * Z
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)

            # Derivatives: ∂/∂x_j (a sin + b cos)(k·x) = k_j(a cos - b sin)
            # ∂A3/∂y = ky * (a3 cos(phase) - b3 sin(phase))
            dA3_dy = ky * (a3 * cos_p - b3 * sin_p)
            # ∂A2/∂z = kz * (a2 cos(phase) - b2 sin(phase))
            dA2_dz = kz * (a2 * cos_p - b2 * sin_p)
            # ∂A1/∂z = kz * (a1 cos(phase) - b1 sin(phase))
            dA1_dz = kz * (a1 * cos_p - b1 * sin_p)
            # ∂A3/∂x = kx * (a3 cos(phase) - b3 sin(phase))
            dA3_dx = kx * (a3 * cos_p - b3 * sin_p)
            # ∂A2/∂x = kx * (a2 cos(phase) - b2 sin(phase))
            dA2_dx = kx * (a2 * cos_p - b2 * sin_p)
            # ∂A1/∂y = ky * (a1 cos(phase) - b1 sin(phase))
            dA1_dy = ky * (a1 * cos_p - b1 * sin_p)

            u += dA3_dy - dA2_dz
            v += dA1_dz - dA3_dx
            w += dA2_dx - dA1_dy

        return u, v, w

    def u0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        u, _, _ = self._eval_curl_A(X, Y, Z)
        return u

    def v0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        _, v, _ = self._eval_curl_A(X, Y, Z)
        return v

    def w0(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        _, _, w = self._eval_curl_A(X, Y, Z)
        return w

    def velocity(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Efficient joint evaluation (one Fourier loop, not three)."""
        return self._eval_curl_A(X, Y, Z)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_IC_REGISTRY: dict[str, type] = {
    "taylor_green": TaylorGreenVortex3D,
    "abc": ABCFlow,
    "fourier_random": FourierRandomDivFree,
}


def make_ic(name: str, **kwargs) -> DivFreeIC:
    """Construct an IC by name with optional keyword arguments.

    Parameters
    ----------
    name : str
        One of 'taylor_green', 'abc', 'fourier_random'.
    **kwargs :
        Passed to the IC constructor.

    Returns
    -------
    DivFreeIC instance.
    """
    if name not in _IC_REGISTRY:
        raise ValueError(
            f"Unknown IC '{name}'. Available: {list(_IC_REGISTRY.keys())}"
        )
    return _IC_REGISTRY[name](**kwargs)
