"""
src/physics/derivatives.py
===========================
Automatic differentiation engine for PDE residuals.

All functions accept PyTorch tensors and use torch.autograd.grad.
Inputs must have requires_grad=True for derivatives to be computed.

Design principles
-----------------
- Every function is pure: no global state, no side effects.
- create_graph=True is always set so that higher-order derivatives and
  second-order optimizers work correctly.
- retain_graph defaults to True so multiple derivative calls on the
  same graph do not trigger graph deletion.
- Functions return tensors of the same shape as their inputs.

Performance note on Laplacians
-------------------------------
For a scalar field f(x,y,z,t) the Laplacian Δf = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
requires three separate second-order AD calls (one per spatial coordinate).
This is O(d_space) in memory and compute, which is acceptable for d=3.
An alternative is `torch.autograd.functional.hessian`, but it computes
the full Hessian (O(d²) entries) which is wasteful here.
"""
from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# First-order partial derivatives
# ---------------------------------------------------------------------------

def grad_scalar(
    f: Tensor,
    x: Tensor,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """Compute ∂f/∂x via automatic differentiation.

    Parameters
    ----------
    f : Tensor, shape (N,) or (N,1)
        Scalar field values (must be result of differentiable computation
        with respect to x).
    x : Tensor, shape (N,) or (N,1), requires_grad=True
        The variable to differentiate with respect to.

    Returns
    -------
    Tensor of same shape as x containing ∂f/∂x at each point.
    If f does not depend on x, returns a zero tensor of the same shape.
    """
    (df_dx,) = torch.autograd.grad(
        outputs=f,
        inputs=x,
        grad_outputs=torch.ones_like(f),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    if df_dx is None:
        # f does not depend on x — derivative is identically zero
        return torch.zeros_like(f)
    # Additional guard: with create_graph=True, PyTorch may return a tensor
    # that has grad_fn set but requires_grad=False when the gradient is a
    # constant w.r.t. all leaves (e.g., derivative of sin(y) w.r.t. x=0 via
    # the chain rule, where the intermediate result is a constant function).
    # We detect this by checking requires_grad OR grad_fn, matching PyTorch's
    # own internal check in run_backward.
    if not (df_dx.requires_grad or df_dx.grad_fn is not None):
        return torch.zeros_like(f)
    return df_dx


def partial(
    f: Tensor,
    x: Tensor,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """Alias for grad_scalar for readability (∂f/∂x)."""
    return grad_scalar(f, x, create_graph=create_graph, retain_graph=retain_graph)


# ---------------------------------------------------------------------------
# Second-order partial derivatives
# ---------------------------------------------------------------------------

def partial2(
    f: Tensor,
    x: Tensor,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """Compute ∂²f/∂x² via two successive AD passes.

    If f does not depend on x (first derivative is zero), returns zeros
    without attempting a second AD pass that would fail on the zero tensor.

    Parameters
    ----------
    f : Tensor, shape (N,) or (N,1)
    x : Tensor, shape (N,) or (N,1), requires_grad=True

    Returns
    -------
    Tensor of same shape as f containing ∂²f/∂x².
    """
    df_dx = partial(f, x, create_graph=True, retain_graph=retain_graph)
    # Short-circuit: if df_dx has no grad structure, f is independent of x
    # so the second derivative is also identically zero.
    if not (df_dx.requires_grad or df_dx.grad_fn is not None):
        return torch.zeros_like(f)
    try:
        d2f_dx2 = partial(df_dx, x, create_graph=create_graph, retain_graph=retain_graph)
    except RuntimeError:
        # Second AD pass failed — f's first derivative is constant w.r.t. x
        # (e.g., ∂(const)/∂x = 0). Return zero tensor.
        return torch.zeros_like(f)
    return d2f_dx2


def mixed_partial(
    f: Tensor,
    x: Tensor,
    y: Tensor,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """Compute ∂²f/(∂x ∂y)."""
    df_dx = partial(f, x, create_graph=True, retain_graph=True)
    d2f_dxdy = partial(df_dx, y, create_graph=create_graph, retain_graph=retain_graph)
    return d2f_dxdy


# ---------------------------------------------------------------------------
# Vector-calculus operators
# ---------------------------------------------------------------------------

def laplacian_scalar(
    f: Tensor,
    spatial_vars: list[Tensor],
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """Compute the Laplacian Δf = Σᵢ ∂²f/∂xᵢ² for a scalar field f.

    Parameters
    ----------
    f : Tensor, shape (N,) or (N,1)
    spatial_vars : list of Tensors, each shape (N,) or (N,1)
        The spatial coordinates [x, y, z] to differentiate over.

    Returns
    -------
    Tensor of shape (N,) containing Δf.
    """
    lap = torch.zeros_like(f)
    for xi in spatial_vars:
        lap = lap + partial2(f, xi, create_graph=create_graph, retain_graph=retain_graph)
    return lap


def divergence(
    F: list[Tensor],
    X: list[Tensor],
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """Compute ∇·F = Σᵢ ∂Fᵢ/∂xᵢ for a vector field F = (F₀, F₁, ...).

    Parameters
    ----------
    F : list of Tensors, each shape (N,)
        Vector field components, len(F) must equal len(X).
    X : list of Tensors, each shape (N,), requires_grad=True
        Coordinate tensors paired with F components.

    Returns
    -------
    Tensor of shape (N,) containing ∇·F.
    """
    assert len(F) == len(X), "F and X must have the same number of components."
    div = torch.zeros_like(F[0])
    for fi, xi in zip(F, X):
        div = div + partial(fi, xi, create_graph=create_graph, retain_graph=retain_graph)
    return div


def curl3d(
    F: list[Tensor],
    X: list[Tensor],
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> list[Tensor]:
    """Compute ∇×F for a 3D vector field F = (Fx, Fy, Fz).

    Parameters
    ----------
    F : [Fx, Fy, Fz] — vector field components, each shape (N,)
    X : [x, y, z]   — coordinate tensors, each shape (N,), requires_grad=True

    Returns
    -------
    [ωx, ωy, ωz] where
        ωx = ∂Fz/∂y − ∂Fy/∂z
        ωy = ∂Fx/∂z − ∂Fz/∂x
        ωz = ∂Fy/∂x − ∂Fx/∂y
    """
    assert len(F) == 3 and len(X) == 3, "curl3d requires exactly 3 components."
    Fx, Fy, Fz = F
    x, y, z = X

    dFz_dy = partial(Fz, y, create_graph=create_graph, retain_graph=retain_graph)
    dFy_dz = partial(Fy, z, create_graph=create_graph, retain_graph=retain_graph)
    dFx_dz = partial(Fx, z, create_graph=create_graph, retain_graph=retain_graph)
    dFz_dx = partial(Fz, x, create_graph=create_graph, retain_graph=retain_graph)
    dFy_dx = partial(Fy, x, create_graph=create_graph, retain_graph=retain_graph)
    dFx_dy = partial(Fx, y, create_graph=create_graph, retain_graph=retain_graph)

    omega_x = dFz_dy - dFy_dz
    omega_y = dFx_dz - dFz_dx
    omega_z = dFy_dx - dFx_dy

    return [omega_x, omega_y, omega_z]


# ---------------------------------------------------------------------------
# Batch derivative utility: compute all first-order partials of a vector
# field simultaneously — used for the full Jacobian (dFi/dxj)
# ---------------------------------------------------------------------------

def jacobian_row(
    fi: Tensor,
    X: list[Tensor],
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> list[Tensor]:
    """Compute [∂fi/∂x₀, ∂fi/∂x₁, ...] for a single scalar output fi.

    Parameters
    ----------
    fi : Tensor, shape (N,)
    X : list of Tensors, each shape (N,), requires_grad=True

    Returns
    -------
    list of Tensors [∂fi/∂x₀, ∂fi/∂x₁, ...], same length as X.
    """
    return [
        partial(fi, xj, create_graph=create_graph, retain_graph=retain_graph)
        for xj in X
    ]


# ---------------------------------------------------------------------------
# Convenience: prepare input tensors for differentiation
# ---------------------------------------------------------------------------

def split_xyzt(
    pts: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split a (N, 4) collocation tensor into (x, y, z, t) flat tensors.

    Each output tensor has shape **(N,)** and requires_grad=True.
    Using flat tensors avoids (N,1) vs (N,) broadcasting errors throughout
    the derivative engine.

    Parameters
    ----------
    pts : Tensor, shape (N, 4)
        Columns: [x, y, z, t]

    Returns
    -------
    x, y, z, t : Tensors of shape (N,), each with requires_grad=True
    """
    x = pts[:, 0].detach().requires_grad_(True)
    y = pts[:, 1].detach().requires_grad_(True)
    z = pts[:, 2].detach().requires_grad_(True)
    t = pts[:, 3].detach().requires_grad_(True)
    return x, y, z, t
