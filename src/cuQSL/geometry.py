"""Geometry helpers shared by the Cartesian and spherical backends.

Grid conventions
----------------
``grid_xyz = [xgrid, ygrid, zgrid]`` -- three 1-D arrays giving the physical
coordinates of the grid nodes.  Any of them may be geometrically spaced; the
caller supplies the matching normalised coordinate mapping.
"""

from __future__ import annotations

import numpy as np
import torch


def grid_spacing(grid_xyz, rtol=1e-6):
    """Return ``dxyz`` for a uniform grid, raising if the grid is stretched."""
    axes = [np.asarray(a, dtype=float).ravel() for a in grid_xyz]
    if any(a.size < 2 for a in axes):
        raise ValueError("each grid axis needs at least two nodes")
    dxyz = np.array([a[1] - a[0] for a in axes], dtype=float)
    for axis, step in zip(axes, dxyz):
        if not np.allclose(np.diff(axis), step, rtol=rtol, atol=rtol * abs(step)):
            raise ValueError(
                "grid_xyz must be uniformly spaced; resample the field onto a "
                "regular grid before constructing the solver"
            )
    return dxyz


def normalise(xyz, lo, hi):
    """Map physical coordinates to ``[-1, 1]`` (one scale per axis)."""
    lo = torch.as_tensor(lo, dtype=xyz.dtype, device=xyz.device)
    hi = torch.as_tensor(hi, dtype=xyz.dtype, device=xyz.device)
    span = hi - lo
    if torch.any(span <= 0):
        raise ValueError("grid extent must be positive on every axis")
    return 2.0 * (xyz - lo) / span - 1.0


def unit_vector(B):
    """Return ``(bhat, bad)`` where ``bad`` marks cells with ``|B| ~ 0``."""
    nrm = torch.linalg.vector_norm(B, dim=-1, keepdim=True)
    scale = nrm.amax().clamp_min(1.0)
    bad = nrm <= 1e-12 * scale
    safe = torch.where(bad, torch.ones_like(nrm), nrm)
    return B / safe, bad.squeeze(-1)


def cartesian_curl(Bxyz, dxyz):
    """``curl B`` on a uniform Cartesian grid, shape ``(nx, ny, nz, 3)``."""
    d = np.asarray(dxyz, dtype=float)
    # gradients[i][..., c] = d B_c / d x_i
    grad = np.gradient(np.asarray(Bxyz, dtype=float), axis=(0, 1, 2))
    dbx_dx, dby_dy, dbz_dz = grad[0][..., 0], grad[1][..., 1], grad[2][..., 2]
    dby_dx, dbz_dx = grad[0][..., 1], grad[0][..., 2]
    dbx_dy, dbz_dy = grad[1][..., 0], grad[1][..., 2]
    dbx_dz, dby_dz = grad[2][..., 0], grad[2][..., 1]

    curl = np.empty_like(np.asarray(Bxyz, dtype=float))
    curl[..., 0] = (dbz_dy - dby_dz) / d[1]
    curl[..., 1] = (dbx_dz - dbz_dx) / d[2]
    curl[..., 2] = (dby_dx - dbx_dy) / d[0]
    return curl


def cartesian_jacobian(bhat, dxyz):
    """``d bhat_c / d x_d`` in physical units, shape ``(nx, ny, nz, 3, 3)``.

    Element ``J[..., d, c]`` is the derivative of component ``c`` with respect
    to the physical coordinate ``x_d`` -- the layout consumed by the line
    integrator (``dU_c/ds = sum_d U_d J[d, c]``).
    """
    d = np.asarray(dxyz, dtype=float)
    grad = np.gradient(np.asarray(bhat, dtype=float), axis=(0, 1, 2))
    jac = np.stack([g for g in grad], axis=-2)  # (..., d, c)
    return jac / d[None, None, None, :, None]


def orthonormal_frame(bhat):
    """Two unit vectors orthogonal to ``bhat`` (deterministic, vectorised)."""
    comp = bhat.abs()
    axis = comp.argmin(dim=-1)
    e = torch.zeros_like(bhat)
    e.scatter_(1, axis.unsqueeze(1), 1.0)
    u = torch.linalg.cross(bhat, e, dim=-1)
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1e-300)
    v = torch.linalg.cross(bhat, u, dim=-1)
    return u, v


def frame_components(vec_a, vec_b, e1, e2):
    """Coordinates of ``(vec_a, vec_b)`` in the orthonormal frame ``(e1, e2)``.

    Returns a ``(N, 2, 2)`` matrix whose columns are the two vectors, so the
    determinant is the signed area they span in that plane.
    """
    row1 = torch.stack([(vec_a * e1).sum(-1), (vec_b * e1).sum(-1)], dim=-1)
    row2 = torch.stack([(vec_a * e2).sum(-1), (vec_b * e2).sum(-1)], dim=-1)
    return torch.stack([row1, row2], dim=-2)


def rtp_from_xyz(xyz):
    """Cartesian -> spherical; ``theta = arccos(z/r)``, ``phi = atan2(y, x)``."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x * x + y * y + z * z)
    safe = r.clamp_min(1e-300)
    theta = torch.arccos((z / safe).clamp(-1.0, 1.0))
    phi = torch.atan2(y, x)
    phi = torch.where(phi < 0, phi + 2.0 * torch.pi, phi)
    return torch.stack((r, theta, phi), dim=-1)


def xyz_from_rtp(rtp):
    r, theta, phi = rtp[..., 0], rtp[..., 1], rtp[..., 2]
    st = torch.sin(theta)
    return torch.stack(
        (r * st * torch.cos(phi), r * st * torch.sin(phi), r * torch.cos(theta)),
        dim=-1,
    )


def spherical_basis(theta, phi):
    """Unit vectors :math:`(\\hat r, \\hat\\theta, \\hat\\phi)` at ``(theta, phi)``."""
    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)
    zero = torch.zeros_like(phi)
    one = torch.ones_like(phi)
    e_r = torch.stack((st * cp, st * sp, ct), dim=-1)
    e_t = torch.stack((ct * cp, ct * sp, -st), dim=-1)
    e_p = torch.stack((-sp, cp, zero), dim=-1)
    del one
    return e_r, e_t, e_p


def spherical_to_cartesian_vectors(vec_rtp, theta, phi):
    """Rotate a field given in physical spherical components to Cartesian axes."""
    e_r, e_t, e_p = spherical_basis(theta, phi)
    out = (
        vec_rtp[..., 0:1] * e_r
        + vec_rtp[..., 1:2] * e_t
        + vec_rtp[..., 2:3] * e_p
    )
    return out
