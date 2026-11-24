"""Cartesian backend: a field sampled on a uniform ``(nx, ny, nz)`` grid."""

from __future__ import annotations

import numpy as np
import torch

from .geometry import cartesian_curl, cartesian_jacobian, grid_spacing, normalise
from .interpolate import TensorFieldInterpolator
from .solver import _BaseSolver

_CHANNELS = 15  # |B|(3) + d bhat / dx (9) + curl B (3), sampled in one pass


def _unit(B):
    nrm = np.linalg.norm(B, axis=-1, keepdims=True)
    scale = float(np.max(nrm)) if nrm.size else 1.0
    nrm = np.where(nrm <= 1e-12 * max(scale, 1e-30), 1.0, nrm)
    return B / nrm


class CartesianGeometry:
    """Uniform-grid geometry with a rectangular domain boundary.

    All three sampled fields (``B``, ``d bhat / dx`` and ``curl B``) live in a
    single 15-channel interpolation grid, so one ``grid_sample`` call returns
    everything the right-hand side and the twist integrand need.
    """

    def __init__(self, Bxyz, grid_xyz, dtype=torch.float64):
        B = np.asarray(Bxyz, dtype=float)
        if B.ndim != 4 or B.shape[-1] != 3:
            raise ValueError(f"Bxyz must be (nx, ny, nz, 3), got {B.shape}")
        self.dxyz = grid_spacing(grid_xyz)
        self.lo = np.array([float(np.min(a)) for a in grid_xyz])
        self.hi = np.array([float(np.max(a)) for a in grid_xyz])
        self.scale = float(self.dxyz.min())
        self.suggested_step = float(self.dxyz.min())
        self.dtype = dtype

        bhat = _unit(B)
        jac = cartesian_jacobian(bhat, self.dxyz)
        curl = cartesian_curl(B, self.dxyz)
        packed = np.concatenate((B, jac.reshape(*jac.shape[:3], 9), curl), axis=-1)
        assert packed.shape[-1] == _CHANNELS
        self._packed = TensorFieldInterpolator(packed, dtype=dtype)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        self._packed.to(self.device)
        return self

    def _coords(self, x):
        return normalise(x, self.lo, self.hi)

    # ------------------------------------------------------- geometry API
    def inside(self, x):
        return (self._coords(x).abs() <= 1.0).all(dim=-1)

    def exit_fraction(self, start, end):
        """Fraction of the straight segment ``start -> end`` inside the box.

        Exact for an axis-aligned box, so the kernel no longer needs a 40-step
        bisection per field line that leaves the domain.
        """
        c0 = self._coords(start)
        c1 = self._coords(end)
        d = c1 - c0
        out = (c1.abs() > 1.0)
        sign = torch.where(c1 > 1.0, torch.ones_like(c1), -torch.ones_like(c1))
        with torch.no_grad():
            t = (sign - c0) / torch.where(d.abs() > 1e-300, d, torch.full_like(d, 1e-300))
        t = torch.where(out, t.clamp(0.0, 1.0), torch.ones_like(t))
        return t.amin(dim=-1)

    def sample(self, x):
        """One interpolation pass: ``(B, bhat, jacb, curl)`` at ``x``."""
        raw = self._packed(self._coords(x))
        B = raw[:, :3]
        jacb = raw[:, 3:12].reshape(-1, 3, 3)
        curl = raw[:, 12:15]
        nrm = torch.linalg.vector_norm(B, dim=-1, keepdim=True)
        bhat = B / nrm.clamp_min(1e-30)
        return B, bhat, jacb, curl

    # ------------------------------------------------ individual accessors
    def Bvec(self, x):
        return self.sample(x)[0]

    def bhat(self, x):
        return self.sample(x)[1]

    def jacb(self, x):
        return self.sample(x)[2]

    def curl(self, x):
        return self.sample(x)[3]


class qsl_solver_cat(_BaseSolver):
    """QSL map for a Cartesian field.

    Parameters
    ----------
    Bxyz : array-like, ``(nx, ny, nz, 3)``
        Magnetic field on a uniform grid.
    grid_xyz : list of three 1-D arrays
        Physical coordinates of the grid nodes (``[xgrid, ygrid, zgrid]``).
    dtype : torch.dtype
        Working precision of the integration (``float64`` by default).
    """

    def __init__(self, Bxyz, grid_xyz, *, dtype=torch.float64, **kwargs):
        super().__init__(CartesianGeometry(Bxyz, grid_xyz, dtype=dtype),
                         dtype=dtype, **kwargs)
