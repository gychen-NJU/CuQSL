"""Cartesian backend: a field sampled on a uniform ``(nx, ny, nz)`` grid."""

from __future__ import annotations

import numpy as np
import torch

from .geometry import cartesian_curl, cartesian_jacobian, grid_spacing, normalise
from .interpolate import TensorFieldInterpolator
from .solver import _BaseSolver


def _unit(B):
    nrm = np.linalg.norm(B, axis=-1, keepdims=True)
    scale = float(np.max(nrm)) if nrm.size else 1.0
    nrm = np.where(nrm <= 1e-12 * max(scale, 1e-30), 1.0, nrm)
    return B / nrm


class CartesianGeometry:
    """Uniform-grid geometry with a rectangular domain boundary."""

    def __init__(self, Bxyz, grid_xyz, dtype=torch.float64):
        B = np.asarray(Bxyz, dtype=float)
        if B.ndim != 4 or B.shape[-1] != 3:
            raise ValueError(f"Bxyz must be (nx, ny, nz, 3), got {B.shape}")
        self.dxyz = grid_spacing(grid_xyz)
        self.lo = np.array([float(np.min(a)) for a in grid_xyz])
        self.hi = np.array([float(np.max(a)) for a in grid_xyz])
        self.scale = float(self.dxyz.min())
        self.suggested_step = float(self.dxyz.min())

        bhat = _unit(B)
        self._bvec = TensorFieldInterpolator(B, dtype=dtype)
        self._jac = TensorFieldInterpolator(cartesian_jacobian(bhat, self.dxyz), dtype=dtype)
        self._curl = TensorFieldInterpolator(cartesian_curl(B, self.dxyz), dtype=dtype)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        for field in (self._bvec, self._jac, self._curl):
            field.to(self.device)
        return self

    def _coords(self, x):
        return normalise(x, self.lo, self.hi)

    def inside(self, x):
        return (self._coords(x).abs() <= 1.0).all(dim=-1)

    def Bvec(self, x):
        return self._bvec(self._coords(x))

    def bhat(self, x):
        B = self.Bvec(x)
        nrm = torch.linalg.vector_norm(B, dim=-1, keepdim=True)
        return B / nrm.clamp_min(1e-30)

    def jacb(self, x):
        return self._jac(self._coords(x))

    def curl(self, x):
        return self._curl(self._coords(x))


class qsl_solver_cat(_BaseSolver):
    """QSL map for a Cartesian field.

    Parameters
    ----------
    Bxyz : array-like, ``(nx, ny, nz, 3)``
        Magnetic field on a uniform grid.
    grid_xyz : list of three 1-D arrays
        Physical coordinates of the grid nodes (``[xgrid, ygrid, zgrid]``).
    """

    def __init__(self, Bxyz, grid_xyz, **kwargs):
        super().__init__(CartesianGeometry(Bxyz, grid_xyz), **kwargs)
