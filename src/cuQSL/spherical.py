"""Spherical backend: resample ``(r, theta, phi)`` data onto a Cartesian cube.

Doing the interpolation once, on a Cartesian cube that covers the shell, means
the spherical backend reuses the *same* line integrator as the Cartesian one.
That removes the connection terms (and their truncation errors) that the old
spherical kernel needed, and it guarantees that both backends agree to within
interpolation error.
"""

from __future__ import annotations

import numpy as np
import torch

from .cartesian import CartesianGeometry
from .geometry import rtp_from_xyz, spherical_basis
from .solver import _BaseSolver

TWO_PI = 2.0 * np.pi


class SphericalGeometry:
    """A field given on an ``(r, theta, phi)`` grid, integrated in Cartesian space."""

    def __init__(self, Brtp, rtp_range, stretched=(False, False, False),
                 dtype=torch.float64, max_nodes=192):
        B = np.asarray(Brtp, dtype=float)
        if B.ndim != 4 or B.shape[-1] != 3:
            raise ValueError(f"Brtp must be (nr, ntheta, nphi, 3), got {B.shape}")
        if len(stretched) > 1 and (stretched[1] or stretched[2]):
            raise NotImplementedError(
                "only the radial axis supports a stretched grid; resample theta/phi "
                "on a uniform grid before constructing the solver"
            )
        nr, ntheta, nphi, _ = B.shape
        r0, r1 = (float(v) for v in np.asarray(rtp_range[0], dtype=float).ravel()[:2])
        t0, t1 = (float(v) for v in np.asarray(rtp_range[1], dtype=float).ravel()[:2])
        p0, p1 = (float(v) for v in np.asarray(rtp_range[2], dtype=float).ravel()[:2])
        self.bounds = (r0, r1, t0, t1, p0, p1)

        radial_log = bool(stretched[0])
        r_axis = np.geomspace(r0, r1, nr) if radial_log else np.linspace(r0, r1, nr)
        t_axis = np.linspace(t0, t1, ntheta)
        p_axis = np.linspace(p0, p1, nphi)

        spacing = min(
            float(np.diff(r_axis).min()),
            max(r0, 1e-9) * (t1 - t0) / max(ntheta - 1, 1),
            max(r0 * np.sin(min(t1, np.pi - t0)), 1e-9) * (p1 - p0) / max(nphi - 1, 1),
        )
        self.suggested_step = spacing

        lo, hi = self._bounding_box(r0, r1, t0, t1, p0, p1, spacing)
        counts = [min(max_nodes, max(8, int(np.ceil((hi[i] - lo[i]) / spacing)) + 1)) for i in range(3)]
        xg = np.linspace(lo[0], hi[0], counts[0])
        yg = np.linspace(lo[1], hi[1], counts[1])
        zg = np.linspace(lo[2], hi[2], counts[2])
        cube = self._resample(B, r_axis, t_axis, p_axis, xg, yg, zg, radial_log)
        self._box = CartesianGeometry(cube, [xg, yg, zg], dtype=dtype)
        self.scale = spacing
        self.device = torch.device("cpu")

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _bounding_box(r0, r1, t0, t1, p0, p1, pad):
        theta = np.linspace(t0, t1, 33)
        phi = np.linspace(p0, p1, 65)
        tt, pp = np.meshgrid(theta, phi, indexing="ij")
        pts = []
        for radius in (r0, r1):
            st, ct, sp, cp = np.sin(tt), np.cos(tt), np.sin(pp), np.cos(pp)
            pts.append(np.stack(
                (radius * st * cp, radius * st * sp, radius * ct), axis=-1
            ).reshape(-1, 3))
        allpts = np.concatenate(pts)
        return allpts.min(0) - pad, allpts.max(0) + pad

    @staticmethod
    def _resample(B, r_axis, t_axis, p_axis, xg, yg, zg, radial_log):
        gamma = np.stack(np.meshgrid(xg, yg, zg, indexing="ij"), axis=-1).reshape(-1, 3)
        pts = torch.as_tensor(gamma, dtype=torch.float64)
        rtp = rtp_from_xyz(pts)
        r, theta, phi = rtp[:, 0].numpy(), rtp[:, 1].numpy(), rtp[:, 2].numpy()

        r0, r1 = r_axis[0], r_axis[-1]
        t0, t1 = t_axis[0], t_axis[-1]
        p0, p1 = p_axis[0], p_axis[-1]
        if radial_log:
            ri = 2.0 * np.log(max(r, 1e-300) / r0) / np.log(r1 / r0) - 1.0
        else:
            ri = 2.0 * (r - r0) / (r1 - r0) - 1.0
        ti = 2.0 * (theta - t0) / (t1 - t0) - 1.0
        pi = 2.0 * (phi - p0) / (p1 - p0) - 1.0

        grid = np.stack((pi, ti, ri), axis=-1).reshape(xg.size, yg.size, zg.size, 3)
        tensor = torch.as_tensor(B, dtype=torch.float64).permute(3, 0, 1, 2).unsqueeze(0)
        grid_t = torch.as_tensor(grid, dtype=torch.float64).unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(
            tensor, grid_t, mode="bilinear", padding_mode="border", align_corners=True
        )
        vec = sampled[0].permute(1, 2, 3, 0)  # (nx, ny, nz, 3) in spherical components

        tt = torch.as_tensor(theta, dtype=torch.float64).reshape(xg.size, yg.size, zg.size)
        pp = torch.as_tensor(phi, dtype=torch.float64).reshape(xg.size, yg.size, zg.size)
        e_r, e_t, e_p = spherical_basis(tt, pp)
        cart = vec[..., 0:1] * e_r + vec[..., 1:2] * e_t + vec[..., 2:3] * e_p
        return cart.numpy()

    # -------------------------------------------------------- geometry API
    def to(self, device):
        self._box.to(device)
        self.device = torch.device(device)
        return self

    def inside(self, x):
        r0, r1, t0, t1, p0, p1 = self.bounds
        rtp = rtp_from_xyz(x)
        r, theta, phi = rtp[..., 0], rtp[..., 1], rtp[..., 2]
        span = p1 - p0
        tol = 1e-9
        ok_r = (r >= r0 * (1 - tol)) & (r <= r1 * (1 + tol))
        ok_t = (theta >= t0 - tol) & (theta <= t1 + tol)
        if span >= TWO_PI - 1e-6:
            ok_p = torch.ones_like(ok_r)
        else:
            shifted = p0 + torch.remainder(phi - p0, TWO_PI)
            ok_p = (shifted >= p0 - tol) & (shifted <= p1 + tol)
        return ok_r & ok_t & ok_p

    def Bvec(self, x):
        return self._box.Bvec(x)

    def bhat(self, x):
        return self._box.bhat(x)

    def jacb(self, x):
        return self._box.jacb(x)

    def curl(self, x):
        return self._box.curl(x)

    def sample(self, x):
        """Packed interpolation of ``(B, bhat, jacb, curl)`` on the resampled cube."""
        return self._box.sample(x)


class qsl_solver_sph(_BaseSolver):
    """QSL map for a field given in spherical coordinates.

    Parameters
    ----------
    Brtp : array-like, ``(nr, ntheta, nphi, 3)``
        Field components ``(B_r, B_theta, B_phi)`` in the local orthonormal basis.
    rtp_range : sequence of three ``(min, max)`` pairs
        Extent of the ``(r, theta, phi)`` grid.
    stretched : sequence of three bool
        Log-spaced radial axis when ``stretched[0]`` is true.  ``theta``/``phi``
        must be uniform.
    """

    def __init__(self, Brtp, rtp_range, stretched=(False, False, False), *,
                 dtype=torch.float64, **kwargs):
        super().__init__(SphericalGeometry(Brtp, rtp_range, stretched, dtype=dtype),
                         dtype=dtype, **kwargs)

    def __call__(self, points=None, **kwargs):
        if points is None:
            points = self.default_points()
        return super().__call__(points, **kwargs)

    def default_points(self, n_r=24, n_t=24, n_p=48):
        """A coarse grid over the shell, used when ``points`` is omitted."""
        r0, r1, t0, t1, p0, p1 = self.geometry.bounds
        grid = np.stack(np.meshgrid(
            np.linspace(r0, r1, n_r),
            np.linspace(t0, t1, n_t),
            np.linspace(p0, p1, n_p),
            indexing="ij",
        ), axis=-1).reshape(-1, 3)
        return self._rtp_to_xyz(grid)

    @staticmethod
    def _rtp_to_xyz(rtp):
        r, theta, phi = rtp[:, 0], rtp[:, 1], rtp[:, 2]
        st = np.sin(theta)
        return np.stack(
            (r * st * np.cos(phi), r * st * np.sin(phi), r * np.cos(theta)), axis=-1
        )
