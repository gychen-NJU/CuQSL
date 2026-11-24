"""Small I/O helpers: coordinate conversion and VTK export.

The field arrays used throughout the package are ordinary NumPy arrays shaped
``(nx, ny, nz, ...)`` with the first three axes following the grid arrays passed
to the solvers.  Nothing here is required by the solvers themselves.
"""

from __future__ import annotations

import numpy as np


def xyz2rtp(xyz):
    """Convert Cartesian coordinates ``(..., 3)`` to ``(r, theta, phi)``.

    ``theta`` is the polar angle in ``[0, pi]`` and ``phi`` the azimuth in
    ``[0, 2 pi)``.
    """
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    safe = np.where(r == 0.0, 1.0, r)
    theta = np.arccos(np.clip(z / safe, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    return np.stack((r, theta, phi), axis=-1)


def rtp2xyz(rtp):
    """Convert ``(r, theta, phi)`` to Cartesian coordinates ``(..., 3)``."""
    rtp = np.asarray(rtp, dtype=float)
    r, theta, phi = rtp[..., 0], rtp[..., 1], rtp[..., 2]
    st, ct = np.sin(theta), np.cos(theta)
    return np.stack((r * st * np.cos(phi), r * st * np.sin(phi), r * ct), axis=-1)


def data2vts(filename, grid_xyz, data=None, name=None):
    """Write a structured grid to a ``.vts`` file through :mod:`pyevtk`.

    Parameters
    ----------
    filename : str
        Output path; the ``.vts`` suffix is added by pyevtk.
    grid_xyz : sequence of three 1-D arrays
        Node coordinates along each axis.
    data : array-like or sequence of array-like, optional
        Scalar/vector arrays shaped ``(nx, ny, nz)`` or ``(nx, ny, nz, 3)``.
    name : str or sequence of str, optional
        One name per array in ``data``.
    """
    try:
        from pyevtk.hl import gridToVTK
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("data2vts requires pyevtk: pip install pyevtk") from exc

    if data is None:
        raise ValueError("data2vts requires at least one array to write")

    arrays = [np.asarray(item, dtype=float)
              for item in (data if isinstance(data, (list, tuple)) else [data])]
    names = [name] if isinstance(name, str) else list(name or [])
    if len(names) != len(arrays):
        raise ValueError("`name` must have one entry per array in `data`")

    expected = tuple(len(np.asarray(a).ravel()) for a in grid_xyz)
    for arr, key in zip(arrays, names):
        if arr.shape[:3] != expected:
            raise ValueError(
                f"{key!r} has shape {arr.shape}, expected leading axes {expected}"
            )
    payload = dict(zip(names, arrays))
    return gridToVTK(filename, *[np.asarray(a, dtype=float).ravel() for a in grid_xyz],
                     pointData=payload)
