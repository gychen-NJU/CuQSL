"""cuQSL -- quasi-separatrix layer diagnostics for magnetic fields.

Two back ends share one integrator:

``cuQSL.cat``
    Field sampled on a uniform Cartesian grid.
``cuQSL.sph``
    Field sampled on a spherical ``(r, theta, phi)`` grid; resampled onto a
    Cartesian cube before integration, so no spherical connection terms are
    needed.

Both return ``(logQ, Length, Twist)`` for a set of seed points.
"""

from __future__ import annotations

from .cartesian import CartesianGeometry, qsl_solver_cat
from .interpolate import TensorFieldInterpolator
from .solver import Diagnostics
from .spherical import SphericalGeometry, qsl_solver_sph

__version__ = "2.0.0"
__author__ = "Guoyin Chen"

cat = qsl_solver_cat
sph = qsl_solver_sph

__all__ = [
    "cat",
    "sph",
    "qsl_solver_cat",
    "qsl_solver_sph",
    "CartesianGeometry",
    "SphericalGeometry",
    "TensorFieldInterpolator",
    "Diagnostics",
    "__version__",
]
