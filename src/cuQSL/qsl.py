"""Quasi-separating layer diagnostic from the two field-line legs.

Definition
----------
Follow the field line from the seed ``x0`` forward to the boundary at ``x_F``
and backward to ``x_B``.  Lie-transport the two tangents of the seed's
perpendicular plane along both legs (Tassev & Savcheva 2017); at each endpoint
express them in an orthonormal frame transverse to the field there.  That
gives the two Jacobians

.. math::

    A = \\partial x_F / \\partial x_0, \\qquad C = \\partial x_B / \\partial x_0

and, writing :math:`J = A C^{-1}` for the map from the backward plane to the
forward plane,

.. math::

    Q = \\frac{\\|J\\|_F^2}{|\\det J|}.

Expressing ``A`` and ``C`` in the frames of their *own* endpoint (rather than
both in the seed frame) is what keeps

.. math::

    |\\det A| = B_0 / B_F, \\qquad |\\det C| = B_0 / B_B

by flux conservation, and therefore keeps ``Q`` at or above its mathematical
floor.  For any invertible ``2x2`` ``J`` the singular values obey
``s_1^2 + s_2^2 >= 2 s_1 s_2``, so ``Q >= 2``; values below ``2`` indicate a
numerical failure, never a genuinely weak squashing.  ``Q = 2`` is the
identity map.

The equivalent form quoted by Tassev & Savcheva (2017) eq. (11),

.. math::

    N^2 = |a_F|^2 |b_B|^2 + |a_B|^2 |b_F|^2 - 2 (a_F \\cdot b_F)(a_B \\cdot b_B),
    \\qquad Q = N^2 \\frac{B_F B_B}{B_0^2},

is the same quantity once the tangents are resolved in a common frame and the
field-strength ratios are used for the two determinants.
"""

from __future__ import annotations

import torch

TWO_PI = 2.0 * torch.pi
FLOOR = 2.0 - 1e-6


def q_from_jacobians(a, c):
    """Return ``(Q, log10 Q, below_floor)`` from the two ``2x2`` Jacobians.

    ``a`` maps seed-plane coordinates to the forward endpoint's transverse
    frame, ``c`` to the backward endpoint's; both are ``(N, 2, 2)``.
    """
    j = a @ torch.linalg.inv(c)
    n2 = (j * j).sum(dim=(-1, -2))
    det = (j[..., 0, 0] * j[..., 1, 1] - j[..., 0, 1] * j[..., 1, 0]).abs()
    q = n2 / det.clamp_min(1e-300)
    log_q = torch.log10(q)
    below_floor = torch.isfinite(q) & (q < FLOOR)
    return q, log_q, below_floor


def twist_integrand(b, curl):
    """Integrand ``g(s) = (B . curl B) / (4 pi |B|^2)``.

    The twist number is ``T = (1/4 pi) int (B . curl B)/|B|^2 ds``, so the
    caller accumulates ``0.5 * (g0 + g1) * |ds|`` (trapezoid) per step with
    the *arc length* of the step.  Both legs are traversed forward along the
    field line, so both contribute with a positive ``ds``; ``g`` itself
    carries the sign of ``B . curl B`` and the two legs together cover the
    whole line from the backward to the forward boundary.
    """
    b2 = (b * b).sum(-1)
    return (b * curl).sum(-1) / (4.0 * torch.pi * b2)
