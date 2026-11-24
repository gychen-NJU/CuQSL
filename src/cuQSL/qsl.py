"""Quasi-separatrix layer diagnostic from two field-line legs.

``Q`` follows Tassev & Savcheva (2017) eq. (11) (equivalently Scott et al.
2017, ``Q = N^2/|Delta|``)::

    N^2 = |aF|^2 |bB|^2 + |aB|^2 |bF|^2 - 2 (aF.bB) (aB.bF)
    Det = B0^2 / (BF * BB)
    Q   = N^2 / Det

``Q = 2`` is the identity map and is also the mathematical floor for any
mapping, so values below ``2`` indicate a numerical failure rather than a very
small squashing.
"""

from __future__ import annotations

import torch

TWO_PI = 2.0 * torch.pi


def q_from_legs(B0, BF, BB, aF, bF, aB, bB):
    """Return ``(Q, log10 Q, below_floor)`` from the projected tangent vectors."""
    n2 = (
        (aF * aF).sum(-1) * (bB * bB).sum(-1)
        + (aB * aB).sum(-1) * (bF * bF).sum(-1)
        - 2.0 * (aF * bF).sum(-1) * (aB * bB).sum(-1)
    )
    det = B0 * B0 / (BF * BB)
    q = n2 / det
    log_q = torch.log10(q)
    below_floor = torch.isfinite(q) & (q < 2.0 - 1e-6)
    return q, log_q, below_floor


def twist_integrand(B, curl):
    """Integrand ``g(s) = (B . curl B) / (4 pi |B|^2)``.

    The twist number is ``T = (1/4 pi) int (B . curl B)/|B|^2 ds``, so the
    caller accumulates ``0.5 * (g0 + g1) * ds`` (trapezoid) per step, with
    ``ds`` signed by the integration direction: a backward leg then contributes
    the correct (negative) share, and the two legs together give the
    field-aligned integral.
    """
    b2 = (B * B).sum(-1)
    return (B * curl).sum(-1) / (4.0 * torch.pi * b2)
