"""Adaptive Runge-Kutta-Fehlberg 4(5) stepping, vectorised over many lines.

The interface is intentionally small: :meth:`Fehlberg45.attempt` performs one
trial step and returns the 5th-order solution together with the normalised
error estimate.  Accepting or rejecting a step is the caller's job, which keeps
per-line step sizes independent (some field lines cross a boundary long before
their neighbours).
"""

from __future__ import annotations

import torch

# Classical Fehlberg 4(5) tableaux (Roberts & Shipman / Fehlberg 1969).
_NODES = (0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0)
_COEF = (
    (),
    (1.0 / 4.0,),
    (3.0 / 32.0, 9.0 / 32.0),
    (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0),
    (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0),
    (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0),
)
# 4th order weights (k6 has weight 0) and 5th order weights.
_B4 = (25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0)
_B5 = (16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0)


class Fehlberg45:
    """Fehlberg 4(5) integrator with a fixed set of tolerances."""

    def __init__(self, atol_pos=1e-9, atol_vec=1e-9, rtol=1e-9, safety=0.9,
                 min_factor=0.1, max_factor=5.0):
        self.atol_pos = float(atol_pos)
        self.atol_vec = float(atol_vec)
        self.rtol = float(rtol)
        self.safety = float(safety)
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)

    def attempt(self, rhs, y, h):
        """Try one step of (signed) size ``h`` from state ``y`` (N, m).

        Returns ``(y5, err)`` where ``err`` is the normalised error per point
        (``err <= 1`` means the step is acceptable).
        """
        k = []
        for stage, (node, coef) in enumerate(zip(_NODES, _COEF)):
            if stage == 0:
                yi = y
            else:
                acc = torch.zeros_like(y)
                for j, c in enumerate(coef):
                    acc = acc + c * k[j]
                yi = y + h.unsqueeze(1) * acc
            k.append(rhs(yi))

        acc4 = torch.zeros_like(y)
        acc5 = torch.zeros_like(y)
        for j in range(6):
            if _B4[j]:
                acc4 = acc4 + _B4[j] * k[j]
            if _B5[j]:
                acc5 = acc5 + _B5[j] * k[j]
        y4 = y + h.unsqueeze(1) * acc4
        y5 = y + h.unsqueeze(1) * acc5

        return y5, self.error_norm(y4 - y5, y5)

    def error_norm(self, err, y5):
        """Mixed absolute/relative error norm, position and tangent parts separately."""
        atol = torch.full_like(y5, self.atol_vec)
        atol[:, :3] = self.atol_pos
        scale = atol + self.rtol * y5.abs()
        return (err.abs() / scale).amax(dim=1)

    def next_dt(self, h, err):
        """Proposed next step size (signed, same sign as ``h``)."""
        err = torch.clamp(err, min=1e-12)
        factor = self.safety * err.pow(-0.2)
        factor = torch.clamp(factor, self.min_factor, self.max_factor)
        return h * factor
