"""Trace field lines both ways and evaluate the quasi-separatrix layer map.

The kernel is geometry agnostic: the geometry object provides

* ``inside(x)``  -- boolean mask, ``True`` for points inside the domain;
* ``bhat(x)``    -- unit field, shape ``(N, 3)``;
* ``jacb(x)``    -- ``d bhat_c / d x_d`` in physical units, shape ``(N, 3, 3)``;
* ``Bvec(x)`` / ``curl(x)`` -- field and its curl, used for the twist;
* ``scale``      -- a length scale used for the "seed sits on the wall" probe.

Each line integrates the augmented state ``y = (x, U, V)`` with

.. math:: dx/ds = \\hat b, \\quad dU/ds = (U \\cdot \\nabla) \\hat b, \\quad dV/ds = (V \\cdot \\nabla) \\hat b

so ``U`` and ``V`` are the Lie-transported tangents of Tassev & Savcheva
(2017).  The step size is dropped when the line reaches the boundary, and the
exit point is located by bisection on the geometry's ``inside`` predicate.
"""

from __future__ import annotations

import torch

from .geometry import frame_components, orthonormal_frame
from .integrator import Fehlberg45
from .qsl import q_from_jacobians, twist_integrand

# Status codes for a single leg.
RUNNING = 0
HIT_BOUNDARY = 1
MAX_STEPS = 2
INVALID = 3
BOUNDARY_SEED = 4

# Reported per-point status; 0 means the two legs both reached the boundary.
REPORT_OK = 0
REPORT_MAX_STEPS = 1
REPORT_INVALID = 2
REPORT_BOUNDARY_SEED = 3


class FieldLineKernel:
    """Integrate field lines and evaluate ``Q`` / ``L`` / ``T``."""

    def __init__(self, geometry, step_size, *, max_steps=200_000, atol_pos=1e-9,
                 atol_vec=1e-9, rtol=1e-9, hmin_factor=1e-4, hmax_factor=20.0,
                 n_bisect=40):
        self.geom = geometry
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.hmin = self.step_size * float(hmin_factor)
        self.hmax = self.step_size * float(hmax_factor)
        self.n_bisect = int(n_bisect)
        self.rk = Fehlberg45(atol_pos=atol_pos, atol_vec=atol_vec, rtol=rtol)
        self._sample = getattr(geometry, "sample", None)
        self._exit = getattr(geometry, "exit_fraction", None)

    # -------------------------------------------------------------- compile
    def compile(self, **kwargs):
        """Wrap the right-hand side in ``torch.compile`` (fusion + fewer launches)."""
        kwargs.setdefault("dynamic", False)
        self._rhs = torch.compile(self._rhs, **kwargs)
        return self

    # ------------------------------------------------------------------ rhs
    def _rhs(self, y):
        x, U, V = y[:, :3], y[:, 3:6], y[:, 6:9]
        if self._sample is not None:
            _B, B, J, _curl = self._sample(x)
        else:  # geometry without a packed sampler
            B = self.geom.bhat(x)
            J = self.geom.jacb(x)
        dU = torch.einsum("nd,ndc->nc", U, J)
        dV = torch.einsum("nd,ndc->nc", V, J)
        return torch.cat((B, dU, dV), dim=1)

    def _twist_at(self, x):
        if self._sample is not None:
            B, _bhat, _J, curl = self._sample(x)
        else:
            B = self.geom.Bvec(x)
            curl = self.geom.curl(x)
        return twist_integrand(B, curl)

    # -------------------------------------------------------------- one leg
    def _trace(self, points, sign):
        dev, dtype = points.device, points.dtype
        n = points.shape[0]
        zero = torch.zeros(n, dtype=dtype, device=dev)

        B0 = self.geom.Bvec(points)
        b0 = self.geom.bhat(points)
        U, V = orthonormal_frame(b0)
        x = points.clone()

        length, twist = zero.clone(), zero.clone()
        steps = torch.zeros(n, dtype=torch.long, device=dev)
        h = torch.full((n,), self.step_size * sign, dtype=dtype, device=dev)
        g = self._twist_at(x)

        status = torch.zeros(n, dtype=torch.long, device=dev)
        bad_seed = ~torch.isfinite(B0).all(dim=-1) | (B0.pow(2).sum(-1) <= 0)
        status = torch.where(bad_seed, torch.full_like(status, INVALID), status)
        probe = x + (sign * 1e-9 * self.geom.scale) * b0
        status = torch.where(
            self.geom.inside(probe), status, torch.full_like(status, BOUNDARY_SEED)
        )

        while True:
            active = status == RUNNING
            if not bool(active.any()):
                break

            y = torch.cat((x, U, V), dim=1)
            y_try, err = self.rk.attempt(self._rhs, y, h)
            at_floor = h.abs() <= self.hmin
            accept = active & ((err <= 1.0) | at_floor)

            # Locate the wall for lines that left the domain during the step.
            crossed = accept & ~self.geom.inside(y_try[:, :3])
            y_end = y_try
            if bool(crossed.any()):
                idx = crossed.nonzero(as_tuple=False).squeeze(1)
                yc, hc = y[idx], h[idx]
                if self._exit is not None:
                    frac = self._exit(yc[:, :3], y_try[idx, :3]).to(dtype=dtype)
                else:
                    lo = torch.zeros(idx.numel(), dtype=dtype, device=dev)
                    hi = torch.ones(idx.numel(), dtype=dtype, device=dev)
                    for _ in range(self.n_bisect):
                        mid = 0.5 * (lo + hi)
                        with torch.no_grad():
                            ym, _ = self.rk.attempt(self._rhs, yc, hc * mid)
                        inside_mid = self.geom.inside(ym[:, :3])
                        lo = torch.where(inside_mid, mid, lo)
                        hi = torch.where(inside_mid, hi, mid)
                    frac = torch.where(inside_mid, hi, lo)
                with torch.no_grad():
                    y_cut, _ = self.rk.attempt(self._rhs, yc, hc * frac)
                y_end = y_try.index_copy(0, idx, y_cut)
                ds_cut = (hc * frac).abs()
            else:
                ds_cut = torch.zeros(0, dtype=dtype, device=dev)

            ds = torch.where(accept, h.abs(), zero)
            if ds_cut.numel():
                ds = ds.index_copy(0, crossed.nonzero(as_tuple=False).squeeze(1), ds_cut)
            ds = torch.where(accept, ds, zero)
            keep = accept.unsqueeze(1)
            x = torch.where(keep, y_end[:, :3], x)
            U = torch.where(keep, y_end[:, 3:6], U)
            V = torch.where(keep, y_end[:, 6:9], V)

            finite_end = torch.isfinite(y_end).all(dim=-1)
            g_end = self._twist_at(x)
            g_end = torch.where(torch.isfinite(g_end), g_end, torch.zeros_like(g_end))
            # Arclength element is positive on both legs: the twist integral is
            # field-aligned, and the two legs together cover B -> F.
            twist = twist + torch.where(accept, 0.5 * (g + g_end) * ds, zero)
            length = length + torch.where(accept, ds, zero)
            g = torch.where(accept, g_end, g)
            steps = steps + accept.long()

            # Advance the step size: grow/shrink after a success, shrink on retry.
            proposed = self.rk.next_dt(h, torch.where(torch.isfinite(err), err, torch.ones_like(err)))
            proposed = proposed.abs().clamp(min=self.hmin, max=self.hmax)
            proposed = torch.where(h < 0, -proposed, proposed)
            shrunk = h * 0.5
            h = torch.where(accept, proposed, shrunk)

            done = accept & crossed
            status = torch.where(done, torch.full_like(status, HIT_BOUNDARY), status)
            status = torch.where(
                accept & ~finite_end, torch.full_like(status, INVALID), status
            )
            status = torch.where(
                accept & ~crossed & (steps >= self.max_steps),
                torch.full_like(status, MAX_STEPS),
                status,
            )

        B_end = self.geom.sample(x)[0] if self._sample is not None else self.geom.Bvec(x)
        return {
            "point": x,
            "U": U,
            "V": V,
            "B": B_end.pow(2).sum(-1).sqrt(),
            "length": length,
            "twist": twist,
            "status": status,
            "steps": steps,
        }

    # ------------------------------------------------------------ full pass
    def run(self, points):
        """Evaluate ``Q``, ``L`` and ``T`` for ``points`` (N, 3)."""
        fwd = self._trace(points, +1.0)
        bwd = self._trace(points, -1.0)

        # Resolve each leg's transported tangents in the frame of its *own*
        # endpoint: only then does |det A| = B0/BF hold by flux conservation,
        # which is what keeps Q at or above its floor of 2.
        frame_f = orthonormal_frame(self.geom.bhat(fwd["point"]))
        frame_b = orthonormal_frame(self.geom.bhat(bwd["point"]))
        a = frame_components(fwd["U"], fwd["V"], *frame_f)
        c = frame_components(bwd["U"], bwd["V"], *frame_b)
        q, log_q, below_floor = q_from_jacobians(a, c)

        ok = (fwd["status"] == HIT_BOUNDARY) & (bwd["status"] == HIT_BOUNDARY)
        ok = ok & torch.isfinite(log_q)

        def report(status):
            out = torch.full_like(status, REPORT_INVALID)
            out = torch.where(status == HIT_BOUNDARY, torch.zeros_like(out), out)
            out = torch.where(status == MAX_STEPS, torch.full_like(out, REPORT_MAX_STEPS), out)
            out = torch.where(status == BOUNDARY_SEED, torch.full_like(out, REPORT_BOUNDARY_SEED), out)
            return out

        status = torch.maximum(report(fwd["status"]), report(bwd["status"]))
        status = torch.where(ok, torch.zeros_like(status), status)

        return {
            "logq": log_q,
            "q": q,
            "length": fwd["length"] + bwd["length"],
            "twist": fwd["twist"] + bwd["twist"],
            "status": status,
            "below_floor": below_floor,
            "steps": fwd["steps"] + bwd["steps"],
            "start": points,
            "end_forward": fwd["point"],
            "end_backward": bwd["point"],
        }
