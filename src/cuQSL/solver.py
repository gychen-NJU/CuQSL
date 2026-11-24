"""Solver front ends: batching, persistence and the public ``__call__`` API."""

from __future__ import annotations

import pickle

import numpy as np
import torch

from .kernel import FieldLineKernel

DEFAULT_BATCH = 262_144


class _BaseSolver:
    """Shared batching / persistence layer for the two geometries."""

    def __init__(self, geometry, *, step_size=None, max_steps=200_000,
                 batch_size=DEFAULT_BATCH, device=None, dtype=torch.float64,
                 atol_pos=1e-9, atol_vec=1e-9, rtol=1e-9):
        self.geometry = geometry
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.geometry.to(self.device)
        self.step_size = float(step_size if step_size is not None else geometry.suggested_step)
        self.kernel = FieldLineKernel(
            geometry, self.step_size, max_steps=max_steps,
            atol_pos=atol_pos, atol_vec=atol_vec, rtol=rtol,
        )
        self.last_result = None

    # ------------------------------------------------------------- call API
    def __call__(self, points, **kwargs):
        """Return ``(logQ, Length, Twist)`` for ``points`` of shape ``(N, 3)``."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if kwargs.get("batch_size"):
            self.batch_size = int(kwargs["batch_size"])
        if kwargs.get("device"):
            self.device = torch.device(kwargs["device"])
            self.geometry.to(self.device)

        chunks = []
        for start in range(0, pts.shape[0], self.batch_size):
            block = torch.as_tensor(pts[start:start + self.batch_size],
                                    dtype=self.dtype, device=self.device)
            chunks.append(self._run_block(block))

        logq = np.concatenate([c["logq"].cpu().numpy() for c in chunks])
        length = np.concatenate([c["length"].cpu().numpy() for c in chunks])
        twist = np.concatenate([c["twist"].cpu().numpy() for c in chunks])
        status = np.concatenate([c["status"].cpu().numpy() for c in chunks])
        below = np.concatenate([c["below_floor"].cpu().numpy() for c in chunks])
        steps = np.concatenate([c["steps"].cpu().numpy() for c in chunks])
        self.last_result = {
            "logq": logq, "length": length, "twist": twist,
            "status": status, "below_floor": below, "steps": steps,
        }
        return logq, length, twist

    def _run_block(self, block):
        with torch.no_grad():
            return self.kernel.run(block)

    # ---------------------------------------------------------- persistence
    def save(self, file_name="cuQSL.pkl"):
        with open(file_name, "wb") as handle:
            pickle.dump(self, handle)
        return file_name

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as handle:
            return pickle.load(handle)


class Diagnostics:
    """Small helpers to summarise a solver run."""

    STATUS_NAMES = {0: "ok", 1: "max_steps", 2: "invalid", 3: "boundary_seed", 4: "unused"}

    @staticmethod
    def summary(result):
        status = np.asarray(result["status"])
        total = status.size
        counts = {name: int((status == code).sum()) for code, name in _STATUS_CODES.items()}
        return {
            "total": int(total),
            "ok_fraction": float(counts.get("ok", 0) / max(total, 1)),
            "counts": counts,
            "below_floor": int(np.asarray(result["below_floor"]).sum()),
            "logq_min": float(np.nanmin(result["logq"])) if total else float("nan"),
            "logq_max": float(np.nanmax(result["logq"])) if total else float("nan"),
        }


# Kernel status codes: 0 ok, 1 hit boundary, 2 max steps, 3 invalid, 4 seed on wall.
_STATUS_CODES = {0: "ok", 1: "max_steps", 2: "invalid", 3: "boundary_seed"}
