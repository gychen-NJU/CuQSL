"""Trilinear interpolation of tensor fields on a regular grid.

The grid is addressed in *normalised index space*: each geometry maps a
physical point to coordinates in ``[-1, 1]^3`` (one value per grid axis,
ordered ``x, y, z``).  This module is deliberately geometry-agnostic -- it
only ever sees those normalised coordinates.

Supported channel layouts
-------------------------
``C = 3``   vector field, ``out[n] = b(x_n)``
``C = 9``   matrix field, row-major ``J[..., d, c] = d b_c / d x_d``
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class TensorFieldInterpolator:
    """Trilinear (or linear) interpolation of a ``(nx, ny, nz, C)`` grid."""

    def __init__(self, field, dtype=torch.float64):
        tensor = torch.as_tensor(np.asarray(field), dtype=dtype)
        if tensor.ndim == 5 and tensor.shape[-2:] == (3, 3):
            tensor = tensor.reshape(*tensor.shape[:3], 9)
        if tensor.ndim != 4:
            raise ValueError(f"field must be (nx, ny, nz, C), got {tuple(tensor.shape)}")
        self.shape = tuple(int(s) for s in tensor.shape[:3])
        self.channels = int(tensor.shape[3])
        if self.channels not in (3, 9):
            raise ValueError("only 3 (vector) or 9 (3x3 matrix) channels are supported")
        # (nx, ny, nz, C) -> (1, C, nx, ny, nz)  (grid_sample wants W = last axis)
        self._grid = tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        self.dtype = dtype
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        self._grid = self._grid.to(self.device)
        return self

    def __call__(self, points_norm):
        """Sample at ``points_norm`` (N, 3) given in normalised index coordinates."""
        if not torch.is_tensor(points_norm):
            points_norm = torch.as_tensor(np.asarray(points_norm))
        pts = points_norm.to(dtype=self._grid.dtype, device=self._grid.device)
        # grid_sample reads the last coordinate against W (= nz), so pass (z, y, x).
        grid = torch.stack((pts[:, 2], pts[:, 1], pts[:, 0]), dim=-1)
        grid = grid.view(1, 1, -1, 1, 3)
        out = F.grid_sample(
            self._grid,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        out = out.view(self.channels, -1).t()
        if self.channels == 9:
            out = out.view(-1, 3, 3)
        return out
