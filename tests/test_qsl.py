"""Analytic regression tests for the two back ends.

Every case has a closed-form answer, so a regression shows up as a number that
moves -- not merely as a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from cuQSL import cat, sph

LOG10_2 = np.log10(2.0)


def uniform_field(n=21, spacing=1.0):
    axis = np.linspace(-1.0, 1.0, n) * spacing
    B = np.zeros((n, n, n, 3))
    B[..., 2] = 1.0
    return B, [axis, axis, axis]


def shear_field(n=21, extent=1.0):
    axis = np.linspace(-extent, extent, n)
    y = axis[None, :, None]
    B = np.zeros((n, n, n, 3))
    B[..., 0] = y
    B[..., 2] = 1.0
    return B, [axis, axis, axis]


class TestCartesian:
    def test_uniform_field_gives_q_two(self):
        B, grid = uniform_field()
        solver = cat(B, grid)
        logq, length, twist = solver(np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.4]]))
        np.testing.assert_allclose(logq, LOG10_2, atol=1e-6)
        np.testing.assert_allclose(length, 2.0, atol=1e-6)
        np.testing.assert_allclose(twist, 0.0, atol=1e-9)

    def test_shear_field_matches_analytic_q(self):
        B, grid = shear_field()
        solver = cat(B, grid, step_size=0.1)
        logq, length, twist = solver(np.array([[0.0, 0.5, 0.0]]))
        # Q for the flux-tube definition (Tassev & Savcheva 2017 eq. 11).
        np.testing.assert_allclose(logq[0], 0.716003, atol=5e-3)
        np.testing.assert_allclose(length[0], 2.0 * np.sqrt(1.25), atol=1e-4)
        expected_twist = -(2.0 * np.sqrt(1.25)) / (4.0 * np.pi * 1.25)
        np.testing.assert_allclose(twist[0], expected_twist, atol=1e-3)

    def test_spacing_does_not_change_q(self):
        """Regression for the missing ``/ dxyz`` in the tangent transport."""
        results = []
        for n in (11, 21):
            B, grid = shear_field(n=n)
            solver = cat(B, grid, step_size=0.1)
            logq, _, _ = solver(np.array([[0.0, 0.5, 0.0]]))
            results.append(logq[0])
        np.testing.assert_allclose(results[1], results[0], atol=1e-2)

    def test_q_is_never_below_two(self):
        B, grid = shear_field(n=11)
        solver = cat(B, grid, step_size=0.2)
        solver(np.array([[0.1, 0.3, -0.2], [-0.4, 0.6, 0.1]]))
        logq = solver.last_result["logq"]
        assert np.all(logq[np.isfinite(logq)] >= LOG10_2 - 1e-6)
        assert solver.last_result["below_floor"].sum() == 0

    def test_save_and_load_roundtrip(self, tmp_path):
        B, grid = shear_field(n=11)
        solver = cat(B, grid)
        path = tmp_path / "solver.pkl"
        solver.save(str(path))
        restored = cat.load(str(path))
        first = solver(np.array([[0.0, 0.5, 0.0]]))[0]
        second = restored(np.array([[0.0, 0.5, 0.0]]))[0]
        np.testing.assert_allclose(first, second, atol=1e-12)


class TestSpherical:
    @staticmethod
    def monopole(nr=31, nt=31, nph=61):
        rtp_range = ([1.0, 2.0], [0.3, 2.84], [0.0, 2.0 * np.pi])
        r = np.linspace(*rtp_range[0], nr)
        theta = np.linspace(*rtp_range[1], nt)
        phi = np.linspace(*rtp_range[2], nph)
        R, _, _ = np.meshgrid(r, theta, phi, indexing="ij")
        B = np.zeros((nr, nt, nph, 3))
        B[..., 0] = 1.0 / R**2  # div B = 0
        return B, rtp_range

    def test_radial_monopole(self):
        B, rtp_range = self.monopole()
        solver = sph(B, rtp_range, step_size=0.01, max_steps=50_000)
        seed_rtp = np.array([1.5, 0.8, 0.5])
        seed = np.array([[
            seed_rtp[0] * np.sin(seed_rtp[1]) * np.cos(seed_rtp[2]),
            seed_rtp[0] * np.sin(seed_rtp[1]) * np.sin(seed_rtp[2]),
            seed_rtp[0] * np.cos(seed_rtp[1]),
        ]])
        logq, length, twist = solver(seed)
        np.testing.assert_allclose(logq[0], LOG10_2, atol=5e-3)
        np.testing.assert_allclose(length[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(twist[0], 0.0, atol=1e-4)

    def test_default_grid_does_not_crash(self):
        B, rtp_range = self.monopole(nr=9, nt=9, nph=17)
        solver = sph(B, rtp_range, step_size=0.05, max_steps=5_000)
        logq, length, twist = solver()  # no points -> built-in grid
        assert logq.shape == length.shape == twist.shape


@pytest.mark.parametrize("bad", [np.zeros((4, 4, 4, 2)), np.zeros((4, 4, 3))])
def test_invalid_field_shape_raises(bad):
    with pytest.raises(ValueError):
        cat(bad, [np.linspace(0, 1, 4)] * 3)
