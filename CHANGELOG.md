# Changelog

## 2.0.2 (frozen branch) -- 2026-11-25

Archived on Zenodo: [10.5281/zenodo.24508867](https://doi.org/10.5281/zenodo.24508867)
(concept DOI for all versions: [10.5281/zenodo.24508866](https://doi.org/10.5281/zenodo.24508866)).

Performance only: results are unchanged to numerical accuracy.

### Throughput

| change | effect |
| --- | --- |
| `B`, `bhat`, `J(bhat)` and `curl B` are packed into one 15-channel interpolation grid (`CartesianGeometry.sample`), so each Runge-Kutta stage issues one `grid_sample` instead of two | 2.5x on CPU alone for the TD slice (`22.6 -> 8.95 ms` per seed) |
| the domain exit point is computed analytically by clipping the step against the axis-aligned box (`geometry` exits via `exit_fraction`) instead of 40 bisection iterations, each of which cost a full RK attempt | one step instead of ~40 per boundary crossing |
| `dtype=` is accepted by both solvers (`torch.float64` default, `torch.float32` halves memory) and forwarded to the geometry; `float32` no longer trips the in-place dtype check on the exit mask | GPU runs are numerically identical to the float64 CPU path to `4e-5` median in `logQ` |
| `kernel.compile(backend=...)` / `bench` hooks wrap the RHS in `torch.compile` | `cudagraphs`: 1.6x when lines are long (4096 seeds: 15.4 s -> 9.4 s); no gain when they are short, since capture dominates. `inductor` (needs `triton-windows<3.3`) shows no gain: the bottleneck is `grid_sample` plus the Python stage loop |

Measured on a 6241-point TD flux-rope slice (81x81x51, `step_size = 0.055`,
median 19 steps per line), RTX 4060 Laptop: 2.0.1 CPU `141.5 s` ->
2.0.2 CPU `55.9 s` -> 2.0.2 GPU `15.7 s` (`float64`) or `16.5 s` (`float32`).

### Tests

`test_float32_works_and_matches_float64` and
`test_analytic_exit_fraction_matches_inside` added; suite is 12 passed.

## 2.0.1 (frozen branch) -- 2026-11-24

### Correctness

| # | defect | fix |
| --- | --- | --- |
| P0-8 | the tangents of both legs were resolved in the *seed* frame. That is exact only while `bhat` stays parallel along the line; a generic field makes `|det A| != B_0/B_F`, which breaks the flux identity and pushes `Q` below its mathematical floor of 2. Found while running the TD flux-rope field: several slices gave `logQ` down to `-0.73` (`Q ~ 0.2`), with the seed-frame determinant off by up to `30x` | each leg is resolved in the orthonormal frame of *its own* endpoint (`geometry.frame_components`), and `Q = \\|A C^{-1}\\|_F^2 / \\|det A C^{-1}\\|` (`qsl.q_from_jacobians`). New regression: `B = (3 z, 0, 1)`, a translation map with `Q = 2` but a threefold change of `|B|` and a rotating `bhat`, gives `logQ = log10(2)` (the old seed-frame resolution gives `0.474`) |

## 2.0.0 (frozen branch) -- 2026-11-24

Rewritten kernel, with every defect found in the 1.0.0 review addressed.  The
public entry points (`cuQSL.cat`, `cuQSL.sph`, `save` / `load`) keep their
names and constructor arguments.

### Correctness

| # | 1.0.0 defect | 2.0.0 behaviour |
| --- | --- | --- |
| P0-1 | tangent Jacobian differentiated per grid index, never divided by the grid spacing, so `Q` depended on the mesh (`cuQSL_cat_scripts.py:34`) | `geometry.cartesian_jacobian` returns physical derivatives; a regression test checks that two grid spacings give the same `Q` |
| P0-2 | backward leg added the twist with the wrong sign (`:303`), cancelling the forward leg | both legs accumulate `0.5 (g + g') |ds|`; the two legs together span `B -> F`; the shear-field twist matches the closed form to `1e-6` |
| P0-3 | `sf[~check_stop][mask] += ...` wrote to an advanced-indexing copy, silently dropping the last step of every line | masked assignments operate on the state tensors themselves |
| P0-4 | spherical stepper clamped a *signed* step with `max(hmin)`, so every backward leg started by turning around (`cuQSL_sph_scripts.py:180`) | step size carries a sign; the floor is applied to its magnitude |
| P0-5 | `self.streched` vs `self.stretched` broke `solver()` with no points (`cuQSL_sph.py:82`) | the spherical solver builds its own grid; `solver()` works, covered by a test |
| P0-6 | Fehlberg 4(5) weights were wrong (`y4` gained `k6/4`, `y5` used non-classical weights): each step advanced `0.632 dt` while `t` advanced `dt`, inflating `L` and `T` by 1.58x | classical tableau in `integrator.py`; `L` now converges with step size and matches the exact arclength |
| P0-7 | `B/|B|` on a zero cell produced NaN that spread through `np.gradient` and every line crossing a null | the field is interpolated first and normalised per query point with a clamped norm; a null simply stalls the line, reported through the status code |

### Robustness

* Out-of-grid sampling uses `padding_mode="border"`, and the domain is tested
  explicitly: a line stops when the segment leaves the domain, and the exit
  point is located by bisection instead of being clamped to the last node.
* Lines that reach `max_steps`, start on a wall, or see a non-finite field are
  flagged in `last_result["status"]` instead of returning a plausible-looking
  `Q`.
* `Q < 2` is reported as `below_floor`; it is never silently clipped.
* No subprocesses, no `shell=True`, no temporary directories, no pickled tensor
  grids: batching happens in-process, so failures raise instead of printing.
* Single-point batches work (`N=1` no longer hits a `squeeze` crash), and the
  three outputs are consistently shaped `(N,)`.
* `pyproject.toml` replaces `setup.py`; `scipy` and `matplotlib` are no longer
  imported, and `pyevtk` is an optional extra.  A clean install imports both
  back ends without extra packages.

### Structure

* One Cartesian kernel (`kernel.py`) serves both back ends; the spherical
  back end resamples its shell onto a Cartesian cube (`spherical.py`), which
  removes the spherical connection terms and the frame-transport error they
  carried.
* Shared helpers (`interpolate.py`, `geometry.py`, `integrator.py`, `qsl.py`)
  are small, documented modules with a single responsibility each.
* Tests in `tests/test_qsl.py` pin the analytic answers: uniform field
  (`logQ = log10 2`), shear field (`logQ`, `L`, `T`), grid-spacing invariance,
  `Q >= 2`, save/load round trip, radial monopole on a shell.

### Known limitations

* The spherical back end interpolates twice (grid -> cube -> line), so its
  accuracy is set by the cube resolution; refinements converge at second order
  but slower than the Cartesian error.
* Only the radial axis supports a stretched (log-spaced) grid; `theta` and
  `phi` must be uniform, and a stretched angle raises `NotImplementedError`.
* `Q` uses the flux-tube normalisation `Q_perp` (ratios of `|B|`).  The
  boundary-normal `Q_d` of Scott et al. (2017) is not exposed.
* The gradient of the interpolated field is piecewise linear per cell, so
  extremely thin current sheets inside one cell are smoothed.
