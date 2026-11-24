# Changelog

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
