# cuQSL

Quasi-separatrix layer (QSL) diagnostics for magnetic fields: the squashing
factor `Q`, the field-line length `L` and the twist number `T`, integrated
along field lines that are traced from a seed point in both directions until
they reach the domain boundary.

Two back ends share a single Cartesian kernel:

| back end | input | use |
| --- | --- | --- |
| `cuQSL.cat` | `Bxyz` on a uniform `(x, y, z)` grid | box-shaped domains |
| `cuQSL.sph` | `(B_r, B_theta, B_phi)` on an `(r, theta, phi)` grid | spherical shells |

## Install

```bash
pip install -e .
```

Requires Python >= 3.9, NumPy and PyTorch. `pyevtk` is only needed for
`cuQSL.io.data2vts` (`pip install -e ".[io]"`).

## Quick start

```python
import numpy as np
from cuQSL import cat

nx = ny = nz = 64
x = np.linspace(-1.0, 1.0, nx)
y = np.linspace(-1.0, 1.0, ny)
z = np.linspace(-1.0, 1.0, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

Bxyz = np.empty((nx, ny, nz, 3))
Bxyz[..., 0] = Y          # a current sheet along y = 0
Bxyz[..., 1] = 0.0
Bxyz[..., 2] = 1.0

solver = cat(Bxyz, [x, y, z])
points = np.array([[0.0, 0.5, 0.0], [0.1, 0.5, 0.2]])
logQ, length, twist = solver(points)
```

Each output is a NumPy array of shape `(N,)`, one value per seed.  `logQ` is
`log10(Q)`; `Q = 2` (equivalently `logQ = 0.30103`) is the identity map and the
mathematical floor for any mapping, so a value below `log10(2)` means the line
did not close numerically.

### Spherical shells

```python
from cuQSL import sph

r = np.linspace(1.0, 2.0, 48)
theta = np.linspace(0.1, np.pi - 0.1, 48)
phi = np.linspace(0.0, 2.0 * np.pi, 96)
R, T, P = np.meshgrid(r, theta, phi, indexing="ij")

Brtp = np.zeros((r.size, theta.size, phi.size, 3))
Brtp[..., 0] = 1.0 / R**2   # radial monopole: div B = 0
Brtp[..., 1] = 0.0
Brtp[..., 2] = 0.0

solver = sph(Brtp, ([1.0, 2.0], [0.1, np.pi - 0.1], [0.0, 2.0 * np.pi]))
seed = np.array([[1.5 * np.sin(0.8), 0.0, 1.5 * np.cos(0.8)]])
logQ, length, twist = solver(seed)
```

The spherical back end resamples the shell onto a Cartesian cube (spacing taken
from the finest grid spacing) and then runs the same kernel.  This avoids the
spherical connection terms entirely, at the price of one interpolation step;
accuracy therefore scales with the cube resolution, not with the integrator
tolerance.

Calling `solver()` with no points evaluates a coarse built-in grid over the
shell.

## What is computed

With `bhat = B/|B|`, transport along the line the two tangents `U`, `V` of the
seed's perpendicular plane, `dU/ds = (U . grad) bhat`, `dV/ds = (V . grad) bhat`,
from the seed to the forward (`F`) and backward (`B`) boundary, and resolve
them **in the orthonormal frame of the endpoint they reach**.  With `A` the
`2x2` matrix so obtained at `F` and `C` at `B`, and `J = A C^{-1}`,

```
Q = |J|_F^2 / |det J|
```

This is the flux-tube (`Q_perp`) definition: the tangent plane at each end is
perpendicular to the local field, so `|det A| = B_0/B_F` and `|det C| = B_0/B_B`
by flux conservation.  The equivalent textbook form is Tassev & Savcheva (2017)
eq. (11),

```
N^2 = |U_F|^2 |V_B|^2 + |U_B|^2 |V_F|^2 - 2 (U_F . V_F) (U_B . V_B)
Q   = N^2 * B_F B_B / B_0^2
```

and it is the same as `Q = N^2 / |J|` of Scott, Pontin & Hornig (2017) under
that normalisation: the ratio uses `|B|`, not the boundary-normal component.
Resolving both legs in the *seed* frame instead silently breaks the flux
identity whenever `|B|` or `bhat` changes along the line, and can push `Q`
below its mathematical floor of 2.  The twist number is

```
T = (1 / 4 pi) \int (B . curl B) / |B|^2 ds
```

accumulated with a positive arclength element on both legs, so the two legs
together cover the full line `B -> F`.

## Accuracy and cost

* Default dtype is `float64`; pass `dtype=torch.float32` to halve the memory
  footprint, and `device="cuda"` to run the whole batch on the GPU. The step
  size is adaptive (Fehlberg 4(5), classical coefficient table) with per-line
  control; the exit point is found analytically by clipping the step against
  the axis-aligned box (a fallback bisection is not needed).
* On analytic test fields: uniform field gives `logQ = log10(2)` to machine
  precision; a shear field `B = (y, 0, 1)` gives `logQ = 0.7160` (flux-tube
  convention) with a second-order convergence in grid spacing
  (`-8.6e-4`, `-2.1e-4`, `-5.4e-5` for spacing `0.1`, `0.05`, `0.025`), and
  `T = -0.14235` matching the closed form to `1e-6`.
* A slanted field `B = (3 z, 0, 1)` (translation map, `Q = 2`, but `|B|`
  changes threefold and `bhat` rotates) gives `logQ = log10(2)`; resolving in
  the seed frame gives `logQ = 0.474` instead.
* A radial monopole on a spherical shell gives `logQ = 0.30103` within
  `5e-3` at 21x21x41, improving with resolution.
* Seeds are evaluated in batches (`solver.batch_size`, default 262144); the
  solver keeps the last run in `solver.last_result` together with per-point
  status codes, see `cuQSL.Diagnostics.summary`.

### Throughput

`B`, `bhat`, the Jacobian of `bhat` and `curl B` are packed into a single
15-channel grid, so every Runge-Kutta stage costs one `grid_sample` call
instead of three, and a boundary crossing costs one step instead of a
40-iteration bisection. On a 6241-point slice through a Titov-Demoulin flux
rope (81x81x51 grid, `step_size = 0.055`, median 19 steps per line) on an
RTX 4060 Laptop:

| configuration | wall time | per point | vs 2.0.1 |
| --- | --- | --- | --- |
| 2.0.1, CPU, float64 | 141.5 s | 22.6 ms | 1x |
| 2.0.2, CPU, float64 | 55.9 s | 8.95 ms | 2.5x |
| 2.0.2, GPU, float32 | 16.5 s | 2.65 ms | 8.5x |
| 2.0.2, GPU, float64 | 15.7 s | 2.51 ms | 9.0x |

Both dtypes cost the same on the GPU because the loop is launch-bound, not
arithmetic-bound; `float32` buys memory, not speed. GPU `float32` agrees with
CPU `float64` to `4e-5` in the median exponent (`p99 = 1.2e-3`,
`max = 6e-3` on nearly singular QSL ridges).

`solver.kernel.compile(backend="cudagraphs")` wraps the right-hand side in
CUDA graphs. It pays off only when lines are long: at 4096 seeds through the
same slice it turns 15.4 s into 9.4 s (1.6x), but at 6241 seeds, where the
median line is only 20 steps, graph capture dominates and it is slower.
`backend="inductor"` (needs `triton-windows<3.3`) gives no speedup here --
the bottleneck is `grid_sample` plus the Python-level stage loop, not
element-wise fusion. Leave compilation off unless you profiling long lines.

## Status codes

`solver.last_result["status"]` is `0` for a line that reached the boundary in
both directions and non-zero otherwise:

| code | meaning |
| --- | --- |
| 0 | ok |
| 1 | hit `max_steps` before the boundary |
| 2 | invalid (non-finite field or geometry) |
| 3 | seed sits on the boundary |

`last_result["below_floor"]` counts points with `Q < 2`, which flags numerical
failure rather than a physical result.

## Repository layout

```
src/cuQSL/
  interpolate.py   trilinear interpolation on the normalised grid
  geometry.py      grids, curls, Jacobians, spherical basis
  integrator.py    Fehlberg 4(5) step with error control
  kernel.py        two-leg field-line tracing, boundary bisection
  qsl.py           Q / twist formulas
  solver.py        batching, persistence, diagnostics
  cartesian.py     cat back end
  spherical.py     sph back end (resampling + shell predicate)
  io.py            coordinate conversion, VTK export
tests/             analytic regression tests
```

## References

* Tassev, S. & Savcheva, A. 2017, ApJ, 840, 89,
  [doi:10.3847/1538-4357/aa6f06](https://doi.org/10.3847/1538-4357/aa6f06)
* Scott, R. B., Pontin, D. I. & Hornig, G. 2017, ApJ, 848, 117,
  [doi:10.3847/1538-4357/aa8a64](https://doi.org/10.3847/1538-4357/aa8a64)

## License

MIT -- see [LICENSE](LICENSE).
