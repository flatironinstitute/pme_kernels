# pme_kernels

Kernels for particle mesh Ewald (PME) methods, from the paper

> S. Jiang, J. Liang, and L. Lu, *Kernel Design in Particle Mesh Ewald Methods: Splitting, Interpolation, and
> Upsampling*, 2026.

PME methods compute electrostatic interactions with two kernels: a splitting kernel, which divides the interaction into
a local part summed over nearby pairs and a smooth part computed on a grid, and a spreading and interpolation (SI)
kernel, which transfers charges to the grid and potentials and forces back. This repository provides both, as data that
any code can read and as a small NumPy package:

- **C² splits** of `1/r^ν`, `ν = 1, …, 6` (from the Coulomb to the dispersion interaction), for the tolerances
  `10⁻²` to `10⁻⁸`. The local part and its first two derivatives vanish at the cutoff, so the energy of a pair is twice
  continuously differentiable where the pair crosses it, which removes a source of energy drift in long simulations.
- **Optimized SI kernels** `φ*_P` for the Coulomb interaction, orders `P = 4, …, 10`, tolerances `10⁻²` to `10⁻⁸`. Each
  is twice continuously differentiable, conserves charge and dipole by construction (as B-splines do), and meets its
  tolerance on the coarsest grid for its order, never finer than that of the B-spline of the same order.

The kernels depend only on the order and the tolerance, not on the grid or the box, so they are tabulated once. A PME
code adopts them by replacing tables: see [`examples/README.md`](examples/README.md).

## Contents

```
data/
  splits/c2_splits.json      C² window functions of the splits of 1/r^ν (Legendre coefficients, c_s, truncation errors)
  si/si_kernels.json         the optimized SI kernels (Legendre coefficients of g, c_s, s*, h/r_c, errors)
  tables/phi_P*_eps*.txt     the SI kernels and their first two derivatives at 128 parts of each grid interval
pme_kernels/                 Python package (NumPy only)
  si.py                      SIKernel: values, derivatives, Fourier transform, stencil weights; HermiteTable; BSpline
  split.py                   C2Split: window function, local part and its derivative, smooth part, Fourier transform
  influence.py               optimal (analytic differentiation) and natural influence functions
  params.py                  the grid for a tolerance, an order, a cutoff, and a box (Section 4.7 of the paper)
  pme.py                     a minimal particle mesh Ewald method with analytic differentiation
  ewald.py                   an accurate Ewald sum for checks
include/pme_kernels.h        header-only C evaluator of the tables (stencil weights and derivatives)
examples/                    PME with the new kernels against Ewald; notes on using the kernels in a code
tests/                       conservation, smoothness, tables, splits, and PME against Ewald
design/                      the programs that computed the kernels, and their records
paper/                       Tables 5.1 and 5.2 and Figures 2.1, 5.1, and 5.2 of the paper
```

## Quick start

```python
import pme_kernels as pk

k = pk.SIKernel(5, 1e-4)                 # phi_5^* for the tolerance 1e-4
k(0.3), k(0.3, 1)                        # value and derivative at t = 0.3 (grid units)
j0, w, dw = k.stencil(0.3)               # grid points j0, ..., j0 + 4 of a particle at x = 0.3, weights, derivatives

s = pk.C2Split(1, 1e-4, rc=9.0)          # C^2 Coulomb split with the cutoff 9
s.local(4.5), s.local(4.5, 1)            # local part and its derivative at r = 4.5

par = pk.choose(1e-4, 5, rc=9.0, L=50.0) # grid for a box of side 50: par['n'] points per dimension
```

`python examples/pme_coulomb.py` computes the forces of random charges with the new kernels and compares them with an
Ewald sum; with `--eps 1e-4 --P 5` the relative RMS force error is below `10⁻⁴`, about ten times smaller than that of
the B-spline of the same order on the same grid.

Install with `pip install -e .` (the data directory is found next to the package), or set `PME_KERNELS_DATA` to the
data directory.

## Grids

The largest grid spacing `h/r_c = π/(c_s s*)` on which `φ*_P` meets the tolerance (Table 5.2 of the paper; a dash means
that no grid with `s ≤ 4` suffices). A box of side `L` takes the first FFT-friendly grid with at least `L/h` points in
each dimension; the error decreases on finer grids.

| tolerance | c_s   | P = 4 | P = 5 | P = 6 |
|-----------|-------|-------|-------|-------|
| 1e-2      | 6.83  | 0.419 | 0.460 | 0.460 |
| 1e-3      | 9.47  | 0.205 | 0.315 | 0.332 |
| 1e-4      | 12.02 | 0.0894| 0.199 | 0.248 |
| 1e-5      | 14.52 | –     | 0.115 | 0.171 |
| 1e-6      | 16.98 | –     | 0.0589| 0.112 |
| 1e-7      | 19.42 | –     | –     | 0.0720|
| 1e-8      | 21.84 | –     | –     | 0.0451|

Orders 7 to 10 are in `data/si/si_kernels.json`.

## Data formats

- `data/si/si_kernels.json`: for each kernel, `g(t) = Σ_k a_k P_k(t/a)` on `|t| ≤ a = (P − 2)/2` (Legendre
  polynomials `P_k`, coefficients `legendre_coefficients`), and `φ = B₂ * g` with `B₂(t) = max(1 − |t|, 0)`. The
  kernel is even and supported on `[−P/2, P/2]` in units of the grid spacing.
- `data/tables/phi_P{P}_eps{eps}.txt`: columns `t, φ(t), φ'(t), φ''(t)` at `t = k/128`, `0 ≤ t ≤ P/2`. On each part,
  `φ` is the polynomial of degree five that matches the three columns at both ends; this table changes the errors of
  the paper by less than 0.4%.
- `data/splits/c2_splits.json`: the window function `w(t) = Σ_k a_k P_k(t)` on `[−1, 1]`, `w(0) = 1`. With
  `t = r/r_c`, the local part is `L_ν(r) = r^{−ν} R_ν(t)` for `r < r_c`, zero beyond, with `R₁(t) = ∫_t^1 w / ∫_0^1 w`,
  `R₂ = w`, and `R_{ν+2} = R_ν − (t/ν) R_ν'`. For the Coulomb interaction, `χ = w/∫_{−1}^1 w` is the splitting kernel
  and `Ŝ(k) = 4π χ̂(|k| r_c)/|k|²`.

## Tests

```
python tests/run.py        # or: pytest tests
```

## Reproducing the kernels and the paper

[`design/README.md`](design/README.md) regenerates `data/` from scratch; [`paper/README.md`](paper/README.md)
regenerates two tables and three figures of the paper.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0; see [`LICENSE`](LICENSE).
