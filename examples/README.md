# Examples

## `pme_coulomb.py`

A particle mesh Ewald calculation of the Coulomb forces of random charges with the C² Coulomb split and the optimized
SI kernel, checked against an accurate Ewald sum, with the B-spline of the same order on the same grid for comparison.

```
python examples/pme_coulomb.py --eps 1e-4 --P 5 --N 1000 --L 10
```

## Using the kernels in an existing PME code

A particle mesh Ewald code keeps its structure: the local part is summed over the pairs within the cutoff, and the
smooth part is computed by spreading, FFT, the influence function, inverse FFT, and interpolation. With the new
kernels, four things change, all of them tables:

1. **Local part.** Replace the real-space kernel (for example `erfc(αr)/r` in Gaussian Ewald methods) by the local
   part `L₁(r)` of the C² split and its derivative, `C2Split(1, eps, rc).local(r)` and `.local(r, 1)`; tabulate them
   in `r` as the code tabulates its kernel. The local part vanishes, with its first two derivatives, at `r = r_c`.
2. **Self energy.** The interaction of a charge with its own smooth part is `S(0) = 2χ(0)/r_c`,
   `C2Split(1, eps, rc).self_potential()`; the energy subtracts `S(0) Σ q_i²/2`.
3. **Spreading and interpolation.** Replace the weights of the SI kernel (B-spline weights in SPME and P3M) by those
   of `φ*_P`, from its table: `include/pme_kernels.h` in C, `HermiteTable` in Python. A particle at the coordinate
   `x` (in grid units) is coupled to the grid points `j0, …, j0 + P - 1`, `j0 = floor(x - P/2) + 1`, with the
   one-dimensional weights `φ(x - j)`; analytic differentiation uses `φ'(x - j)/h`.
4. **Influence function.** Compute it once for each grid from the Fourier transforms of the two kernels:
   `pme_kernels.influence.optimal` gives the optimal influence function for analytic differentiation (Ballenegger,
   Cerdà, and Holm; Stern and Calkins), and `pme_kernels.influence.natural` the natural one, `Ŝ/Φ̂²`. The B-spline factors of SPME are not used: the influence function already compensates the SI kernel.

The grid follows from the tolerance, the order, and the cutoff: the spacing is at most `π r_c/(c_s s*)`, given for
every tabulated kernel as `h_over_rc` in `data/si/si_kernels.json` (see `pme_kernels.choose`).
