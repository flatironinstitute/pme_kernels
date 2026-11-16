# Design programs

These are the programs that computed the kernels of the paper, kept as they were run; they write their results to
`records/`, and `export.py` turns the records into `../data/`. NumPy only; run with `OPENBLAS_NUM_THREADS=1`.

```
cd design
OPENBLAS_NUM_THREADS=1 python review_inverse_splits.py                # C^2 splits of 1/r^nu  -> records/review_inverse_splits_record.json
OPENBLAS_NUM_THREADS=1 python review_si_coarsest.py --workers 12      # SI kernels            -> records/review_si_coarsest_record.json
python export.py                                                       # records -> ../data
```

- `review_inverse_splits.py`: for each `ν = 1, …, 6` and tolerance, the window function of the C² split, an
  expansion in the fewest even PSWFs that admit the conditions `R_ν^(j)(1) = 0`, `j ≤ 2` (null vector of a small linear
  system; a closed form for `ν ≤ 3`), and its truncation error; about 16 seconds in all.
- `review_si_coarsest.py`: for each order `P = 4, …, 10` and tolerance, the minimum of the AD aliasing error over the
  admissible kernels `φ = B₂ * g` (`g` an expansion in 64 even PSWFs) on a sequence of grids, and the coarsest grid on
  which it meets the tolerance, with the B-spline and the `ψ₀` kernel for comparison. Each minimization takes a few
  seconds, and the search for `s*` about a minute for each order and tolerance; the whole study, with its comparisons
  and checks, about 15 minutes on 14 cores. Other options: `--checks` (constraints between collocation points, errors
  on finer grids), `--tables 5 128` (accuracy of the tables), `--curves` and `--charge` (data of Figures 5.2 and 2.1).
- `review_si_figures.py`: data of Figure 5.1.
- The other `review_*.py` files are modules that these programs import: PSWF bases, cell integrals of the alias sums,
  the AD error, and the sequential quadratic programming method.

The docstrings of these research programs may refer to the numbering of earlier drafts of the paper.
