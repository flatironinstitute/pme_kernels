# Tables and figures of the paper

- `python tables.py` writes `table_5_1.tex` (the smallest order that meets each tolerance on the grid that ESP uses)
  and `table_5_2.tex` (the largest grid spacings for orders 4, 5, and 6) from `../design/records/review_si_coarsest_record.json`.
- `figures/plot_assigned_charge.m`, `figures/plot_si_kernels.m`, and `figures/plot_si_coarsest_curves.m` draw
  Figures 2.1, 5.1, and 5.2 (MATLAB) from the data in `../design/records/`, which `../design/review_si_coarsest.py
  --charge`, `../design/review_si_figures.py`, and `../design/review_si_coarsest.py --curves` produce; the PDFs are
  written next to the scripts.

The complete Coulomb calculations of Section 5.3 and the simulations of water of Section 5.4 are not included.
