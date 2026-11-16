"""Write data/ from the records of the design programs.

records/review_inverse_splits_record.json (review_inverse_splits.py) gives the C^2 splits of 1/r^nu, nu = 1, ..., 6;
records/review_si_coarsest_record.json (review_si_coarsest.py) gives the optimized SI kernels. This program writes
data/splits/c2_splits.json, data/si/si_kernels.json, and the tables data/tables/phi_P{P}_eps{eps}.txt (values and first
two derivatives of each SI kernel at 128 equal parts of every grid interval, as in Section 4.7 of the paper).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORDS = Path(__file__).resolve().parent/'records'
DATA = ROOT/'data'
PARTS = 128


def trim(coefficients):
    """Drop the trailing zero coefficients of a Legendre series."""
    c = list(coefficients)
    while c and c[-1] == 0:
        c.pop()
    return c


def splits():
    rows = json.loads((RECORDS/'review_inverse_splits_record.json').read_text())['rows']
    out = []
    for r in sorted((r for r in rows if r['d'] == 2), key=lambda r: (r['nu'], -r['tolerance'])):
        out.append(dict(nu=r['nu'], tolerance=r['tolerance'], c_s=r['c'], terms=r['terms'],
                        pswf_coefficients=r['relative_coefficients'], legendre_coefficients=trim(r['legendre_coefficients']),
                        truncation_error=r['truncation']))
    doc = dict(
        description='C^2 splits of 1/r^nu (Section 3.3): the window function w on [-1, 1], an expansion in the fewest '
                    'even PSWFs psi_0^c, ..., psi_2m^c (c = c_s) that makes the local part twice continuously '
                    'differentiable at the cutoff, normalized by w(0) = 1.',
        representation='w(t) = sum_k legendre_coefficients[k] P_k(t), |t| <= 1; pswf_coefficients are the coefficients of '
                       'psi_0^c, psi_2^c, ... relative to the first. With t = r/r_c, the local part is '
                       'L_nu(r) = r^-nu R_nu(t) for r < r_c and 0 beyond, R_1(t) = int_t^1 w / int_0^1 w, R_2 = w, '
                       'R_{nu+2} = R_nu - (t/nu) R_nu\'. For nu = 1, chi = w/int_{-1}^1 w is the splitting kernel.',
        bandwidth='c_s from ESP\'s rule chi(1) = eps for the psi_0 split, one bandwidth for all nu',
        truncation_error='relative truncation error (3.5) of the split on the band |xi| <= c_s/r_c, limit of large boxes',
        splits=out)
    (DATA/'splits'/'c2_splits.json').write_text(json.dumps(doc, indent=1)+'\n')
    return len(out)


def si_kernels():
    results = json.loads((RECORDS/'review_si_coarsest_record.json').read_text())['results']
    out = []
    for r in sorted((r for r in results if r.get('sigma') is not None), key=lambda r: (-r['tolerance'], r['P'])):
        out.append(dict(P=r['P'], tolerance=r['tolerance'], c_s=r['c_s'], s_star=r['sigma'],
                        h_over_rc=float(np.pi/(r['c_s']*r['sigma'])), error=r['error'], c_w=r['cw'],
                        legendre_coefficients=trim(r['legendre_coefficients']),
                        comparison=dict(B_P_s_star=r['spline_sigma'], psi0_s_star_box_10rc=r['prolate_sigma'])))
    doc = dict(
        description='Optimized SI kernels phi_P^* (Section 4.5): for each order P and tolerance eps, the admissible '
                    'kernel phi = B_2 * g that meets eps on the coarsest grid, h = h_0/s*, h_0 = pi r_c/c_s, in the '
                    'relative AD aliasing error of the band |xi| <= c_s/r_c (limit of large boxes), with the C^2 Coulomb '
                    'split and the optimal influence function.',
        representation='g(t) = sum_k legendre_coefficients[k] P_k(t/a) for |t| <= a = (P - 2)/2, g = 0 beyond; '
                       'phi = B_2 * g, B_2(t) = max(1 - |t|, 0); phi is supported on [-P/2, P/2], in grid units.',
        fields='c_s: splitting bandwidth (ESP\'s rule); s_star: coarsest grid ratio; h_over_rc: largest grid spacing, '
               'pi/(c_s s*); error: relative AD aliasing error on that grid; c_w: bandwidth of the PSWF basis of g; '
               'comparison: coarsest ratios of B_P (large boxes) and of the psi_0 kernel (box of side 10 r_c), None if '
               'above 4',
        kernels=out)
    (DATA/'si'/'si_kernels.json').write_text(json.dumps(doc, indent=1)+'\n')
    return len(out)


def tables():
    sys.path.insert(0, str(ROOT))
    from pme_kernels.si import SIKernel
    from pme_kernels import _data
    _data.load.cache_clear()
    count = 0
    for e in json.loads((DATA/'si'/'si_kernels.json').read_text())['kernels']:
        k = SIKernel(e['P'], e['tolerance'])
        t, f, df, d2f = k.table(PARTS)
        header = (f"Optimized SI kernel phi_P^* of order P = {e['P']} for the tolerance {e['tolerance']:g} "
                  f"(c_s = {e['c_s']:.6f}, s* = {e['s_star']:.6f}, h/r_c = {e['h_over_rc']:.6f}).\n"
                  f"phi is even and supported on [-P/2, P/2] (grid units); columns: t, phi(t), phi'(t), phi''(t) at "
                  f"t = k/{PARTS}.\nOn each part, evaluate phi by the quintic Hermite polynomial that matches the "
                  f"three columns at both ends.")
        np.savetxt(DATA/'tables'/f"phi_P{e['P']}_eps{e['tolerance']:.0e}.txt", np.c_[t, f, df, d2f],
                   fmt='%.17g', header=header)
        count += 1
    return count


if __name__ == '__main__':
    for sub in ('splits', 'si', 'tables'):
        (DATA/sub).mkdir(parents=True, exist_ok=True)
    print('splits', splits(), 'SI kernels', si_kernels(), 'tables', tables())
