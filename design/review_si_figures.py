"""Data for the figures of the SI kernels: their shapes and transforms, and the AD force error of each mode.

For the cases (P, eps, sigma) below, the kernels are B_P, the psi_0 kernel with its bandwidth tuned on the grid sigma
in the box of side BOX r_c (as in review_si_coarsest.py), and the kernel of review_si_coarsest.py for (P, eps), whose
shape does not change with the grid. A grid of spacing h has the ratio sigma = (pi/h)/(c_s/r_c) of its Nyquist
frequency to the band edge of the C^2 Coulomb split of eps.

The shapes are phi(t) on [0, P/2] and |phi_hat(q)| for 0 <= q = h xi <= 6 pi (review_si_kernels_space.csv,
review_si_kernels_frequency.csv), with the jump of the psi_0 kernel and the transforms at 2 pi, 4 pi, the band edge
pi/s, and its alias 2 pi - pi/s in review_si_figures.json. The error of a
mode xi of the band is the relative mean-square AD force error of review_finite_grids.ad_squared_error, with the
pair-optimal influence function and all aliases; its average over the directions of xi at |xi| = rho c_s/r_c, for
10^-3 <= rho <= 1, is eta(rho) (review_si_modes.csv). The error of the kernel, the square root of the average of eta
weighted by the mean-square force chi_hat(c_s rho)^2 of the modes, is that of the tables: review_si_coarsest.evaluate
in the limit of large boxes (spherical rule), and in the box of side BOX r_c (review_si_figures.json).
NumPy only. Run with OPENBLAS_NUM_THREADS=1.
"""
import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import review_si_coarsest as co

DEST = Path(__file__).resolve().parent/'records'
BOX = co.BOX
# Shapes: the coarse grid of P = 5 at 1e-3 and the upsampled grid of P = 4 at 1e-4, each kernel's coarsest grid.
SHAPES = ((5, 1e-3), (4, 1e-4))
# Mode errors: P = 4 at 1e-4 on ESP's grid, twofold upsampled, and on the coarsest grid of the new kernel.
MODES = ((4, 1e-4, 1.), (4, 1e-4, 2.), (4, 1e-4, None))


def record(P, eps):
    rows = json.loads((DEST/'review_si_coarsest_record.json').read_text())['results']
    return next(r for r in rows if r['P'] == P and abs(r['tolerance']-eps) < 1e-15)


def parts(P, eps, sigma):
    """The three kernels on the grid sigma, with the bandwidth of the tuned psi_0 kernel."""
    cw, _ = co.tune(P, sigma, co.band(eps, L=BOX))
    r = record(P, eps)
    return cw, {'spline': co.kernel_parts(P, 'spline'), 'prolate': co.kernel_parts(P, 'prolate', c=cw),
                'design': co.kernel_parts(P, 'design', g=r['legendre_coefficients'])}


def mode_errors(P, sigma, kernel, rho, na=24):
    """eta(rho): the relative mean-square AD error of the modes at |xi| = rho c_s/r_c, averaged over directions by a
    Gauss product rule in cos(theta) and phi on the octant."""
    z, wz = leggauss(na)
    z, wz = .5*z+.5, .5*wz
    p, wp = leggauss(na)
    p, wp = .25*np.pi*(p+1), .25*np.pi*wp
    Z, Q = np.meshgrid(z, p, indexing='ij')
    u = np.stack([np.sqrt(1-Z**2)*np.cos(Q), np.sqrt(1-Z**2)*np.sin(Q), Z], axis=-1).reshape(-1, 3)
    w = np.outer(wz, wp).ravel()
    w = w/w.sum()
    # Coordinates q = h xi on the grid sigma of every node (rho, u).
    q = (np.pi*rho[:, None, None]*u[None, :, :]/sigma).reshape(-1)
    alias, eps, eta, f = co.terms(P, q, kernel)
    shape = (len(rho), len(w), 3)
    alias, eps, eta, q = (x.reshape(shape) for x in (alias, eps, eta, q))
    q2 = np.sum(q*q, axis=2)
    L = -2*np.sum(np.log1p(alias), axis=2)+2*np.log1p(-np.sum(eps, axis=2)/q2)-np.log1p(np.sum(eta, axis=2)/q2)
    return (-np.expm1(L))@w


def shapes():
    info = {}
    t = np.linspace(0, 1, 1601)
    q = np.linspace(0, 6*np.pi, 6001)
    space, frequency = {}, {'q': q}
    for P, eps in SHAPES:
        r = record(P, eps)
        sigma = r['sigma']
        cw, kernels = parts(P, eps, sigma)
        key = f'P{P}'
        space[key+'_t'] = P/2*t
        for name, k in kernels.items():
            phi, _, edge, transform = k
            values = phi(P/2*t)
            # The psi_0 kernel jumps at the end of its support; B_P and the new kernel vanish there.
            values[-1] = edge if name == 'prolate' else 0.
            space[f'{key}_{name}'] = values
            frequency[f'{key}_{name}'] = np.abs(transform(q))
        # The transforms at the aliases 2 pi m of the longest waves and at the alias 2 pi - pi/s of the band edge.
        points = np.array([2*np.pi, 4*np.pi, 2*np.pi-np.pi/sigma, np.pi/sigma])
        values = {name: np.abs(k[3](points)).tolist() for name, k in kernels.items()}
        info[key] = dict(P=P, tolerance=eps, sigma=sigma, c_s=r['c_s'], prolate_cw=cw,
                         prolate_edge=float(kernels['prolate'][2]),
                         transform={name: dict(zip(('2pi', '4pi', 'alias_of_band_edge', 'band_edge'), v))
                                    for name, v in values.items()})
    for name, columns in (('review_si_kernels_space.csv', space), ('review_si_kernels_frequency.csv', frequency)):
        with (DEST/name).open('w') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(list(columns))
            writer.writerows(np.array(list(columns.values())).T.tolist())
    return info


def modes():
    rho = np.geomspace(1e-3, 1, 241)
    columns, info = {'rho': rho}, []
    for P, eps, sigma in MODES:
        r = record(P, eps)
        sigma = r['sigma'] if sigma is None else sigma
        cw, kernels = parts(P, eps, sigma)
        limit, box = co.sphere(eps, depth=40), co.band(eps, L=BOX)
        entry = dict(P=P, tolerance=eps, sigma=sigma, c_s=r['c_s'], prolate_cw=cw,
                     lowest_mode={f'{L:g}': 2*np.pi/(r['c_s']*L) for L in (10., 80.)})
        for name, k in kernels.items():
            columns[f'P{P}_s{sigma:.3f}_{name}'] = mode_errors(P, sigma, k, rho)
            entry[name] = dict(limit=co.evaluate(P, sigma, limit, k), box=co.evaluate(P, sigma, box, k))
        info.append(entry)
        print(entry, flush=True)
    with (DEST/'review_si_modes.csv').open('w') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(list(columns))
        writer.writerows(np.array(list(columns.values())).T.tolist())
    return info


if __name__ == '__main__':
    import sys
    # The errors per mode, for a figure no longer in the paper, only on request.
    out = dict(shapes=shapes(), **(dict(modes=modes()) if '--modes' in sys.argv else {}))
    (DEST/'review_si_figures.json').write_text(json.dumps(out, indent=1)+'\n')
