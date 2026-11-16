"""SI kernels that minimize the AD force error with a bound on the third derivative.

The RMS AD force error depends on phi and phi' only through L^2 quantities,
which do not see the values of phi' and phi'' at the ends of the support.
Its infimum over C^2 kernels is therefore its minimum over C^0 kernels,
approached by kernels whose derivatives change in ever thinner layers at
the ends, and the C^2 problem has no minimizer. A bound |phi'''| <= M
restores one, because the C^2 kernels with the bound form a compact set in
C^2. With M = max|B_P'''|, the kernel B_P is admissible.

For P = 4, 5, 6 and the grids s = 1, 2 of review_ad_grids.py (the psi_0
split with c_s = 12.024 and n0 = 40), the program minimizes the error over
N = 16, 24, 32, 40, 48 even PSWFs of bandwidth c_w, for the c_w of the
24-term C^2 kernel of review_ad_grids.py and for that of the tuned psi_0
kernel:
- c2: C^2 only, by multistart BFGS as in review_ad_grids.py with more
  iterations;
- bounded: C^2 and |phi'''(t)| <= M at 400 points of [0, P/2], by an
  augmented Lagrangian method with BFGS inner solves, in coordinates
  whitened by the Gauss-Newton metric of the error plus the mean square of
  phi'''/M, with the BFGS matrix carried from one round to the next.
It reports the error, the largest |phi'''| on 20001 points, the sup-norm
change of phi from the previous N, and the error with a finer cell
quadrature. NumPy only. Run with OPENBLAS_NUM_THREADS=1.
"""
import csv
import json
import sys
from math import comb, factorial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legder, legval

import review_ad_grids as grids
import review_finite_grids as design
from review_complete_si import window_profile

DEST = Path(__file__).resolve().parent/'records'
STEM = 'review_si_bounded'
COUNTS = (16, 24, 32, 40, 48)
# x = 2t/P on [0, 1]; phi''' is odd, and the points cluster at the end of the support.
POINTS = np.sin(.5*np.pi*np.linspace(0, 1, 400))
FINE = np.linspace(0, 1, 20001)
ITERATIONS = 2000
ROUND_ITERATIONS = 1000


def spline_third(P, t):
    """B_P''' for the centered cardinal B-spline of support P >= 4."""
    y = np.asarray(t)+P/2
    return sum((-1)**k*comb(P, k)*np.where(y > k, np.abs(y-k)**(P-4), 0.) for k in range(P+1))/factorial(P-4)


def spline_start(window):
    # The L^2 projection of B_P onto the PSWFs, a start that is nearly admissible.
    x, w = leggauss(4000)
    return (legval(x, window.co.T).T*w[:, None]).T@window_profile(.5*window.P*x, window.P, None)


def third_rows(window, x):
    # Rows map PSWF coefficients to phi''' at t = P x/2, in grid units.
    return (2/window.P)**3*legval(x, legder(window.co.T, 3)).T


def whitened_chart(window, s, scale, G=None):
    # The C^2 chart of review_finite_grids.fit, whitened by the same Gauss-Newton
    # metric, plus the mean square of the rows of G if G is given.
    point, Z, leading = window.chart(2)
    reference = point+Z@leading
    cell, defects, transforms = window.parts[s]
    alias, _, f = window.aliases(reference, s)
    _, derivative = design.squared_error(alias)
    defect = np.einsum('j,jqu->qu', reference, defects)
    J = (np.einsum('jl,jqu->lqu', Z, defects)
         - (Z.T@transforms)[:, :, None]*defect[None, :, :]/f[None, :, None])/f[None, :, None]
    metric = 2*np.einsum('jqu,lqu,q,u->jl', J.conj(), J, derivative, cell.wu).real/scale
    if G is not None:
        metric = metric+Z.T@(G.T@G)@Z/len(G)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    root = np.sqrt(np.maximum(eigenvalues, 1e-12*eigenvalues[-1]))
    return point, Z@(eigenvectors/root), (root[:, None]*eigenvectors.T)@Z.T


def error(window, a, s, scale):
    # The stability constraints of review_ad_grids.py.
    if np.linalg.norm(a) > 10:
        return np.inf, None
    terms, d_terms, f = window.ad_terms(a, s)
    if f.min() < 1e-3:
        return np.inf, None
    value, d_value = design.ad_squared_error(*terms, window.parts[s][0].q)
    return value/scale, sum(x@y for x, y in zip(d_value, d_terms))/scale


def bfgs(fun, x, H, iterations):
    """design.bfgs from a given inverse Hessian approximation, which it also returns."""
    value, grad = fun(x)
    iteration = -1
    for iteration in range(iterations):
        if np.linalg.norm(grad, np.inf) < 2e-9:
            break
        direction = -H@grad
        if direction@grad >= 0:
            H = np.eye(len(x)); direction = -grad
        step = 1.
        for _ in range(45):
            next_value, next_grad = fun(x+step*direction)
            if np.isfinite(next_value) and next_value <= value+1e-4*step*(grad@direction):
                break
            step *= .5
        else:
            break
        dx = step*direction; dg = next_grad-grad
        curvature = dx@dg
        if curvature > 1e-12*np.linalg.norm(dx)*np.linalg.norm(dg):
            V = np.eye(len(x))-np.outer(dx, dg)/curvature
            H = V@H@V.T+np.outer(dx, dx)/curvature
        else:
            H = np.eye(len(x))
        x = x+dx; value, grad = next_value, next_grad
    return value, x, float(np.linalg.norm(grad, np.inf)), iteration+1, H


def minimize(window, s, scale, starts, G=None):
    """Multistart minimization of the loss on the C^2 chart, with G a <= 1 if G is given."""
    point, Zw, to_z = whitened_chart(window, s, scale, G)

    def solve(z, lam, rho, H):
        def objective(z):
            a = point+Zw@z
            value, gradient = error(window, a, s, scale)
            if not np.isfinite(value):
                return np.inf, np.zeros_like(z)
            if G is not None:
                t = np.maximum(0., lam+rho*(G@a-1))
                value += (t@t-lam@lam)/(2*rho)
                gradient = gradient+G.T@t
            return value, Zw.T@gradient
        if not np.isfinite(objective(z)[0]):
            return None
        return bfgs(objective, z, H, ITERATIONS if G is None else ROUND_ITERATIONS)

    results = []
    for start in starts:
        z = to_z@(start-point)
        if G is None:
            out = solve(z, None, None, np.eye(len(z)))
            if out is None:
                continue
            _, z, gradient, iterations, _ = out
            rounds, violation = 0, 0.
        else:
            lam, rho, violation, previous, H = np.zeros(len(G)), 1., np.inf, np.inf, np.eye(len(z))
            for rounds in range(1, 16):
                out = solve(z, lam, rho, H)
                if out is None:
                    break
                _, z, gradient, iterations, H = out
                c = G@(point+Zw@z)-1
                lam = np.maximum(0., lam+rho*c)
                new = max(float(c.max()), 0.)
                value = error(window, point+Zw@z, s, scale)[0]
                if new < 1e-7 and abs(value-previous) < 1e-8*value:
                    break
                if new > .25*violation:
                    rho *= 10
                violation, previous = new, value
            if out is None:
                continue
            violation = max(float((G@(point+Zw@z)-1).max()), 0.)
        a = point+Zw@z
        value = error(window, a, s, scale)[0]
        if np.isfinite(value):
            results.append(dict(loss=value, a=a, violation=violation, projected_gradient=gradient,
                                iterations=iterations, rounds=rounds))
    feasible = [r for r in results if r['violation'] < 1e-5] or results
    best = min(feasible, key=lambda r: r['loss'])
    best['starts'] = len(results)
    return best


def task(item):
    P, s, label, cw, scale, record = item
    t = .5*P*FINE
    M = float(np.abs(spline_third(P, t)).max())
    rng = np.random.default_rng(20261102)
    out, previous = [], {}
    for N in COUNTS:
        size = max(grids.SIZE, N+40)
        window = design.Window(P, cw, N, size=size, scales=(s,))
        verify = design.Window(P, cw, N, 160, size, scales=(s,))
        G = third_rows(window, POINTS)/M
        G = np.vstack([G, -G])
        lead = np.zeros(N); lead[0] = 1/window.beta[0]
        base = [lead, np.zeros(N), spline_start(window)]
        if record is not None and N >= len(record):
            base.append(np.concatenate([record, np.zeros(N-len(record))]))
        for kind in ('c2', 'bounded'):
            if kind == 'c2':
                starts = base+[lead+.025*rng.normal(size=N) for _ in range(4)]
            else:
                # The C^2 minimizer for the same N, B_P, and psi_0.
                starts = [previous['c2'], base[2], lead]
            if kind in previous:
                starts.append(np.concatenate([previous[kind], np.zeros(N-len(previous[kind]))]))
            result = minimize(window, s, scale, starts, G if kind == 'bounded' else None)
            a = result['a']
            profile = a@window.co
            values = legval(FINE, profile)
            third = (2/P)**3*legval(FINE, legder(profile, 3))
            row = dict(P=P, s=s, label=label, cw=cw, N=N, kind=kind, bound=M,
                       error=float(np.sqrt(result['loss']*scale)),
                       verified_error=grids.ad_error(verify, a, s),
                       max_third=float(np.abs(third).max()), max_third_over_bound=float(np.abs(third).max()/M),
                       third_at_end=float(abs(third[-1])),
                       violation=result['violation'], projected_gradient=result['projected_gradient'],
                       iterations=result['iterations'], rounds=result['rounds'], starts=result['starts'],
                       integral=float(window.beta@a), coefficients=a.tolist(),
                       profile=values[::200].tolist())
            if kind in previous:
                old = legval(FINE, previous[kind+'_profile'])
                row['sup_change'] = float(np.abs(values-old).max()/np.abs(values).max())
            previous[kind], previous[kind+'_profile'] = a, profile
            out.append(row)
            print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in row.items()
                   if k not in ('coefficients', 'profile')}, flush=True)
    return out


def main(workers, selection):
    record = {(x['P'], x['name']): x for x in json.loads((DEST/'review_ad_grids_record.json').read_text())['fits']}
    baseline = {(int(r['P']), int(float(r['s']))): r for r in csv.DictReader((DEST/'review_ad_grids.csv').open())}
    items = []
    for P in (4, 5, 6):
        for s in (1, 2):
            if selection and (P, s) not in selection:
                continue
            fit = record[P, f'c2_s{s}']
            scale = float(baseline[P, s]['prolate'])**2
            items.append((P, s, 'c2_record', fit['cw'], scale, np.array(fit['coefficients'])))
            items.append((P, s, 'psi0_tuned', float(baseline[P, s]['prolate_cw']), scale, None))
    if workers > 1:
        with Pool(workers) as pool:
            rows = [r for out in pool.map(task, items, chunksize=1) for r in out]
    else:
        rows = [r for item in items for r in task(item)]
    name = STEM if not selection else STEM+'_'+'_'.join(f'{P}{s}' for P, s in selection)
    (DEST/(name+'_record.json')).write_text(json.dumps(dict(
        counts=COUNTS, constraint_points=len(POINTS), iterations=ITERATIONS,
        optimizer='multistart; augmented Lagrangian with BFGS inner solves for the bound', rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    workers = int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 1
    selection = [tuple(int(c) for c in x) for x in sys.argv[sys.argv.index('--only')+1].split(',')] \
        if '--only' in sys.argv else []
    main(workers, selection)
