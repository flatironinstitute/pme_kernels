"""SI kernels that meet a tolerance on the coarsest grid.

For a tolerance eps, the C^2 Coulomb split (c2_split_fewest.py) has the bandwidth c_s of c2_bandwidth: ESP's
rule chi(1) = eps for the psi_0 split, raised where needed so that the truncation error of the split is at most eps,
with r_c = 1. A grid of spacing h is described by the ratio sigma = (pi/h)/(c_s/r_c)
of its Nyquist frequency to the band edge; in the variable q = h xi of ad_grids.py, the band model depends on
the grid only through sigma. ESP's rule gives sigma >= 1, close to one in large boxes; sigma = 1 is the smallest
ratio considered here.

The error E(phi; sigma) is the relative AD aliasing error of the band |xi| <= c_s/r_c in the limit of large boxes, an
integral over the ball. For kernels whose translates conserve charge and dipole, the integrand vanishes at the bottom
of the band, and the program evaluates it on the lattice of band() with each shell |k| = const weighted by a radial
trapezoid rule; for every kernel, sphere() gives a spherical product rule, refined geometrically toward the bottom of
the band for the psi_0 kernel, whose relative error there tends to one. In a box of side L, the band is the lattice
of the box's wavenumbers, and the error of the psi_0 kernel grows with L.

For each order P, the kernel phi = B_2 * g of si_design.FactoredWindow (g an expansion in even PSWFs; C^2,
phi' <= 0 on [0, P/2], |phi'''| <= max|B_P'''|) minimizes E(phi; sigma) on one grid; E*(sigma) is the minimum. The
coarsest ratio sigma*(P, eps) is the smallest sigma >= 1 with E*(sigma) <= eps, and the kernel of order P for the
tolerance eps is the minimizer on that grid. Since B_P = B_2 * B_{P-2} is admissible, E*(sigma) <= E(B_P; sigma) and
sigma*(P, eps) is at most the coarsest ratio of B_P. For comparison, the program finds the coarsest ratios of B_P in the
limit and of the psi_0 kernel with its bandwidth tuned on each grid, in the box of side BOX r_c.
Ratios above SIGMA_MAX are reported as missing. Each kernel is checked on finer grids, with 16 more terms, with the
spherical rule, and in boxes of side 10, 20, 40 r_c. With --checks, the program evaluates the constraints of each kernel
between the points at which they are imposed and its errors on finer grids up to SIGMA_MAX; with --tables degree parts,
the change of the errors when each kernel is replaced by its twice continuously differentiable HermiteTable. NumPy
only. Run with OPENBLAS_NUM_THREADS=1.
"""
import json
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder, leggauss, legval

import ad_grids as grids
import c2_split_fewest as splits
import refinement as ref
import si_design as d
from complete_si import factored

DEST = Path(__file__).resolve().parent/'records'
STEM = 'si_coarsest'
TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
ORDERS = (4, 5, 6, 7, 8, 9, 10)
SIGMA_MIN, SIGMA_MAX, STEP = 1., 4., 1.1
# Runs with fewer terms seed the search; the kernel of the search is checked against a longer expansion.
WARMUP, COUNT, LONGER = (24, 32, 48), 64, 80
# Every coarsest ratio is bracketed to this relative width, and the upper end of the bracket is reported.
RESOLUTION = 1e-3
BOX = 10.


def bandwidth(eps):
    """ESP's splitting bandwidth: chi(1) = eps for chi = psi_0^c/int psi_0^c on [-1, 1]."""
    x, w = leggauss(200)
    f = lambda c: np.log(legval(1., ref.coeff(c))/(w@legval(x, ref.coeff(c))))-np.log(eps)
    lo, hi = 1., 40.
    for _ in range(60):
        mid = .5*(lo+hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return .5*(lo+hi)


@lru_cache(maxsize=None)
def c2_bandwidth(eps):
    """The bandwidth of the C^2 Coulomb split for the tolerance eps: ESP's bandwidth, raised where needed so that the
    relative truncation error of the split in the limit of large boxes (inverse_splits.truncation) is at most
    eps, which happens for eps >= 1e-3."""
    import inverse_splits as powers

    def excess(c):
        _, profile, _, _ = splits.fewest(c, 1)
        return np.log(powers.truncation(profile, 1, c, 2)[0]/eps)
    lo = bandwidth(eps)
    if excess(lo) <= 0:
        return lo
    hi = 1.02*lo
    while excess(hi) > 0:
        hi *= 1.02
    for _ in range(40):
        mid = .5*(lo+hi)
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return hi


def chi_transform(c):
    """The transform chi_hat(omega) of the profile of the C^2 split with bandwidth c, normalized by chi_hat(0) = 1."""
    _, profile, _, _ = splits.fewest(c, 1)
    t, w = leggauss(400)
    values = legval(t, profile)/(2*profile[0])
    return lambda omega: np.concatenate([np.cos(np.outer(x, t))@(w*values) for x in
                                         np.array_split(np.atleast_1d(omega), max(1, np.size(omega)//4096))])


def band(eps, R=80., L=None):
    """The band of the C^2 split of eps on the lattice whose wavenumbers 2 pi k/L reach the band edge c_s/r_c at |k| = R.

    With L given, this is the band of the box of side L, R = c_s L/(2 pi r_c), and each mode is weighted by its
    mean-square force, proportional to chi_hat(r_c |xi|)^2/|xi|^2. Otherwise it is a quadrature for the limit of large
    boxes: the modes of each shell |k| = const share the weight that a radial trapezoid rule on the shell radii gives
    the shell in the integral of chi_hat^2 over the radius, normalized by the exact integral. The rule takes the
    integrand to vanish at the bottom of the band, as it does for kernels whose translates conserve charge and dipole.
    In the variable q = h xi, the band is the lattice (pi/(R sigma)) k, |k| <= R. Returns c_s, the coordinates q on
    the grid sigma = 1, and the classes of modes under permutations and sign changes (sorted triples) with weights."""
    c = c2_bandwidth(eps)
    if L is not None:
        R = c*L/(2*np.pi*ref.rc)
    K = int(np.floor(R))
    triples, counts = [], []
    for a in range(K+1):
        for b in range(a, K+1):
            top = int(np.floor(R*R))-a*a-b*b
            if top < b*b:
                break
            e = np.arange(b, int(np.floor(np.sqrt(top)))+1)
            if a == 0 and b == 0:
                e = e[e > 0]
            k = np.stack([np.full(len(e), a), np.full(len(e), b), e], axis=1)
            permutations = np.where((a == b) & (b == e), 1, np.where((a == b) | (b == e), 3, 6))
            signs = 2**((a > 0)+(b > 0)+(e > 0))
            triples.append(k); counts.append(permutations*signs)
    index, count = np.concatenate(triples), np.concatenate(counts)
    shell = np.sum(index**2, axis=1)
    shells, inverse = np.unique(shell, return_inverse=True)
    chi = chi_transform(c)
    rho = np.sqrt(shells)/R
    if L is not None:
        # The mean-square force of a mode is proportional to chi_hat(r_c |xi|)^2/|xi|^2.
        W = count*chi(c*rho)[inverse]**2/shell
        return c, (np.pi/R)*np.arange(K+1), (index, W/W.sum())
    gaps = np.diff(np.concatenate([[0.], rho]))
    radial = .5*(gaps+np.concatenate([gaps[1:], [0.]]))
    x, wx = leggauss(200)
    total = .5*wx@chi(.5*c*(x+1))**2
    population = np.bincount(inverse, weights=count)
    W = count*(radial*chi(c*rho)**2/population)[inverse]/total
    return c, (np.pi/R)*np.arange(K+1), (index, W)


def sphere(eps, depth=12, uniform=16, nr=10, na=16, floor=0.):
    """A spherical product rule for the limit of large boxes: Gauss-Legendre in the radius on the intervals between
    the points k/uniform, the first refined geometrically down to 2^-depth/uniform, and in cos(theta) and phi on the
    octant, by symmetry. Returns c_s, the coordinates q on the grid sigma = 1 (three per node), and the nodes with
    their weights. The psi_0 kernel needs depth = 40, since its relative error tends to one at the bottom of the band;
    for kernels whose translates conserve charge and dipole, the integrand is O(rho^2) there. With floor > 0, the rule
    covers only the radii floor <= |xi|/(c_s/r_c) <= 1, as the modes of a box of side 2 pi r_c/(c_s floor) do."""
    c = c2_bandwidth(eps)
    chi = chi_transform(c)
    breaks = np.array(sorted({floor} | {x for x in [2.**-j/uniform for j in range(1, depth+1)] if x > floor}
                             | {k/uniform for k in range(1, uniform+1)}))
    t, w = leggauss(nr)
    rho = np.concatenate([.5*(hi-lo)*t+.5*(hi+lo) for lo, hi in zip(breaks[:-1], breaks[1:])])
    wr = np.concatenate([.5*(hi-lo)*w for lo, hi in zip(breaks[:-1], breaks[1:])])
    z, wz = leggauss(na)
    z, wz = .5*z+.5, .5*wz
    p, wp = leggauss(na)
    p, wp = .25*np.pi*(p+1), .25*np.pi*wp
    Z, Q = np.meshgrid(z, p, indexing='ij')
    u = np.stack([np.sqrt(1-Z**2)*np.cos(Q), np.sqrt(1-Z**2)*np.sin(Q), Z], axis=-1).reshape(-1, 3)
    W = ((wr*chi(c*rho)**2)[:, None]*np.outer(wz, wp).ravel()[None, :]).ravel()
    n = len(W)
    return c, (np.pi*rho[:, None, None]*u[None, :, :]).ravel(), (np.arange(3*n).reshape(n, 3), W/W.sum())


def kernel_parts(P, kind, c=None, g=None):
    """phi, its derivative inside the support, phi(P/2^-), and the transform of phi, for B_P ('spline'), for the
    psi_0 kernel with bandwidth c ('prolate'), and for B_2 * g with g given by Legendre coefficients in t/a ('design')."""
    if kind == 'spline':
        return (lambda x: grids.window_profile(x, P, None),
                lambda x: grids.window_profile(x+.5, P-1, None)-grids.window_profile(x-.5, P-1, None),
                0., lambda q: np.sinc(q/(2*np.pi))**P)
    if kind == 'prolate':
        co = d.design.pswf_basis(c, 1, grids.SIZE)[0]
        z = P*co[0]
        t, w = leggauss(200)
        return (lambda x: legval(2*x/P, co)/z, lambda x: (2/P)*legval(2*x/P, legder(co))/z, legval(1., co)/z,
                lambda q: (P/2)*np.cos(np.outer(q, P/2*t))@(w*legval(t, co))/z)
    a, g = (P-2)/2, np.asarray(g, dtype=float)
    y, wy = leggauss(len(g)+200)
    return (lambda x: factored(x, P, g, 0), lambda x: factored(x, P, g, 1), 0.,
            lambda q: np.sinc(q/(2*np.pi))**2*(a*np.cos(np.outer(q, a*y))@(wy*legval(y, g))))


def terms(P, q, parts, nu=80, chunk=40000, cuts=None):
    """The per-coordinate inputs (alias, eps, eta) of ad_squared_error at the coordinates q, and the transform, by
    the cell integrals of finite_grids.Window.ad_terms, with exp(i q (l-u)) factored as exp(i q l) exp(-i q u).
    A fifth element of parts, if present, gives v(q) = -int phi'(t) sin(q t) dt for the derivative used inside the
    support, for kernels such as piecewise polynomial tables that can jump between the ends of the support. The cell
    integrals use nu Gauss points between consecutive cuts, the points of the cell at which the kernels of the stencil
    can lose smoothness: by default the ends of the grid intervals of the stencil."""
    phi, dphi, edge, transform = parts[:4]
    sine = parts[4] if len(parts) > 4 else None
    t, w = leggauss(nu)
    if cuts is None:
        cuts = [0., .5, 1.] if P % 2 else [0., 1.]
    u = np.concatenate([(b-a)/2*t+(a+b)/2 for a, b in zip(cuts[:-1], cuts[1:])])
    wu = np.concatenate([(b-a)/2*w for a, b in zip(cuts[:-1], cuts[1:])])
    ell = np.arange(-P//2-1, P//2+3)
    x = ell[:, None]-u[None, :]
    inside = np.abs(x) <= P/2
    F, dF = phi(x)*inside, dphi(x)*inside
    out = [np.empty(len(q)) for _ in range(4)]
    for i in range(0, len(q), chunk):
        qq = q[i:i+chunk]
        E, back = np.exp(1j*np.outer(qq, ell)), np.exp(-1j*np.outer(qq, u))
        f = transform(qq)
        # delta = q f - v, with v the sine transform of the interior derivative.
        delta = qq*f-sine(qq) if sine is not None else 2*edge*np.sin(qq*P/2)
        D0 = np.abs((E@F)*back-f[:, None])**2@wu
        DV = np.abs((E@dF)*back+1j*(qq*f-delta)[:, None])**2@wu
        out[0][i:i+chunk] = D0/f**2
        out[1][i:i+chunk] = qq*delta/f
        out[2][i:i+chunk] = (-delta*(2*qq*f-delta)+DV-qq*qq*D0)/(f*f+D0)
        out[3][i:i+chunk] = f
    return out


def evaluate(P, sigma, data, parts, nu=80, cuts=None, floor=d.FLOOR):
    """E(phi; sigma) on the band data = (c_s, q, (index, weights)) of band() or sphere(); infinite if phi_hat falls
    below floor on the band (by default the numerical barrier of si_design)."""
    _, q, b = data
    alias, eps, eta, f = terms(P, q/sigma, parts, nu, cuts=cuts)
    if f.min() < floor:
        return float('inf')
    value = d.design.ad_squared_error(alias, eps, eta, q/sigma, b)[0]
    return float(np.sqrt(value)) if np.isfinite(value) else float('inf')


def tune(P, sigma, data):
    """The bandwidth c_w of the psi_0 kernel that minimizes its error on the grid sigma, and that error."""
    c, value = ref.minimize(lambda x: evaluate(P, sigma, data, kernel_parts(P, 'prolate', c=x)),
                            np.linspace(.3, 2.5, 111)*np.pi*P)
    return float(c), float(value)


def design(P, sigma, data, cw, N, warm=None):
    """The minimizer of E(phi; sigma) over phi = B_2 * g with N terms; returns the error and the coefficients."""
    c, base, b = data
    spline = grids.spline_error(P, sigma, base=base, band=b)
    problem = d.Problem(P, cw, N, b, base, {sigma: spline}, scales=(sigma,), factored=True)
    lead = np.zeros(N); lead[0] = 1/problem.window.beta[0]
    starts = [d.spline_start(problem.window), lead]
    if warm is not None:
        starts.append(np.concatenate([warm, np.zeros(N-len(warm))]))
    problem.whiten(starts[-1])
    feasible, other = None, None
    for start in starts:
        run = d.sqp(problem, problem.to_z@(start-problem.point))
        if run is None:
            continue
        z, _, history = run
        (f, _, _), cons = problem.evaluate(z)
        violation = problem.violation(z, cons)
        if violation < 1e-6:
            if feasible is None or f < feasible[0]:
                feasible = f, problem.coefficients(z), history['iterations']
        elif other is None or violation < other[3]:
            other = f, problem.coefficients(z), history['iterations'], violation
    # With few terms the constraints may admit no kernel; the least violating one then seeds more terms.
    best = feasible or other
    return float(np.sqrt(best[0])*spline), best[1], best[2]


def coarsest_ratio(f, eps):
    """The smallest sigma in [SIGMA_MIN, SIGMA_MAX] with f(sigma) <= eps, for f decreasing, by bisection in log sigma
    to a bracket of relative width RESOLUTION, whose upper end is returned."""
    if f(SIGMA_MIN) <= eps:
        return SIGMA_MIN
    if f(SIGMA_MAX) > eps:
        return None
    lo, hi = np.log(SIGMA_MIN), np.log(SIGMA_MAX)
    while hi-lo > np.log1p(RESOLUTION):
        mid = .5*(lo+hi)
        lo, hi = (lo, mid) if f(np.exp(mid)) <= eps else (mid, hi)
    return float(np.exp(hi))


def task(item):
    eps, P = item
    t0 = time.time()
    data = band(eps)
    c, base, b = data
    box = band(eps, L=BOX)
    spline = kernel_parts(P, 'spline')
    # The bandwidth of the PSWF basis of g: that of the psi_0 kernel tuned on the coarsest grid in the box.
    cw = tune(P, SIGMA_MIN, box)[0]
    # The minimum E*(sigma) on a geometric sequence of ratios, warm-started from the previous ratio.
    warm = None
    for N in WARMUP:
        warm = design(P, SIGMA_MIN, data, cw, N, warm)[1]
    samples, sigma = [], SIGMA_MIN
    while True:
        value, warm, iterations = design(P, sigma, data, cw, COUNT, warm)
        samples.append(dict(sigma=sigma, error=value, coefficients=warm.tolist(), iterations=iterations,
                            spline=evaluate(P, sigma, data, spline), prolate=tune(P, sigma, box)[1]))
        print(dict(eps=eps, P=P, sigma=round(sigma, 4), error=value, spline=samples[-1]['spline'],
                   prolate=samples[-1]['prolate']), flush=True)
        if value <= eps or sigma >= SIGMA_MAX:
            break
        sigma = min(sigma*STEP, SIGMA_MAX)
    result = dict(tolerance=eps, c_s=c, P=P, cw=cw, samples=samples)
    if samples[-1]['error'] > eps:
        result['sigma'] = None
    elif len(samples) == 1:
        result.update(sigma=SIGMA_MIN, error=samples[0]['error'], coefficients=samples[0]['coefficients'])
    else:
        # Bisection in log sigma between the last two ratios, warm-started from the nearer kernel.
        lo, hi = samples[-2], samples[-1]
        while hi['sigma'] > (1+RESOLUTION)*lo['sigma']:
            mid = float(np.sqrt(lo['sigma']*hi['sigma']))
            value, a, _ = design(P, mid, data, cw, COUNT, np.array(hi['coefficients']))
            point = dict(sigma=mid, error=value, coefficients=a.tolist())
            lo, hi = (lo, point) if value <= eps else (point, hi)
        result.update(sigma=hi['sigma'], error=hi['error'], coefficients=hi['coefficients'])
    # The coarsest ratios of B_P in the limit and of the psi_0 kernel tuned on each grid in the box of side BOX.
    result['spline_sigma'] = coarsest_ratio(lambda s: evaluate(P, s, data, spline), eps)
    result['prolate_sigma'] = coarsest_ratio(lambda s: tune(P, s, box)[1], eps)
    if result['sigma'] is not None:
        # The kernel is the minimizer of the search on its coarsest grid; a longer expansion checks the discretization.
        sigma, a = result['sigma'], np.array(result['coefficients'])
        g = a@d.FactoredWindow(P, cw, COUNT, size=max(grids.SIZE, COUNT+40), q=base, scales=(sigma,)).co
        parts = kernel_parts(P, 'design', g=g)
        longer = design(P, sigma, data, cw, LONGER, a)[0]
        result.update(final_error=result['error'], legendre_coefficients=g.tolist(),
                      longer=dict(count=LONGER, error=longer, change=longer/result['error']-1))
        result['finer'] = {f'{x:g}': evaluate(P, sigma*x, data, parts) for x in (1.25, 1.5, 2., 3.)}
        # The quadratures of the limit, and boxes of side 10, 20 and 40 r_c.
        accurate = sphere(eps)
        result['dense_160'] = evaluate(P, sigma, band(eps, 160.), parts)
        result['sphere'] = evaluate(P, sigma, accurate, parts)
        result['spline_sphere'] = evaluate(P, sigma, accurate, spline)
        result['boxes'] = {f'{L:g}': evaluate(P, sigma, band(eps, L=L), parts) for L in (10., 20., 40.)}
    result['seconds'] = round(time.time()-t0)
    print('RESULT', {k: v for k, v in result.items() if k not in ('samples', 'coefficients', 'final_coefficients',
                                                                  'legendre_coefficients')}, flush=True)
    return result


def safe_task(item):
    """task, with each result saved as it is found, so that a failure loses only its own case."""
    parts = DEST/(STEM+'_parts')
    parts.mkdir(exist_ok=True)
    path = parts/f'{item[0]:g}_P{item[1]}.json'
    if path.exists():
        return json.loads(path.read_text())
    try:
        result = task(item)
    except Exception as failure:
        print('FAILED', item, repr(failure), flush=True)
        return dict(tolerance=item[0], P=item[1], failure=repr(failure))
    path.write_text(json.dumps(result)+'\n')
    return result


def main(workers, tolerances, orders):
    items = [(eps, P) for eps in tolerances for P in orders]
    with Pool(workers) as pool:
        results = pool.map(safe_task, items, chunksize=1)
    name = STEM if (tuple(tolerances), tuple(orders)) == (TOLERANCES, ORDERS) else \
        STEM+'_'+'_'.join(f'{t:g}' for t in tolerances)+'_P'+''.join(str(P) for P in orders)
    (DEST/(name+'_record.json')).write_text(json.dumps(dict(
        definition=f'sigma = (pi/h)/(c_s/r_c); sigma* = smallest sigma >= {SIGMA_MIN:g} with min E(phi; sigma) <= eps, '
                   f'bracketed to a relative width {RESOLUTION:g}',
        kernel=f'phi = B_2 * g, g an expansion in {COUNT} even PSWFs (checked with {LONGER}); C^2; phi\' <= 0 on '
               '[0, P/2]; |phi\'\'\'| <= max|B_P\'\'\'|',
        band='limit of large boxes: the lattice of radius 80 with shells weighted by a radial trapezoid rule; '
             'checked with radius 160, with a spherical product rule, and in boxes of 10, 20, 40 r_c',
        box=BOX, sigma_range=[SIGMA_MIN, SIGMA_MAX], step=STEP, results=results), indent=1)+'\n')


def curve_task(item):
    """Errors on one grid s: the kernel of (P, eps) and B_P in the limit of large boxes, and the psi_0 kernel with its
    bandwidth tuned on the grid, in the box of side BOX r_c."""
    eps, P, result, sigmas = item
    data, box = band(eps), band(eps, L=BOX)
    parts, spline = kernel_parts(P, 'design', g=result['legendre_coefficients']), kernel_parts(P, 'spline')
    rows = []
    for sigma in sigmas:
        cw, tuned = tune(P, sigma, box)
        rows.append(dict(tolerance=eps, P=P, s=float(sigma), design_s=result['sigma'],
                         design=evaluate(P, sigma, data, parts), spline=evaluate(P, sigma, data, spline),
                         prolate=tuned, prolate_cw=cw))
        print(rows[-1], flush=True)
    return rows


def curves(workers, selection):
    record = json.loads((DEST/(STEM+'_record.json')).read_text())
    sigmas = np.exp(np.linspace(np.log(SIGMA_MIN), np.log(SIGMA_MAX), 29))
    items = [(r['tolerance'], r['P'], r, [s]) for r in record['results']
             if (r['tolerance'], r['P']) in selection and r['sigma'] is not None for s in sigmas]
    with Pool(workers) as pool:
        rows = [x for out in pool.map(curve_task, items, chunksize=1) for x in out]
    import csv
    path = DEST/(STEM+'_curves.csv')
    if path.exists():
        # Keep the curves of other cases.
        rows = [dict(r) for r in csv.DictReader(path.open()) if (float(r['tolerance']), int(r['P'])) not in selection]+rows
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def boxes(cases=((1e-4, 4), (1e-4, 6), (1e-6, 6))):
    """On the coarsest grid of the kernel of (P, eps): its error, that of B_P, and that of the psi_0 kernel with its
    bandwidth tuned for each box, in boxes of side 10, 20, 40, 80 r_c and in the limit of large boxes."""
    record = {(r['tolerance'], r['P']): r for r in json.loads((DEST/(STEM+'_record.json')).read_text())['results']}
    rows = []
    for eps, P in cases:
        r = record[eps, P]
        sigma = r['sigma']
        parts, spline = kernel_parts(P, 'design', g=r['legendre_coefficients']), kernel_parts(P, 'spline')
        for L in (10., 20., 40., 80., None):
            if L is None:
                data = sphere(eps)
                cw = tune(P, sigma, sphere(eps, depth=40, nr=6, na=8))[0]
                prolate = evaluate(P, sigma, sphere(eps, depth=40), kernel_parts(P, 'prolate', c=cw))
            else:
                data = band(eps, L=L)
                cw, prolate = tune(P, sigma, data)
            rows.append(dict(tolerance=eps, P=P, s=sigma, L=L, design=evaluate(P, sigma, data, parts),
                             spline=evaluate(P, sigma, data, spline), prolate=prolate, prolate_cw=cw))
            print(rows[-1], flush=True)
    (DEST/(STEM+'_boxes.json')).write_text(json.dumps(rows, indent=1)+'\n')


def growth(cases=((1e-4, 4), (1e-4, 6), (1e-6, 6)), sides=(10., 1e2, 1e3, 1e4, 1e5, 1e6, 1e8)):
    """The error of the psi_0 kernel against the side L of the box, on the coarsest grid of the kernel of (P, eps),
    with the bandwidth tuned in the box of side BOX r_c: the spherical rule restricted to |xi| >= 2 pi/L, which
    approximates the modes of the box for large L, and the limit of large boxes."""
    record = {(r['tolerance'], r['P']): r for r in json.loads((DEST/(STEM+'_record.json')).read_text())['results']}
    rows = []
    for eps, P in cases:
        sigma = record[eps, P]['sigma']
        c = c2_bandwidth(eps)
        cw = tune(P, sigma, band(eps, L=BOX))[0]
        parts = kernel_parts(P, 'prolate', c=cw)
        # The integral of |S'|^2 over a cell: the excess derivative alias power at q = 0.
        defect = float(terms(P, np.zeros(1), parts)[2][0])
        for L in sides+(None,):
            floor = 0. if L is None else 2*np.pi*ref.rc/(c*L)
            rows.append(dict(tolerance=eps, P=P, s=sigma, L=L, prolate_cw=cw, charge_derivative_power=defect,
                             prolate=evaluate(P, sigma, sphere(eps, depth=40, floor=floor), parts)))
            print(rows[-1], flush=True)
    (DEST/(STEM+'_growth.json')).write_text(json.dumps(rows, indent=1)+'\n')


def assigned_charge(eps=1e-4, P=4, n=2000):
    """The deviation S(theta)-1 of the charge S(theta) = sum_l phi(l-theta) that a unit charge assigns to a
    one-dimensional grid, and its derivative, over one cell at the midpoints of n subintervals: B_P, the psi_0 kernel
    with the bandwidth of the record (tuned on the coarsest grid in the box of side BOX r_c), and the kernel of
    (P, eps)."""
    import csv
    r = {(x['tolerance'], x['P']): x for x in json.loads((DEST/(STEM+'_record.json')).read_text())['results']}[eps, P]
    theta = (np.arange(n)+.5)/n-.5
    ell = np.arange(-P, P+1)
    x = ell[:, None]-theta[None, :]
    inside = np.abs(x) <= P/2
    columns = dict(theta=theta)
    for name, parts in (('spline', kernel_parts(P, 'spline')), ('prolate', kernel_parts(P, 'prolate', c=r['cw'])),
                        ('design', kernel_parts(P, 'design', g=r['legendre_coefficients']))):
        phi, dphi = parts[0], parts[1]
        columns[name+'_S'] = np.sum(phi(x)*inside, axis=0)-1
        columns[name+'_dS'] = -np.sum(dphi(x)*inside, axis=0)
    with (DEST/'assigned_charge.csv').open('w') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(list(columns))
        writer.writerows(np.array(list(columns.values())).T.tolist())
    (DEST/'assigned_charge.json').write_text(json.dumps(dict(tolerance=eps, P=P, prolate_cw=r['cw'], **{
        k+'_max': float(np.max(np.abs(v))) for k, v in columns.items() if k != 'theta'}), indent=1)+'\n')


class HermiteTable:
    """A twice continuously differentiable table of the kernel phi = B_2 * g (Legendre coefficients g in t/a) of order
    P: on each of `parts` equal parts of every grid interval of the stencil, the polynomial of a degree >= 5 that
    matches phi, phi' and phi'' at the two ends of the part and phi at degree - 5 interior points (the interior
    Chebyshev extrema of degree degree - 4). Adjacent parts share their end values, so the table is C^2 on the line
    and vanishes with two derivatives at +-P/2. Every grid interval is divided and interpolated in the same way, so
    the sums over the stencil of the table and of t times the table are the interpolants of those of phi: the integer
    translates of the table conserve charge and dipole up to rounding errors."""

    def __init__(self, P, g, degree=5, parts=64):
        from numpy.polynomial import chebyshev as cheb
        self.P, self.parts, self.h, self.cheb = P, parts, 1/parts, cheb
        n = degree-5
        z = np.cos(np.pi*np.arange(1, n+1)/(n+1))
        I = np.eye(degree+1)
        rows = [cheb.chebvander(z, degree)]
        for k in range(3):
            for end in (-1., 1.):
                rows.append((cheb.chebval(end, cheb.chebder(I, k)) if k else cheb.chebval(end, I))[None, :])
        solve = np.linalg.inv(np.vstack(rows))
        ends = -P/2+self.h*np.arange(P*parts+1)
        # phi, phi' and phi'' at the ends of the parts, in the local variable z in [-1, 1] of each part.
        trace = np.stack([factored(ends, P, g, k)*(self.h/2)**k for k in range(3)])
        trace[:, [0, -1]] = 0.
        lo = ends[:-1]
        inner = factored(lo[:, None]+self.h*(.5+.5*z[None, :]), P, g, 0)
        data = np.concatenate([inner, np.stack([trace[k, i:len(lo)+i] for k in range(3) for i in (0, 1)], axis=1)],
                              axis=1)
        self.coefficients = data@solve.T

    def derivative(self, order):
        """The Chebyshev coefficients of the derivative of an order on each part, in the variable t."""
        coefficients = self.coefficients
        for _ in range(order):
            coefficients = self.cheb.chebder(coefficients, axis=1)*(2/self.h)
        return coefficients

    def __call__(self, x, order=0):
        x = np.asarray(x, dtype=float)
        P, h = self.P, self.h
        index = np.clip(np.floor((x+P/2)/h).astype(int), 0, P*self.parts-1)
        z = 2*(x-(-P/2+h*index))/h-1
        c = self.derivative(order)[index]
        # Clenshaw's recurrence for each point with the coefficients of its part.
        b1, b2 = np.zeros(x.shape), np.zeros(x.shape)
        for k in range(c.shape[-1]-1, 0, -1):
            b1, b2 = 2*z*b1-b2+c[..., k], b1
        return (z*b1-b2+c[..., 0])*(np.abs(x) <= P/2)

    def kernel_parts(self, nq=None):
        """phi, phi', phi(P/2^-) = 0 and the transform, for evaluate() with cuts at the ends of the parts."""
        t, w = leggauss(nq or self.coefficients.shape[1]+4)
        P, h = self.P, self.h
        x = ((-P/2+h*np.arange(P*self.parts))[:, None]+h*(.5+.5*t)[None, :]).ravel()
        weights = np.tile(.5*h*w, P*self.parts)
        values = self(x)
        return (lambda y: self(y, 0), lambda y: self(y, 1), 0., lambda q: np.cos(np.outer(q, x))@(weights*values))

    def defects(self, n=4000):
        """The largest jumps of the table and of its first two derivatives at the ends of the parts, including the
        ends of the support, and the RMS over the cell of S - 1, S', S_1 and S_1' for the translates of the table."""
        jumps = []
        for k in range(3):
            c = self.derivative(k)
            left, right = self.cheb.chebval(-1., c.T), self.cheb.chebval(1., c.T)
            jumps.append(float(max(np.abs(right[:-1]-left[1:]).max(), abs(left[0]), abs(right[-1]))))
        u = (np.arange(n)+.5)/n
        x = np.arange(-self.P, self.P+1)[:, None]-u[None, :]
        F, dF = self(x), self(x, 1)
        moments = [F.sum(0)-1, dF.sum(0), (x*F).sum(0), (F+x*dF).sum(0)]
        return jumps, [float(np.sqrt(np.mean(m**2))) for m in moments]


def tables(degree=5, parts=64):
    """The change of the error on the coarsest grid when phi is replaced by its HermiteTable, for every kernel of the
    record, and the defects of the tables. The cell integrals are split at the ends of the parts."""
    record = json.loads((DEST/(STEM+'_record.json')).read_text())
    rows = []
    for r in record['results']:
        if r.get('sigma') is None:
            continue
        P, sigma, eps = r['P'], r['sigma'], r['tolerance']
        data = band(eps)
        g = np.array(r['legendre_coefficients'])
        reference = evaluate(P, sigma, data, kernel_parts(P, 'design', g=g))
        table = HermiteTable(P, g, degree, parts)
        # The condition on phi_hat belongs to the design; the table is evaluated without it.
        tabulated = evaluate(P, sigma, data, table.kernel_parts(), nu=degree+6, cuts=np.linspace(0, 1, parts+1),
                             floor=-np.inf)
        jumps, moments = table.defects()
        rows.append(dict(tolerance=eps, P=P, s=sigma, error=r['final_error'], reference=reference, tabulated=tabulated,
                         change=tabulated/reference-1, jumps=jumps, moments=moments))
        print(rows[-1], flush=True)
    (DEST/(STEM+f'_tables_d{degree}_m{parts}.json')).write_text(json.dumps(dict(
        degree=degree, parts=parts, table='HermiteTable: C^2, values and two derivatives at the ends of each part',
        jumps='largest jumps of the table and of its first two derivatives at the ends of the parts',
        moments="RMS over the cell of S - 1, S', S_1 and S_1'", rows=rows), indent=1)+'\n')


def checks(steps=24):
    """For every kernel of the record: the constraints of the problem between the points at which they are imposed
    (the normalization, g(a), the largest phi' on (0, P/2), and the largest |phi'''| relative to max|B_P'''|), the
    smallest phi_hat on the band |q| <= pi/s* and on |q| <= pi, and its errors on the grids s* (4/s*)^(k/steps),
    k = 0, ..., steps, which show whether it stays below the tolerance on every finer grid up to s = 4; and whether
    the minima sampled in the search for s* decrease with s."""
    import si_bounded as bounded
    record = json.loads((DEST/(STEM+'_record.json')).read_text())
    rows = []
    for r in record['results']:
        if r.get('sigma') is None:
            continue
        P, sigma, eps = r['P'], r['sigma'], r['tolerance']
        a, g = (P-2)/2, np.array(r['legendre_coefficients'])
        M = float(np.abs(bounded.spline_third(P, .5*P*np.linspace(0, 1, 20001))).max())
        dg = legder(g)
        inner = lambda y: legval(np.clip(y/a, -1, 1), dg)*(np.abs(y) < a)/a
        slope, third = -np.inf, 0.
        breaks = sorted(set([0.]+[P/2-k for k in (2, 1) if P/2-k > 0]+[P/2]))
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            x = np.linspace(lo, hi, 20001)[1:-1]
            slope = max(slope, float(factored(x, P, g, 1).max()))
            third = max(third, float(np.abs(inner(x+1)-2*inner(x)+inner(x-1)).max()/M))
        parts = kernel_parts(P, 'design', g=g)
        q = np.linspace(0, np.pi, 4001)
        f = parts[3](q)
        data = band(eps)
        finer = sigma*(SIGMA_MAX/sigma)**(np.arange(steps+1)/steps)
        errors = [evaluate(P, s, data, parts) for s in finer]
        samples = [x['error'] for x in r['samples']]
        rows.append(dict(tolerance=eps, P=P, s=sigma, normalization=float(2*a*g[0]-1), g_end=float(legval(1., g)),
                         largest_slope=slope, third_over_bound=third,
                         smallest_transform_band=float(f[q <= np.pi/sigma].min()), smallest_transform=float(f.min()),
                         finer_grids=finer.tolist(), finer_errors=errors,
                         finer_below_tolerance=bool(max(errors) <= eps*(1+1e-12)),
                         finer_decreasing=bool(np.all(np.diff(errors) <= 0)),
                         samples_decreasing=bool(np.all(np.diff(samples) <= 0))))
        print({k: v for k, v in rows[-1].items() if k not in ('finer_grids', 'finer_errors')}, flush=True)
    (DEST/(STEM+'_checks.json')).write_text(json.dumps(dict(rows=rows), indent=1)+'\n')


if __name__ == '__main__':
    if '--tables' in sys.argv:
        i = sys.argv.index('--tables')
        tables(*(int(x) for x in sys.argv[i+1:i+3]))
        sys.exit()
    if '--checks' in sys.argv:
        checks()
        sys.exit()
    if '--boxes' in sys.argv:
        boxes()
        sys.exit()
    if '--growth' in sys.argv:
        growth()
        sys.exit()
    if '--charge' in sys.argv:
        assigned_charge()
        sys.exit()
    if '--curves' in sys.argv:
        curves(15, [(1e-4, P) for P in (4, 5, 6)]+[(1e-8, P) for P in (7, 8, 10)])
        sys.exit()
    workers = int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 12
    tolerances = tuple(float(x) for x in sys.argv[sys.argv.index('--tolerances')+1].split(',')) \
        if '--tolerances' in sys.argv else TOLERANCES
    orders = tuple(int(x) for x in sys.argv[sys.argv.index('--orders')+1].split(',')) \
        if '--orders' in sys.argv else ORDERS
    main(workers, tolerances, orders)
