"""One C^2 SI kernel for each order, for all grids from the base grid to the twofold upsampled one.

For the C^2 Coulomb split of a tolerance (c2_split_fewest.py: the
bandwidth c_s, and the base grid of n0 points whose Nyquist frequency first
reaches c_s/r_c, in the box L = 10 with r_c = 1) and an order P, the
kernel phi(t) = sum_j a_j psi_{2j}^{c_w}(2t/P), |t| <= P/2, solves

    minimize   E_1(phi)
    over       phi even, C^2 on the line, int phi = 1, phi' <= 0 on [0, P/2],
               |phi'''| <= M_P = max|B_P'''|, phi_hat > 0 on the band,
    subject to E_s(phi) <= E_s(B_P) for s = 1.25, 1.5, 1.75, 2,

where E_s is the relative AD aliasing error of the band (Section 4.5) on
the grid h0/s, with the band |xi| <= c_s/r_c weighted by the C^2 split and
the pair-optimal influence of each kernel. The bounds are imposed at 400
points of [0, P/2] that cluster at the end of the support and checked on
20001 points; the errors are checked on every grid of n0 to 2 n0 points.

The problem is solved by a trust-region SQP method with the l_inf exact
penalty, in coordinates of the C^2 chart whitened at the start. Its
Hessians are Gauss-Newton forms of the alias and derivative-alias powers,
and its quadratic subproblems are solved by a primal-dual interior-point
method with one elastic variable. NumPy only. Run with OPENBLAS_NUM_THREADS=1.
"""
import csv
import json
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legder, legval

import ad_grids as grids
import c2_split_fewest as splits
import finite_grids as design
import refinement as ref
import si_bounded as bounded

DEST = Path(__file__).resolve().parent/'records'
STEM = 'si_design'
# The ESP tolerances of the paper and their C^2 splits.
TOLERANCES = (1e-3, 1e-4, 1e-5)
# Grids on which the kernel must not be worse than B_P; s = 1 is the objective.
UPSAMPLED = (1.25, 1.5, 1.75, 2.)
SCALES = (1.,)+UPSAMPLED
POINTS = bounded.POINTS
FINE = np.linspace(0, 1, 20001)
ITERATIONS = 1000
# A numerical barrier: the ratios of the alias sums need phi_hat > 0 on the band. Only kernels of orders well above
# the lowest one that meets the tolerance on the grid s = 1 come near it: there the minimizer drives phi_hat toward
# zero at the Nyquist frequency, with errors far below the tolerance.
FLOOR = 1e-6


def c2_band(tolerance):
    """Base-grid coordinates, band (triples, weights) and n0 for the C^2 Coulomb split of a tolerance."""
    record = json.loads((DEST/'c2_split_fewest_record.json').read_text())
    split = next(x for x in record['splits'] if x['nu'] == 1 and x['tolerance'] == tolerance)
    c, n0, profile = split['c'], split['n0'], np.array(split['legendre_coefficients'])
    ks = np.arange(-int(np.floor(c/ref.dxi)), int(np.floor(c/ref.dxi))+1)
    K1, K2, K3 = np.meshgrid(ks, ks, ks, indexing='ij')
    xi = ref.dxi*np.sqrt(K1*K1+K2*K2+K3*K3)
    keep = (xi > 0) & (xi <= c/ref.rc)
    # The transform of the density chi = w/int w at omega = r_c xi, by Gauss quadrature.
    t, w = leggauss(400)
    values = legval(t, profile)/(2*profile[0])
    omega = ref.rc*xi[keep]
    chi = np.empty(len(omega))
    for start in range(0, len(omega), 4096):
        chi[start:start+4096] = np.cos(np.outer(omega[start:start+4096], t))@(w*values)
    W = (chi/xi[keep])**2
    W = W/W.sum()
    triples = np.sort(np.abs(np.stack([K1[keep], K2[keep], K3[keep]], axis=1)), axis=1)
    index, inverse = np.unique(triples, axis=0, return_inverse=True)
    base = (ref.L/n0)*ref.dxi*np.arange(ks.max()+1)
    return c, n0, base, (index, np.bincount(inverse.ravel(), weights=W))


def loss(window, a, s, band, hessian=False):
    """The squared AD error V_s, its gradient, and a Gauss-Newton form of its Hessian."""
    terms, d_terms, f = window.ad_terms(a, s)
    if f.min() < FLOOR:
        return np.inf, None, None
    cell = window.parts[s][0]
    value, d_value = design.ad_squared_error(*terms, cell.q, band)
    gradient = sum(x@y for x, y in zip(d_value, d_terms))
    if not hessian:
        return value, gradient, None
    _, defects, transforms = window.parts[s]
    ddefects = window.dparts[s][0]
    defect = np.einsum('j,jqu->qu', a, defects)
    ddefect = np.einsum('j,jqu->qu', a, ddefects)
    S0 = f*f+np.abs(defect)**2@cell.wu
    # The alias power is a sum of squares of r = defect/f, and the derivative
    # alias power of rho = ddefect/sqrt(S0); the negative curvature of -q^2 D0 is dropped.
    Jr = (defects-transforms[:, :, None]*(defect/f[:, None])[None])/f[None, :, None]
    Jrho = (ddefects-(transforms*f)[:, :, None]*(ddefect/S0[:, None])[None])/np.sqrt(S0)[None, :, None]
    H = 0.
    for J, c in ((Jr, d_value[0]), (Jrho, d_value[2])):
        weight = np.sqrt(np.maximum(c, 0.)[:, None]*cell.wu[None, :])
        M = (J*weight[None]).reshape(len(a), -1)
        H = H+2*(M.conj()@M.T).real
    return value, gradient, H


def qp(H, g, C, d, tol=1e-10, iterations=80):
    """min 1/2 x'Hx + g'x subject to Cx <= d, by Mehrotra's primal-dual interior-point method."""
    m, n = C.shape
    x = np.zeros(n)
    s = np.maximum(d-C@x, 1.)
    lam = np.ones(m)
    scale = 1+max(np.abs(g).max(), np.abs(d).max())
    for _ in range(iterations):
        rd = H@x+g+C.T@lam
        rp = C@x+s-d
        mu = s@lam/m
        if np.abs(rd).max() < tol*scale and np.abs(rp).max() < tol*scale and mu < tol*scale:
            break
        K = H+(C.T*(lam/s))@C
        try:
            factor = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            factor = np.linalg.cholesky(K+1e-12*np.trace(K)/n*np.eye(n))

        def solve(rc):
            rhs = -rd-C.T@((rc+lam*rp)/s)
            dx = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
            ds = -rp-C@dx
            dl = (rc-lam*ds)/s
            return dx, ds, dl

        def step(v, dv):
            neg = dv < 0
            return min(1., float((-v[neg]/dv[neg]).min())) if neg.any() else 1.

        dx, ds, dl = solve(-s*lam)
        alpha = min(step(s, ds), step(lam, dl))
        mu_aff = (s+alpha*ds)@(lam+alpha*dl)/m
        sigma = (mu_aff/mu)**3
        dx, ds, dl = solve(-s*lam-ds*dl+sigma*mu)
        alpha = .995*min(step(s, ds), step(lam, dl))
        x, s, lam = x+alpha*dx, s+alpha*ds, lam+alpha*dl
    return x, lam


class FactoredWindow(design.Window):
    """The kernels phi = B_2 * g, with g(t) = sum_j b_j psi_{2j}^c(t/a) on [-a, a], a = (P-2)/2.

    The transform sinc(q/2)^2 g_hat(q) vanishes to second order at every nonzero multiple of 2 pi, so the
    integer translates of phi sum to one and those of t phi(t) to zero. With G1 and G2 the first and second
    antiderivatives of g from -a, phi = G2(t+1)-2G2(t)+G2(t-1), and phi'' = g(t+1)-2g(t)+g(t-1), so phi is
    C^2 on the line if and only if g(a) = 0, and phi and phi' vanish at P/2 for every g."""

    def __init__(self, P, c, count=10, nq=80, size=40, q=None, scales=(1, 2)):
        self.P, self.c, self.a = P, c, (P-2)/2
        self.co = design.pswf_basis(c, count, size)
        self.beta = 2*self.a*self.co[:, 0]
        self.edge = np.zeros(count)
        self.gtrace = legval(1., self.co.T)
        # phi, phi' and phi'' at P/2.
        self.traces = np.vstack([np.zeros(count), np.zeros(count), self.gtrace])
        self.I1 = np.polynomial.legendre.legint(self.co.T, lbnd=-1)
        self.I2 = np.polynomial.legendre.legint(self.I1, lbnd=-1)
        y, wy = leggauss(2*size+40)
        self.parts, self.dparts = {}, {}
        for s in scales:
            cell = ref.Cell(P, (design.qbase if q is None else q)/s, nq)
            profile = self.rows(cell.x, 0).reshape(count, *cell.x.shape)
            response = np.einsum('qlu,jlu->jqu', cell.phase, profile)
            ghat = self.a*(legval(y, self.co.T)*wy)@np.cos(np.outer(self.a*y, cell.q))
            transform = (np.sinc(cell.q/(2*np.pi))**2)[None, :]*ghat
            self.parts[s] = (cell, response-transform[:, :, None], transform)
            dprofile = self.rows(cell.x, 1).reshape(count, *cell.x.shape)
            dresponse = np.einsum('qlu,jlu->jqu', cell.phase, dprofile)
            # The sine transform of phi' is -q phi_hat, since phi vanishes at the ends of its support.
            dtransform = -cell.q[None, :]*transform
            self.dparts[s] = (dresponse-1j*dtransform[:, :, None], dtransform)

    def rows(self, t, order):
        """phi_j^(order)(t) for the basis, as an array of shape (count, len(t))."""
        t = np.asarray(t, dtype=float).ravel()
        a = self.a

        def antiderivative(x, k):
            # k = 2, 1: the antiderivatives G2, G1 of g from -a; k = 0: g; k = -1: g'; zero to the left of -a.
            y = np.clip(x/a, -1, 1)
            inside, right = np.abs(x) <= a, x > a
            if k == 0:
                return legval(y, self.co.T)*inside
            if k == -1:
                return legval(y, legder(self.co.T))*inside/a
            if k == 1:
                return a*legval(y, self.I1)*(inside | right)
            return a*a*legval(y, self.I2)*(inside | right)+np.outer(a*legval(1., self.I1), np.maximum(x-a, 0))
        k = {0: 2, 1: 1, 2: 0, 3: -1}[order]
        return antiderivative(t+1, k)-2*antiderivative(t, k)+antiderivative(t-1, k)

    def chart(self, order=None):
        # The normalization and, for C^2, g(a) = 0; phi and phi' vanish at P/2 for every g.
        E = np.vstack([self.beta, self.gtrace]) if order is not None and order >= 2 else self.beta[None]
        target = np.eye(len(E))[0]
        scale = np.linalg.norm(E, axis=1)
        E, target = E/scale[:, None], target/scale
        u, sigma, vt = np.linalg.svd(E, full_matrices=True)
        point = vt[:len(target)].T@((u.T@target)/sigma)
        nullspace = vt[len(target):].T
        leading = np.zeros(len(self.beta)); leading[0] = 1/self.beta[0]
        return point, nullspace, nullspace.T@(leading-point)


def kernel_rows(window, t, order):
    """The values of the basis kernels' derivatives of an order at the points t in [0, P/2]: (count, len(t))."""
    if isinstance(window, FactoredWindow):
        return window.rows(t, order)
    P = window.P
    return (2/P)**order*legval(2*np.asarray(t)/P, legder(window.co.T, order) if order else window.co.T)


@lru_cache(maxsize=None)
def gauss_rule(n):
    """The n-point Gauss-Legendre rule, computed once: leggauss solves a dense n x n eigenvalue problem, which for
    n = 4000 took most of the time of each minimization."""
    return leggauss(n)


def spline_start(window):
    """The L^2 projection of B_P on the basis, or of B_{P-2} on the basis of g for phi = B_2 * g."""
    if not isinstance(window, FactoredWindow):
        return bounded.spline_start(window)
    y, w = gauss_rule(4000)
    return (legval(y, window.co.T).T*w[:, None]).T@bounded.window_profile(window.a*y, window.P-2, None)/(
        (legval(y, window.co.T)**2)@w)


def constraint_points(window, per_piece=None):
    """Points of [0, P/2] at which the bounds are imposed. A PSWF expansion is smooth on [0, P/2), and the points
    cluster at P/2. For phi = B_2 * g, the third derivative jumps at P/2-1 and P/2-2, so the points are Chebyshev
    points of each interval between breakpoints, moved inward by 1e-9 so that both one-sided limits are included."""
    P = window.P
    if not isinstance(window, FactoredWindow):
        return .5*P*POINTS
    # The degree of g grows with the number of terms, and so does the number of points.
    per_piece = per_piece or max(200, 8*len(window.beta))
    breaks = sorted(set([0.]+[P/2-k for k in (2, 1) if P/2-k > 0]+[P/2]))
    out = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        x = np.clip(.5*(1-np.cos(np.pi*np.arange(per_piece+1)/per_piece)), 1e-9, 1-1e-9)
        out.append(lo+(hi-lo)*x)
    return np.concatenate(out)


class Problem:
    """The design problem in whitened coordinates of the C^2 chart."""

    def __init__(self, P, cw, N, band, base, spline_errors, scales=SCALES, monotone=True, bounded_third=True,
                 factored=False):
        # scales[0] is the objective's grid; spline_errors gives E_s(B_P), which normalizes each E_s.
        self.P, self.N, self.band, self.base, self.scales = P, N, band, base, scales
        self.window = (FactoredWindow if factored else design.Window)(P, cw, N, size=max(grids.SIZE, N+40), q=base,
                                                                       scales=scales)
        self.scale = {s: spline_errors[s]**2 for s in scales}
        M = float(np.abs(bounded.spline_third(P, .5*P*FINE)).max())
        slope = float(np.abs(np.diff(bounded.window_profile(.5*P*FINE, P, None))).max()/(.5*P*(FINE[1]-FINE[0])))
        t = constraint_points(self.window)
        G = kernel_rows(self.window, t, 3).T/M if bounded_third else np.zeros((0, N))
        D = kernel_rows(self.window, t[(t > 0) & (t < P/2)], 1).T/slope if monotone else np.zeros((0, N))
        self.M, self.rows = M, np.vstack([G, -G, D])
        self.limits = np.concatenate([np.ones(2*len(G)), np.zeros(len(D))])
        self.point, Z, _ = self.window.chart(2)
        self.Z = Z

    def whiten(self, a):
        # The Gauss-Newton metric of the objective and the constraints at a, plus the mean square of the bound rows.
        # Grids on which a violates the stability bound are left out of the metric.
        H = sum(h/self.scale[s] for s in self.scales
                for h in [loss(self.window, a, s, self.band, True)[2]] if h is not None)
        metric = self.Z.T@H@self.Z if np.ndim(H) else 0.
        metric = metric+self.Z.T@(self.rows.T@self.rows)@self.Z/len(self.rows)
        values, vectors = np.linalg.eigh(metric)
        root = np.sqrt(np.maximum(values, 1e-12*values[-1]))
        self.Zw = self.Z@(vectors/root)
        self.to_z = (root[:, None]*vectors.T)@self.Z.T
        self.A, self.b = self.rows@self.Zw, self.limits-self.rows@self.point

    def coefficients(self, z):
        return self.point+self.Zw@z

    def evaluate(self, z, hessian=False):
        """Objective, nonlinear constraints (E_s/E_s(B_P))^2-1 <= 0, and their derivatives in z."""
        a = self.coefficients(z)
        out = []
        for s in self.scales:
            v, g, H = loss(self.window, a, s, self.band, hessian)
            if not np.isfinite(v):
                return None
            out.append((v/self.scale[s], self.Zw.T@g/self.scale[s],
                        None if H is None else self.Zw.T@H@self.Zw/self.scale[s]))
        f = out[0]
        cons = [(v-1, g, H) for v, g, H in out[1:]]
        return f, cons

    def violation(self, z, cons):
        linear = float(np.max(self.A@z-self.b))
        return max([0., linear]+[c[0] for c in cons])


def subproblem(problem, z, g, H, J, constants, radius, nu):
    """The SQP step d and its multipliers: min g'd + d'Hd/2 + nu tau subject to the linearized constraints
    relaxed by tau >= 0 (constants + J d <= tau, A(z+d) <= b + tau) and |d_i| <= radius."""
    n = len(z)
    Hx = np.zeros((n+1, n+1)); Hx[:n, :n] = H; Hx[n, n] = 1e-12*np.trace(H)/n
    rows = [np.hstack([J, -np.ones((len(J), 1))]), np.hstack([problem.A, -np.ones((len(problem.A), 1))]),
            np.hstack([np.eye(n), np.zeros((n, 1))]), np.hstack([-np.eye(n), np.zeros((n, 1))]),
            np.concatenate([np.zeros(n), [-1.]])[None]]
    limits = np.concatenate([-constants, problem.b-problem.A@z, radius*np.ones(2*n), [0.]])
    x, mult = qp(Hx, np.concatenate([g, [nu]]), np.vstack(rows), limits)
    return x[:n], max(x[n], 0.), np.maximum(mult[:len(J)], 0.)


def sqp(problem, z, iterations=ITERATIONS, radius=1.):
    """Trust-region SQP with the l_inf penalty and second-order corrections; returns z, its multipliers,
    and a history summary."""
    evaluation = problem.evaluate(z, True)
    if evaluation is None:
        return None
    (f, g, Hf), cons = evaluation
    lam = np.zeros(len(cons))
    nu, n = 1., len(z)
    history = dict(accepted=0, corrected=0, rejected=0)
    for iteration in range(iterations):
        H = Hf+sum(l*c[2] for l, c in zip(lam, cons))
        H = .5*(H+H.T)+1e-12*np.trace(H)/n*np.eye(n)
        J = np.array([c[1] for c in cons]).reshape(len(cons), n)
        c0 = np.array([c[0] for c in cons]).reshape(len(cons))
        violation = problem.violation(z, cons)
        d, tau, mult = subproblem(problem, z, g, H, J, c0, radius, nu)
        if violation > 1e-9 and tau > .9*violation and nu < 1e8:
            # The step barely reduces the linearized violation: weight feasibility more.
            nu *= 10
            d, tau, mult = subproblem(problem, z, g, H, J, c0, radius, nu)
        phi = f+nu*violation
        predicted = phi-(f+g@d+.5*d@H@d+nu*tau)
        if predicted <= 1e-15*max(1., abs(phi)):
            break
        accepted = None
        trial = problem.evaluate(z+d, True)
        if trial is not None:
            (ft, gt, Ht), ct = trial
            if (phi-(ft+nu*problem.violation(z+d, ct)))/predicted >= .1:
                accepted = d, trial
            else:
                # Second-order correction: the linearization at z shifted to the constraint values at z + d.
                shifted = np.array([c[0] for c in ct])-J@d
                p, _, _ = subproblem(problem, z, g, H, J, shifted, radius, nu)
                correction = problem.evaluate(z+p, True)
                if correction is not None:
                    (fc, gc, Hc), cc = correction
                    if (phi-(fc+nu*problem.violation(z+p, cc)))/predicted >= .1:
                        accepted = p, correction
                        history['corrected'] += 1
        if accepted is not None:
            step, ((f, g, Hf), cons) = accepted
            z = z+step
            lam = mult
            history['accepted'] += 1
            if np.abs(step).max() > .9*radius:
                radius = min(2*radius, 1e3)
            if problem.violation(z, cons) < 1e-9:
                # At a feasible point the merit function equals f for every penalty.
                nu = max(1., 2*float(np.sum(lam)))
        else:
            history['rejected'] += 1
            radius = .25*np.abs(d).max()
        nu = max(nu, 2*float(np.sum(lam)))
        if radius < 1e-12 or (np.abs(d).max() < 1e-10 and problem.violation(z, cons) < 1e-9):
            break
    history.update(iterations=iteration+1, radius=float(radius), penalty=float(nu))
    return z, lam, history


def kernel_report(problem, a, spline, fine_scales):
    P, window = problem.P, problem.window
    factored = isinstance(window, FactoredWindow)
    values = a@kernel_rows(window, .5*P*FINE, 0)
    third = a@kernel_rows(window, .5*P*FINE, 3)
    slope = a@kernel_rows(window, .5*P*FINE[1:-1], 1)
    errors = {}
    for s in fine_scales:
        w = window if s in window.parts else type(window)(P, window.c, problem.N, size=max(grids.SIZE, problem.N+40),
                                                          q=problem.base, scales=(s,))
        errors[s] = grids.ad_error(w, a, s, problem.band)
    return dict(max_third_over_bound=float(np.abs(third).max()/problem.M), third_at_end=float(abs(third[-1])),
                max_slope=float(slope.max()), minimum=float(values.min()), integral=float(window.beta@a),
                traces=(window.traces@a).tolist(), errors={str(s): e for s, e in errors.items()},
                ratios={str(s): errors[s]/spline[s] for s in fine_scales if s in spline},
                profile=values[::200].tolist(), factored=factored,
                legendre_coefficients=(a@window.co).tolist())


def task(item):
    tolerance, P, cw, counts = item[:4]
    bounded_third = item[4] if len(item) > 4 else True
    factored = item[5] if len(item) > 5 else False
    c, n0, base, band = c2_band(tolerance)
    fine_scales = tuple(n/n0 for n in range(n0, 2*n0+1, 2))
    spline = {s: grids.spline_error(P, s, base=base, band=band) for s in sorted(set(SCALES+fine_scales))}
    out, previous = [], None
    for N in counts:
        t0 = time.time()
        problem = Problem(P, cw, N, band, base, {s: spline[s] for s in SCALES}, bounded_third=bounded_third,
                          factored=factored)
        lead = np.zeros(N); lead[0] = 1/problem.window.beta[0]
        starts = {'psi0': lead, 'spline': spline_start(problem.window)}
        if previous is not None:
            starts['previous'] = np.concatenate([previous, np.zeros(N-len(previous))])
        problem.whiten(starts.get('previous', lead))
        results = []
        for name, start in starts.items():
            z = problem.to_z@(start-problem.point)
            run = sqp(problem, z)
            if run is None:
                continue
            z, lam, history = run
            (f, _, _), cons = problem.evaluate(z)
            results.append(dict(start=name, z=z, objective=f, violation=problem.violation(z, cons),
                                multipliers=lam.tolist(), history=history))
        feasible = [r for r in results if r['violation'] < 1e-6] or results
        best = min(feasible, key=lambda r: r['objective'])
        a = problem.coefficients(best['z'])
        row = dict(tolerance=tolerance, c_s=c, n0=n0, P=P, cw=cw, N=N, bounded_third=bounded_third, factored=factored,
                   start=best['start'],
                   error=float(np.sqrt(best['objective']*problem.scale[1.])), violation=best['violation'],
                   multipliers=best['multipliers'], history=best['history'],
                   starts={r['start']: float(np.sqrt(r['objective']*problem.scale[1.])) for r in results},
                   spline={str(s): v for s, v in spline.items()}, coefficients=a.tolist(),
                   seconds=round(time.time()-t0))
        row.update(kernel_report(problem, a, spline, fine_scales))
        profile = a@kernel_rows(problem.window, .5*P*FINE, 0)
        if previous is not None:
            row['sup_change'] = float(np.abs(profile-previous_profile).max()/np.abs(previous_profile).max())
        previous, previous_profile = a, profile
        out.append(row)
        print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in row.items()
               if k not in ('coefficients', 'profile', 'legendre_coefficients', 'spline', 'errors', 'ratios', 'traces')},
              'ratios', {k: round(v, 3) for k, v in row['ratios'].items() if float(k) in SCALES}, flush=True)
    return out


def test():
    """The interior-point solver against the KKT conditions, and the SQP method on a per-grid problem of si_bounded."""
    rng = np.random.default_rng(1)
    for n, m in ((5, 12), (30, 900)):
        B = rng.normal(size=(n, n)); H = B@B.T+1e-3*np.eye(n)
        g = rng.normal(size=n); C = rng.normal(size=(m, n)); d = rng.uniform(.1, 1., m)
        x, lam = qp(H, g, C, d)
        kkt = max(np.abs(H@x+g+C.T@lam).max(), max(0., (C@x-d).max()), np.abs(lam*(d-C@x)).max(), max(0., -lam.min()))
        print('qp', n, m, 'KKT residual', kkt, flush=True)
        assert kkt < 1e-7
    record = json.loads((DEST/'si_bounded_record.json').read_text())['rows']
    for P, N in ((4, 24), (5, 24)):
        row = next(r for r in record if (r['P'], r['s'], r['label'], r['kind'], r['N']) == (P, 1, 'c2_record', 'bounded', N))
        scale = {1.: row['error']}
        problem = Problem(P, row['cw'], N, None, None, scale, scales=(1.,), monotone=False)
        lead = np.zeros(N); lead[0] = 1/problem.window.beta[0]
        t0 = time.time()
        for name, start in (('previous record', np.array(row['coefficients'])), ('psi0', lead),
                            ('spline', bounded.spline_start(problem.window))):
            problem.whiten(start)
            run = sqp(problem, problem.to_z@(start-problem.point))
            if run is None:
                print(P, N, name, 'start violates the stability bound', flush=True)
                continue
            z, lam, history = run
            (f, _, _), cons = problem.evaluate(z)
            print(P, N, name, 'SQP error', np.sqrt(f)*row['error'], 'record', row['error'], 'violation',
                  problem.violation(z, cons), history, f'{time.time()-t0:.0f}s', flush=True)


def main(workers, tolerances, counts):
    items, bandwidths = [], {}
    for tolerance in tolerances:
        c, n0, base, band = c2_band(tolerance)
        for P in (4, 5, 6):
            # The bandwidths of the psi_0 kernel tuned for AD on the base and the twofold grid; the first is the
            # basis of the design, and at the tolerance 1e-4 the second checks that the kernel does not depend on it.
            tuned = [ref.minimize(lambda b: grids.prolate_error(P, b, s, base=base, band=band),
                                  np.linspace(.3, 2.5, 111)*np.pi*P) for s in (1., 2.)]
            bandwidths[tolerance, P] = [float(cw) for cw, _ in tuned]
            for cw in bandwidths[tolerance, P][:2 if tolerance == 1e-4 else 1]:
                items.append((tolerance, P, float(cw), counts))
    if workers > 1:
        with Pool(workers) as pool:
            rows = [r for out in pool.map(task, items, chunksize=1) for r in out]
    else:
        rows = [r for item in items for r in task(item)]
    name = STEM if tuple(tolerances) == TOLERANCES else STEM+'_'+'_'.join(f'{t:g}' for t in tolerances)
    (DEST/(name+'_record.json')).write_text(json.dumps(dict(
        problem='minimize E_1 over even C^2 phi with int phi = 1, phi\' <= 0 on [0, P/2], |phi\'\'\'| <= max|B_P\'\'\'|, '
                'phi_hat > 0 on the band, subject to E_s <= E_s(B_P) for s in '+str(UPSAMPLED),
        split='the C^2 Coulomb split of c2_split_fewest_record.json', box=dict(L=ref.L, rc=ref.rc),
        constraint_points=len(POINTS), check_points=len(FINE), iterations=ITERATIONS,
        method='trust-region SQP, l_inf penalty, Gauss-Newton Hessians, interior-point subproblems; '
               'starts: the psi_0 kernel, the projection of B_P, and the kernel for the previous N',
        tuned_psi0_bandwidths={f'{t:g}_{P}': v for (t, P), v in bandwidths.items()}, rows=rows), indent=1)+'\n')


def compare_task(item):
    """Band-model errors of the designed kernel, of B_P, and of the psi_0 kernel tuned for each grid."""
    tolerance, P, N, row = item
    c, n0, base, band = c2_band(tolerance)
    a = np.array(row['coefficients'])
    out = []
    for n in list(range(n0, 2*n0+1, 2))+[int(round(x*n0)) for x in (2.5, 3., 3.5, 4.)]:
        s = n/n0
        window = design.Window(P, row['cw'], N, size=max(grids.SIZE, N+40), q=base, scales=(s,))
        verify = design.Window(P, row['cw'], N, 160, size=max(grids.SIZE, N+40), q=base, scales=(s,))
        value, checked = grids.ad_error(window, a, s, band), grids.ad_error(verify, a, s, band)
        assert abs(checked/value-1) < 1e-6
        cw, prolate = ref.minimize(lambda b: grids.prolate_error(P, b, s, base=base, band=band),
                                   np.linspace(.3, 2.5, 111)*np.pi*P)
        assert .3*np.pi*P < cw < 2.5*np.pi*P
        assert abs(grids.prolate_error(P, cw, s, 160, base, band)/prolate-1) < 1e-6
        spline = grids.spline_error(P, s, base=base, band=band)
        assert abs(grids.spline_error(P, s, 320, base, band)/spline-1) < 1e-8
        out.append(dict(tolerance=tolerance, c_s=c, n0=n0, P=P, n=n, s=s, design=value, spline=spline,
                        prolate=prolate, prolate_cw=float(cw)))
        print(out[-1], flush=True)
    return out


def compare(workers, count):
    """The errors of the kernels with count terms and the first bandwidth, on the grids n0 to 4 n0."""
    record = json.loads((DEST/(STEM+'_record.json')).read_text())
    items = []
    for tolerance in TOLERANCES:
        for P in (4, 5, 6):
            cw = record['tuned_psi0_bandwidths'][f'{tolerance:g}_{P}'][0]
            row = next(r for r in record['rows'] if (r['tolerance'], r['P'], r['N']) == (tolerance, P, count)
                       and abs(r['cw']-cw) < 1e-12)
            items.append((tolerance, P, count, row))
    with Pool(workers) as pool:
        rows = [r for out in pool.map(compare_task, items, chunksize=1) for r in out]
    record['reported_count'] = count
    (DEST/(STEM+'_record.json')).write_text(json.dumps(record, indent=1)+'\n')
    with (DEST/(STEM+'.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def unbounded(workers, counts):
    """The problem without the bound on the third derivative, at the tolerance 1e-4, with the first bandwidths."""
    record = json.loads((DEST/(STEM+'_record.json')).read_text())
    items = [(1e-4, P, record['tuned_psi0_bandwidths'][f'{1e-4:g}_{P}'][0], counts, False) for P in (4, 5, 6)]
    with Pool(workers) as pool:
        rows = [r for out in pool.map(task, items, chunksize=1) for r in out]
    (DEST/(STEM+'_unbounded_record.json')).write_text(json.dumps(dict(
        problem='as si_design_record.json without the bound on the third derivative', rows=rows), indent=1)+'\n')


def extended(workers, counts):
    """The problem at the tolerance 1e-4 with the first bandwidths and up to 128 terms, to show convergence in N."""
    record = json.loads((DEST/(STEM+'_record.json')).read_text())
    items = [(1e-4, P, record['tuned_psi0_bandwidths'][f'{1e-4:g}_{P}'][0], counts) for P in (4, 5, 6)]
    with Pool(workers) as pool:
        rows = [r for out in pool.map(task, items, chunksize=1) for r in out]
    (DEST/(STEM+'_extended_record.json')).write_text(json.dumps(dict(
        problem='as si_design_record.json, tolerance 1e-4, the first bandwidths, up to 128 terms', rows=rows), indent=1)+'\n')


def factored_study(workers, tolerances, counts, name='factored'):
    """The problem for the kernels phi = B_2 * g, whose translates conserve charge and dipole exactly."""
    record = json.loads((DEST/(STEM+'_record.json')).read_text())
    items = []
    for tolerance in tolerances:
        for P in (4, 5, 6):
            for cw in record['tuned_psi0_bandwidths'][f'{tolerance:g}_{P}'][:2 if tolerance == 1e-4 else 1]:
                items.append((tolerance, P, cw, counts, True, True))
    with Pool(workers) as pool:
        rows = [r for out in pool.map(task, items, chunksize=1) for r in out]
    (DEST/(STEM+'_'+name+'_record.json')).write_text(json.dumps(dict(
        problem='as si_design_record.json for phi = B_2 * g, g an expansion in even PSWFs on [-(P-2)/2, (P-2)/2]',
        rows=rows), indent=1)+'\n')


if __name__ == '__main__':
    if '--test' in sys.argv:
        test()
    elif '--unbounded' in sys.argv:
        unbounded(3, (16, 24, 32, 40, 48, 56, 64))
    elif '--factored' in sys.argv:
        tolerances = tuple(float(x) for x in sys.argv[sys.argv.index('--tolerances')+1].split(',')) \
            if '--tolerances' in sys.argv else (1e-4,)
        factored_study(int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 6, tolerances,
                       (16, 24, 32, 40, 48, 56, 64))
    elif '--extended' in sys.argv:
        extended(3, (16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128))
    elif '--compare' in sys.argv:
        compare(int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 1,
                int(sys.argv[sys.argv.index('--compare')+1]))
    else:
        workers = int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 1
        tolerances = tuple(float(x) for x in sys.argv[sys.argv.index('--tolerances')+1].split(',')) \
            if '--tolerances' in sys.argv else TOLERANCES
        counts = tuple(int(x) for x in sys.argv[sys.argv.index('--counts')+1].split(',')) \
            if '--counts' in sys.argv else (16, 24, 32, 40, 48)
        main(workers, tolerances, counts)
