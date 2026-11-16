"""Controlled PSWF-sum design on the finite grids of the review.

NumPy only. Every candidate uses the same band, force weights, exact
cell alias integral, and pair-optimal influence as review_refinement.py.
The optimization is a deterministic multistart BFGS search, not a proof
of a global optimum. Coefficients and verification residuals are saved.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder, legval

import review_refinement as ref


def pswf_basis(c, count=10, size=40):
    k = np.arange(0, 2*size, 2, dtype=float)
    diagonal = k*(k+1) + c*c*(2*k*(k+1)-1)/((2*k+3)*(2*k-1))
    off = c*c*(k[:-1]+2)*(k[:-1]+1)/((2*k[:-1]+3)*np.sqrt((2*k[:-1]+1)*(2*k[:-1]+5)))
    _, vectors = np.linalg.eigh(np.diag(diagonal)+np.diag(off, 1)+np.diag(off, -1))
    co = np.zeros((count, 2*size))
    co[:, ::2] = vectors[:, :count].T*np.sqrt(k+.5)
    co *= np.where(legval(0., co.T) < 0, -1., 1.)[:, None]
    return co


# Coordinate permutations and sign changes have identical tensor factors.
triples = np.sort(np.abs(ref.ks[np.array(ref.ix).T]), axis=1)
indices, inverse = np.unique(triples, axis=0, return_inverse=True)
weights = np.bincount(inverse, weights=ref.W)
qbase = ref.h0*ref.dxi*np.arange(len(ref.ks)//2+1)


def squared_error(alias, band=None):
    index, weight = (indices, weights) if band is None else band
    logs = np.log1p(alias)
    lt = logs[index].sum(axis=1)
    residual = -np.expm1(-2*lt)
    value = weight@residual
    mode_gradient = 2*weight*np.exp(-2*lt)
    derivative = sum(np.bincount(index[:, j], weights=mode_gradient,
                                 minlength=len(alias)) for j in range(3))/(1+alias)
    return value, derivative


def ad_squared_error(alias, eps, eta, q, band=None):
    """Band-model relative AD force error with the pair-optimal influence.

    Per coordinate q, alias is the alias ratio of the kernel, eps = q*delta/f
    the endpoint term that interior differentiation omits, and eta the excess
    derivative alias power S_V/S_0-q^2. For a mode with coordinates q_a the
    error is 1-(1-E)^2/((1+H)*prod(1+alias)^2), with E and H the sums of eps
    and eta divided by sum(q_a^2). Returns the loss and its derivatives with
    respect to the three per-coordinate arrays."""
    index, weight = (indices, weights) if band is None else band
    lt = np.log1p(alias)[index].sum(axis=1)
    q2 = (q*q)[index].sum(axis=1)
    E = eps[index].sum(axis=1)/q2
    H = eta[index].sum(axis=1)/q2
    L = -2*lt+2*np.log1p(-E)-np.log1p(H)
    value = weight@(-np.expm1(L))
    common = -weight*np.exp(L)
    spread = lambda w: sum(np.bincount(index[:, j], weights=w, minlength=len(alias)) for j in range(3))
    return value, (spread(-2*common)/(1+alias), spread(-2*common/((1-E)*q2)), spread(-common/((1+H)*q2)))


def ad_natural_error(alias, eps, eta, q, band=None):
    """The AD error of ad_squared_error with natural deconvolution instead.

    The influence divides by the squared main-copy transform, so each mode
    has error T^2 (1+H)-1+2E, where T=prod(1+alias)."""
    index, weight = (indices, weights) if band is None else band
    lt = np.log1p(alias)[index].sum(axis=1)
    q2 = (q*q)[index].sum(axis=1)
    E = eps[index].sum(axis=1)/q2
    H = eta[index].sum(axis=1)/q2
    growth = np.exp(2*lt)*(1+H)
    value = weight@(growth-1+2*E)
    spread = lambda w: sum(np.bincount(index[:, j], weights=w, minlength=len(alias)) for j in range(3))
    return value, (spread(2*weight*growth)/(1+alias), spread(2*weight/q2), spread(weight*np.exp(2*lt)/q2))


class Window:
    def __init__(self, P, c, count=10, nq=80, size=40, q=None, scales=(1, 2)):
        # q: one-dimensional grid-unit coordinates of the band on the base grid;
        # scales: the upsampling factors s of the grids h0/s on which the kernel is evaluated.
        self.P, self.c = P, c
        self.co = pswf_basis(c, count, size)
        self.beta = P*self.co[:, 0]
        # Traces of the basis and its first two derivatives at the support endpoint.
        self.traces = np.array([legval(1., legder(self.co.T, l)) for l in range(3)])
        self.edge = self.traces[0]
        self.parts = {}
        for s in scales:
            cell = ref.Cell(P, (qbase if q is None else q)/s, nq)
            profile = np.array([legval(cell.xleg, co)*cell.ins for co in self.co])
            response = np.einsum('qlu,jlu->jqu', cell.phase, profile)
            transform = (P/2)*np.array([cell.cos@(cell.w*legval(cell.t, co)) for co in self.co])
            self.parts[s] = (cell, response-transform[:, :, None], transform)
        # Interior derivatives of the basis kernels, for analytic differentiation (AD).
        derivative = legder(self.co.T).T
        self.dparts = {}
        for s in scales:
            cell = self.parts[s][0]
            profile = np.array([(2/P)*legval(cell.xleg, co)*cell.ins for co in derivative])
            response = np.einsum('qlu,jlu->jqu', cell.phase, profile)
            sine = np.sin((P/2)*cell.q[:, None]*cell.t[None, :])
            # The cell mean of the derivative response is i*transform.
            transform = np.array([sine@(cell.w*legval(cell.t, co)) for co in derivative])
            self.dparts[s] = (response-1j*transform[:, :, None], transform)

    def aliases(self, a, s):
        cell, defect_basis, transform_basis = self.parts[s]
        f = a@transform_basis
        defect = np.einsum('j,jqu->qu', a, defect_basis)
        numerator = np.abs(defect)**2@cell.wu
        alias = numerator/f**2
        gradient = (2*np.einsum('qu,jqu,u->qj', defect.conj(), defect_basis,
                               cell.wu).real/f[:, None]**2
                    - 2*alias[:, None]*transform_basis.T/f[:, None])
        return alias, gradient, f

    def ad_terms(self, a, s):
        """Per-coordinate inputs of ad_squared_error and their gradients."""
        cell, defect_basis, transform_basis = self.parts[s]
        ddefect_basis, dtransform_basis = self.dparts[s]
        q, T = cell.q, transform_basis.T
        f = a@transform_basis
        # The AD transform is v=-a@dtransform. Integration by parts gives
        # delta=q*f-v=2 g(P/2) sin(qP/2), which avoids cancellation in q*f-v.
        sine = 2*np.sin(q*self.P/2)
        delta = sine*(a@self.edge)
        d_delta = sine[:, None]*self.edge[None, :]
        defect = np.einsum('j,jqu->qu', a, defect_basis)
        ddefect = np.einsum('j,jqu->qu', a, ddefect_basis)
        D0, DV = np.abs(defect)**2@cell.wu, np.abs(ddefect)**2@cell.wu
        d_D0 = 2*np.einsum('qu,jqu,u->qj', defect.conj(), defect_basis, cell.wu).real
        d_DV = 2*np.einsum('qu,jqu,u->qj', ddefect.conj(), ddefect_basis, cell.wu).real
        S0, d_S0 = f*f+D0, 2*f[:, None]*T+d_D0
        alias = D0/f**2
        d_alias = d_D0/f[:, None]**2-2*alias[:, None]*T/f[:, None]
        eps = q*delta/f
        d_eps = q[:, None]*(d_delta-delta[:, None]*T/f[:, None])/f[:, None]
        numerator = -delta*(2*q*f-delta)+DV-q*q*D0
        d_numerator = (-d_delta*(2*q*f-delta)[:, None]-delta[:, None]*(2*q[:, None]*T-d_delta)
                       + d_DV-(q*q)[:, None]*d_D0)
        eta = numerator/S0
        d_eta = (d_numerator-eta[:, None]*d_S0)/S0[:, None]
        return (alias, eps, eta), (d_alias, d_eps, d_eta), f

    def chart(self, order=None):
        # order=k imposes g^(l)(P/2)=0 for l<=k, so the zero-extended kernel is C^k.
        E = np.vstack([self.beta, self.traces[:0 if order is None else order+1]])
        target = np.eye(len(E))[0]
        if order is not None and order > 0:
            # Derivative traces can exceed the integral row by orders of magnitude.
            scale = np.linalg.norm(E, axis=1)
            E, target = E/scale[:, None], target/scale
        u, sigma, vt = np.linalg.svd(E, full_matrices=True)
        point = vt[:len(target)].T@((u.T@target)/sigma)
        nullspace = vt[len(target):].T
        leading = np.zeros(len(self.beta)); leading[0] = 1/self.beta[0]
        return point, nullspace, nullspace.T@(leading-point)


def bfgs(fun, start, iterations=180):
    x = np.array(start, dtype=float)
    value, grad = fun(x)
    H = np.eye(len(x))
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
        x += dx; value, grad = next_value, next_grad
    return value, x, float(np.linalg.norm(grad, np.inf)), iteration+1


def fit(window, grids, scales, order=None, force='spectral', random_starts=3, iterations=180, band=None,
        start=None):
    # start: optional coefficients of a kernel, projected onto the constraints and added to the starts.
    point, Z, leading = window.chart(order)

    # Whiten the coefficient chart with the Gauss-Newton metric of the
    # normalized cell defects. Raw PSWF coefficients can have curvatures
    # differing by many orders of magnitude on a narrow retained band.
    reference = point+Z@leading
    metric = np.zeros((Z.shape[1], Z.shape[1]))
    for s in grids:
        cell, defects, transforms = window.parts[s]
        alias, _, f = window.aliases(reference, s)
        _, derivative = squared_error(alias, band)
        defect = np.einsum('j,jqu->qu', reference, defects)
        J = (np.einsum('jl,jqu->lqu', Z, defects)
             - (Z.T@transforms)[:, :, None]*defect[None, :, :]/f[None, :, None])/f[None, :, None]
        metric += 2*np.einsum('jqu,lqu,q,u->jl', J.conj(), J, derivative,
                             cell.wu).real/scales[s]/len(grids)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    root = np.sqrt(np.maximum(eigenvalues, 1e-12*eigenvalues[-1]))
    transform = eigenvectors/root
    inverse_transform = root[:, None]*eigenvectors.T
    raw_leading = leading.copy()
    raw_start = None if start is None else Z.T@(start-point)
    Z = Z@transform
    leading = inverse_transform@raw_leading

    def objective(z):
        a = point+Z@z
        value, gradient = 0., np.zeros_like(a)
        for s in grids:
            if force in ('ad', 'ad_natural'):
                terms, d_terms, f = window.ad_terms(a, s)
            else:
                alias, d_alias, f = window.aliases(a, s)
            # Stable, positive deconvolution at every retained coordinate.
            if np.min(f) < 1e-3 or np.linalg.norm(a) > 10:
                return float('inf'), np.zeros_like(z)
            if force in ('ad', 'ad_natural'):
                error = ad_squared_error if force == 'ad' else ad_natural_error
                loss, d_loss = error(*terms, window.parts[s][0].q, band)
                value += loss/scales[s]/len(grids)
                gradient += sum(x@y for x, y in zip(d_loss, d_terms))/scales[s]/len(grids)
                continue
            loss, d_loss = squared_error(alias, band)
            value += loss/scales[s]/len(grids)
            gradient += (d_loss@d_alias)/scales[s]/len(grids)
        return value, Z.T@gradient

    # Deterministic starts include the projected leading prolate.
    rng = np.random.default_rng(20261020)
    starts = [leading, np.zeros_like(leading)]
    starts += [inverse_transform@(raw_leading+.025*rng.normal(size=len(leading))) for _ in range(random_starts)]
    if start is not None:
        starts.append(inverse_transform@raw_start)
    results = [bfgs(objective, start, iterations) for start in starts
               if np.isfinite(objective(start)[0])]
    if not results:
        return None
    result = min(results, key=lambda r: r[0])
    value, z, gradient, iterations = result
    return dict(loss=value, a=point+Z@z, projected_gradient=gradient,
                iterations=iterations, starts=len(results))


def main():
    dest = Path(__file__).resolve().parent/'records'
    with (dest/'review_retuned_refinement.csv').open() as stream:
        baselines = {(int(row['P']), int(row['s'])): {key: float(value) for key, value in row.items()}
                     for row in csv.DictReader(stream)}
    records, fits = [], []
    # Name, design grids, endpoint regularity order, and PSWF count. The C^1 and
    # C^2 fits use two and three more PSWFs than the unconstrained fits, so all
    # three keep nine free coefficients.
    specs = [('sum_s1', (1,), None, 10), ('sum_s2', (2,), None, 10),
             ('edge_s1', (1,), 0, 10), ('edge_s2', (2,), 0, 10), ('joint', (1, 2), None, 10),
             ('c1_s1', (1,), 1, 12), ('c1_s2', (2,), 1, 12),
             ('c2_s1', (1,), 2, 13), ('c2_s2', (2,), 2, 13)]
    for P in (4, 5, 6):
        scales = {s: baselines[P, s]['pro']**2 for s in (1, 2)}
        bandwidths = sorted(set([baselines[P, s]['cw'] for s in (1, 2)]
                                + [factor*np.pi*P for factor in (.2, .3, .4, .5, .6, .8, 1., 1.2, 1.4,
                                                                 1.6, 1.8, 2., 2.4, 2.8, 3.2, 4.)]))
        best = {}
        for c in bandwidths:
            windows = {count: Window(P, c, count) for count in sorted({spec[3] for spec in specs})}
            for name, grids, order, count in specs:
                result = fit(windows[count], grids, scales, order)
                if result is not None and (name not in best or result['loss'] < best[name][1]['loss']):
                    best[name] = windows[count], result, order
        for name, (window, result, order) in best.items():
            a = result['a']
            verify = Window(P, window.c, len(a), nq=160)
            values = {}; worst = 0.; minimum = float('inf')
            for s in (1, 2):
                alias, _, f = window.aliases(a, s)
                value = np.sqrt(squared_error(alias)[0])
                checked = np.sqrt(squared_error(verify.aliases(a, s)[0])[0])
                discrepancy = abs(checked/value-1)
                assert discrepancy < 2e-7
                worst = max(worst, discrepancy); minimum = min(minimum, float(f.min()))
                values[s] = value
            assert abs(window.beta@a-1) < 2e-13
            if order is not None:
                assert abs(window.edge@a) < 2e-13
                assert all(abs(t@a) < 1e-12*np.linalg.norm(t)*np.linalg.norm(a) for t in window.traces[1:order+1])
            item = dict(P=P, name=name, cw=window.c, count=len(a), order=order, coefficients=a.tolist(),
                        native_coefficients=(a/((-1.)**np.arange(len(a))*np.sqrt(2/(4*np.arange(len(a))+1)))).tolist(),
                        legendre_coefficients=(a@window.co).tolist(),
                        endpoint=float(window.edge@a), endpoint_traces=(window.traces@a).tolist(),
                        integral=float(window.beta@a),
                        error_s1=values[1], error_s2=values[2], minimum_response=minimum,
                        quadrature_relative_difference=worst,
                        projected_gradient=result['projected_gradient'], starts=result['starts'])
            fits.append(item)
            print(item | {'legendre_coefficients': '(saved)', 'coefficients': a.tolist()}, flush=True)
        for s in (1, 2):
            row = dict(P=P, s=s, spline=baselines[P, s]['B'], prolate=baselines[P, s]['pro'])
            for field, name in [('sum', f'sum_s{s}'), ('endpoint_sum', f'edge_s{s}'), ('joint_sum', 'joint'),
                                ('c1_sum', f'c1_s{s}'), ('c2_sum', f'c2_s{s}')]:
                window, result, _ = best[name]
                row[field] = float(np.sqrt(squared_error(window.aliases(result['a'], s)[0])[0]))
            records.append(row); print('SUMMARY', row, flush=True)
    with (dest/'review_finite_grids.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    metadata = dict(basis='even L2-normalized PSWFs, signs positive at the origin; count gives their number',
                    regularity='order k imposes g^(l)(P/2)=0 for l<=k, so the zero-extended kernel is C^k; null means no endpoint condition',
                    native_basis='native_coefficients use norm squared 2/(2n+1), even origin sign (-1)^(n/2)',
                    pswf_legendre_size=40, cell_quadrature=80, verification_quadrature=160,
                    bandwidth_search='0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4,2.8,3.2,4.0 times pi P, plus both retuned leading bandwidths',
                    stability='f(q)>=1e-3 at all retained coordinates, coefficient norm <=10',
                    joint_loss='mean of squared RMS divided by retuned-leading squared RMS at each grid',
                    influence='pair-optimal for the same truncated spherical reference',
                    optimizer='multistart BFGS with analytic gradients; no global-optimality certificate',
                    fits=fits)
    (dest/'review_finite_grids_record.json').write_text(json.dumps(metadata, indent=2)+'\n')


def verify_selected():
    dest = Path(__file__).resolve().parent/'records'
    data = json.loads((dest/'review_finite_grids_record.json').read_text())
    worst, minimum = 0., float('inf')
    for item in data['fits']:
        P, c = item['P'], item['cw']; a = np.array(item['coefficients'])
        co = pswf_basis(c, count=len(a), size=60)
        for s in (1, 2):
            cell = ref.Cell(P, qbase/s, 160)
            g = sum(x*legval(cell.xleg, y)*cell.ins for x, y in zip(a, co))
            f = (P/2)*sum(x*(cell.cos@(cell.w*legval(cell.t, y))) for x, y in zip(a, co))
            response = np.einsum('qlu,lu->qu', cell.phase, g)
            alias = np.abs(response-f[:, None])**2@cell.wu/f**2
            error = np.sqrt(squared_error(alias)[0])
            difference = abs(error/item[f'error_s{s}']-1)
            worst = max(worst, difference)
            assert difference < 2e-7
            if item['name'] == 'joint' or item['name'].endswith(f's{s}'):
                q = np.linspace(0., qbase[-1]/s, 1001)
                values = (P/2)*sum(x*(np.cos((P/2)*q[:, None]*cell.t)
                                            @(cell.w*legval(cell.t, y))) for x, y in zip(a, co))
                minimum = min(minimum, float(values.min()))
                assert values.min() >= 1e-3
    window = Window(4, 11.)
    point, Z, z = window.chart()
    alias, jacobian, _ = window.aliases(point+Z@z, 2)
    _, derivative = squared_error(alias)
    gradient = (derivative@jacobian)@Z
    direction = np.arange(1., len(z)+1); direction /= np.linalg.norm(direction)
    step = 1e-6
    plus = squared_error(window.aliases(point+Z@(z+step*direction), 2)[0])[0]
    minus = squared_error(window.aliases(point+Z@(z-step*direction), 2)[0])[0]
    discrepancy = abs(((plus-minus)/(2*step))/(gradient@direction)-1)
    assert discrepancy < 1e-5
    verification = dict(shapes=len(data['fits']), verification_legendre_size=60, verification_quadrature=160,
                        maximum_relative_RMS_change=worst, minimum_dense_working_response=minimum,
                        directional_gradient_relative_error=float(discrepancy))
    print('VERIFICATION', verification, flush=True)
    (dest/'review_finite_grids_verification.json').write_text(json.dumps(verification, indent=2)+'\n')


if __name__ == '__main__':
    if '--verify' not in sys.argv:
        main()
    verify_selected()
