"""Complete Coulomb validation of the stored finite-grid SI designs.

NumPy only. All windows use the full-reference pair-optimal spectral-force
influence. The physical setup matches finite_grids.py; no window is
fitted to the validation configurations. Run with OPENBLAS_NUM_THREADS=1.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder, leggauss, legint, legval

import refinement as ref


ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent/'records'


def density(kappa, co, nq):
    t, w = leggauss(nq)
    values = np.empty(len(kappa))
    weights = w*legval(t, co)/(2*co[0])
    for start in range(0, len(kappa), 2048):
        sl = slice(start, start+2048)
        values[sl] = np.cos(kappa[sl, None]*t)@weights
    return values


def factored(x, P, g, order):
    """phi^(order)(x) for phi = B_2 * g, with g given by Legendre coefficients in t/a on [-a, a], a = (P-2)/2.

    With G1, G2 the antiderivatives of g from -a, phi = G2(x+1)-2G2(x)+G2(x-1), phi' is the same second
    difference of G1, and phi'' that of g."""
    a = (P-2)/2
    g = np.asarray(g)
    I1 = legint(g, lbnd=-1)
    I2 = legint(I1, lbnd=-1)

    def anti(y, k):
        u = np.clip(y/a, -1, 1)
        inside, right = np.abs(y) <= a, y > a
        if k == 0:
            return legval(u, g)*inside
        if k == 1:
            return a*legval(u, I1)*(inside | right)
        return a*a*legval(u, I2)*(inside | right)+a*legval(1., I1)*np.maximum(y-a, 0)
    x = np.asarray(x, dtype=float)
    k = 2-order
    return anti(x+1, k)-2*anti(x, k)+anti(x-1, k)


def window_profile(x, P, co):
    if isinstance(co, dict) and 'g' in co:
        return factored(x, P, co['g'], 0)
    if isinstance(co, dict) and 'beta' in co:
        return window_profile(x, P, None)+co['beta']*(
            window_profile(np.asarray(x)+1, P-2, None)
            -2*window_profile(x, P-2, None)
            +window_profile(np.asarray(x)-1, P-2, None))
    if co is None:
        # The centered cardinal B-spline of support P.
        y = np.asarray(x)+P/2
        g = np.zeros_like(y)
        inside = np.abs(x) < P/2
        for j in range(P+1):
            g[inside] += (-1)**j*math.comb(P, j)*np.maximum(y[inside]-j, 0)**(P-1)
        return g/math.factorial(P-1)
    return legval(np.clip(2*np.asarray(x)/P, -1, 1), co)*(np.abs(x) < P/2)


def window_transform(q, P, co):
    if isinstance(co, dict) and 'g' in co:
        a = (P-2)/2
        t, w = leggauss(200)
        ghat = a*(np.cos(a*np.asarray(q)[:, None]*t)@(w*legval(t, np.asarray(co['g']))))
        return np.sinc(np.asarray(q)/(2*np.pi))**2*ghat
    if isinstance(co, dict) and 'beta' in co:
        return np.sinc(np.asarray(q)/(2*np.pi))**P*(1-co['beta']*np.asarray(q)**2)
    if co is None:
        return np.sinc(np.asarray(q)/(2*np.pi))**P
    t, w = leggauss(160)
    return (P/2)*(np.cos((P/2)*np.asarray(q)[:, None]*t)@(w*legval(t, co)))


def window_derivative(x, P, co):
    """Interior derivative of window_profile, for B_P (co=None) or Legendre coefficients."""
    x = np.asarray(x)
    if isinstance(co, dict) and 'g' in co:
        return factored(x, P, co['g'], 1)
    if co is None:
        return window_profile(x+.5, P-1, None)-window_profile(x-.5, P-1, None)
    return (2/P)*legval(np.clip(2*x/P, -1, 1), legder(co))*(np.abs(x) < P/2)


def derivative_transform(q, P, co):
    """v(q), where i*v(q) is the transform of window_derivative; v=q*f for a continuous kernel."""
    if co is None or (isinstance(co, dict) and 'g' in co):
        return np.asarray(q)*window_transform(q, P, co)
    t, w = leggauss(160)
    return -np.sin((P/2)*np.asarray(q)[:, None]*t)@(w*legval(t, legder(co)))


def alias_moment(q, P, co, derivative=False):
    cell = ref.Cell(P, q, 100)
    g = (window_derivative if derivative else window_profile)(cell.x, P, co)*cell.ins
    response = np.einsum('qlu,lu->qu', cell.phase, g)
    # Sum_m |f(q+2 pi m)|^2, or Sum_m v(q+2 pi m)^2, evaluated without alias truncation.
    return np.abs(response)**2@cell.wu


def influence(n, P, co, shat, k, D, index, rule='spectral'):
    q = 2*np.pi*k/n
    moment = alias_moment(q, P, co)
    power = moment[index[0]]*moment[index[1]]*moment[index[2]]
    A = np.sum(D*D, axis=1)*power**2
    shifts = np.arange(-3, 4)
    f = window_transform((q[:, None]+2*np.pi*shifts).ravel(), P, co).reshape(n, 7)
    if rule == 'ad':
        # Each output alias carries its own derivative multiplier, from the SI kernel's derivative.
        derivative = alias_moment(q, P, co, derivative=True)
        s, d = [moment[ix] for ix in index], [derivative[ix] for ix in index]
        A = (n/ref.L)**2*(d[0]*s[1]*s[2]+s[0]*d[1]*s[2]+s[0]*s[1]*d[2])*power
        v = derivative_transform((q[:, None]+2*np.pi*shifts).ravel(), P, co).reshape(n, 7)
    B = np.zeros(n**3)
    B2 = np.zeros_like(B)
    K = np.column_stack([k[ix] for ix in index])
    for m1 in shifts:
        for m2 in shifts:
            for m3 in shifts:
                km = K+n*np.array([m1, m2, m3])
                shell = np.sum(km*km, axis=1)
                U2 = f[index[0], m1+3]**2*f[index[1], m2+3]**2*f[index[2], m3+3]**2
                term = (2*np.pi/ref.L)*np.sum(D*km, axis=1)*shat[shell]*U2
                if rule == 'ad':
                    F = [f[index[a], m+3] for a, m in enumerate((m1, m2, m3))]
                    W = sum(v[index[a], m+3]*km[:, a]*F[(a+1) % 3]*F[(a+2) % 3]
                            for a, m in enumerate((m1, m2, m3)))
                    term = (n/ref.L)*(2*np.pi/ref.L)*W*F[0]*F[1]*F[2]*shat[shell]
                B += term
                if max(abs(m1), abs(m2), abs(m3)) <= 2:
                    B2 += term
    I = np.divide(B, A, out=np.zeros_like(B), where=A > 0)
    I2 = np.divide(B2, A, out=np.zeros_like(B2), where=A > 0)
    # Omit Nyquist planes, giving the symmetric retained set |k_a|<n/2.
    omitted = np.any(np.abs(K) == n//2, axis=1)
    I[omitted] = 0.; I2[omitted] = 0.
    base = shat[np.sum(K*K, axis=1)]
    normalization = np.sum(np.sum(D*D, axis=1)*base**2)
    discrepancy = np.sqrt(np.sum(A*(I-I2)**2)/normalization)
    assert discrepancy < 2e-7, (n, P, discrepancy)
    I = I.reshape((n,)*3)
    neg = (-np.arange(n)) % n
    symmetry = np.max(np.abs(I-I[np.ix_(neg, neg, neg)]))/max(np.max(np.abs(I)), 1e-300)
    assert symmetry < 1e-9
    return .5*(I+I[np.ix_(neg, neg, neg)]), float(discrepancy)


def reciprocal_force(r, charges, K, S, L):
    force = np.zeros_like(r)
    for start in range(0, len(S), 4096):
        sl = slice(start, start+4096)
        phase = np.exp(1j*(K[sl]@r.T))
        spectrum = S[sl]*(phase.conj()@charges)
        force += (phase.T@(K[sl]*spectrum[:, None])).imag
    return force/L**3


def gaussian_reference(r, q, alpha):
    L = ref.L
    cutoff = np.sqrt(-np.log(1e-15))/alpha
    assert cutoff < L/2
    dr = r[:, None, :]-r[None, :, :]
    dr -= L*np.round(dr/L)
    rr = np.linalg.norm(dr, axis=2)
    use = (rr > 0) & (rr < cutoff)
    z = rr[use]
    erfc = np.array([math.erfc(alpha*x) for x in z])
    factor = np.zeros_like(rr)
    factor[use] = erfc/z**3+2*alpha/np.sqrt(np.pi)*np.exp(-(alpha*z)**2)/z**2
    near = np.einsum('ij,ijc,j->ic', factor, dr, q)
    kcut = 2*alpha*np.sqrt(-np.log(1e-15))
    nk = int(np.ceil(kcut*L/(2*np.pi)))
    kk = np.arange(-nk, nk+1)
    K = (2*np.pi/L)*np.array(np.meshgrid(kk, kk, kk, indexing='ij')).reshape(3, -1).T
    k2 = np.sum(K*K, axis=1)
    keep = (k2 > 0) & (k2 <= kcut**2)
    K, k2 = K[keep], k2[keep]
    S = 4*np.pi*np.exp(-k2/(4*alpha**2))/k2
    return near+reciprocal_force(r, q, K, S, L)


def compact_near(r, q, co):
    dr = r[:, None, :]-r[None, :, :]
    dr -= ref.L*np.round(dr/ref.L)
    rr = np.linalg.norm(dr, axis=2)
    use = (rr > 0) & (rr < ref.rc)
    z = rr[use]; t = z/ref.rc
    primitive = legint(co)
    residual = 1-(legval(t, primitive)-legval(0., primitive))/co[0]
    factor = np.zeros_like(rr)
    factor[use] = residual/z**3+legval(t, co)/(co[0]*ref.rc*z**2)
    return np.einsum('ij,ijc,j->ic', factor, dr, q)


def mesh_force(r, charges, n, P, co, I, wave, rule='spectral'):
    rho = np.zeros((n,)*3)
    indices, weights, factors = [], [], []
    for rj, qj in zip(r, charges):
        X = rj*n/ref.L
        nodes = np.floor(X-P/2).astype(int)[None, :]+1+np.arange(P)[:, None]
        w = window_profile(nodes-X, P, co)
        ww = (w[:, 0, None, None]*w[None, :, 1, None]*w[None, None, :, 2]).ravel()
        ix = np.meshgrid(*(nodes[:, j] % n for j in range(3)), indexing='ij')
        ind = np.ravel_multi_index(tuple(x.ravel() for x in ix), (n,)*3)
        np.add.at(rho.ravel(), ind, qj*ww)
        indices.append(ind); weights.append(ww)
        factors.append((w, window_derivative(nodes-X, P, co)*(n/ref.L)) if rule == 'ad' else None)
    spectrum = np.fft.fftn(rho)*I*n**3/ref.L**3
    force = np.zeros_like(r)
    if rule == 'ad':
        # Minus the gradient of the interpolated potential, differentiating the SI kernel.
        potential = np.fft.ifftn(spectrum).real.ravel()
        for i, ((w, dw), ind) in enumerate(zip(factors, indices)):
            for axis in range(3):
                f = [dw[:, b] if b == axis else w[:, b] for b in range(3)]
                force[i, axis] = (f[0][:, None, None]*f[1][None, :, None]*f[2][None, None, :]).ravel()@potential[ind]
        return force
    for axis in range(3):
        field = np.fft.ifftn(-1j*wave[axis]*spectrum).real.ravel()
        force[:, axis] = [w@field[ix] for w, ix in zip(weights, indices)]
    return force


def self_matrix(n, P, I):
    """Mesh potential at stencil node j of a unit grid charge at stencil node j'."""
    green = np.fft.ifftn(I).real*n**3/ref.L**3
    j = np.array(np.meshgrid(*(np.arange(P),)*3, indexing='ij')).reshape(3, -1).T
    d = (j[:, None, :]-j[None, :, :]) % n
    return green[d[..., 0], d[..., 1], d[..., 2]]


def self_field(r, n, P, co, matrix, chunk=4096):
    """AD field that each unit charge at r receives from its own spread charge."""
    field = np.zeros_like(r)
    for start in range(0, len(r), chunk):
        X = r[start:start+chunk]*n/ref.L
        x = np.floor(X-P/2).astype(int)[:, None, :]+1+np.arange(P)[None, :, None]-X[:, None, :]
        w, dw = window_profile(x, P, co), window_derivative(x, P, co)*(n/ref.L)
        weights = np.einsum('ia,ib,ic->iabc', w[:, :, 0], w[:, :, 1], w[:, :, 2]).reshape(len(X), -1)
        potential = (weights@matrix.T).reshape(len(X), P, P, P)
        for axis in range(3):
            f = [dw[:, :, b] if b == axis else w[:, :, b] for b in range(3)]
            field[start:start+chunk, axis] = np.einsum('ia,ib,ic,iabc->i', *f, potential)
    return field


def self_rms(n, P, co, matrix, points):
    """RMS self field of a unit charge over uniform positions, by Gauss quadrature on one cell.

    The stencil is fixed for frac(X-P/2) in (0,1), where the field is smooth."""
    t, w = leggauss(points)
    t, w = (t+1)/2, w/2
    theta = np.array(np.meshgrid(t, t, t, indexing='ij')).reshape(3, -1).T
    weight = np.einsum('i,j,k->ijk', w, w, w).ravel()
    field = self_field((theta+P/2)*ref.L/n, n, P, co, matrix)
    return float(np.sqrt(weight@np.sum(field**2, axis=1)))


def force_metrics(force, retained, truth, charges):
    """Retain absolute charge-scaled errors and configuration-relative errors."""
    norm = np.linalg.norm(truth)
    pair_weight = np.sum(charges**2)**2-np.sum(charges**4)
    return dict(relative_force=float(np.linalg.norm(force-truth)/norm),
                mesh_difference=float(np.linalg.norm(force-retained)/norm),
                tail=float(np.linalg.norm(retained-truth)/norm),
                pair_scaled_squared_error=float(np.sum((force-truth)**2)/pair_weight),
                pair_scaled_squared_mesh=float(np.sum((force-retained)**2)/pair_weight),
                pair_scaled_squared_tail=float(np.sum((retained-truth)**2)/pair_weight))


def charge_metrics_from_relative(row, truth, charges):
    """Recover absolute scaling of a saved norm using its unchanged reference."""
    result = dict(row)
    scale = np.sum(truth**2)/(np.sum(charges**2)**2-np.sum(charges**4))
    for absolute, relative in [('pair_scaled_squared_error', 'relative_force'),
                               ('pair_scaled_squared_mesh', 'mesh_difference'),
                               ('pair_scaled_squared_tail', 'tail')]:
        value = float(row[relative])**2*scale
        if absolute in result:
            assert np.isclose(result[absolute], value, rtol=2e-12, atol=0)
        result[absolute] = float(value)
    return result


def mesh_setup(n, co_split):
    """Grid modes, the split multiplier on all physical shells, and the retained set of an n^3 mesh."""
    k = np.r_[np.arange(n//2), np.arange(-n//2, 0)]
    ii = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    index = [x.ravel() for x in ii]
    K = np.column_stack([k[ix] for ix in index])
    D = (2*np.pi/ref.L)*K
    wave = [(2*np.pi/ref.L)*k[x] for x in ii]
    maximum = 3*(n//2+3*n)**2
    shell = np.arange(maximum+1)
    kappa = ref.rc*(2*np.pi/ref.L)*np.sqrt(shell)
    nq = max(160, int(np.ceil(kappa[-1]+40)))
    den = density(kappa, co_split, nq)
    probe = np.linspace(0, maximum, 503, dtype=int)
    quad_difference = np.max(np.abs(den[probe]-density(kappa[probe], co_split, 2*nq)))
    assert quad_difference < 2e-12
    shat = np.zeros_like(den)
    shat[1:] = 4*np.pi*den[1:]/((2*np.pi/ref.L)**2*shell[1:])
    keep = np.all(np.abs(K) < n//2, axis=1) & np.any(K != 0, axis=1)
    retained_S = shat[np.sum(K[keep]*K[keep], axis=1)]
    return k, index, D, wave, shat, keep, retained_S, quad_difference


def main():
    fits = json.loads((DEST/'finite_grids_record.json').read_text())['fits']
    spline_fits = {(int(x['P']), x['name']): float(x['beta'])
                   for x in csv.DictReader((DEST/'spline_derivative.csv').open())}
    baselines = {(int(x['P']), int(x['s'])): x for x in csv.DictReader((DEST/'retuned_refinement.csv').open())}
    co_split = ref.coeff(ref.c)
    rng = np.random.default_rng(20261021)
    samples = []
    for trial in range(16):
        r = rng.uniform(0., ref.L, (64, 3))
        q = np.r_[np.ones(32), -np.ones(32)]
        reference = gaussian_reference(r, q, 1.3)
        check = gaussian_reference(r, q, 1.5)
        discrepancy = np.linalg.norm(reference-check)/np.linalg.norm(reference)
        assert discrepancy < 2e-11, discrepancy
        samples.append(dict(r=r, q=q, truth=reference, near=compact_near(r, q, co_split), reference_difference=float(discrepancy)))
    rows, checks = [], []
    for s in (1, 2):
        n = 40*s
        k, index, D, wave, shat, keep, retained_S, quad_difference = mesh_setup(n, co_split)
        for sample in samples:
            retained = reciprocal_force(sample['r'], sample['q'], D[keep], retained_S, ref.L)
            sample['retained'] = sample['near']+retained
            sample['tail'] = float(np.linalg.norm(sample['retained']-sample['truth'])/np.linalg.norm(sample['truth']))
        for P in (4, 5, 6):
            cw = float(baselines[P, s]['cw'])
            lead = ref.coeff(cw)/(P*ref.coeff(cw)[0])
            candidates = [('spline', None), ('prolate', lead)]
            candidates.append(('spline_derivative', dict(beta=spline_fits[P, f's{s}'])))
            for method, name in [('sum', f'sum_s{s}'), ('endpoint_sum', f'edge_s{s}'), ('joint_sum', 'joint'),
                                 ('c2_sum', f'c2_s{s}')]:
                fit = next(x for x in fits if x['P'] == P and x['name'] == name)
                candidates.append((method, np.array(fit['legendre_coefficients'])))
            for method, co in candidates:
                I, shell_difference = influence(n, P, co, shat, k, D, index)
                checks.append(dict(P=P, s=s, method=method, influence_shell_difference=shell_difference, transform_difference=float(quad_difference)))
                for trial, sample in enumerate(samples, 1):
                    force = sample['near']+mesh_force(sample['r'], sample['q'], n, P, co, I, wave)
                    rows.append(dict(P=P, s=s, n=n, method=method, sample=trial,
                                     **force_metrics(force, sample['retained'], sample['truth'], sample['q'])))
                errors = [x['relative_force'] for x in rows if (x['P'], x['s'], x['method']) == (P, s, method)]
                print(P, s, method, 'mean', np.mean(errors), 'tail', np.mean([x['tail'] for x in samples]), 'alias check', shell_difference, flush=True)
    with (DEST/'complete_si.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    record = dict(L=ref.L, rc=ref.rc, c_s=ref.c, N=64, trials=len(samples), grids=[40, 80],
                  influence='full-reference pair-optimal spectral force, exact SI alias moment; physical aliases |m_a|<=3',
                  retained_modes='|k_a|<n/2; Nyquist planes omitted for every method',
                  reference='Gaussian Ewald alpha=1.3 and 1.5; exponential cutoff 1e-15',
                  maximum_reference_difference=max(x['reference_difference'] for x in samples),
                  checks=checks,
                  configurations=[dict(positions=x['r'].tolist(), charges=x['q'].tolist()) for x in samples])
    (DEST/'complete_si_record.json').write_text(json.dumps(record, indent=2)+'\n')


def main_ad():
    """AD forces with the SI kernels designed for AD by ad_grids.py.

    The influence is the full-reference pair-optimal AD influence. The AD
    field of a particle includes a self field from its own spread charge;
    it is computed separately, so the error is reported with and without
    it, and the RMS self field of a unit charge is integrated over a cell."""
    fits = {(x['P'], x['name']): x for x in json.loads((DEST/'ad_grids_record.json').read_text())['fits']}
    tuned = {(int(x['P']), int(x['s'])): float(x['prolate_cw'])
             for x in csv.DictReader((DEST/'ad_grids.csv').open())}
    co_split = ref.coeff(ref.c)
    samples = []
    for item in json.loads((DEST/'complete_si_record.json').read_text())['configurations']:
        r, q = np.array(item['positions']), np.array(item['charges'])
        samples.append(dict(r=r, q=q, truth=gaussian_reference(r, q, 1.3), near=compact_near(r, q, co_split)))
    rows, kernels = [], []
    for s in (1, 2):
        n = 40*s
        k, index, D, wave, shat, keep, retained_S, quad_difference = mesh_setup(n, co_split)
        for sample in samples:
            sample['retained'] = sample['near']+reciprocal_force(sample['r'], sample['q'], D[keep], retained_S, ref.L)
        for P in (4, 5, 6):
            lead = ref.coeff(tuned[P, s])/(P*ref.coeff(tuned[P, s])[0])
            candidates = [('spline', None), ('prolate', lead),
                          ('c2_sum', np.array(fits[P, f'c2_s{s}']['legendre_coefficients']))]
            for method, co in candidates:
                I, shell_difference = influence(n, P, co, shat, k, D, index, rule='ad')
                matrix = self_matrix(n, P, I)
                # The self field is the AD field of a lone unit charge. The FFT route
                # loses relative accuracy as the self field becomes small.
                probe = samples[0]['r'][:3]
                lone = np.array([mesh_force(x[None], np.ones(1), n, P, co, I, wave, rule='ad')[0] for x in probe])
                self_difference = float(np.abs(self_field(probe, n, P, co, matrix)-lone).max()/np.abs(lone).max())
                assert self_difference < 1e-8, self_difference
                rms, check = self_rms(n, P, co, matrix, 40), self_rms(n, P, co, matrix, 48)
                assert abs(check/rms-1) < 1e-8
                kernels.append(dict(P=P, s=s, n=n, method=method, self_field_rms=check,
                                    self_quadrature_relative_difference=abs(check/rms-1),
                                    self_matrix_difference=self_difference,
                                    influence_shell_difference=shell_difference,
                                    transform_difference=float(quad_difference)))
                for trial, sample in enumerate(samples, 1):
                    force = sample['near']+mesh_force(sample['r'], sample['q'], n, P, co, I, wave, rule='ad')
                    own = sample['q'][:, None]*self_field(sample['r'], n, P, co, matrix)
                    weight = np.sum(sample['q']**2)**2-np.sum(sample['q']**4)
                    rows.append(dict(P=P, s=s, n=n, method=method, sample=trial,
                                     **force_metrics(force, sample['retained'], sample['truth'], sample['q']),
                                     pair_scaled_squared_without_self=float(np.sum((force-own-sample['truth'])**2)/weight),
                                     pair_scaled_squared_self=float(np.sum(own**2)/weight)))
                group = [x for x in rows if (x['P'], x['s'], x['method']) == (P, s, method)]
                print(P, s, method, 'RMS', np.sqrt(np.mean([x['pair_scaled_squared_error'] for x in group])),
                      'without self', np.sqrt(np.mean([x['pair_scaled_squared_without_self'] for x in group])),
                      'unit self field RMS', check, flush=True)
    with (DEST/'complete_si_ad.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    record = dict(L=ref.L, rc=ref.rc, c_s=ref.c, N=64, trials=len(samples), grids=[40, 80],
                  force='AD: the derivative of the interior polynomial of the SI kernel',
                  influence='full-reference pair-optimal AD force, exact SI alias moments; physical aliases |m_a|<=3',
                  retained_modes='|k_a|<n/2; Nyquist planes omitted for every method',
                  reference='Gaussian Ewald alpha=1.3; the configurations of complete_si_record.json',
                  self_field='field of a unit charge from its own spread charge; RMS over uniform positions by 48^3 Gauss points on one cell',
                  kernels=kernels)
    (DEST/'complete_si_ad_record.json').write_text(json.dumps(record, indent=2)+'\n')


def design_inputs(tolerance=1e-4):
    """The C^2 Coulomb split of a tolerance, the designed SI kernel of each order, and the tuned psi_0 bandwidths."""
    split = next(x for x in json.loads((DEST/'c2_split_fewest_record.json').read_text())['splits']
                 if x['nu'] == 1 and x['tolerance'] == tolerance)
    record = json.loads((DEST/'si_design_record.json').read_text())
    table = [x for x in csv.DictReader((DEST/'si_design.csv').open()) if float(x['tolerance']) == tolerance]
    count = int(record['reported_count'])
    kernels = {}
    for P in (4, 5, 6):
        cw = record['tuned_psi0_bandwidths'][f'{tolerance:g}_{P}'][0]
        row = next(r for r in record['rows'] if (r['tolerance'], r['P'], r['N']) == (tolerance, P, count)
                   and abs(r['cw']-cw) < 1e-12)
        kernels[P] = np.array(row['legendre_coefficients'])
    tuned = {(int(x['P']), int(x['n'])): float(x['prolate_cw']) for x in table}
    return np.array(split['legendre_coefficients']), split['c'], kernels, tuned


def main_design(grids=(40, 60, 80)):
    """AD forces with the C^2 Coulomb split and one designed SI kernel per order for every grid.

    As main_ad, for the configurations of complete_si_record.json, with the C^2 split of the tolerance
    1e-4 (c2_split_fewest_record.json) in place of the psi_0 split, the kernels of si_design.py,
    the psi_0 kernel tuned for AD on each grid for the same split, and B_P, on the grids 40^3, 60^3 and 80^3."""
    co_split, c_s, designed, tuned = design_inputs()
    samples = []
    for item in json.loads((DEST/'complete_si_record.json').read_text())['configurations']:
        r, q = np.array(item['positions']), np.array(item['charges'])
        samples.append(dict(r=r, q=q, truth=gaussian_reference(r, q, 1.3), near=compact_near(r, q, co_split)))
    rows, kernels, tails = [], [], {}
    for n in grids:
        s = n/40
        k, index, D, wave, shat, keep, retained_S, quad_difference = mesh_setup(n, co_split)
        for sample in samples:
            sample['retained'] = sample['near']+reciprocal_force(sample['r'], sample['q'], D[keep], retained_S, ref.L)
        weight = [np.sum(x['q']**2)**2-np.sum(x['q']**4) for x in samples]
        tails[n] = float(np.sqrt(np.mean([np.sum((x['retained']-x['truth'])**2)/w for x, w in zip(samples, weight)])))
        print('grid', n, 'Fourier tail', tails[n], flush=True)
        for P in (4, 5, 6):
            lead = ref.coeff(tuned[P, n])/(P*ref.coeff(tuned[P, n])[0])
            for method, co in [('spline', None), ('prolate', lead), ('design', designed[P])]:
                I, shell_difference = influence(n, P, co, shat, k, D, index, rule='ad')
                matrix = self_matrix(n, P, I)
                probe = samples[0]['r'][:3]
                lone = np.array([mesh_force(x[None], np.ones(1), n, P, co, I, wave, rule='ad')[0] for x in probe])
                self_difference = float(np.abs(self_field(probe, n, P, co, matrix)-lone).max()/np.abs(lone).max())
                assert self_difference < 1e-8, self_difference
                rms, check = self_rms(n, P, co, matrix, 40), self_rms(n, P, co, matrix, 48)
                assert abs(check/rms-1) < 1e-8
                kernels.append(dict(P=P, s=s, n=n, method=method, self_field_rms=check,
                                    self_quadrature_relative_difference=abs(check/rms-1),
                                    self_matrix_difference=self_difference,
                                    influence_shell_difference=shell_difference,
                                    transform_difference=float(quad_difference)))
                for trial, (sample, w) in enumerate(zip(samples, weight), 1):
                    force = sample['near']+mesh_force(sample['r'], sample['q'], n, P, co, I, wave, rule='ad')
                    own = sample['q'][:, None]*self_field(sample['r'], n, P, co, matrix)
                    rows.append(dict(P=P, s=s, n=n, method=method, sample=trial,
                                     **force_metrics(force, sample['retained'], sample['truth'], sample['q']),
                                     pair_scaled_squared_without_self=float(np.sum((force-own-sample['truth'])**2)/w),
                                     pair_scaled_squared_self=float(np.sum(own**2)/w)))
                group = [x for x in rows if (x['n'], x['P'], x['method']) == (n, P, method)]
                print(n, P, method, 'RMS', np.sqrt(np.mean([x['pair_scaled_squared_error'] for x in group])),
                      'without self', np.sqrt(np.mean([x['pair_scaled_squared_without_self'] for x in group])),
                      'unit self field RMS', check, flush=True)
    with (DEST/'complete_design.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    record = dict(L=ref.L, rc=ref.rc, c_s=c_s, split='C^2 Coulomb split, tolerance 1e-4', N=64, trials=len(samples),
                  grids=list(grids), force='AD: the derivative of the interior polynomial of the SI kernel',
                  kernels_source='si_design_record.json (one kernel per order) and si_design.csv (tuned psi_0)',
                  influence='full-reference pair-optimal AD force, exact SI alias moments; physical aliases |m_a|<=3',
                  retained_modes='|k_a|<n/2; Nyquist planes omitted for every method',
                  reference='Gaussian Ewald alpha=1.3; the configurations of complete_si_record.json',
                  self_field='field of a unit charge from its own spread charge; RMS over uniform positions by 48^3 Gauss points on one cell',
                  tail_rms={str(n): v for n, v in tails.items()}, kernels=kernels)
    (DEST/'complete_design_record.json').write_text(json.dumps(record, indent=2)+'\n')


def coarsest_inputs(tolerance):
    """The C^2 split of a tolerance, and for each order the kernel of si_coarsest.py with the coarsest grid
    ratios s = (pi/h)/(c_s/r_c) of that kernel, of B_P, and of the psi_0 kernel tuned on each grid."""
    import c2_split_fewest as splits
    record = json.loads((DEST/'si_coarsest_record.json').read_text())
    results = {r['P']: r for r in record['results'] if r['tolerance'] == tolerance and 'failure' not in r}
    c_s = next(iter(results.values()))['c_s']
    return np.array(splits.fewest(c_s, 1)[1]), c_s, results


def main_coarsest(tolerance=1e-4, orders=(4, 5, 6, 7)):
    """Complete AD forces with the C^2 split and, for each order, the kernels on the coarsest grids.

    For the configurations of complete_si_record.json in the box L = 10 r_c, each order is computed on the
    coarsest even grid on which the kernel of si_coarsest.py meets the tolerance in the limit of large boxes,
    on that of B_P, and on that of the psi_0 kernel in the box L = 10 r_c. On each grid, the kernels are that kernel,
    B_P, and the psi_0 kernel with its bandwidth tuned for the grid in the band of this box."""
    import si_coarsest as co
    co_split, c_s, results = coarsest_inputs(tolerance)
    box = co.band(tolerance, L=ref.L)
    samples = []
    for item in json.loads((DEST/'complete_si_record.json').read_text())['configurations']:
        r, q = np.array(item['positions']), np.array(item['charges'])
        samples.append(dict(r=r, q=q, truth=gaussian_reference(r, q, 1.3), near=compact_near(r, q, co_split)))
    weight = [np.sum(x['q']**2)**2-np.sum(x['q']**4) for x in samples]
    unit = ref.L*c_s/(np.pi*ref.rc)
    rows, kernels, tails, setups = [], [], {}, {}
    for P in orders:
        result = results[P]
        ratios = [x for x in (result['sigma'], result['spline_sigma'], result['prolate_sigma']) if x is not None]
        for n in sorted(set(int(2*np.ceil(x*unit/2)) for x in ratios)):
            s = n/unit
            if n not in setups:
                setups[n] = mesh_setup(n, co_split)
                k, index, D, wave, shat, keep, retained_S, _ = setups[n]
                for sample in samples:
                    sample[n] = sample['near']+reciprocal_force(sample['r'], sample['q'], D[keep], retained_S, ref.L)
                tails[n] = float(np.sqrt(np.mean([np.sum((x[n]-x['truth'])**2)/w for x, w in zip(samples, weight)])))
                print('grid', n, 's', s, 'Fourier tail', tails[n], flush=True)
            k, index, D, wave, shat, keep, retained_S, quad_difference = setups[n]
            cw = co.tune(P, s, box)[0]
            lead = ref.coeff(cw)/(P*ref.coeff(cw)[0])
            for method, co_k in [('spline', None), ('prolate', lead),
                                 ('design', dict(g=np.array(result['legendre_coefficients'])))]:
                I, shell_difference = influence(n, P, co_k, shat, k, D, index, rule='ad')
                matrix = self_matrix(n, P, I)
                rms = self_rms(n, P, co_k, matrix, 48)
                kernels.append(dict(P=P, n=n, s=s, method=method, cw=cw if method == 'prolate' else None,
                                    self_field_rms=rms, influence_shell_difference=shell_difference))
                for trial, (sample, w) in enumerate(zip(samples, weight), 1):
                    force = sample['near']+mesh_force(sample['r'], sample['q'], n, P, co_k, I, wave, rule='ad')
                    own = sample['q'][:, None]*self_field(sample['r'], n, P, co_k, matrix)
                    rows.append(dict(P=P, s=s, n=n, method=method, sample=trial,
                                     **force_metrics(force, sample[n], sample['truth'], sample['q']),
                                     pair_scaled_squared_without_self=float(np.sum((force-own-sample['truth'])**2)/w),
                                     pair_scaled_squared_self=float(np.sum(own**2)/w)))
                group = [x for x in rows if (x['n'], x['P'], x['method']) == (n, P, method)]
                print(P, n, method, 'relative', np.sqrt(np.mean([x['relative_force']**2 for x in group])),
                      'without self', np.sqrt(np.mean([x['pair_scaled_squared_without_self'] for x in group])), flush=True)
    stem = f'complete_coarsest_{tolerance:g}'
    with (DEST/(stem+'.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (DEST/(stem+'_record.json')).write_text(json.dumps(dict(
        L=ref.L, rc=ref.rc, c_s=c_s, tolerance=tolerance, split='C^2 Coulomb split', N=64, trials=len(samples),
        force='AD', influence='full-reference pair-optimal AD force', retained_modes='|k_a|<n/2',
        reference='Gaussian Ewald alpha=1.3; the configurations of complete_si_record.json',
        grids={str(n): n/unit for n in setups}, tail_rms={str(n): v for n, v in tails.items()},
        kernels=kernels), indent=2)+'\n')


if __name__ == '__main__':
    if '--coarsest' in sys.argv:
        main_coarsest()
    elif '--design' in sys.argv:
        main_design()
    else:
        main_ad() if '--ad' in sys.argv else main()
