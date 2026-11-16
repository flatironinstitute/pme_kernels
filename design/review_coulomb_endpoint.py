"""Join Coulomb cutoff matching and complete-field SI validation.

At fixed c_s, r_c, grids and configurations, impose integral one and
chi^(l)(1)=0 for l<order on the compact split density, so the compact
residual is C^order. Optimize the remaining PSWF coefficients using the
full natural-influence force quadratic with B_6 held fixed on one grid, or
on several grids with each quadratic scaled by the psi_0 split's value.
Validate stored SI shapes with full-reference optimal influence. NumPy only.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder, leggauss, legval

import review_complete_si as complete
import review_finite_grids as design
import review_refinement as ref


DEST = Path(__file__).resolve().parent/'records'
STEM = 'review_coulomb_endpoint'


def validation_fingerprint(fits, spline_fits, baselines, configurations):
    return hashlib.sha256(json.dumps(dict(
        fits=fits, spline_fits=sorted((P, name, beta) for (P, name), beta in spline_fits.items()),
        baselines=sorted((P, s, item) for (P, s), item in baselines.items()),
        configurations=configurations), sort_keys=True).encode()).hexdigest()


def transform_table(profiles, maximum, nq):
    shell = np.arange(maximum+1)
    kappa = ref.rc*ref.dxi*np.sqrt(shell)
    t, w = leggauss(nq)
    weighted = w[:, None]*legval(t, profiles).T
    result = np.zeros((maximum+1, profiles.shape[1]))
    for start in range(1, maximum+1, 2048):
        sl = slice(start, min(maximum+1, start+2048))
        result[sl] = (4*np.pi/(ref.dxi**2*shell[sl, None]))*(np.cos(kappa[sl, None]*t)@weighted)
    return result


def split_quadratic(n, profiles, max_alias=4, mesh=True, P=6, window=None):
    """Natural-influence force quadratic of the density profiles on n^3.

    The SI kernel of support P has Legendre coefficients window, or is B_P
    when window is None. Physical aliases |m_a|<=max_alias include the tail
    and cross term; the second quadratic stops at max_alias-1. With
    mesh=False the retained modes are exact, so only the physical tail
    remains."""
    half = n//2
    k = np.r_[np.arange(half), np.arange(-half, 0)]
    ii = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    ix = [v.ravel() for v in ii]
    K = np.column_stack([k[v] for v in ix]); norm0 = np.sum(K*K, axis=1)
    use = np.all(np.abs(K) < half, axis=1)
    D = K*use[:, None]
    q = 2*np.pi*k/n
    cell = ref.Cell(P, q, 160)
    g = complete.window_profile(cell.x, P, window)*cell.ins
    response = np.einsum('qlu,lu->qu', cell.phase, g)
    f0 = complete.window_transform(q, P, window)
    alias = np.abs(response-f0[:, None])**2@cell.wu/f0**2
    logT = sum(np.log1p(alias[v]) for v in ix)
    eta = np.expm1(2*logT) if mesh else 0*logT
    maximum = 3*(half+max_alias*n)**2
    nq = max(160, int(np.ceil(ref.rc*ref.dxi*np.sqrt(maximum)+40)))
    table = transform_table(profiles, maximum, nq)
    probe = np.linspace(0, maximum, 503, dtype=int)
    t, w = leggauss(2*nq)
    checked = np.zeros((len(probe), profiles.shape[1]))
    nz = probe > 0
    checked[nz] = 4*np.pi/(ref.dxi**2*probe[nz, None])*(
        np.cos((ref.rc*ref.dxi*np.sqrt(probe[nz]))[:, None]*t)@(w[:, None]*legval(t, profiles).T))
    transform_difference = float(np.max(np.abs(table[probe]-checked)))
    assert transform_difference < 1e-10
    base = table[norm0]
    Hbase = base.T@((ref.dxi**2*norm0*np.where(use, eta, 1.))[:, None]*base)
    H2, H3 = Hbase.copy(), Hbase.copy()
    shifted = {m: complete.window_transform(q+2*np.pi*m, P, window) for m in range(-max_alias, max_alias+1)}
    for m1 in range(-max_alias, max_alias+1):
        for m2 in range(-max_alias, max_alias+1):
            for m3 in range(-max_alias, max_alias+1):
                if m1 == m2 == m3 == 0:
                    continue
                shift = (m1, m2, m3)
                km = K+n*np.array(shift); normm = np.sum(km*km, axis=1)
                value = table[normm]
                term = value.T@((ref.dxi**2*normm)[:, None]*value)
                if mesh:
                    ratio = np.ones(n**3)
                    for axis in range(3):
                        ratio *= (shifted[shift[axis]][ix[axis]]/f0[ix[axis]])**2
                    cross = base.T@((ref.dxi**2*np.sum(D*km, axis=1)*ratio)[:, None]*value)
                    term = term-cross-cross.T
                H3 += term
                if max(abs(v) for v in shift) <= max_alias-1:
                    H2 += term
    return .5*(H3+H3.T), H2, transform_difference


def probe_windows(n, objective):
    """SI kernels whose natural-influence error the split design minimizes on n^3."""
    if objective == 'b6':
        return [(6, None)]
    if objective == 'tail':
        return [None]
    fits = json.loads((DEST/'review_finite_grids_record.json').read_text())['fits']
    return [(P, np.array(next(x for x in fits if x['P'] == P and x['name'] == f'c2_s{n//40}')['legendre_coefficients']))
            for P in (4, 5, 6)]


def construct_split(grids=(80,), order=1, count=4, max_alias=4, objective='b6'):
    basis = design.pswf_basis(ref.c, count=count)
    # Integral one and chi^(l)(1)=0 for l<order: the compact residual is C^order.
    E = np.array([2*basis[:, 0]]+[legval(1., legder(basis.T, l)) for l in range(order)])
    target = np.eye(len(E))[0]
    first = np.linalg.solve(E[:, :len(E)], target)
    point = np.r_[first, np.zeros(count-len(E))]
    _, _, vt = np.linalg.svd(E, full_matrices=True)
    Z = vt[len(E):].T
    profiles = (np.column_stack([point, Z]).T@basis).T
    # The last profile is the psi_0 split. A design with several grids or SI
    # kernels averages their quadratics, each scaled by the psi_0 split's value.
    lead = ref.coeff(ref.c)/(2*ref.coeff(ref.c)[0])
    terms = [(n, w) for n in grids for w in probe_windows(n, objective)]
    H3, H2, quadratics, leading_scores, transform_difference = 0., 0., {}, {}, 0.
    for n, w in terms:
        full, truncated, difference = split_quadratic(n, np.column_stack([profiles, lead]), max_alias,
                                                      mesh=w is not None, **(dict(P=w[0], window=w[1]) if w else {}))
        scale = full[-1, -1]*len(terms) if len(terms) > 1 else 1.
        H3 = H3+full[:-1, :-1]/scale; H2 = H2+truncated[:-1, :-1]/scale
        key = f'{n}' if w is None or w[1] is None else f'{n},P={w[0]}'
        quadratics[key] = full[:-1, :-1]; leading_scores[key] = float(full[-1, -1])
        transform_difference = max(transform_difference, difference)
    assert np.linalg.eigvalsh(H3[1:, 1:]).min() > 0
    z = -np.linalg.solve(H3[1:, 1:], H3[1:, 0]); v = np.r_[1., z]
    a = point+Z@z; co = a@basis
    residual = float(np.linalg.norm(E@a-target))
    shell_difference = float(abs(v@H2@v/(v@H3@v)-1))
    print('QUADRATIC CHECK', H3[0, 0], v@H3@v, shell_difference, flush=True)
    assert residual < 1e-12
    assert shell_difference < 1e-3
    result = dict(c_s=ref.c, rc=ref.rc, L=ref.L, n=grids[0] if len(grids) == 1 else list(grids),
        window=dict(b6='B6', tail=None, c2='stored C2 SI kernels, P=4,5,6')[objective],
        influence=None if objective == 'tail' else 'natural',
        objective=dict(b6='B6 natural-influence force error',
                       tail='physical force tail beyond the retained modes',
                       c2='mean natural-influence force error of the stored C2 SI kernels at P=4,5,6, '
                          'each scaled by its psi_0 split value')[objective],
        order=order, count=count,
        coefficient_convention='L2-orthonormal PSWF basis, even degrees '
            + ','.join(str(2*j) for j in range(count))+', positive values at the origin',
        constraints=f'integral over [-1,1] equals 1; chi^(l)(1)=0 for l<{order}; {count} even PSWFs',
        coefficients=a.tolist(), legendre_coefficients=co.tolist(),
        two_term_coefficients=point.tolist(), two_term_legendre_coefficients=(point@basis).tolist(),
        quadratic=H3.tolist(), feasible_score=float(H3[0, 0]), optimized_score=float(v@H3@v),
        leading_scores=leading_scores,
        scores_relative_to_leading={key: float(v@quadratics[key]@v/leading_scores[key]) for key in quadratics},
        constraint_residual=residual, physical_alias_half_width=max_alias,
        alias_shell_relative_difference=shell_difference,
        transform_absolute_difference=transform_difference)
    print('SPLIT', result, flush=True)
    (DEST/(STEM+'_split.json')).write_text(json.dumps(result, indent=2)+'\n')
    return result


def validate(result, reuse=False):
    co_split = np.array(result['legendre_coefficients'])
    fits = json.loads((DEST/'review_finite_grids_record.json').read_text())['fits']
    spline_fits = {(int(x['P']), x['name']): float(x['beta'])
                   for x in csv.DictReader((DEST/'review_spline_derivative.csv').open())}
    baselines = {(int(x['P']), int(x['s'])): x for x in csv.DictReader((DEST/'review_retuned_refinement.csv').open())}
    configurations = json.loads((DEST/'review_complete_si_record.json').read_text())['configurations']
    inputs_digest = validation_fingerprint(fits, spline_fits, baselines, configurations)
    samples = []
    for item in configurations:
        r = np.array(item['positions']); charges = np.array(item['charges'])
        truth = complete.gaussian_reference(r, charges, 1.3)
        samples.append(dict(r=r, q=charges, truth=truth,
                            near=complete.compact_near(r, charges, co_split)))
    previous, previous_checks = [], []
    if reuse and (DEST/(STEM+'.csv')).exists():
        old_record = json.loads((DEST/(STEM+'_record.json')).read_text())
        assert old_record['split']['legendre_coefficients'] == result['legendre_coefficients']
        assert old_record.get('validation_inputs_sha256') == inputs_digest, \
            'validation inputs changed or have no fingerprint; rerun without --validate-only'
        for item in csv.DictReader((DEST/(STEM+'.csv')).open()):
            previous.append({k: int(v) if k in ('P', 's', 'n', 'sample') else
                             v if k == 'method' else float(v) for k, v in item.items()})
        previous_checks = [dict(s=2, **x) if 's' not in x else x for x in old_record['checks']]
    rows, checks = [], []
    for s in (1, 2):
        validate_grid(result, samples, fits, spline_fits, baselines, s,
                      previous, previous_checks, rows, checks)
    with (DEST/(STEM+'.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (DEST/(STEM+'_record.json')).write_text(json.dumps(dict(
        split=result, configurations=16, validation_grids=[40, 80],
        validation_inputs_sha256=inputs_digest,
        same_configurations_as='review_complete_si_record.json',
        shape_selection='unchanged fixed-band SI shapes at each validation grid; full-reference influence recomputed',
        checks=checks), indent=2)+'\n')


def validate_grid(result, samples, fits, spline_fits, baselines, s,
                  previous, previous_checks, rows, checks):
    co_split = np.array(result['legendre_coefficients'])
    n = 40*s; half = n//2
    k = np.r_[np.arange(half), np.arange(-half, 0)]
    ii = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    ix = [x.ravel() for x in ii]
    K = np.column_stack([k[v] for v in ix]); D = ref.dxi*K
    maximum = 3*(half+3*n)**2
    nq = max(160, int(np.ceil(ref.rc*ref.dxi*np.sqrt(maximum)+40)))
    shat = transform_table(co_split[:, None], maximum, nq)[:, 0]
    keep = np.all(np.abs(K) < half, axis=1) & np.any(K != 0, axis=1)
    retained_S = shat[np.sum(K[keep]*K[keep], axis=1)]
    for sample in samples:
        retained = sample['near']+complete.reciprocal_force(sample['r'], sample['q'], D[keep], retained_S, ref.L)
        sample['retained'] = retained
        sample['tail'] = float(np.linalg.norm(retained-sample['truth'])/np.linalg.norm(sample['truth']))
    wave = [ref.dxi*k[x] for x in ii]
    for P in (4, 5, 6):
        cw = float(baselines[P, s]['cw'])
        lead = ref.coeff(cw)/(P*ref.coeff(cw)[0])
        candidates = [('spline', None), ('prolate', lead)]
        candidates.append(('spline_derivative', dict(beta=spline_fits[P, f's{s}'])))
        for method, name in [('sum', f'sum_s{s}'), ('endpoint_sum', f'edge_s{s}'), ('c2_sum', f'c2_s{s}')]:
            fit = next(x for x in fits if x['P'] == P and x['name'] == name)
            candidates.append((method, np.array(fit['legendre_coefficients'])))
        for method, co in candidates:
            cached = [r for r in previous if (r['P'], r['s'], r['method']) == (P, s, method)]
            if len(cached) == len(samples):
                rows.extend(complete.charge_metrics_from_relative(
                    row, samples[row['sample']-1]['truth'], samples[row['sample']-1]['q'])
                    for row in cached)
                checks.extend(x for x in previous_checks if
                              (x['P'], x['s']) == (P, s) and x.get('method') == method)
                continue
            I, shell_difference = complete.influence(n, P, co, shat, k, D, ix)
            checks.append(dict(P=P, s=s, method=method, influence_shell_difference=shell_difference))
            for trial, sample in enumerate(samples, 1):
                force = sample['near']+complete.mesh_force(sample['r'], sample['q'], n, P, co, I, wave)
                rows.append(dict(P=P, s=s, n=n, method=method, sample=trial,
                    **complete.force_metrics(force, sample['retained'], sample['truth'], sample['q'])))
            values = [r['relative_force'] for r in rows if
                      (r['P'], r['s'], r['method']) == (P, s, method)]
            print('COMPLETE', P, s, method, np.mean(values), 'tail', np.mean([x['tail'] for x in samples]), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--design-grid', type=int, nargs='+', choices=[40, 80], default=[80])
    parser.add_argument('--order', type=int, choices=[1, 2], default=1,
                        help='cutoff regularity C^order of the compact residual')
    parser.add_argument('--count', type=int, default=4, help='number of even PSWFs in the density')
    parser.add_argument('--objective', choices=['b6', 'tail', 'c2'], default='b6',
                        help='B_6 natural-influence force error, the physical tail alone, '
                             'or the mean error of the stored C2 SI kernels at P=4,5,6')
    parser.add_argument('--validate-only', action='store_true',
                        help='reuse the saved split and previously evaluated grid/window cases')
    args = parser.parse_args()
    if args.order > 1:
        STEM = f'review_coulomb_c{args.order}'
    if sorted(args.design_grid) == [40]:
        STEM += '_coarse'
    elif sorted(args.design_grid) == [40, 80]:
        STEM += '_joint'
    result = (json.loads((DEST/(STEM+'_split.json')).read_text()) if args.validate_only
              else construct_split(tuple(args.design_grid), args.order, args.count, objective=args.objective))
    validate(result, reuse=args.validate_only)
