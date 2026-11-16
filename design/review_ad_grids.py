"""SI kernels designed for AD forces on the finite grids of the review.

NumPy only. The band, force weights, exact cell alias integrals and
pair-optimal influence are those of review_finite_grids.py. The force
differentiates the interior polynomial of the SI kernel, as MD codes do
(analytic differentiation, AD). The baselines are B_P and the leading
prolate with its bandwidth retuned for AD. The optimization is a
deterministic multistart BFGS search, not a proof of a global optimum.
Run with OPENBLAS_NUM_THREADS=1; --workers only distributes bandwidths.
"""
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legder, legval

import review_finite_grids as design
import review_refinement as ref
from review_complete_si import window_profile


DEST = Path(__file__).resolve().parent/'records'
COUNT = 24
# Even Legendre terms per PSWF; the AD error needs more than the spectral fits at large c_w.
SIZE = 60
# Name, design grid, and endpoint regularity order: C^0 is the least the AD
# error needs to stay bounded under refinement, and C^2 is the kernel we keep.
SPECS = [('c0_s1', (1,), 0), ('c0_s2', (2,), 0), ('c2_s1', (1,), 2), ('c2_s2', (2,), 2)]
FACTORS = (.2, .3, .4, .5, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.4, 2.8, 3.2, 4.)


def ad_error(window, a, s, band=None):
    terms, _, f = window.ad_terms(a, s)
    if f.min() < 1e-3:
        return float('inf')
    value = design.ad_squared_error(*terms, window.parts[s][0].q, band)[0]
    return float(np.sqrt(value)) if np.isfinite(value) else float('inf')


def spectral_error(window, a, s):
    return float(np.sqrt(design.squared_error(window.aliases(a, s)[0])[0]))


def spline_error(P, s, nq=160, base=None, band=None):
    # base, band: the band coordinates on the base grid and the band of review_finite_grids, by default.
    # B_P' = B_{P-1}(x+1/2)-B_{P-1}(x-1/2); B_P is continuous, so the endpoint term is zero.
    cell = ref.Cell(P, (design.qbase if base is None else base)/s, nq)
    q = cell.q
    f = np.sinc(q/(2*np.pi))**P
    g = window_profile(cell.x, P, None)*cell.ins
    dg = (window_profile(cell.x+.5, P-1, None)-window_profile(cell.x-.5, P-1, None))*cell.ins
    D0 = np.abs(np.einsum('qlu,lu->qu', cell.phase, g)-f[:, None])**2@cell.wu
    # The cell mean of the derivative response is -i q f.
    DV = np.abs(np.einsum('qlu,lu->qu', cell.phase, dg)+1j*(q*f)[:, None])**2@cell.wu
    eta = (DV-q*q*D0)/(f*f+D0)
    return float(np.sqrt(design.ad_squared_error(D0/f**2, 0*q, eta, q, band)[0]))


def prolate_error(P, c, s, nq=80, base=None, band=None):
    window = design.Window(P, c, 1, nq, SIZE, base, scales=(s,))
    return ad_error(window, np.array([1/window.beta[0]]), s, band)


def fit_bandwidth(task):
    P, c, scales = task
    window = design.Window(P, c, COUNT, size=SIZE)
    results = {}
    for name, grids, order in SPECS:
        result = design.fit(window, grids, scales, order, force='ad', random_starts=12, iterations=600)
        if result is not None:
            results[name] = result
    return P, c, results


def main(workers):
    baselines = {(int(r['P']), int(r['s'])): {k: float(v) for k, v in r.items()}
                 for r in csv.DictReader((DEST/'review_retuned_refinement.csv').open())}
    spectral = {(x['P'], x['name']): x for x in
                json.loads((DEST/'review_finite_grids_record.json').read_text())['fits']}
    tasks = []
    for P in (4, 5, 6):
        scales = {s: baselines[P, s]['pro']**2 for s in (1, 2)}
        bandwidths = sorted(set([baselines[P, s]['cw'] for s in (1, 2)]
                                + [factor*np.pi*P for factor in FACTORS]))
        tasks += [(P, c, scales) for c in bandwidths]
    if workers > 1:
        with Pool(workers) as pool:
            outcomes = pool.map(fit_bandwidth, tasks, chunksize=1)
    else:
        outcomes = [fit_bandwidth(task) for task in tasks]
    best = {}
    for P, c, results in outcomes:
        for name, result in results.items():
            if (P, name) not in best or result['loss'] < best[P, name][1]['loss']:
                best[P, name] = c, result
    fits, records = [], []
    for P in (4, 5, 6):
        for name, grids, order in SPECS:
            c, result = best[P, name]
            a = result['a']
            window, verify = design.Window(P, c, COUNT, size=SIZE), design.Window(P, c, COUNT, 160, SIZE)
            assert abs(window.beta@a-1) < 2e-13
            assert all(abs(t@a) < 1e-12*np.linalg.norm(t)*np.linalg.norm(a) for t in window.traces[:order+1])
            item = dict(P=P, name=name, cw=c, count=COUNT, order=order, coefficients=a.tolist(),
                        legendre_coefficients=(a@window.co).tolist(),
                        endpoint_traces=(window.traces@a).tolist(), integral=float(window.beta@a))
            worst = 0.
            for s in (1, 2):
                value, checked = ad_error(window, a, s), ad_error(verify, a, s)
                worst = max(worst, abs(checked/value-1))
                item[f'ad_error_s{s}'] = value
                item[f'spectral_error_s{s}'] = spectral_error(window, a, s)
            assert worst < 1e-6
            item.update(quadrature_relative_difference=worst, loss=result['loss'],
                        projected_gradient=result['projected_gradient'],
                        iterations=result['iterations'], starts=result['starts'])
            fits.append(item)
            print({k: v for k, v in item.items() if 'coefficients' not in k}, flush=True)
        for s in (1, 2):
            # The leading prolate, with its bandwidth retuned for AD on this grid.
            cw, value = ref.minimize(lambda c: prolate_error(P, c, s), np.linspace(.3, 2.5, 111)*np.pi*P)
            assert .3*np.pi*P < cw < 2.5*np.pi*P
            checked = prolate_error(P, cw, s, nq=160)
            assert abs(checked/value-1) < 1e-6
            spline = spline_error(P, s)
            assert abs(spline_error(P, s, nq=320)/spline-1) < 1e-8
            row = dict(P=P, s=s, spline=spline, prolate=value, prolate_cw=cw)
            for field, name in [('c0_sum', f'c0_s{s}'), ('c2_sum', f'c2_s{s}')]:
                row[field] = next(x[f'ad_error_s{s}'] for x in fits if (x['P'], x['name']) == (P, name))
            # The spectral-force error of the same tuned psi_0 kernel, and the mean-square
            # deviation of the total assigned charge from one (the alias ratio at q=0).
            window = design.Window(P, cw, 1, size=SIZE)
            lead = np.array([1/window.beta[0]])
            row['prolate_spectral'] = spectral_error(window, lead, s)
            assert design.qbase[0] == 0
            row['prolate_charge_defect'] = float(window.aliases(lead, 1)[0][0])
            x = next(x for x in fits if (x['P'], x['name']) == (P, f'c2_s{s}'))
            row['c2_charge_defect'] = float(design.Window(P, x['cw'], COUNT, size=SIZE).aliases(np.array(x['coefficients']), 1)[0][0])
            # Kernels designed for spectral forces, used with AD.
            for field, name in [('spectral_sum', f'sum_s{s}'), ('spectral_c2', f'c2_s{s}')]:
                x = spectral[P, name]
                row[field] = ad_error(design.Window(P, x['cw'], x['count']), np.array(x['coefficients']), s)
            records.append(row)
            print('SUMMARY', row, flush=True)
    with (DEST/'review_ad_grids.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    metadata = dict(force='AD: the analytic derivative of the interior polynomial of the SI kernel',
                    observable='band-model AD force pair RMS, relative to the retained exact force power',
                    influence='pair-optimal for the AD force and the same truncated spherical reference',
                    basis=f'{COUNT} even L2-normalized PSWFs, signs positive at the origin',
                    regularity='order k imposes g^(l)(P/2)=0 for l<=k, so the zero-extended kernel is C^k',
                    pswf_legendre_size=SIZE, cell_quadrature=80, verification_quadrature=160,
                    bandwidth_search='0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4,2.8,3.2,4.0 times pi P, plus both retuned leading bandwidths',
                    prolate_search='0.3 to 2.5 times pi P, 111 points, golden-section refinement at each local minimum',
                    stability='f(q)>=1e-3 at all retained coordinates, coefficient norm <=10',
                    optimizer='multistart BFGS with analytic gradients, 12 random starts, 600 iterations; no global-optimality certificate',
                    spectral_columns='kernels of review_finite_grids_record.json (sum and c2), evaluated with AD',
                    fits=fits)
    (DEST/'review_ad_grids_record.json').write_text(json.dumps(metadata, indent=2)+'\n')


def independent_error(P, c, a, s, size=80, nq=160):
    """The AD error from a larger PSWF expansion, built without Window."""
    co = a@design.pswf_basis(c, count=len(a), size=size)
    dco = legder(co)
    cell = ref.Cell(P, design.qbase/s, nq)
    q = cell.q
    t, w = leggauss(200)
    g = legval(cell.xleg, co)*cell.ins
    dg = (2/P)*legval(cell.xleg, dco)*cell.ins
    f = (P/2)*np.cos((P/2)*q[:, None]*t)@(w*legval(t, co))
    v = -np.sin((P/2)*q[:, None]*t)@(w*legval(t, dco))
    D0 = np.abs(np.einsum('qlu,lu->qu', cell.phase, g)-f[:, None])**2@cell.wu
    DV = np.abs(np.einsum('qlu,lu->qu', cell.phase, dg)+1j*v[:, None])**2@cell.wu
    # The endpoint term delta=q f-v=2 g(P/2) sin(qP/2) is what interior differentiation omits.
    delta = 2*legval(1., co)*np.sin(q*P/2)
    assert np.allclose(q*f-v, delta, rtol=0, atol=1e-12)
    eta = (DV-q*q*D0-delta*(2*q*f-delta))/(f*f+D0)
    return float(np.sqrt(design.ad_squared_error(D0/f**2, q*delta/f, eta, q)[0]))


def verify():
    data = json.loads((DEST/'review_ad_grids_record.json').read_text())
    worst = 0.
    for item in data['fits']:
        for s in (1, 2):
            value = independent_error(item['P'], item['cw'], np.array(item['coefficients']), s)
            difference = abs(value/item[f'ad_error_s{s}']-1)
            worst = max(worst, difference)
            assert difference < 1e-6, (item['P'], item['name'], s, difference)
    # The analytic gradient of the AD loss against a central difference.
    item = next(x for x in data['fits'] if (x['P'], x['name']) == (4, 'c2_s2'))
    window = design.Window(4, item['cw'], COUNT, size=SIZE)
    a = np.array(item['coefficients'])
    q = window.parts[2][0].q

    def loss(b):
        terms, d_terms, _ = window.ad_terms(b, 2)
        value, d_value = design.ad_squared_error(*terms, q)
        return value, sum(x@y for x, y in zip(d_value, d_terms))

    direction = np.random.default_rng(1).normal(size=len(a)); direction /= np.linalg.norm(direction)
    step = 1e-7
    difference = (loss(a+step*direction)[0]-loss(a-step*direction)[0])/(2*step)
    gradient_error = abs(difference/(loss(a)[1]@direction)-1)
    assert gradient_error < 1e-5
    verification = dict(shapes=len(data['fits']), verification_legendre_size=80, verification_quadrature=160,
                        maximum_relative_RMS_change=worst, directional_gradient_relative_error=float(gradient_error))
    print('VERIFICATION', verification, flush=True)
    (DEST/'review_ad_grids_verification.json').write_text(json.dumps(verification, indent=2)+'\n')


if __name__ == '__main__':
    if '--verify' not in sys.argv:
        workers = int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 1
        main(workers)
    verify()
