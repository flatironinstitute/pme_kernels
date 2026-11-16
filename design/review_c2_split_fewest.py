"""The C^2 Coulomb split with the fewest PSWF terms, at the bandwidths of ESP's rule.

The local part of 1/r is twice continuously differentiable at the cutoff if
and only if chi(1) = chi'(1) = 0 (Proposition 3.1), and the C^2 split uses
the fewest even PSWFs that admit these m = 2 conditions, psi_0^c, psi_2^c,
psi_4^c: the conditions leave one direction, and the profile is normalized
by int chi = 1. The coefficients are proportional to
1/(psi_2i(1) prod_{k != i}(gamma_2i - gamma_2k)) (Theorem 3.2), which the
program checks against the null vector of the conditions. Other powers:
review_inverse_splits.py.

For each tolerance eps = 1e-2, ..., 1e-8 and the bandwidth c of ESP's rule
chi(1) = eps for the psi_0 split (review_si_coarsest.bandwidth), the program
reports:
- the PSWF coefficients relative to the first, and their largest relative
  difference from the closed form;
- the jump at the cutoff of the third derivative of the local part of the
  pair energy, relative to the third derivative of the bare interaction,
  |chi''(1)|/3;
- the minimum of the profile over [-1, 1], relative to its value at 0;
- the truncation error of both splits on cubic retained sets |k_a| < n/2,
  relative to the RMS force of the smooth part of the same split (and
  also without this normalization), against the Nyquist frequency in units
  of c/r_c (L = 10, r_c = 1), with
  the ratio of the two on the grid that ESP uses (the first even n with
  Nyquist frequency at least c/r_c, ESP's rule), the grid at which the C^2 split
  reaches the truncation error of the psi_0 split on that grid, and the
  grid above which it is the smaller one.
NumPy only. Run with OPENBLAS_NUM_THREADS=1.
"""
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder, leggauss, legmulx, legval

import review_coulomb_endpoint as endpoint
import review_cutoff_jumps as jumps
import review_finite_grids as design
import review_refinement as ref
import review_split_truncation as truncation

DEST = Path(__file__).resolve().parent/'records'
STEM = 'review_c2_split_fewest'
TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
SHELLS = 1_000_000


def conditions(nu):
    return 2 if nu == 1 else nu//2+2


def fewest(c, nu):
    if nu != 1:
        raise ValueError('the splits of other powers are review_inverse_splits.split')
    m = conditions(nu)
    basis = design.pswf_basis(c, count=m+1, size=60)
    E = np.array([legval(1., legder(basis.T, j)) for j in range(m)])
    a = np.linalg.svd(E, full_matrices=True)[2][-1]
    profile = a@basis
    scale = 2*profile[0] if nu % 2 else legval(0., profile)
    return a/scale, profile/scale, basis, float(np.linalg.norm(E@(a/scale)))


def leading(c, nu):
    lead = design.pswf_basis(c, count=1, size=60)[0]
    return lead/(2*lead[0] if nu % 2 else legval(0., lead))


def out_of_band(profile, c):
    t, wt = leggauss(800)
    x, wx = leggauss(1600)
    omega = .5*c*(x+1.)
    v = legval(t, profile)
    transform = np.cos(np.outer(omega, t))@(wt*v)
    return 2*np.pi*(wt@v**2)-c*(wx@transform**2)


def third_derivative_jump(profile, nu):
    if nu == 1:
        return abs(legval(1., legder(profile, 2)))/3
    # R_6 = w - 5/8 t w' + 1/8 t^2 w''; only R_6''' survives at t = 1.
    w1, w2 = legder(profile), legder(profile, 2)
    R6 = np.zeros(len(profile)+2)
    R6[:len(profile)] += profile
    tw1 = legmulx(w1)
    R6[:len(tw1)] -= 5/8*tw1
    t2w2 = legmulx(legmulx(w2))
    R6[:len(t2w2)] += 1/8*t2w2
    return abs(legval(1., legder(R6, 3)))/336


def truncation_curve(profiles, c, n0):
    nq = int(np.ceil(ref.rc*ref.dxi*np.sqrt(SHELLS)))+200
    table = endpoint.transform_table(profiles, SHELLS, nq)
    weight = ref.dxi**2*np.arange(SHELLS+1)
    counts = truncation.sphere_counts(SHELLS)
    # Beyond |k|^2 = SHELLS, every mode is omitted; the transform of a split whose density jumps at t = 1 decays
    # like 2 chi(1) sin(kappa)/kappa there, and sin^2 averages to 1/2.
    chi1 = legval(1., profiles)
    R = ref.dxi*np.sqrt(SHELLS)
    tail = (ref.L/(2*np.pi))**3*4*np.pi*16*np.pi**2*2*chi1**2/(ref.rc**2*R)
    # The mean-square force of the smooth part, all modes: the denominator of the relative truncation error.
    total = (counts*weight)@(table**2)+tail
    rows = []
    for n in range(n0, int(2.4*n0)+1, 2):
        k = np.arange(-n//2+1, n//2)
        K2 = (k[:, None, None]**2+k[None, :, None]**2+k[None, None, :]**2).ravel()
        retained = np.bincount(K2[K2 > 0], minlength=SHELLS+1)[:SHELLS+1]
        # The omitted modes are summed directly, since the difference of the sums over all and over the retained
        # modes loses all precision once the truncation error is below about 1e-7 of the force.
        omitted = counts-retained
        omitted[0] = 0
        omitted_energy = (omitted*weight)@(table**2)+tail
        sigma = np.sqrt(omitted_energy/ref.L**6)
        relative = np.sqrt(omitted_energy/total)
        rows.append(dict(n=n, nyquist_over_band=float(np.pi*n/ref.L/(c/ref.rc)),
                         psi0=float(relative[0]), c2=float(relative[1]),
                         psi0_absolute=float(sigma[0]), c2_absolute=float(sigma[1])))
    return rows


def crossing(x, a, b):
    d = np.log(np.asarray(b))-np.log(np.asarray(a))
    for i in range(len(d)-1):
        if d[i] > 0 >= d[i+1]:
            return float(x[i]+(x[i+1]-x[i])*d[i]/(d[i]-d[i+1]))
    return None


def main():
    import review_inverse_splits as powers
    import review_si_coarsest as co
    t = np.linspace(-1, 1, 4001)
    out = []
    for tolerance in TOLERANCES:
        c = co.bandwidth(tolerance)
        a, profile, basis, residual = fewest(c, 1)
        closed, _ = powers.closed_form(c, 2)
        lead = leading(c, 1)
        item = dict(tolerance=tolerance, c=c, nu=1, terms=len(a),
                    coefficients=a.tolist(), relative_coefficients=(a/a[0]).tolist(),
                    closed_form_difference=float(np.max(np.abs(closed/closed[0]-a/a[0])/np.abs(a/a[0]))),
                    legendre_coefficients=profile.tolist(), constraint_residual=residual,
                    third_derivative_jump=float(third_derivative_jump(profile, 1)),
                    minimum_relative=float(legval(t, profile).min()/legval(0., profile)))
        n0 = int(2*np.ceil(c*ref.L/(np.pi*ref.rc)/2))
        width = max(len(lead), len(profile))
        profiles = np.zeros((width, 2))
        profiles[:len(lead), 0] = lead
        profiles[:len(profile), 1] = profile
        rows = truncation_curve(profiles, c, n0)
        x = [r['nyquist_over_band'] for r in rows]
        item.update(n0=n0, truncation=rows,
                    base_ratio=rows[0]['c2']/rows[0]['psi0'],
                    reaches_psi0_base_at=crossing(x, [rows[0]['psi0']]*len(rows), [r['c2'] for r in rows]),
                    smaller_than_psi0_from=crossing(x, [r['psi0'] for r in rows], [r['c2'] for r in rows]))
        out.append(item)
        print({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in item.items()
               if k not in ('coefficients', 'legendre_coefficients', 'truncation', 'relative_coefficients')},
              'rel coeff', [f'{x:.2e}' for x in item['relative_coefficients']], flush=True)
    (DEST/'review_c2_split_coulomb_record.json').write_text(json.dumps(dict(
        L=ref.L, rc=ref.rc, bandwidth="ESP's rule chi(1) = eps for the psi_0 split", splits=out), indent=2)+'\n')


if __name__ == '__main__':
    main()
