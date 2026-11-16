"""C^d splits of 1/r^nu, nu = 1, ..., 6, with the fewest PSWF terms, and their truncation errors.

The profile w is even on [-1, 1] with w(0) = 1, and the split is that of review_c2_split_fewest.py (Section 2.2):
the local part t^-nu R_nu(t), with R_2 = w, R_1 = int_t^1 w / int_0^1 w and R_{nu+2} = R_nu - (t/nu) R_nu', vanishes
for t > 1, and the smooth part s_nu = t^-nu - t^-nu R_nu is t^-nu there. The local part, and with it the smooth part,
is d times continuously differentiable at t = 1 if and only if R_nu^(j)(1) = 0 for j = 0, ..., d (Proposition 5.1),
d conditions for nu = 1, where R_1(1) = 0, and d + 1 for nu >= 2. The C^d split uses the fewest even PSWFs
psi_0^c, ..., psi_2m^c that admit the m conditions.

For each nu, d = 2, 3, 4, and each tolerance eps with the bandwidth c of ESP's rule chi(1) = eps (review_si_coarsest),
the program reports:
- the coefficients of the profile relative to the first and its minimum on [0, 1];
- the jump at the cutoff of the derivative of order d + 1 of the local part, relative to that of t^-nu;
- the relative force truncation error of the split in the limit of large boxes, the square root of the force energy
  of the smooth part outside the ball |xi| <= c/r_c relative to its total, from the three-dimensional Fourier
  transform of s_nu: 4 pi/xi [int_0^1 r s_nu(r) sin(xi r) dr + Im E_{nu-1}(-i xi)] for nu >= 2, with the generalized
  exponential integral E_n(z) = int_1^inf e^{-zt} t^-n dt, and 4 pi chi_hat(xi)/xi^2 for nu = 1; the smooth part on
  [0, 1] is computed by exact division of Legendre series, the integral by Gauss quadrature up to |xi| = 80 c, and
  the rest is bounded by integrations by parts (tail_bound);
and, for the psi_0 split, the first discontinuous derivative of the local part at the cutoff relative to that of
t^-nu: the force for nu = 1 and the energy itself for nu >= 2, whose smooth part then jumps and has no finite force
truncation error. NumPy only. Run with OPENBLAS_NUM_THREADS=1.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder, legdiv, leggauss, legint, legmulx, legroots, legsub, legval

import review_c2_split_fewest as F
import review_finite_grids as design
import review_si_coarsest as co

DEST = Path(__file__).resolve().parent/'records'
STEM = 'review_inverse_splits'
TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
POWERS = (1, 2, 3, 4, 5, 6)
ORDERS = (2, 3, 4)


gauss = lru_cache(maxsize=None)(leggauss)


def absolute_integral(c):
    """int_0^1 |p(t)| dt for the Legendre series p, exactly, by splitting at the real roots of p in (0, 1)."""
    roots = legroots(c) if len(c) > 1 else np.array([])
    roots = np.sort(roots[(np.abs(np.imag(roots)) < 1e-9) & (np.real(roots) > 0) & (np.real(roots) < 1)].real)
    ends = np.concatenate([[0.], roots, [1.]])
    return float(np.sum(np.abs(np.diff(legval(ends, legint(c))))))


def residual(profile, nu, normalized=True):
    """R_nu as a Legendre series on [-1, 1]; for odd nu, without the factor 1/int_0^1 w if not normalized."""
    profile = np.asarray(profile, dtype=float)
    if nu % 2:
        R, k = -legint(profile, lbnd=1), 1
        if normalized:
            R = R/profile[0]
    else:
        R, k = profile.copy(), 2
    while k < nu:
        R = legsub(R, legmulx(legder(R))/k)
        k += 2
    return R


def conditions(basis, nu, d):
    """The rows R_nu^(j)(1) = 0, j = 0, ..., d (j >= 1 for nu = 1), as a matrix acting on the coefficients."""
    if nu == 1:
        # R_1^(j)(1) = -w^(j-1)(1)/int_0^1 w; the rows of review_c2_split_fewest.fewest.
        return np.array([legval(1., legder(basis.T, j)) for j in range(d)])
    return np.array([[legval(1., legder(residual(b, nu, False), j)) for b in basis] for j in range(d+1)])


def split(c, nu, d):
    """The C^d split of 1/r^nu with the fewest even PSWFs of bandwidth c: the coefficients and the profile, w(0) = 1."""
    m = d if nu == 1 else d+1
    basis = design.pswf_basis(c, count=m+1, size=60)
    E = conditions(basis, nu, d)
    a = np.linalg.svd(E, full_matrices=True)[2][-1]
    profile = a@basis
    scale = legval(0., profile)
    return a/scale, profile/scale, float(np.linalg.norm(E@(a/scale))/np.linalg.norm(E))


def prolate_eigenvalues(c, count, size=60):
    """The eigenvalues gamma_0 < gamma_2 < ... of the prolate differential operator
    -((1 - t^2) psi')' + c^2 t^2 psi for the first count even PSWFs, from the matrix of
    review_finite_grids.pswf_basis."""
    k = np.arange(0, 2*size, 2, dtype=float)
    diagonal = k*(k+1)+c*c*(2*k*(k+1)-1)/((2*k+3)*(2*k-1))
    off = c*c*(k[:-1]+2)*(k[:-1]+1)/((2*k[:-1]+3)*np.sqrt((2*k[:-1]+1)*(2*k[:-1]+5)))
    return np.linalg.eigvalsh(np.diag(diagonal)+np.diag(off, 1)+np.diag(off, -1))[:count]


def closed_form(c, m):
    """The profile in the span of psi_0^c, ..., psi_2m^c with w^(j)(1) = 0, j < m, and w(0) = 1: its coefficients
    are proportional to 1/(psi_2i(1) prod_{k != i}(gamma_2i - gamma_2k)) (Theorem 3.2)."""
    basis = design.pswf_basis(c, count=m+1, size=60)
    gamma = prolate_eigenvalues(c, m+1)
    end = legval(1., basis.T)
    a = np.array([1/(end[i]*np.prod([gamma[i]-gamma[k] for k in range(m+1) if k != i])) for i in range(m+1)])
    profile = a@basis
    scale = legval(0., profile)
    return a/scale, profile/scale


def smooth_part(profile, nu):
    """The smooth part s_nu on [-1, 1] as an even Legendre series, from s_1 = int_0^t w/(t int_0^1 w) or
    s_2 = (1 - w)/t^2 and s_{nu+2} = -s_nu'/(nu t), each division by t or t^2 done exactly in the Legendre basis
    (the remainders vanish to rounding error, since the numerators vanish at t = 0 to the order of the divisor)."""
    if nu % 2:
        s, k = legdiv(legint(profile, lbnd=0), [0., 1.])[0]/profile[0], 1
    else:
        s, k = legdiv(legsub([1.], profile), [1/3, 0., 2/3])[0], 2
    while k < nu:
        s, k = -legdiv(legder(s), [0., 1.])[0]/k, k+2
    return s


def expint(n, z, tolerance=1e-15):
    """E_n(z) = int_1^inf e^{-zt} t^-n dt by its continued fraction (modified Lentz), for |z| > 1, Re z >= 0."""
    z = np.asarray(z, dtype=complex)
    b = z+n
    c = np.full(z.shape, 1e300, dtype=complex)
    dd = 1/b
    h = dd.copy()
    for i in range(1, 100000):
        a = -i*(n-1+i)
        b = b+2
        dd = 1/(a*dd+b)
        c = b+a/c
        delta = c*dd
        h = h*delta
        if np.max(np.abs(delta-1)) < tolerance:
            break
    return h*np.exp(-z)


def transform(profile, nu, xi):
    """The three-dimensional Fourier transform of the smooth part s_nu (r_c = 1) at |xi| = xi > 1."""
    xi = np.asarray(xi, dtype=float)
    t, w = gauss(max(400, int(2*xi.max())+200))
    if nu == 1:
        chi = legval(t, profile)/(2*profile[0])
        return 4*np.pi*(np.cos(np.outer(xi, t))@(w*chi))/xi**2
    r, wr = .5*(t+1), .5*w
    s = legval(r, smooth_part(profile, nu))
    inner = np.concatenate([np.sin(np.outer(x, r))@(wr*r*s) for x in np.array_split(xi, max(1, len(xi)//256))])
    return 4*np.pi/xi*(inner+expint(nu-1, -1j*xi).imag)


def tail_bound(profile, nu, d, top, integrations=8):
    """A bound on 4 pi int_{xi > top} xi^4 S_hat(xi)^2 dxi, for the split whose local part is d times continuously
    differentiable at the cutoff, by m = integrations integrations by parts.

    For nu >= 2, S_hat(xi) = (4 pi/xi) int_0^inf f(r) sin(xi r) dr with f = r s_nu, which is odd and smooth on
    [0, 1], equals r^(1-nu) beyond, and has continuous derivatives of orders up to d at r = 1. The even derivatives
    of f vanish at 0, so the terms at r = 0 drop out, and
        |S_hat(xi)| <= (4 pi/xi) (sum_{j=d+1}^{m-1} |[f^(j)](1)|/xi^(j+1) + int_0^inf |f^(m)(r)| dr/xi^m),
    with [f^(j)](1) the jump of f^(j) at r = 1. For nu = 1, S_hat = 4 pi chi_hat/xi^2 with chi even on [-1, 1] and
    chi^(j)(1) = 0 for j < d, and in the same way
        |chi_hat(xi)| <= 2 (sum_{j=d}^{m-1} |chi^(j)(1)|/xi^(j+1) + int_0^1 |chi^(m)(t)| dt/xi^m).
    The derivatives below the first nonzero jump are zero for the exact split and are not used. The integrals of
    |f^(m)| are evaluated exactly, on [0, 1] by splitting at the roots of f^(m)."""
    if nu == 1:
        chi = np.asarray(profile, dtype=float)/(2*profile[0])
        a = [abs(legval(1., legder(chi, j))) for j in range(d, integrations)]
        a.append(absolute_integral(legder(chi, integrations)))
        p = list(range(d+1, integrations+1))+[integrations]
        p = p[:len(a)-1]+[integrations]
        scale, shift = 4*np.pi*(2*4*np.pi)**2, 1
    else:
        f = legmulx(smooth_part(profile, nu))
        outside = lambda j: (-1)**j*np.prod(np.arange(nu-1, nu-1+j, dtype=float))
        a = [abs(outside(j)-legval(1., legder(f, j))) for j in range(d+1, integrations)]
        a.append(absolute_integral(legder(f, integrations))+np.prod(np.arange(nu-1, nu+integrations-2, dtype=float)))
        p = list(range(d+2, integrations+1))+[integrations]
        scale, shift = 4*np.pi*(4*np.pi)**2, 3
    a, p = np.array(a), np.array(p, dtype=float)
    # int_top^inf xi^(shift-1) (sum_i a_i xi^-p_i)^2 dxi, with xi^4 S_hat^2 <= scale/(4 pi) xi^(shift-1) (...)^2.
    P, Q = np.meshgrid(p, p, indexing='ij')
    return float(scale*np.sum(np.outer(a, a)*top**(shift-P-Q)/(P+Q-shift)))


def truncation(profile, nu, c, d, top=80.):
    """The relative force truncation error of the split in the limit of large boxes: the square root of
    int_{|xi|>c} |xi|^2 |S_hat|^2 dxi over its value on all of R^3, which is (2 pi)^3 int |grad s_nu|^2 dx.
    d is the order of continuity of the smooth part at r = 1. If the smooth part jumps (d < 0), both integrals are
    infinite and the relative error is not defined. The integral is evaluated by Gauss quadrature for
    c < |xi| < top c, and the bound of tail_bound on the rest is added; the second value is the fraction of the
    total that the bound contributes."""
    if d < 0:
        return None, None
    t, w = leggauss(40)
    edges = c*np.geomspace(1., top, 60)
    xi = np.concatenate([.5*(b-a)*t+.5*(a+b) for a, b in zip(edges[:-1], edges[1:])])
    wx = np.concatenate([.5*(b-a)*w for a, b in zip(edges[:-1], edges[1:])])
    S = transform(profile, nu, xi)
    f = xi**4*S**2
    out = 4*np.pi*wx@f
    tail = tail_bound(profile, nu, d, top*c)
    out = out+tail
    ds = legder(smooth_part(profile, nu) if nu > 1 else smooth_part(profile, 1))
    r, wr = .5*(t+1), .5*w
    rr, ww = leggauss(400)
    rr, ww = .5*(rr+1), .5*ww
    inside = ww@(legval(rr, ds)**2*rr**2)
    total = (2*np.pi)**3*4*np.pi*(inside+nu*nu/(2*nu-1))
    return float(np.sqrt(out/total)), float(tail/out)


def jump(profile, nu, order):
    """|R_nu^(order)(1)| relative to |d^order t^-nu/dt^order| at t = 1, the jump of that derivative of the local part
    when the lower ones vanish."""
    R = residual(profile, nu)
    bare = np.prod(np.arange(nu, nu+order)) if order else 1.
    return float(abs(legval(1., legder(R, order)) if order else legval(1., R))/bare)


def main():
    rows = []
    grid = np.linspace(0, 1, 2001)
    for eps in TOLERANCES:
        c = co.bandwidth(eps)
        lead = F.leading(c, 2)
        for nu in POWERS:
            # The psi_0 split: for nu = 1 the force jumps, for nu >= 2 the local part itself, and then the relative
            # truncation error is not defined.
            trunc, tail = truncation(lead, nu, c, 0 if nu == 1 else -1)
            item = dict(tolerance=eps, c=c, nu=nu, d=None, psi0_jump=jump(lead, nu, 1 if nu == 1 else 0),
                        psi0_truncation=trunc, psi0_tail_fraction=tail)
            rows.append(item)
            for d in ORDERS:
                a, profile, constraint = split(c, nu, d)
                trunc, tail = truncation(profile, nu, c, d)
                rows.append(dict(tolerance=eps, c=c, nu=nu, d=d, terms=len(a), relative_coefficients=(a/a[0]).tolist(),
                                 legendre_coefficients=profile.tolist(), constraint_residual=constraint,
                                 minimum=float(legval(grid, profile).min()), jump=jump(profile, nu, d+1),
                                 truncation=trunc, tail_fraction=tail))
                print({k: (f'{v:.3e}' if isinstance(v, float) else v) for k, v in rows[-1].items()
                       if k not in ('relative_coefficients', 'legendre_coefficients')}, flush=True)
    (DEST/(STEM+'_record.json')).write_text(json.dumps(dict(
        definition='C^d split: fewest even PSWFs with R_nu^(j)(1) = 0, j <= d; w(0) = 1; bandwidth by chi(1) = eps',
        truncation='sqrt(int_{|xi|>c} |xi|^2 S_hat^2 / int_R3 |xi|^2 S_hat^2), limit of large boxes, r_c = 1',
        jump='|R_nu^(d+1)(1)| / (nu (nu+1) ... (nu+d)); psi_0: force jump (nu = 1) or energy jump (nu >= 2) relative',
        rows=rows), indent=1)+'\n')


if __name__ == '__main__':
    main()
