"""Truncation error of the psi_0 and C^2 Coulomb splits against the grid size.

For the retained set {k != 0 : |k_a| < n/2}, the mean-square force of the
truncation error is (3.12) of the review,
    sigma_T^2 = L^-6 sum_{k not retained} |xi_k|^2 S_hat(xi_k)^2,
with S_hat(xi) = 4 pi chi_hat(|xi| r_c)/|xi|^2. The lattice sum is taken over
spheres |k|^2 <= N, with the number of lattice points on each sphere from a
convolution of the one-dimensional counts, and the remainder beyond N is
added from the large-|xi| form of chi_hat, 2 chi(1) sin(omega)/omega, whose
square averages to 2 chi(1)^2/omega^2 (zero for the C^2 split, whose density
vanishes at 1 with its first derivative; its remainder is negligible).

The psi_0 split is that of review_refinement.py (c_s = 12.024, r_c = 1,
L = 10); the C^2 split is read from review_coulomb_c2_ad_split.json.
NumPy only. Run with OPENBLAS_NUM_THREADS=1.
"""
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legval

import review_coulomb_endpoint as endpoint
import review_refinement as ref

DEST = Path(__file__).resolve().parent/'records'
STEM = 'review_split_truncation'
SHELLS = 1_000_000          # |k| <= 1000
GRIDS = list(range(40, 82, 2))


def sphere_counts(maximum):
    """r_3(m), the number of k in Z^3 with |k|^2 = m, for m <= maximum."""
    one = np.zeros(maximum+1)
    j = np.arange(int(np.sqrt(maximum))+1)
    np.add.at(one, j*j, 2.)
    one[0] = 1.
    size = 1 << int(np.ceil(np.log2(3*maximum+3)))
    f = np.fft.rfft(one, size)
    three = np.fft.irfft(f**3, size)[:maximum+1]
    return np.rint(three)


def main():
    split = json.loads((DEST/'review_coulomb_c2_ad_split.json').read_text())
    c2 = np.array(split['legendre_coefficients'])
    psi0 = ref.coeff(ref.c)/(2*ref.coeff(ref.c)[0])
    width = max(len(c2), len(psi0))
    profiles = np.zeros((width, 2))
    profiles[:len(psi0), 0] = psi0
    profiles[:len(c2), 1] = c2
    nq = int(np.ceil(ref.rc*ref.dxi*np.sqrt(SHELLS)))+200
    table = endpoint.transform_table(profiles, SHELLS, nq)
    shell = np.arange(SHELLS+1)
    weight = ref.dxi**2*shell
    counts = sphere_counts(SHELLS)
    total = (counts*weight)@(table**2)
    # Remainder beyond |xi| = R from 2 chi(1)^2/omega^2: L^-6 (L/2 pi)^3 4 pi 16 pi^2 int_R^inf 2 chi(1)^2/(xi r_c)^2 dxi.
    chi1 = legval(1., profiles)
    R = ref.dxi*np.sqrt(SHELLS)
    remainder = (ref.L/(2*np.pi))**3*4*np.pi*16*np.pi**2*2*chi1**2/(ref.rc**2*R)
    total = total+remainder
    rows = []
    for n in GRIDS:
        half = n//2
        k = np.arange(-half+1, half)
        K2 = (k[:, None, None]**2+k[None, :, None]**2+k[None, None, :]**2).ravel()
        K2 = K2[K2 > 0]
        retained = np.bincount(K2, minlength=SHELLS+1)[:SHELLS+1]
        kept = (retained*weight)@(table**2)
        sigma = np.sqrt((total-kept)/ref.L**6)
        rows.append(dict(n=n, nyquist_over_band=float(np.pi*n/ref.L/(ref.c/ref.rc)),
                         psi0_sigma_T=float(sigma[0]), c2_sigma_T=float(sigma[1])))
        print(n, rows[-1], flush=True)
    record = dict(c_s=ref.c, rc=ref.rc, L=ref.L, shells=SHELLS, quadrature_points=nq,
                  chi_at_1=chi1.tolist(), remainder=(remainder/ref.L**6).tolist(),
                  definition='sigma_T^2 = L^-6 sum over k not in {|k_a| < n/2} of |xi_k|^2 S_hat(xi_k)^2',
                  rows=rows)
    (DEST/(STEM+'_record.json')).write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
