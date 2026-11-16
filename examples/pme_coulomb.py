"""Coulomb forces of random charges by a particle mesh Ewald method with the new kernels, checked against Ewald.

For a tolerance eps and an order P, the program takes the C^2 Coulomb split and the optimized SI kernel phi_P^*, the
grid of section 4.7 of the paper (the first FFT-friendly grid with spacing at most pi r_c/(c_s s*)), and the optimal
influence function for analytic differentiation. It reports the relative RMS force error against an accurate Ewald sum,
and that of the B-spline of the same order on the same grid.

    python examples/pme_coulomb.py --eps 1e-4 --P 5 --N 1000 --L 10
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pme_kernels as pk  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eps', type=float, default=1e-4, help='tolerance (a tabulated power of ten)')
    p.add_argument('--P', type=int, default=5, help='order of the SI kernel')
    p.add_argument('--N', type=int, default=1000, help='number of charges (+1 and -1, alternating)')
    p.add_argument('--L', type=float, default=10.0, help='side of the cubic box, in units of the cutoff')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    rc = 1.0
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(0, args.L, (args.N, 3))
    q = np.where(np.arange(args.N) % 2 == 0, 1.0, -1.0)

    t0 = time.time()
    E_ref, F_ref = pk.ewald.ewald(x, q, args.L)
    print(f'Ewald reference: {time.time()-t0:.1f} s')
    par = pk.choose(args.eps, args.P, rc, args.L)
    split = pk.C2Split(1, args.eps, rc)
    print(f"tolerance {args.eps:g}, order {args.P}: c_s = {par['c_s']:.4f}, grid {par['n']}^3, h/r_c = {par['h']:.4f} "
          f"(at most {par['h_max']:.4f}), s = {par['s']:.4f}")
    rms = np.sqrt((F_ref**2).sum(1).mean())
    for name, kernel in ((f'phi_{args.P}^*', par['kernel']), (f'B_{args.P}', pk.BSpline(args.P))):
        t0 = time.time()
        I = pk.influence.optimal(kernel, split, par['n'], args.L)
        E, F = pk.pme.coulomb(x, q, args.L, kernel, split, par['n'], I)
        error = np.sqrt(((F-F_ref)**2).sum(1).mean())/rms
        print(f'{name:>8}: relative RMS force error {error:.2e}, relative energy error {abs(E-E_ref)/abs(E_ref):.1e} '
              f'({time.time()-t0:.1f} s)')


if __name__ == '__main__':
    main()
