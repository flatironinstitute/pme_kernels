"""Choosing the parameters (Section 4.7 of the paper).

A requested tolerance eps, a cutoff r_c, and an order P determine the method: eps fixes the splitting bandwidth c_s
(ESP's rule chi(1) = eps, raised at 1e-2 and 1e-3 so that the truncation error of the C^2 split meets eps), eps
and P fix the optimized kernel phi_P^* and its coarsest grid ratio s*, hence the grid
spacing h = pi r_c/(c_s s*); a box of side L takes the first grid with at least L/h points in each dimension that the
FFT handles efficiently.
"""
import math

from .si import SIKernel


def bandwidth(eps):
    """An approximation of ESP's splitting bandwidth, c_s = ln(1/eps) + 0.86 ln ln(1/eps) + 0.91, which reproduces
    ESP's rule chi(1) = eps to within 0.01 for eps from 1e-2 to 1e-8. The C^2 splits use this bandwidth from 1e-4 on
    and a slightly larger one at 1e-2 and 1e-3, so that their truncation error meets the tolerance; the kernels and
    splits of the data carry the exact values."""
    x = math.log(1/eps)
    return x+0.86*math.log(x)+0.91


def fft_size(m, primes=(2, 3, 5, 7)):
    """The smallest even integer >= m whose prime factors are in `primes`."""
    n = max(2, int(math.ceil(m)))
    n += n % 2
    while True:
        k = n
        for p in primes:
            while k % p == 0:
                k //= p
        if k == 1:
            return n
        n += 2


def choose(eps, P, rc, L):
    """The parameters of the method for the tolerance eps, the order P, the cutoff rc, and the box side L: a dict with
    the kernel, c_s, the largest grid spacing h_max = pi rc/(c_s s*), and the grid size n (n points per dimension,
    spacing L/n <= h_max)."""
    kernel = SIKernel(P, eps)
    h_max = math.pi*rc/(kernel.c_s*kernel.s_star)
    n = fft_size(L/h_max*(1-1e-12))
    h = L/n
    return dict(kernel=kernel, c_s=kernel.c_s, h_max=h_max, n=n, h=h, s=math.pi*rc/(kernel.c_s*h))
