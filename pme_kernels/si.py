"""The optimized SI kernels phi_P^* (Section 4.5 of the paper).

The kernel of order P is phi = B_2 * g, with the linear B-spline B_2(t) = max(1 - |t|, 0) and an even, continuous
function g supported on [-a, a], a = (P - 2)/2, stored by its Legendre coefficients in t/a. Hence phi is even, supported
on [-P/2, P/2], twice continuously differentiable, normalized by int phi = 1, and its integer translates conserve charge
and dipole: sum_j phi(x - j) = 1 and sum_j (j - x) phi(x - j) = 0 for every x.

Coordinates are in units of the grid spacing h: a particle at y spreads to the grid point h j the weight phi(y/h - j).
"""
from functools import lru_cache

import numpy as np
from numpy.polynomial import legendre as leg

from ._data import load, tolerance_key


@lru_cache(maxsize=None)
def _gauss(n):
    return leg.leggauss(n)


class SIKernel:
    """The optimized SI kernel of order P for the tolerance eps.

    Attributes: P, tolerance, c_s (splitting bandwidth), s_star (coarsest grid ratio), h_over_rc (largest grid
    spacing on which the kernel meets the tolerance, h/r_c = pi/(c_s s*)), error (relative AD aliasing error of the
    band on that grid, in the limit of large boxes), coefficients (Legendre coefficients of g in t/a).
    """

    def __init__(self, P, eps):
        entries = [e for e in load('si/si_kernels.json')['kernels'] if e['P'] == P and tolerance_key(eps)(e)]
        if not entries:
            raise ValueError(f'no optimized kernel of order {P} for the tolerance {eps:g} in the data')
        e = entries[0]
        self.P, self.tolerance = P, e['tolerance']
        self.c_s, self.s_star, self.h_over_rc, self.error = e['c_s'], e['s_star'], e['h_over_rc'], e['error']
        self.coefficients = np.array(e['legendre_coefficients'], dtype=float)
        self.a = (P-2)/2
        # Antiderivatives of g in the variable y = t/a from y = -1.
        self._I1 = leg.legint(self.coefficients, lbnd=-1)
        self._I2 = leg.legint(self._I1, lbnd=-1)
        self._dg = leg.legder(self.coefficients)

    def __repr__(self):
        return f'SIKernel(P={self.P}, eps={self.tolerance:g})'

    def _antiderivative(self, x, order):
        """g (order 2), its first antiderivative from -a (order 1), its second (order 0), or g' (order 3)."""
        a = self.a
        y = np.clip(x/a, -1, 1)
        inside, right = np.abs(x) <= a, x > a
        if order == 0:
            return np.where(inside | right, a*a*leg.legval(y, self._I2), 0.)+a*leg.legval(1., self._I1)*np.maximum(x-a, 0)
        if order == 1:
            return np.where(inside | right, a*leg.legval(y, self._I1), 0.)
        if order == 2:
            return np.where(inside, leg.legval(y, self.coefficients), 0.)
        if order == 3:
            return np.where(np.abs(x) < a, leg.legval(y, self._dg)/a, 0.)
        raise ValueError('derivatives of order 0 to 3 only')

    def __call__(self, t, derivative=0):
        """phi^(derivative)(t), from phi = G(t + 1) - 2 G(t) + G(t - 1) with G the second antiderivative of g."""
        t = np.asarray(t, dtype=float)
        G = self._antiderivative
        return G(t+1, derivative)-2*G(t, derivative)+G(t-1, derivative)

    def transform(self, q, chunk=4096):
        """The Fourier transform phi_hat(q) = int phi(t) exp(i q t) dt = (sin(q/2)/(q/2))^2 g_hat(q)."""
        q = np.asarray(q, dtype=float)
        y, w = _gauss(len(self.coefficients)+200)
        gw = leg.legval(y, self.coefficients)*w
        flat = q.ravel()
        ghat = np.concatenate([self.a*np.cos(np.multiply.outer(part*self.a, y))@gw
                               for part in np.array_split(flat, max(1, flat.size//chunk))]) if flat.size else flat
        return (np.sinc(flat/(2*np.pi))**2*ghat).reshape(q.shape)

    def stencil(self, x):
        """For coordinates x in grid units: the first grid index j0 of the P coupled grid points j0, ..., j0 + P - 1,
        the weights phi(x - j), and their derivatives with respect to x, phi'(x - j)."""
        x = np.asarray(x, dtype=float)
        j0 = np.floor(x-self.P/2).astype(int)+1
        offset = x[..., None]-(j0[..., None]+np.arange(self.P))
        return j0, self(offset), self(offset, 1)

    def table(self, parts=128):
        """phi, phi', phi'' at t = k/parts, k = 0, ..., parts P/2 (the kernel is even): the table of the paper."""
        t = np.arange(parts*self.P//2+1)/parts
        return t, self(t), self(t, 1), self(t, 2)


class HermiteTable:
    """A kernel evaluated from its table: on each of the parts of a grid interval, the polynomial of degree five that
    matches phi, phi' and phi'' at both ends. The table is twice continuously differentiable."""

    def __init__(self, t, f, df, d2f):
        self.t, self.f, self.df, self.d2f = (np.asarray(v, dtype=float) for v in (t, f, df, d2f))
        self.step = self.t[1]-self.t[0]
        self.end = self.t[-1]

    @classmethod
    def load(cls, P, eps, parts=128):
        """The table of data/tables for the kernel of order P and the tolerance eps."""
        from ._data import data_dir
        path = data_dir()/'tables'/f'phi_P{P}_eps{eps:.0e}.txt'
        t, f, df, d2f = np.loadtxt(path, unpack=True)
        return cls(t, f, df, d2f)

    def __call__(self, t, derivative=0):
        t = np.asarray(t, dtype=float)
        s, x = np.sign(t), np.abs(t)
        k = np.clip(np.floor(x/self.step).astype(int), 0, len(self.t)-2)
        u = x/self.step-k
        h = self.step
        f0, f1 = self.f[k], self.f[k+1]
        d0, d1 = h*self.df[k], h*self.df[k+1]
        e0, e1 = h*h*self.d2f[k], h*h*self.d2f[k+1]
        u2, u3, u4, u5 = u*u, u**3, u**4, u**5
        if derivative == 0:
            p = (f0*(1-10*u3+15*u4-6*u5)+d0*(u-6*u3+8*u4-3*u5)+e0*(u2-3*u3+3*u4-u5)/2
                 + f1*(10*u3-15*u4+6*u5)+d1*(-4*u3+7*u4-3*u5)+e1*(u3-2*u4+u5)/2)
            return np.where(x <= self.end, p, 0.)
        if derivative == 1:
            p = (f0*(-30*u2+60*u3-30*u4)+d0*(1-18*u2+32*u3-15*u4)+e0*(2*u-9*u2+12*u3-5*u4)/2
                 + f1*(30*u2-60*u3+30*u4)+d1*(-12*u2+28*u3-15*u4)+e1*(3*u2-8*u3+5*u4)/2)/h
            return np.where(x <= self.end, s*p, 0.)
        raise ValueError('values and first derivatives only')


def available():
    """The (P, tolerance) pairs of the optimized kernels in the data."""
    return sorted((e['P'], e['tolerance']) for e in load('si/si_kernels.json')['kernels'])


class BSpline:
    """The centered cardinal B-spline B_P of order P, for comparison, with the same interface as SIKernel."""

    def __init__(self, P):
        from math import comb, factorial
        self.P = P
        self._comb, self._fact = comb, factorial

    def __call__(self, t, derivative=0):
        t = np.asarray(t, dtype=float)
        P, d = self.P, derivative
        y = t+P/2
        # B_P(t) = sum_k (-1)^k C(P, k) (y - k)_+^(P-1)/(P-1)!, and its derivatives.
        out = np.zeros_like(y)
        for k in range(P+1):
            power = P-1-d
            if power < 0:
                continue
            coef = (-1)**k*self._comb(P, k)*self._fact(P-1)/self._fact(power)/self._fact(P-1)
            out = out+coef*np.where(y > k, (y-k)**power, 0.)
        return out*(np.abs(t) < P/2)

    def transform(self, q):
        return np.sinc(np.asarray(q, dtype=float)/(2*np.pi))**self.P

    def stencil(self, x):
        x = np.asarray(x, dtype=float)
        j0 = np.floor(x-self.P/2).astype(int)+1
        offset = x[..., None]-(j0[..., None]+np.arange(self.P))
        return j0, self(offset), self(offset, 1)
