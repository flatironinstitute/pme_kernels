"""The C^2 splits of 1/r^nu (Section 3 of the paper).

With t = r/r_c, the interaction r^-nu = S_nu(r) + L_nu(r) is split into a smooth part S_nu, whose Fourier series is
truncated on the grid, and a local part L_nu(r) = r^-nu R_nu(t) for r < r_c, zero beyond, which is summed directly over
the pairs within the cutoff. The window function w of the C^2 split makes L_nu twice continuously differentiable at
r_c: R_nu(1) = R_nu'(1) = R_nu''(1) = 0. For the Coulomb interaction (nu = 1), chi = w/int_{-1}^1 w is the splitting
kernel, and the smooth part has the Fourier transform S_hat(k) = 4 pi chi_hat(|k| r_c)/|k|^2.
"""
from functools import lru_cache

import numpy as np
from numpy.polynomial import legendre as leg

from ._data import load, tolerance_key


@lru_cache(maxsize=None)
def _gauss(n):
    return leg.leggauss(n)


class C2Split:
    """The C^2 split of 1/r^nu for the tolerance eps, with the cutoff rc.

    Attributes: nu, tolerance, c_s (splitting bandwidth), rc, truncation_error, and the Legendre coefficients of the
    window function w (w(0) = 1) and of R_nu, in t = r/r_c.
    """

    def __init__(self, nu, eps, rc=1.0):
        entries = [e for e in load('splits/c2_splits.json')['splits'] if e['nu'] == nu and tolerance_key(eps)(e)]
        if not entries:
            raise ValueError(f'no C^2 split of 1/r^{nu} for the tolerance {eps:g} in the data')
        e = entries[0]
        self.nu, self.tolerance, self.c_s, self.rc = nu, e['tolerance'], e['c_s'], float(rc)
        self.truncation_error = e['truncation_error']
        self.w = np.array(e['legendre_coefficients'], dtype=float)
        # R_1 = 1 - W(t)/W(1) with W(t) = int_0^t w; R_2 = w; R_{nu+2} = R_nu - (t/nu) R_nu'.
        W = leg.legint(self.w, lbnd=0)
        self.w_integral = float(leg.legval(1., W))            # int_0^1 w
        R = {1: leg.legsub([1.], W/self.w_integral), 2: self.w.copy()}
        for n in range(1, nu-1):
            R[n+2] = leg.legsub(R[n], leg.legmulx(leg.legder(R[n]))/n)
        self.R_coefficients = R[nu]

    def __repr__(self):
        return f'C2Split(nu={self.nu}, eps={self.tolerance:g}, rc={self.rc:g})'

    def window(self, t, derivative=0):
        """The window function w and its derivatives at t in [-1, 1]."""
        c = leg.legder(self.w, derivative) if derivative else self.w
        return leg.legval(np.asarray(t, dtype=float), c)

    def R(self, t, derivative=0):
        """The fraction R_nu(t) of t^-nu carried by the local part, 0 < t < 1, and its derivatives."""
        c = leg.legder(self.R_coefficients, derivative) if derivative else self.R_coefficients
        return leg.legval(np.asarray(t, dtype=float), c)

    def local(self, r, derivative=0):
        """The local part L_nu(r) (derivative=0) or its derivative dL_nu/dr (derivative=1); zero for r >= r_c."""
        r = np.asarray(r, dtype=float)
        nu, rc = self.nu, self.rc
        t = np.minimum(r/rc, 1.)
        inside = r < rc
        safe = np.where(inside, r, 1.)
        if derivative == 0:
            return np.where(inside, safe**-nu*self.R(t), 0.)
        if derivative == 1:
            return np.where(inside, -nu*safe**(-nu-1)*self.R(t)+safe**-nu*self.R(t, 1)/rc, 0.)
        raise ValueError('values and first derivatives only')

    def smooth(self, r):
        """The smooth part S_nu(r) = r^-nu - L_nu(r)."""
        r = np.asarray(r, dtype=float)
        return r**-self.nu-self.local(r)

    # The Coulomb split (nu = 1).

    def chi(self, t):
        """The splitting kernel chi = w/int_{-1}^1 w on [-1, 1] (Coulomb split only)."""
        self._coulomb()
        return np.where(np.abs(t) <= 1, self.window(np.clip(t, -1, 1))/(2*self.w_integral), 0.)

    def chi_transform(self, omega, chunk=4096):
        """chi_hat(omega) = int_{-1}^1 chi(t) cos(omega t) dt."""
        self._coulomb()
        omega = np.asarray(omega, dtype=float)
        t, w = _gauss(len(self.w)+400)
        cw = self.window(t)*w/(2*self.w_integral)
        flat = omega.ravel()
        out = np.concatenate([np.cos(np.multiply.outer(part, t))@cw
                              for part in np.array_split(flat, max(1, flat.size//chunk))]) if flat.size else flat
        return out.reshape(omega.shape)

    def smooth_transform(self, k):
        """S_hat(k) = 4 pi chi_hat(|k| r_c)/|k|^2 for |k| > 0."""
        k = np.asarray(k, dtype=float)
        return 4*np.pi*self.chi_transform(k*self.rc)/k**2

    def self_potential(self):
        """S(0) = 2 chi(0)/r_c: the smooth potential of a unit charge at its own position."""
        self._coulomb()
        return float(self.window(0.))/(self.w_integral*self.rc)

    def _coulomb(self):
        if self.nu != 1:
            raise ValueError('defined for the Coulomb split (nu = 1)')
