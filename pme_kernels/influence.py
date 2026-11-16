"""Influence functions for the Coulomb interaction (Sections 2.5 and 4.2 of the paper).

On the grid of n^3 points of the periodic cubic box of side L (spacing h = L/n), the grid frequencies are
xi_k = 2 pi k/L with k in {-n/2, ..., n/2 - 1}^3, and their aliases are xi_m = xi_k + 2 pi m/h. For analytic
differentiation (AD), the optimal influence function minimizes the mean-square pair force error:

    I_opt(xi_k) = B_k/A_k,  A_k = sum_m Phi_hat(xi_m)^2 sum_m |xi_m|^2 Phi_hat(xi_m)^2,
                            B_k = sum_m |xi_m|^2 Phi_hat(xi_m)^2 S_hat(xi_m),

with Phi_hat(xi) = prod_a phi_hat(h xi_a) and S_hat the transform of the smooth part. The arrays returned here are in
the order of numpy.fft (index k_a at position k_a mod n), with I = 0 at k = 0 and on the planes |k_a| = n/2.
"""
import numpy as np


def _frequencies(n):
    k = np.rint(np.fft.fftfreq(n, 1./n)).astype(int)     # 0, 1, ..., n/2 - 1, -n/2, ..., -1
    return k, np.abs(k) < n/2


def optimal(kernel, split, n, L, aliases=50, source_aliases=2):
    """The optimal influence function for AD of the SI kernel `kernel` (with .transform) and the Coulomb split `split`
    (with .chi_transform and .rc). The sums over the aliases are truncated at |m_a| <= aliases for A_k, which
    factorizes over the coordinates, and at |m_a| <= source_aliases for B_k, in which S_hat decays fast."""
    k, keep = _frequencies(n)
    h = L/n
    q = 2*np.pi*k/n                                    # h xi_a
    m = np.arange(-aliases, aliases+1)
    shifted = q[:, None]+2*np.pi*m[None, :]
    f2 = kernel.transform(shifted)**2
    F = f2.sum(1)                                      # sum_m phi_hat(q + 2 pi m)^2
    G = (shifted**2*f2).sum(1)/h**2                    # sum_m (xi + 2 pi m/h)^2 phi_hat(...)^2
    Fx, Fy, Fz = F[:, None, None], F[None, :, None], F[None, None, :]
    Gx, Gy, Gz = G[:, None, None], G[None, :, None], G[None, None, :]
    A = Fx*Fy*Fz*(Gx*Fy*Fz+Fx*Gy*Fz+Fx*Fy*Gz)
    # B_k = 4 pi sum_m Phi_hat(xi_m)^2 chi_hat(|xi_m| r_c), since |xi|^2 S_hat(xi) = 4 pi chi_hat(|xi| r_c).
    ms = np.arange(-source_aliases, source_aliases+1)
    p2 = kernel.transform(q[None, :]+2*np.pi*ms[:, None])**2          # (2M+1, n)
    km = k[None, :]+n*ms[:, None]                                       # integer frequency vectors per axis
    top = int(3*np.abs(km).max()**2)
    norms = np.arange(top+1)
    chi = split.chi_transform(2*np.pi*np.sqrt(norms)*split.rc/L)       # chi_hat at every integer |k + n m|^2
    B = np.zeros((n, n, n))
    for a in range(len(ms)):
        for b in range(len(ms)):
            for c in range(len(ms)):
                N2 = km[a][:, None, None]**2+km[b][None, :, None]**2+km[c][None, None, :]**2
                B += p2[a][:, None, None]*p2[b][None, :, None]*p2[c][None, None, :]*chi[N2]
    B *= 4*np.pi
    I = np.zeros((n, n, n))
    retained = keep[:, None, None] & keep[None, :, None] & keep[None, None, :]
    retained[0, 0, 0] = False
    I[retained] = B[retained]/A[retained]
    return I


def natural(kernel, split, n, L):
    """The natural influence function I_nat = S_hat(xi)/Phi_hat(xi)^2 (used by ESP), on the retained modes."""
    k, keep = _frequencies(n)
    h = L/n
    f = kernel.transform(2*np.pi*k/n)
    K2 = (2*np.pi/L)**2*(k[:, None, None]**2+k[None, :, None]**2+k[None, None, :]**2)
    retained = keep[:, None, None] & keep[None, :, None] & keep[None, None, :]
    retained[0, 0, 0] = False
    I = np.zeros((n, n, n))
    phi2 = (f[:, None, None]*f[None, :, None]*f[None, None, :])**2
    I[retained] = split.smooth_transform(np.sqrt(K2[retained]))/phi2[retained]
    return I
