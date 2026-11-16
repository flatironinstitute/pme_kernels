"""A reference Ewald sum with the Gaussian split, accurate to about 1e-13 relative, for small test systems."""
import math

import numpy as np


def ewald(x, q, L, tolerance=1e-15, chunk=4096):
    """Energy and forces of neutral charges q at positions x in the periodic cubic box of side L (Coulomb constant
    one, conducting boundary conditions). The Gaussian width is chosen so that the real-space sum needs only the
    minimum image; the reciprocal sum keeps the wave vectors with exp(-k^2/(4 alpha^2)) above `tolerance`."""
    x, q = np.asarray(x, dtype=float), np.asarray(q, dtype=float)
    z = math.sqrt(-math.log(tolerance))
    alpha = 1.05*z/(L/2)
    # Real space, minimum image.
    d = x[:, None, :]-x[None, :, :]
    d -= L*np.round(d/L)
    r = np.sqrt((d*d).sum(-1))
    np.fill_diagonal(r, np.inf)
    qq = q[:, None]*q[None, :]
    erfc = np.vectorize(math.erfc)
    finite = np.isfinite(r)
    E_r = np.zeros_like(r)
    E_r[finite] = erfc(alpha*r[finite])/r[finite]
    energy = 0.5*float((qq*E_r).sum())
    radial = np.zeros_like(r)
    radial[finite] = (E_r[finite]+2*alpha/math.sqrt(math.pi)*np.exp(-(alpha*r[finite])**2))/r[finite]**2
    forces = ((qq*radial)[:, :, None]*d).sum(1)
    # Reciprocal space, over half of the wave vectors.
    kmax = 2*alpha*z
    nmax = int(math.ceil(kmax*L/(2*math.pi)))
    g = np.arange(-nmax, nmax+1)
    G = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    half = (G[:, 0] > 0) | ((G[:, 0] == 0) & (G[:, 1] > 0)) | ((G[:, 0] == 0) & (G[:, 1] == 0) & (G[:, 2] > 0))
    K = 2*math.pi*G[half]/L
    K2 = (K*K).sum(1)
    keep = K2 <= kmax**2
    K, K2 = K[keep], K2[keep]
    V = L**3
    for s in range(0, len(K), chunk):
        k, k2 = K[s:s+chunk], K2[s:s+chunk]
        f = np.exp(-k2/(4*alpha*alpha))/k2
        phase = np.exp(1j*(x@k.T))                     # (N, chunk)
        S = q@phase                                     # structure factors
        energy += float((4*math.pi/V)*(f*np.abs(S)**2).sum())
        forces += (8*math.pi/V)*q[:, None]*((np.imag(np.conj(S)[None, :]*phase)*f[None, :])@k)
    energy -= alpha/math.sqrt(math.pi)*float(q@q)
    return energy, forces
