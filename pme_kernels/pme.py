"""A minimal particle-mesh Ewald method for the Coulomb interaction with analytic differentiation (Section 2.5).

The local part of the split is summed directly over the pairs within the cutoff; the smooth part is computed by the
five steps: spreading with the SI kernel, FFT, multiplication by the influence function, inverse FFT, and interpolation
of the potential and of its gradient with the same kernel. Units: the Coulomb constant is one, the box is periodic and
cubic, and the system is neutral (conducting boundary conditions). Written for clarity, not speed.
"""
import numpy as np


def local(x, q, L, split):
    """Energy and forces of the local part, summed over the pairs within r_c (minimum image; requires L > 2 r_c)."""
    if L <= 2*split.rc:
        raise ValueError('the minimum-image sum requires L > 2 r_c')
    d = x[:, None, :]-x[None, :, :]
    d -= L*np.round(d/L)
    r = np.sqrt((d*d).sum(-1))
    np.fill_diagonal(r, np.inf)
    qq = q[:, None]*q[None, :]
    energy = 0.5*float((qq*split.local(r)).sum())
    # F_i = -sum_j q_i q_j L'(r_ij) (x_i - x_j)/r_ij
    coef = -qq*split.local(r, 1)/r
    forces = (coef[:, :, None]*d).sum(1)
    return energy, forces


def mesh(x, q, L, kernel, n, influence, split=None):
    """Energy and forces of the smooth part on the n^3 grid, with the influence function `influence` (numpy.fft
    order). The energy subtracts the interaction of each charge with its own smooth part, sum q_i^2 S(0)/2, if the
    split is given. Returns (energy, forces)."""
    h = L/n
    P = kernel.P
    s = x/h
    stencils = [kernel.stencil(s[:, a]) for a in range(3)]       # (j0, weights, derivatives) per coordinate
    idx = [(st[0][:, None]+np.arange(P)) % n for st in stencils]
    w = [st[1] for st in stencils]
    dw = [st[2] for st in stencils]
    # Spreading: b = sum_j q_j Phi(h l - x_j), Phi(x) = h^-3 prod_a phi(x_a/h).
    weights = q[:, None, None, None]*w[0][:, :, None, None]*w[1][:, None, :, None]*w[2][:, None, None, :]/h**3
    I0 = np.broadcast_to(idx[0][:, :, None, None], weights.shape)
    I1 = np.broadcast_to(idx[1][:, None, :, None], weights.shape)
    I2 = np.broadcast_to(idx[2][:, None, None, :], weights.shape)
    b = np.zeros((n, n, n))
    np.add.at(b, (I0.ravel(), I1.ravel(), I2.ravel()), weights.ravel())
    # FFT, influence function, inverse FFT: c_l = L^-3 sum_k I(xi_k) b_hat_k exp(-i xi_k . h l).
    c = np.fft.ifftn(influence*np.fft.fftn(b)).real
    cs = c[I0, I1, I2]                                            # (N, P, P, P)
    # Interpolation: u(x_i) = h^3 sum_l c_l Phi(x_i - h l); F_i = -q_i h^3 sum_l c_l grad Phi(x_i - h l).
    u = (cs*w[0][:, :, None, None]*w[1][:, None, :, None]*w[2][:, None, None, :]).sum((1, 2, 3))
    forces = np.stack([
        (cs*dw[0][:, :, None, None]*w[1][:, None, :, None]*w[2][:, None, None, :]).sum((1, 2, 3)),
        (cs*w[0][:, :, None, None]*dw[1][:, None, :, None]*w[2][:, None, None, :]).sum((1, 2, 3)),
        (cs*w[0][:, :, None, None]*w[1][:, None, :, None]*dw[2][:, None, None, :]).sum((1, 2, 3))], axis=1)
    forces *= -q[:, None]/h
    energy = 0.5*float(q@u)
    if split is not None:
        energy -= 0.5*float(q@q)*split.self_potential()
    return energy, forces


def coulomb(x, q, L, kernel, split, n, influence):
    """Energy and forces of N charges q at positions x (N x 3) in the periodic cubic box of side L: the local part of
    `split` summed directly, and the smooth part by the particle-mesh method with `kernel` and `influence`."""
    x, q = np.asarray(x, dtype=float), np.asarray(q, dtype=float)
    e1, f1 = local(x, q, L, split)
    e2, f2 = mesh(x, q, L, kernel, n, influence, split)
    return e1+e2, f1+f2
