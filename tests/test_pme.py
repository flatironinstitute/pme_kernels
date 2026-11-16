"""The particle-mesh method with the new kernels against an accurate Ewald sum."""
import numpy as np

import pme_kernels as pk


def test_forces_meet_the_tolerance():
    rng = np.random.default_rng(1)
    N, L, rc, eps, P = 500, 10.0, 1.0, 1e-4, 5
    x = rng.uniform(0, L, (N, 3))
    q = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    E_ref, F_ref = pk.ewald.ewald(x, q, L)
    par = pk.choose(eps, P, rc, L)
    split = pk.C2Split(1, eps, rc)
    I = pk.influence.optimal(par['kernel'], split, par['n'], L)
    E, F = pk.pme.coulomb(x, q, L, par['kernel'], split, par['n'], I)
    rms = np.sqrt((F_ref**2).sum(1).mean())
    error = np.sqrt(((F-F_ref)**2).sum(1).mean())/rms
    assert error < eps, error
    assert abs(E-E_ref) < eps*abs(E_ref)


def test_new_kernel_beats_the_bspline_on_its_grid():
    rng = np.random.default_rng(2)
    N, L, rc, eps, P = 500, 10.0, 1.0, 1e-4, 5
    x = rng.uniform(0, L, (N, 3))
    q = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    _, F_ref = pk.ewald.ewald(x, q, L)
    par = pk.choose(eps, P, rc, L)
    split = pk.C2Split(1, eps, rc)
    errors = {}
    for name, kernel in (('new', par['kernel']), ('bspline', pk.BSpline(P))):
        I = pk.influence.optimal(kernel, split, par['n'], L)
        _, F = pk.pme.coulomb(x, q, L, kernel, split, par['n'], I)
        errors[name] = np.sqrt(((F-F_ref)**2).sum(1).mean())
    assert errors['bspline'] > 2*errors['new'], errors
