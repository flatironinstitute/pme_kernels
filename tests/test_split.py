"""The C^2 splits of 1/r^nu: smoothness at the cutoff, normalization, derivatives, and the closed form of R_6."""
import numpy as np

from pme_kernels import C2Split

TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def test_local_part_is_C2_at_the_cutoff():
    for nu in range(1, 7):
        for eps in TOLERANCES:
            s = C2Split(nu, eps)
            assert abs(float(s.window(0.))-1) < 1e-12
            for d in range(3):
                assert abs(float(s.R(1., d))) < 1e-10, (nu, eps, d)


def test_R6_closed_form():
    s = C2Split(6, 1e-4)
    t = np.linspace(0.05, 1, 50)
    closed = s.window(t)-5/8*t*s.window(t, 1)+1/8*t**2*s.window(t, 2)
    assert np.abs(s.R(t)-closed).max() < 1e-12


def test_coulomb_splitting_kernel():
    s = C2Split(1, 1e-4, rc=2.)
    assert abs(float(s.chi_transform(0.))-1) < 1e-12
    r = np.array([0.3, 1.1, 1.9])
    assert np.allclose(s.local(r)+s.smooth(r), 1/r)
    assert float(s.local(2.)) == 0. and float(s.local(2.5)) == 0.
    # S(0) = 2 chi(0)/r_c
    assert abs(s.self_potential()-2*float(s.chi(0.))/2.) < 1e-12


def test_local_derivative():
    for nu in (1, 3, 6):
        s = C2Split(nu, 1e-6, rc=1.5)
        r, h = np.array([0.4, 0.9, 1.3, 1.45]), 1e-6
        fd = (s.local(r+h)-s.local(r-h))/(2*h)
        assert np.abs(s.local(r, 1)-fd).max() < 1e-6*np.abs(fd).max()
