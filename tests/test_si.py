"""The optimized SI kernels: normalization, charge and dipole conservation, smoothness, sign, tables, transforms."""
import numpy as np
from numpy.polynomial.legendre import leggauss

from pme_kernels import SIKernel, HermiteTable, available

CASES = [(4, 1e-4), (5, 1e-3), (5, 1e-4), (6, 1e-6), (7, 1e-4), (10, 1e-8)]
rng = np.random.default_rng(0)


def piecewise_integral(f, P, nodes=40):
    """int f over [-P/2, P/2] by Gauss quadrature on the half-unit intervals, where the kernel is smooth."""
    t, w = leggauss(nodes)
    edges = np.arange(-P, P+1)/2
    return sum(0.25*w@f(0.25*t+(lo+hi)/2) for lo, hi in zip(edges[:-1], edges[1:]))


def test_all_kernels_load():
    pairs = available()
    assert len(pairs) == 43
    for P, eps in pairs:
        k = SIKernel(P, eps)
        assert abs(2*k.a*k.coefficients[0]-1) < 1e-12        # int phi = int g = 1


def test_normalization():
    for P, eps in CASES:
        assert abs(piecewise_integral(SIKernel(P, eps), P)-1) < 1e-12


def test_charge_and_dipole_conservation():
    for P, eps in CASES:
        k = SIKernel(P, eps)
        x = rng.uniform(-5, 5, 2000)
        j0, w, _ = k.stencil(x)
        j = j0[:, None]+np.arange(P)
        assert np.abs(w.sum(1)-1).max() < 1e-13
        assert np.abs(((j-x[:, None])*w).sum(1)).max() < 1e-13


def test_twice_continuously_differentiable():
    for P, eps in CASES:
        k = SIKernel(P, eps)
        for d in range(3):
            assert abs(float(k(P/2, d))) < 1e-12 and abs(float(k(-P/2, d))) < 1e-12
            inner = np.array([1., 2., 3.])
            inner = inner[inner < P/2]
            # continuity across the integer and half-integer points where the third derivative may jump
            for x in np.concatenate([inner, inner-0.5]):
                assert abs(float(k(x-1e-9, d)-k(x+1e-9, d))) < 1e-6


def test_positive_and_decreasing():
    for P, eps in CASES:
        k = SIKernel(P, eps)
        t = np.linspace(0, P/2, 4001)[1:-1]
        assert (k(t) > 0).all() and (k(t, 1) <= 1e-12).all()


def test_transform_double_zeros():
    for P, eps in CASES:
        k = SIKernel(P, eps)
        m = 2*np.pi*np.arange(1, 4)
        h = 1e-4
        assert np.abs(k.transform(m)).max() < 1e-13
        assert np.abs((k.transform(m+h)-k.transform(m-h))/(2*h)).max() < 1e-8
        assert abs(float(k.transform(0.))-1) < 1e-12


def test_tables():
    for P, eps in CASES:
        k, T = SIKernel(P, eps), HermiteTable.load(P, eps)
        t = rng.uniform(-P/2, P/2, 5000)
        assert np.abs(T(t)-k(t)).max() < 1e-9
        assert np.abs(T(t, 1)-k(t, 1)).max() < 1e-6
