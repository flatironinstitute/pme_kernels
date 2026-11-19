"""Kernels for particle mesh Ewald methods: C^2 splits of 1/r^nu and optimized SI kernels.

From: S. Jiang, J. Liang, and L. Lu, Kernel Design in Particle Mesh Ewald Methods: Splitting, Interpolation, and
Upsampling, arXiv:2611.23737 (2026).
"""
from .si import SIKernel, HermiteTable, BSpline, available
from .split import C2Split
from .params import choose, bandwidth, fft_size
from . import influence, pme, ewald

__version__ = '0.1.0'
__all__ = ['SIKernel', 'HermiteTable', 'BSpline', 'available', 'C2Split', 'choose', 'bandwidth', 'fft_size',
           'influence', 'pme', 'ewald']
