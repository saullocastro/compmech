r"""
=========================================
Computational Mechanics (:mod:`compmech`)
=========================================

.. currentmodule:: compmech

Python-based toolkit for computational mechanics.

The compmech package contains semi-analytical routines for the analyses:

- linear buckling
- linear static
- non-linear static
- linear flutter
- linear vibration

for:

- unstiffened cylindrical and conical shells
- stiffened and unstiffened panels and plates

The implementation is based on Python and the low level routines are
programmed in C and Cython.

"""
from compmech.version import __version__
INT = int
