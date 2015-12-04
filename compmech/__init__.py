r"""
=========================================
Computational Mechanics (:mod:`compmech`)
=========================================

.. currentmodule:: compmech

Python-based toolkit for computational mechanics.

The compmech package contains semi-analytical routines for linear buckling,
linear static and non-linear static analysis of unstiffened cylindrical and
conical shells (:mod:`.compmech.conecyl`), and panels
(:mod:`.compmech.panels`). The implementation is based on Python and the low
level routines are programmed in Cython.

"""
__version__ = '0.4.1 dev'

import os
lib = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
if os.name == 'nt':
    os.environ['PATH'] = (lib + ';' + os.environ['PATH'])
else:
    os.environ['PATH'] = (lib + ':' + os.environ['PATH'])

