r"""
==============================================================================
Semi-analytical models for Stiffeners (:mod:`compmech.stiffener`)
==============================================================================

.. currentmodule:: compmech.stiffener

This module has all stiffener's formulations used along other classes.

.. autoclass:: BladeStiff1D
    :members:

.. autoclass:: BladeStiff2D
    :members:

.. autoclass:: TStiff2D
    :members:

"""
from . bladestiff1d import BladeStiff1D
from . bladestiff2d import BladeStiff2D
from . tstiff2d import TStiff2D
