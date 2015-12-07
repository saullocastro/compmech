r"""
================================================
Stiffened Panels (:mod:`compmech.stiffpanel1d`)
================================================

.. currentmodule:: compmech.stiffpanel1d

Stiffeners modeled with 1D formulation.

Semi-analytical model for stiffened cylindrical panels. Many stiffeners are
allowed but only one panel property. If different properties are required
(usually for panels between stiffeners), module :mod:`compmech.stiffpanelbay`
is recommended.

.. autoclass:: StiffPanel1D
    :members:

.. autoclass:: Stiffener
    :members:

"""
from stiffpanel1d import load, StiffPanel1D, Stiffener
