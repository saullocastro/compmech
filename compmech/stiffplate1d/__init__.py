r"""
==================================================
Stiffened Plate 1D (:mod:`compmech.stiffplate1d`)
==================================================

Many stiffeners are alowed but only one plate property. If different properties
are required over the circumference, module :mod:`compmech.stiffpanelbay1d` is
recommended, with a very high radius to simulate a plate.

.. currentmodule:: compmech.stiffplate1d

.. autoclass:: StiffPlate1D
    :members:

.. autoclass:: Stiffener
    :members:

"""
from stiffplate1d import load, StiffPlate1D, Stiffener
