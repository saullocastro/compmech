r"""
=======================================================================
Semi-analytical Models for Cones and Cylinder (:mod:`compmech.conecyl`)
=======================================================================

.. currentmodule:: compmech.conecyl

The ``ConeCyl`` class embodies all the methods and functions necessary to
perform:

* linear static analysis
* linear buckling analysis
* non-linear static analysis

in conical and cylindrical structures.

The implementation focused on laminate composite shells with a constant
laminate constitutive relation. This means that the semi-analtical
models were derived doing an integration using a constant `[F]` for the whole
domain. Recalling that `[F]` correlates the strains and the distributed
stresses by the relation:

.. math::
    \{N\} = [F] \{\varepsilon\}


The ``ConeCyl`` object
----------------------

.. autoclass:: ConeCyl
   :members:

Non-linear analysis (:mod:`compmech.conecyl.non_linear`)
--------------------------------------------------------

.. automodule:: compmech.conecyl.non_linear
   :members:


Cone / Cylinder Database (:mod:`compmech.conecyl.conecylDB`)
------------------------------------------------------------

.. automodule:: compmech.conecyl.conecylDB
   :members:

Models' Database (:mod:`compmech.conecyl.modelDB`)
--------------------------------------------------

.. automodule:: compmech.conecyl.modelDB
   :members:

Imperfections (:mod:`compmech.conecyl.imperfections`)
-----------------------------------------------------

.. automodule:: compmech.conecyl.imperfections
   :members:

"""
from .conecyl import load, ConeCyl
