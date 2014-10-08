r"""
==========================================================
Semi-analytical Models for Panels (:mod:`compmech.panels`)
==========================================================

.. currentmodule:: compmech.panels

The ``Panel`` class embodies all the methods and functions necessary to
perform:

* linear static analysis
* linear buckling analysis
* non-linear static analysis

The implementation focused on laminate composite shells with a constant
laminate constitutive relation. This means that the semi-analtical
models were derived doing an integration using a constant `[F]` for the whole
domain. Recalling that `[F]` correlates the strains and the distributed
stresses by the relation:

.. math::
    \{N\} = [F] \{\varepsilon\}


The ``Panel`` object
--------------------

.. autoclass:: Panel
   :members:

"""
