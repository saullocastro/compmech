r"""
=====================================
Integrate <:mod:`compmech.integrate`>
=====================================

.. currentmodule:: compmech.integrate

Brings together many convenient algorithms for numerical integration
that are not available in SciPy, such as the integration of vector valued
functions.

It would be nice to integrate this into SciPy one day.


Examples
--------

Below you can see how to use the available Cython functions.

Integration of scalar functions:

    .. literalinclude:: ../../../../compmech/integrate/tests/test_integrate.py

Integration of vector-valued functions:

    .. literalinclude:: ../../../../compmech/integrate/tests/test_integratev.py


"""
