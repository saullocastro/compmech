Element Stiffness Matrix
========================

The stiffness matrices are integrated numerically where the number of
integration points is fixed for each element type by the parameter ``npts``.
The element has a number of dimensions defined by ``ndim``. Suppose we have
``nel`` 2-D elements (``ndim=2``) for which the stiffness matrix has to be
calculated, the following should be defined:

.. math::
    j = [{j11}_1, {j12}_1, {j21}_1, {j22}_1, ...,
         {j11}_{nel}, {j12}_{nel}, {j21}_{nel}, {j22}_{nel}]

