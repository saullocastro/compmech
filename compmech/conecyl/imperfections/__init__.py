r"""

Convenient routines to transform discrete measured data
into continuous functions applicable in semi-analytical analyses.

.. currentmodule:: compmech.conecyl.imperfections

The implemented geometric imperfection of this module is the one
representing the normal displacement of the mid-surface and will be
called *Measured Geometric Imperfection (MGI)*.

The non-linear analysis using a MGI require the calculation
of the initial imperfection field, called `w_0`, and the corresponding
partial derivatives `{w_0}_{,x}` and `{w_0}_{,\theta}`.

The function ``calc_c0`` described below is implemented to find
the best-fit for a given imperfection data. Three different approximation
functions can be selected using the ``funcnum`` parameter.

The imperfection file should be in the form::

    theta1  height1  imp1
    theta2  height2  imp2
    ...
    thetan  heightn  impn

where ``height`` is measured from the bottom to the top of the cylinder
or cone, parallel to the axial axis, and ``theta`` is the circumferential
coordinate `\theta` measured as shown
:ref:`in the semi-analytical model <conecyl_figure>`.

When implementing a non-linear analysis algorithm, see for example
`fsdt_donnell_bc1_nonlinear.pyx
<https://github.com/compmech/compmech/blob/master/compmech/conecyl/fsdt/fsdt_donnell_bc1_nonlinear.pyx>`_
, the functions to calculate the partial derivatives of the geometric
imperfection are accessible using Cython ::

    from compmech.conecyl.imperfection.mgi cimport cfw0x, cfw0t

The ``cfw0x`` and ``cfw0t`` function headers are::

    cdef void cfw0x(double x, double t, double *c0, double L,
                    int m, int n, double *w0xref, int funcnum) nogil

    cdef void cfw0t(double x, double t, double *c0, double L,
                    int m, int n, double *w0tref, int funcnum) nogil

where ``c0`` is the array containing the coefficients of the approximation
function, ``L`` is the meridional length of the cone or cylinder
(:ref:`as shown here <conecyl_figure>`), ``x`` and ``t`` the `x` and
`\theta` coordinates, ``m`` and ``n`` the number of terms of the
approximation series as described above, and finally,
``w0xref`` and ``w0tref`` are pointers to ``double`` variables.

.. automodule:: compmech.conecyl.imperfections.imperfections
   :members:

"""
from __future__ import absolute_import

from . imperfections import *
