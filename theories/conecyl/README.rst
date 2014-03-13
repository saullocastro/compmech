=======
conecyl
=======

.. sectnum::

.. contents:: Topics:

Current implementations
-----------------------

- static analysis (linear or non-linear)
- buckling analysis (linear or non-linear, using initial imperfections)
- load conditions: axial compression, torsion, pressure and any set of
  surface loads, concentraded or not

Introduction
------------

The equations are symbolically evaluated using a matrix notation, as explained
in FIXME ADD REFERENCES.
Initially the project used SymPy to derive the integrands and to perform the
analytical integrations. Since SymPy showed to be very slow for the integrands
herein evaluated, SymPy was used to derive the integrands and Mathematica used
to perform the integrations. This required a considerable effort on data
management and therefore it was decided to move the derivation of the
integrands inside Mathematica. Now in Mathematica it is derived all the
integrands, the analytical integration calculated and the symbolic
simplifications performed, which is also an important and computationally
expensive step.

SymPy is used to post process the results from Mathematica and print the
matrices in a ready-to-paste-into-Cython form.

In the following sections ``ESL theory`` can be either the
`Classical Laminated Plate Theory (CLPT)
<https://github.com/saullocastro/compmech/blob/master/theories/
conecyl/clpt/README.rst/>`_
or the `First-order Shear Deformation Theory (FSDT)
<https://github.com/saullocastro/compmech/blob/master/theories/
conecyl/fsdt/README.rst/>`_
, and
the ``equation type`` can be either the ``Donnell``'s or ``Sanders``'
equations.

Integration of the linear matrices
----------------------------------

The linear matrices K0 and KG0 are integrated analytically, and the
Mathematica routines necessary for this task are named as::

    {ESL theory}_{equation type}_linear.nb

Each integrated smallest repeatable sub-matrix::

    00, 01, 02, 11, 12 and 22
    
are stored in two text files inside the folder ``./outputs``, one with a
Mathematica format and another in Fortran format, identified with a prefix
``fortran_``. Since the integrated matrices are greatly simplified for the
cylinder case, there are files for both cones and cylinders.

The routines ``print_sparse.py`` are used to convert the Fortran format to a
ready-to-past-in-Cython format, separating in ``row``, ``col`` and ``value``,
which is convenient for the implementation using sparse matrices. The
generated files are stored in ``./sparse_printing``.

The integrated sparse matrices are pasted in the corresponding Cython codes,
currently::

    clpt_linear_donnell.pyx
    clpt_linear_sanders.pyx
    fsdt_linear_donnell.pyx

Integration of the non-linear matrices
--------------------------------------

The non-linear matrices ``K0L``, ``KG`` and ``KLL`` are not integrated
analytically and their integrands are built for the numerical integration
algorithms. Mathematica is used to build the integrands, in the routines
named as::

    {ESL theory}_{equation type}_non_linear_numerical.nb

The output is printed inside ``./numerical`` and the routine
``print_numerical.py`` is used to parse these results into something
that can be directly implemented into Cython, and the new outputs are
stored in ``./numerical_printing``.

For each sub-matrix::

    00, 01, 02, 10, 11, 12, 20, 21, 22

there will be a corresponding file::

    fortran_{cone or cylinder}_{matrix name}_{sub-matrix}.txt

The sub-matrix is identified by a left and a right digit (``0``, ``1`` or
``2``) that indicate the region of the stiffness matrix that this
sub-matrix refers to.

The sub-matrices' integrands are printed in a format ready to be programmed
using sparse matrices::

    c += 1
    rows[c] = 0
    cols[c] = col+4
    out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)

where ``alpha`` and ``beta`` are arguments passed by the numerical integration
algorithm (they give a pre-calculated weight for the point being integrated), 
``pij`` are the left ``qij`` the right components of the integrand.

The definitions of ``pij`` and ``qij`` are given in the corresponding files
with a ``p`` and a ``q`` character after the matrix name::

    fortran_{cone or cylinder}_{matrix name}p_{left digit}.txt
    fortran_{cone or cylinder}_{matrix name}q_{right digit}.txt
    
For example, to calculate the smallest repeatable sub-matrix ``12`` of
``KG``, first the file::

    fortran_cone_kG_12.txt

should be openened, and the definitions for ``pij`` are found in::

    fortran_cone_kGNLp_1.txt

and the definitions for ``qij`` are found in::

    fortran_cone_kGNLq_2.txt

These are implemented in::

    clpt_NL_donnell_numerical.pyx
    clpt_NL_sanders_numerical.pyx
    fsdt_NL_donnell_numerical.pyx
    
Strain equations
----------------

When the non-linear integrands are built another file containing the
strain equations is written::

    fortran_strainFunc.txt

This file is processed using the routine::

    print_strainFunc.py

And another file, the::

    output_strainFunc.txt

which is easy to implement in the Cython codes::

    clpt_commons.pyx
    fsdt_commons.pyx

depending on each case.

