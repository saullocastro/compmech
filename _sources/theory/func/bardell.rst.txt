.. _theory_func_bardell:

Bardell's Hierarchical Functions
================================

Introduction
------------

Bardell (1991) [bardell1991]_ proposed a very convinient set of approximation
functions based on hierarchical polynomials. The convenience comes from the
fast convergence and from the easiness to simulate practically any type of
boundary conditions.

The boundary condition is controlled by the first 4 terms of the approximation
function, herein defined as:

- ``t1``: the translation at extremity 1 (`\xi = -1`)
- ``r1``: the rotation at extremity 1
- ``t2``: the translation at extremity 2 (`\xi = +1`)
- ``r2``: the rotation at extremity 2

Generating Bardell's functions
------------------------------

The following code can be used to generate the Bardell functions for a given
number of terms ``nmax``. The substitution ``replace('**', '^')`` aims to
create an input to Mathematica.

.. literalinclude:: ../../../../theory/func/bardell/bardell.py
    :caption:

The output of the code above is:

.. literalinclude:: ../../../../theory/func/bardell/bardell.txt
    :caption:

In order to calculate the displacement, strain of stress fields using Cython,
the above output is not adequate due to very long integer numbers that will
cause precision overflows. The code below should be used to create an input to
Cython:

.. literalinclude:: ../../../../theory/func/bardell/bardell_floating_point.py
    :caption:

Generating the following output:

.. literalinclude:: ../../../../theory/func/bardell/bardell_floating_point.txt
    :caption:

Implemented modules
-------------------

Three modules were created and they are called in ``compmech`` shared library.
These libraries are available in ``compmech/lib`` after installation.

Three modules are currently available:

- bardell: for the integration along the full domain

.. math::
            \int_{\xi=-1}^{\xi=+1} {f(\xi) g(\xi)}

- bardell_12: for the integration in any interval

.. math::
            \int_{\xi=\xi_1}^{\xi=\xi_2} {f(\xi) g(\xi)}

- bardell_c0c1: for the integration of two linearly dependent field variables
  `\xi` and `\xi'`

.. math::
            \int_{\xi=-1}^{\xi=+1} {f(\xi) g(\xi')}

such that:

.. math::
            \xi' = c_0 + c_1 \xi

with `c_0` and `c_1` being constant values.
