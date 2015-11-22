.. _theory_func_bardell:

Bardell's Hierarchical Functions
================================

Bardell (1991) [bardell1991]_ proposed a very convinient approximation
functions based on hierarchical polynomials. The convenience comes from the
fast convergence and from the easiness to simulate practically any type of
boundary conditions.

The boundary condition is controlled by the first 4 terms of the approximation
function, herein defined as:

- ``t1``: the translation at extremity 1 (`\xi = -1`)
- ``r1``: the rotation at extremity 1
- ``t2``: the translation at extremity 2 (`\xi = +1`)
- ``r2``: the rotation at extremity 2

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
