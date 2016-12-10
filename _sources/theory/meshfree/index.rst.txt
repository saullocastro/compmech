.. _theory_meshfree:

Meshfree Methods
================

Brief Introduction
------------------

Excellent overviews of meshfree methods are found in [belytschko1996]_,
[fries2004]_ and [liu2003]_.

As reviewed by Wallstedt [wallstedt2009]_, one of the methods developed to
circumvent the limitations of the FEM when very large displacements are
required (e.g. extrusion or intrusion simulations where remeshing is
mandatory), is the Smooth Particle Hydrodynamics (SPH) Method developed by
Lucy [lucy1977]_ and Monaghan [monaghan1998]_ and coworkers to model
astrophysics problems and later extended to solid mechanics by Libersky,
Petschek and Randles (see [libersky1990]_ and [randles1996]_). These first
models suffered from instability and lack of convergence, which as corrected
by Swegle [swegle1995]_, Johnson and Beissel [johnson1996]_, Dilts
[dilts2000]_ and others. Despite correcting the stability and increasing the
accuracy of the SPH, these developments introduced some inconvenience, such as
keeping track of additional stress points.

As also reviewed by Wallstedt [wallstedt2009]_, another branch of development
based on the Diffuse Element Method (DEM) of Nayroles and coworkers
[nayroles1992]_. Belytschko, Lu and Gu (see [belytschko1994]_, [lu1994]_ and
[belytschko1994_2]_) reformed the original proposal in terms of Moving Least
Squares (MLS) [#f1]_. This introduced the Element Free Galerkin (EFG) method
which achieved substantial improvements in accuracy.

As stated by [wallstedt2009]_ the use of MSL-based methods is:

   `\checkmark` useful for new classes of problems for which FEM is ill-suited

   `\times` limited regarding the application of essential boundary conditions

   `\times` substantially larger computational cost relative to FEM, mainly
   due to the nearest-neighbor searches

Several MLS-based methods appeared soon, including:

- Reproducing Kernel Particle Method (RKPM) ([liu1993]_, [liu1995]_,
  [liu1995_2]_, [jun1998]_)
- Meshless Local Petrov Galerkin (MLPG) ([atluri1998]_])
- Particle-in-cell (PIC) method ([harlow1963]_)
- Fluid-Implicit Particle (FLIP) ([brackbill1986]_)
- Material Point Method (MPM) ([sulsky1994]_, [sulsky1995]_, [sulsky1996]_)

Other meshfree methods:

- Diffuse Element Method (DEM) ([nayroles1992]_)
- Element Free Galerkin (EFG) ([belytschko1994]_, [lu1994]_ and
  [belytschko1994_2]_). While the DEM suffers from a number of problems due to
  some simplifications, the EFG may be considered a fixed version, being a
  very popular meshfree method [fries2004]_
- Generalized Interpolation Material Point (GIMP) ([bardenhagen2004]_)
- Weighted Least Squares (WLS) (in the GIMP framework) ([wallstedt2009]_).
  Does not require the nearest-neighbor searches and integrates over a
  background as the EFG

.. rubric:: Footnotes

.. [#f1] The moving least squares is a method of reconstructing continuous
   functions from a set of unorganized point samples, as exampled
   `in Wikipedia... <http://en.wikipedia.org/wiki/Moving_least_squares>`_

Gereral Characteristics
-----------------------

Based on the excellent review of [fries2004]_, some interesting aspects of
meshfree methods will be listed in this section. The author will try to
collect the aspects that seem relevant to judge whether further investigation
on these methods will be carried out in order to seek a future implementation
in :mod:`compmech`.

- Computational effort: usually higher than mesh-based counterpart models:

  - meshfree shape functions are more complex than polynomial-like functions
    used in mesh-based methods, requiring many more integration points
    (collocation meshfree methods don't require integration, but this evokes
    accuracy and stability problems)

  - at each integration point the following steps are usually required:
    neighbor search, solution of small systems of equations and small
    matrix-matrix and matrix-vector operations in order to determine the
    derivatives

    .. note::
        this seems to be a perfect task for CUDA implementations...

  - the resulting system of equations has in general a larger band-width for
    meshfree methods than for mesh-based (like finite elements)
    [belytschko1996]_

- Hard to impose essential boundary conditions since you don't have the
  Kronecker delta property in meshfree methods, which will guarantee that a
  given approximation function has a determined value at a given location

