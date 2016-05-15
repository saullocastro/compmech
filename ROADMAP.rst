Version 0.6.0
-------------
- Numerical integration for kG0 of panels, possibility to include non-uniform
  stress states and pretty much any load then (this will pave the way for
  geometrically non-linear analyses)
- T-Stiffened panels using TStiff2D
- Tests for integrate module
- PanelAssembly: an extra assembly options for stiffpanelbay. Before each
  panel was modeled as the integration between y1 and y2 at a given bay
  region. Now there is the option to connect two panels using displacement and
  rotation compatibility such that in overall the bay can be modeled using
  less terms. This could be called a finite-semi-analytical approach, where
  the assembly is accomplished using proper connection matrices and a
  penalizing approach. Additional expected improvements are:
  -- improved numerical stability
  -- possibility to correctly represent displacement fields too complicated to
  be represented using a single domain (e.g. stiffened panels with debonding
  deffects)
  -- considerably faster integration of all structural matrices, preventing
  using large look-up tables associated with bardell_12 or bardell_c0c1
- T-Stiffened panels using PanelAssembly for linear buckling, frequency and
  flutter analyzes

Version 0.6.1
-------------
- using the Bardell's strategy, move many code from Mathematica to SymPy
- check why plate lb with reduced_dof=True is not the same as plate_w
- reconsider reading bardell_12 from the C code and reuse in FORTRAN

Version 0.7.0
-------------
- implement CI compilation and test for FORTRANs
- improve the numerical stability of FORTRAN's eigensolver (cpanelbay)
- skew panel with formulation for stringers not aligned with the panel
  longitudinal direction
- put matrices for cones and cylinders in the new format and unify their
  calculation as we did for compmech.panel
- robustness of sparse-solver for the panel module, it is really a pain to use
  the dense solver because of the relative slowness...
- tune eigenvalue solver for freq in compmech.panel, similarly to what has
  been done for linear buckling analysis....
- add Third-Order Shear Deformation Theory to deal with many papers using
  shear correction factors
- finish implementing compmech.plate for Monteiro

Version 1.0.0 (long term)
--------------------------
- finite element solver
