Version 0.7.1
-------------
- non-linear matrices for PanelAssembly
- elliptical panel, only with numerical integration
- parallel processing to compute kT for PanelAssembly
- strain / stress calculation for panels

Version 0.8.0
-------------
- allow a constant stress state using static results for linear buckling
- mass matrices also valid for variable property along domain (perhaps with
  numerical integration)
- create a boundary class with w1tx etc?
- pass the laminate class instead of plyt etc, make read_stack decided whether
  to use plyt/plyts, laminaprop/laminaprops and so forth
- enhance analysis.freq to support damping
  -- then remove lb, freq and static from Panel
- using the Bardell's strategy, perhaps move code from Mathematica to
  SymPy
- check why plate lb with reduced_dof=True is not the same as plate_w
- reconsider reading bardell_12 from the C code and reuse in FORTRAN

Version 0.9.0
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
