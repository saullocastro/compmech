Version 0.4.0
-------------

- structural damping using alpha*M + beta*K with alpha and beta determined
  based on pre-selected modes, which are determined from a non-damped flutter
  analysis
- simplify and unify setup.py for compmech.conecyl.clpt
- simplify and unify setup.py for compmech.conecyl.fsdt
- finish implementing compmech.aero.pisonplate
- finish implementing compmech.aero.pisonstiffpanel
- tune eigenvalue solver for freq in compmech.aero, similarly to what has been
  done for linear buckling analysis....
- fix mass matrix (kM) calculation for compmech.aero.pisonstiffpanel.fsdt. The
  rotation degrees of freedom should not use F = m*a, but another relation to
  consider the rotational energy properly
- skew panel with formulation for stringers not aligned with the panel
  longitudinal direction
- robustness of sparse-solver for the aero module, it is really a pain to use
  the dense solver because of the relative slowness...

Version 0.5.0
-------------
- make an installable setup.py
- add Third-Order Shear Deformation Theory to deal with many papers using
  shear correction factors

Version 1.0.0 (long term)
--------------------------
- finite element solver
