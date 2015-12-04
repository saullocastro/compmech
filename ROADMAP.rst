Version 0.4.1
-------------
- finish implementing compmech.aero.pistonstiffpanelbay, add pre-load for
  stiffeners
- implement compmech.aero.pistonstiff2Dpanelbay
- fix k0edges for pistonstiff2Dpanelbay
- kG0 for stiffener flange (create a fkG0sf) and base (use kG0y1y2)
- simplify the way we apply pre-stress to be taken as constant or variable
  along the linear buckling analyses.... we can do Nxx_cte and Nxx_var, for
  example

Version 0.4.2
-------------
- finish implementing compmech.plate for Monteiro
- fix mass matrix (kM) calculation for compmech.aero.pistonstiffpanel.fsdt.
  The rotation degrees of freedom should not use F = m*a, but another relation
  to consider the rotational energy properly

Version 0.5.0
-------------
- robustness of sparse-solver for the aero module, it is really a pain to use
  the dense solver because of the relative slowness...
- tune eigenvalue solver for freq in compmech.aero, similarly to what has been
  done for linear buckling analysis....
- skew panel with formulation for stringers not aligned with the panel
  longitudinal direction
- add Third-Order Shear Deformation Theory to deal with many papers using
  shear correction factors

Version 1.0.0 (long term)
--------------------------
- finite element solver
