Version 0.4.1
-------------
- implement compmech.aero.pistonstiff2Dpanelbay
- simplify the way we apply pre-stress to be taken as constant or variable
  along the linear buckling analyses.... we can do Nxx_cte and Nxx_var, for
  example

Version 0.4.2
-------------
- finish implementing compmech.aero.pistonstiffpanelbay, add pre-load for
  stiffeners
- kG0 for stiffener flange (create a fkG0sf) and base (use kG0y1y2)
  (pre-load for 1D stiffeners)
- implement k0edges for pistonstiff2Dpanelbay (not priority)

Version 0.5.0
-------------
- unify the calculation of stiffness matrices for panels (e.g. fk0y1y2)
- unify the calculation of stiffness matrices for plates
- restructuration of modules based on the unification aforementioned
-- currently the aero module does linear buckling analysis
-- create a module panels with stiffeners being optional
-- create a module plates with stiffeners being optional
-- create a module unstiffened panels (faster than stiffened)
-- create a module unstiffened plates (faster than stiffened)
-- the aeroelasticity functionality will be integrated in the freq()
   methods
- finish implementing compmech.plate for Monteiro
- fix mass matrix (kM) calculation for compmech.aero.pistonstiffpanel.fsdt.
  The rotation degrees of freedom should not use F = m*a, but another relation
  to consider the rotational energy properly
- put matrices for cones and cylinders in the new format and unify their
  calculation as we did for plates and panels
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
