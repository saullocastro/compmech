Version 0.5.0
-------------
- implement compmech.aero.pistonstiff2Dpanelbay
- simplify the way we apply pre-stress to be taken as constant or variable
  along the linear buckling analyses.... we can do Nxx_cte and Nxx_var, for
  example
- restructuration of modules
-- create a module stiffpanel1d
-- create a module stiffplate1d
-- create a module stiffpanelbay1d
-- create a module stiffpanelbay2d
-- centralize panel calculation in module panel, include functions k0 and
   k0y1y2, kG0, kG0y1y2 and so forth
-- centralize plate calculation in module panel, same as for panels
- kG0 for stiffener flange (create a fkG0sf) and base (use kG0y1y2)
  (pre-load for 1D stiffeners)




Version 0.5.1
-------------
- implement k0edges for pistonstiff2Dpanelbay (not priority)
- finish implementing pre-load for 1D stiffeners

Version 0.6.0
-------------
Compromise study considering the two following items
- centralize the core modules for clpt and fsdt
- remove plate modules and keep only stiffpanel, with possibility to call:
-- plate (r is None and alpha is None)
-- cylindrical panel (r is not None and alpha is None)
-- conical panel (r is not None and alpha is not None)

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
