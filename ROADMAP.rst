Version 0.5.
-------------
- kM for plate_clt_donnell_bardell and plate_clt_donnell_bardell_w

Version 0.5.1
-------------
- kG0 for stiffener flange (create a fkG0sf) and base (use kG0y1y2)
  (pre-load for 1D stiffeners)
- finish implementing pre-load for 1D stiffeners
- complete the kpanel modulus (quite easy now...)

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
