Version 0.4.0
-------------

- simplify and unify setup.py for compmech.conecyl.clpt
- simplify and unify setup.py for compmech.conecyl.fsdt
- finish implementing compmech.aero.pisonplate
- finish implementing compmech.aero.pisonstiffpanel
- tune eigenvalue solver for freq in compmech.aero, similarly to what has been
  done for linear buckling analysis....
- fix mass matrix (kM) calculation for compmech.aero.pisonstiffpanel.fsdt. The
  rotation degrees of freedom should not use F = m*a, but another relation to
  consider the rotational energy properly


Version 0.5.0
-------------
- make an installable setup.py

Version 1.0.0 (long term)
--------------------------
- finite element solver
