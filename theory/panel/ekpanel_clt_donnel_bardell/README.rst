Elliptical-Conical Panel
------------------------

The elliptical-conical panel has essentially the same kinematic relations as
the conical panel, with the difference that the radius now depends not only on
`x` but also on `\theta`.

Castro (see thesis) demonstrated that for the conical panels it is already not
possible to perform the analytical integration of the structural matrices,
suggesting the use of many cylindrical sub-domains for the efficient analytical
integration. Therefore, the additional radius dependecy on `\theta` would
require additional effort if one decided to integrate the structural matrices
analytically.

Due to the additional difficulty to perform the analytical integration and due
to the known limitation of analytical integration schemes for non-linear
analyses and when the structure does not have homogeneous properties and stress
states, the implemented routines only have numerical integration schemes for
the elliptical panels.
