.. _theory_integrate_legendre_gauss:

Legendre-Gauss Quadrature Points
================================

The related theory is well presented by [weisstein]_.

Code to generate points
-----------------------

The following Mathematica script, taken from [pomax]_ was used to generate the Legendre-Gauss
quadrature points:

.. code-block:: mathematica


    symboliclegendre[n_,x_]:=Solve[LegendreP[n,x]==0];
    legendreprime[n_,a_]:=D[LegendreP[n,x],x]/.x->a;
    weights[n_,x_]:=2/((1-x^2) legendreprime[n,x]^2);

    (*how many terms should be generated*)
    h=64;

    (*what numerical precision is desired?*)
    precision=54;

    str=OpenWrite["out_legendre_gauss_quadrature_points.txt"];
    Do[Write[str];Write[str,"n = "<>ToString[n]];
    nlist=symboliclegendre[n,x];
    xnlist=x/.nlist;
    Do[Write[str,FortranForm[Re[N[Part[xnlist,i],precision]]]],{i,Length[xnlist]}];,{n,2,h}];
    Close[str];
    str=OpenWrite["out_legendre_gauss_quadrature_weights.txt"];
    Do[Write[str];Write[str,"n = "<>ToString[n]];
    slist:=symboliclegendre[n,x];
    xslist=x/.slist;
    Do[Write[str,FortranForm[Re[N[weights[n,Part[xslist,i]],precision]]]],{i,Length[xslist]}];,{n,2,h}];
    Close[str];


points.txt
----------

The points are listed below:

.. literalinclude:: ../../../../theory/integrate/legendre_gauss_quadrature/out_legendre_gauss_quadrature_points.txt

weights.txt
-----------

The weights are listed below:

.. literalinclude:: ../../../../theory/integrate/legendre_gauss_quadrature/out_legendre_gauss_quadrature_weights.txt


