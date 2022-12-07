import sympy
from sympy import sin, cos, pi, Matrix

from mapy.sympytools.doperator import D

sympy.var('x, t, sina, cosa, r, L', commutative=False)
sympy.var('u, v, w', commutative=False)
sympy.var('ux, ut', commutative=False)
sympy.var('vx, vt', commutative=False)
sympy.var('wx, wt', commutative=False)
sympy.var('wxt, wtt', commutative=False)
#
d = Matrix(
    [[     D(x),             0,                             0],
     [ 1/r*sina,      1/r*D(t),                      1/r*cosa],
     [ 1/r*D(t), D(x)-sina*1/r,                             0],
     [        0,             0,                       -D(x,x)],
     [        0,             0, -1/r*(sina*D(x) + 1/r*D(t,t))],
     [        0,             0, -1/r*(2*D(x,t)-1/r*sina*D(t))]])

#
dNL = Matrix(
    [[ux*D(x)/2, vx*D(x)/2, wx*D(x)/2],
     [1/(2*r**2)*(sina**2*u + 2*sina*(cosa*w + vt) - 2*sina*v*D(t) + ut*D(t)),
      1/(2*r**2)*(v + 2*cosa*w*D(t) + vt*D(t)),
      1/(2*r**2)*(cosa**2*w - 2*cosa*v*D(t) + wt*D(t))],
     [1/r*ux*D(t),
      1/r*(-(sina*ux + cosa*wx) + (sina*u + cosa*w + vt)*D(x)),
      1/r*wx*D(t)],
     [0, 0, -ux*D(x,x) - 1/r*vx*D(x,t)],
     [1/(2*r**2)*(-wxt*D(t) + sina*(1/r)*wt*D(t) - sina*(sina*wx + (1/r)*cosa*wtt)),
      1/(2*r**2)*(-(1/r)*wt + sina*(-wx*D(t) + wxt) - (1/r)*wtt*D(t)),
      1/(2*r**2)*((1/r)*cosa*wt*D(t) - cosa*(sina*wx + (1/r)*wtt))],
     [1/r*(-wxt*D(x)),
      1/r*(-sina*wx*D(x)),
      1/r*((1/r)*(sina*ux + cosa*wx)*D(t) + (sina*v - ut)*D(x,x) - (1/r)*(sina*u + vt)*D(x,t)
           -(1/r)*vx*D(t,t) - (1/r)*cosa*wxt)]])
#
P = Matrix(
        [[  0, 0,                      0],
         [  0, 0,                      0],
         [  0, 0, 1/(2*pi*cosa*r)*D(x,x)]])
