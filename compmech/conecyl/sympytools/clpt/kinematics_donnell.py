import sympy
from sympy import sin, cos, pi, Matrix

from compmech.symbolic.doperator import D

sympy.var('x, t, sina, cosa, r, L', commutative=False)
sympy.var('wx, wt, w0x, w0t', commutative=False)
#
d = Matrix(
    [[    D(x),             0,                             0],
     [1/r*sina,      1/r*D(t),                      1/r*cosa],
     [1/r*D(t), D(x)-sina*1/r,                             0],
     [       0,             0,                       -D(x,x)],
     [       0,             0, -1/r*(sina*D(x) + 1/r*D(t,t))],
     [       0,             0, -1/r*(2*D(x,t)-1/r*sina*D(t))]])

#
dNL = Matrix(
    [[ 0, 0,                   wx*D(x)/2],
     [ 0, 0,          1/(2*r**2)*wt*D(t)],
     [ 0, 0, 1/r*(wx*D(t)/2 + wt*D(x)/2)],
     [ 0, 0,                           0],
     [ 0, 0,                           0],
     [ 0, 0,                           0]])
#
P = Matrix(
        [[  0, 0,                      0],
         [  0, 0,                      0],
         [  0, 0, 1/(2*pi*cosa*r)*D(x,x)]])

A_imp = Matrix(
    [[      wx + 2*w0x,                0],
     [               0, 1/r*(wt + 2*w0t)],
     [1/r*(wt + 2*w0t),       wx + 2*w0x],
     [0, 0],
     [0, 0],
     [0, 0]])

A = Matrix(
    [[    wx,      0],
     [     0, 1/r*wt],
     [1/r*wt,     wx],
     [0,           0],
     [0,           0],
     [0,           0]])

Gaux  = Matrix(
    [[0, 0,     D(x)],
     [0, 0, 1/r*D(t)]])
