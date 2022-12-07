import sympy
from sympy import sin, cos, pi, Matrix

from mapy.sympytools.doperator import D

sympy.var('x, t, sina, cosa, r, L', commutative=False)
sympy.var('w, wx, wt, wxt, wtt, w0x, w0t', commutative=False)
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

A = Matrix(
    [[0, wx, 0, 0, 0],
     [1/r*cosa**2*w, 0, 1/r*wt, 0, 0],
     [0, 1/r*wt, wx, 0, 0],
     [0, 0, 0, 0, 0],
     [-cosa*wtt/(2*r**2) -sina*cosa*wx/(2*r), -sina*cosa*w/(2*r**2),
         1/r**2*cosa*wt, 0, -cosa*w/(2*r**2)],
     [-1/r*cosa*wxt, 1/r**2*wt*cosa, 1/r*wx*cosa, -1/r**2*cosa*w, 0]])

Gaux  = Matrix(
    [[0, 0,        1/r],
     [0, 0,       D(x)],
     [0, 0,   1/r*D(t)],
     [0, 0,     D(x,t)],
     [0, 0, 1/r*D(t,t)]])
