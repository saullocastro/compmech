import sympy
from sympy import sin, cos, pi, Matrix

from mapy.sympytools.doperator import D

sympy.var('x, t, sina, cosa, r, L', commutative=False)
sympy.var('ux, ut, v, vx, vt, wx, wt, w0x, w0t', commutative=False)
#
d = Matrix(
    [[     D(x),             0,                             0],
     [ 1/r*sina,      1/r*D(t),                      1/r*cosa],
     [ 1/r*D(t), D(x)-sina*1/r,                             0],
     [        0,             0,                       -D(x,x)],
     [        0, -1/r*(-1/r*D(t)), -1/r*(sina*D(x) + 1/r*D(t,t))],
     [        0, -1/r*(-D(x) + sina/r), -1/r*(2*D(x,t)-1/r*sina*D(t))]])

#
A = Matrix(
    [[0, 0, 0, vx, wx, 0],
     [0, -sina*v/r, -sina*ut/r + v/r - cosa*wt/r, 0, 0, -cosa*v/r + 1/r*wt],
     [-v*sina/r, 0, -sina*ux - cosa*wx, 0, -cosa*v/r + 1/r*wt, wx],
     [0,0,0,0,0,0],
     [0,0,0,0,0,0],
     [0,0,0,0,0,0]])

Gaux  = Matrix([[D(x), 0, 0],
                [1/r*D(t), 0, 0],
                [0, 1/r, 0],
                [0, D(x), 0],
                [0, 0, D(x)],
                [0, 0, 1/r*D(t)]])
