import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

var('i1, k1, i2, j2, k2, l2', integer=True)
var('x, t, xa, xb, L, r, r2, sina, cosa')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('phix, phit, tLA')
sympy.var('c00, c01, c02, c0i, c1i, c2i, c3i, c4i, c5i, c6i')
sympy.var('c0ij, c1ij, c2ij, c3ij, c4ij, c5ij, c6ij')
sympy.var('c7ij, c8ij, c9ij, c10ij, c11ij, c12ij, c13ij')

def List(*e):
    return list(e)

with open(r'.\numerical\fortran_strainFunc.txt') as f:
    lines = [line.strip() for line in f.readlines()]
string = ''.join(lines)
string = string.replace('Pi','pi')
string = string.replace('Sin','sin')
string = string.replace('Cos','cos')
string = string.replace('\\','')
tmp = eval(string)
matrix = Matrix(np.atleast_2d(tmp))

var('sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t')
subs = {
        sin(pi*i1*x/L):sini1x,
        cos(pi*i1*x/L):cosi1x,
        sin(pi*i2*x/L):sini2x,
        cos(pi*i2*x/L):cosi2x,
        sin(j2*t):sinj2t,
        cos(j2*t):cosj2t,
        }

with open('output_strainFunc.txt', 'w') as out:
    out.write('subs\n\n')
    for k, v in subs.items():
        out.write('{0} = {1}\n'.format(k, v))
    out.write('\nexpressions\n\n')
    for (row,col),v in np.ndenumerate(matrix):
        out.write('e[{}] = \n'.format(col))
        for arg in v.args:
            out.write('    {}\n'.format(arg.subs(subs)))
