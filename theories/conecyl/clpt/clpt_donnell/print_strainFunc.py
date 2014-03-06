import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

var('i1, k1, i2, j2, k2, l2', integer=True)
var('x, t, xa, xb, L, r, r2, sina, cosa')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('wx, wt, tLA')
c = sympy.var('c00, c01, c02, c0i, c1i, c2i, c0ij, c1ij, c2ij, c3ij, c4ij, c5ij')

def List(*e):
    return list(e)

filename = r'.\numerical\fortran_strainFunc.txt'

with open(filename) as f:
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

filename = 'output_strainFunc.txt'

with open(filename, 'w') as out:
    out.write('subs\n\n')
    for k, v in subs.items():
        out.write('{0} = {1}\n'.format(k, v))
    out.write('\nexpressions\n\n')
    for (row,col),v in np.ndenumerate(matrix):
        out.write('e[{}] = \n'.format(col))
        for arg in v.args:
            out.write('    {}\n'.format(arg.subs(subs)))