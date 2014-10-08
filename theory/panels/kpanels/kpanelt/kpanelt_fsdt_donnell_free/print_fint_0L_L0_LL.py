import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

num0 = 3
num1 = 3
num2 = 6

var('i1, k1, i2, j2, k2, l2', integer=True)
var('x, t, xa, xb, L, r, r2, sina, cosa')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('wx, wt, w0, w0x, w0t, tLA')
var('Nxx0, Ntt0, Nxt0, Mxx0, Mtt0, Mxt0')
var('NxxL, NttL, NxtL, MxxL, MttL, MxtL')

def List(*e):
    return list(e)

for name in ['fint_L0_0L_LL']:
    filename = r'.\nonlinear_mathematica\fortran_{0}.txt'.format(name)

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

    filename = 'output_{0}.txt'.format(name)

    with open(filename, 'w') as out:
        out.write('subs\n\n')
        for k, value in subs.items():
            out.write('{0} = {1}\n'.format(k, value))
        out.write('\nexpressions\n\n')
        for (row,col),value in np.ndenumerate(matrix):
            colstr = '{0}'.format(col)
            if col+1 > num0:
                colstr = 'col+{0}'.format(col-num0)
            if col+1 > num0 + num1:
                colstr = 'col+{0}'.format(col-num0-num1)
            #out.write('{0}[{1}] = beta*{0}[{1}] + alpha*({2})\n'.format(
                      #name, col, value.subs(subs)))
            out.write('{0}[{1},{2}] = {3}\n'.format(name,
                      row, colstr, value.subs(subs)))
