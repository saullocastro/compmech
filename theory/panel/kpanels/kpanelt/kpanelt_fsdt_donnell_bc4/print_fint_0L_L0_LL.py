import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

num0 = 0
num1 = 6

var('i1, j1', integer=True)
var('x, t, xa, xb, tmin, tmax, L, r, sina, cosa')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('wx, wt, w0, w0x, w0t, tLA')
var('Nxx0, Ntt0, Nxt0, Mxx0, Mtt0, Mxt0, Qt0, Qx0')
var('NxxL, NttL, NxtL, MxxL, MttL, MxtL, QtL, QxL')

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

    var('sini1bx, cosi1bx')
    var('sinj1bt, cosj1bt')
    var('sinj1_bt, cosj1_bt')
    subs = {
            sin(pi*j1*(t-tmin)/(tmax-tmin)): sinj1bt,
            cos(pi*j1*(t-tmin)/(tmax-tmin)): cosj1bt,

            sin(pi*j1*(t-tmin)/(-tmax+tmin)): sinj1_bt,
            cos(pi*j1*(t-tmin)/(-tmax+tmin)): cosj1_bt,

            sin(pi*i1*(0.5*L+x)/L): sini1bx,
            cos(pi*i1*(0.5*L+x)/L): cosi1bx,
            }

    filename = 'output_{0}.txt'.format(name)

    with open(filename, 'w') as out:
        out.write('subs\n\n')
        for k, value in subs.items():
            out.write('{0} = {1}\n'.format(k, value))
        out.write('\nexpressions\n\n')
        for (row, col), value in np.ndenumerate(matrix):
            colstr = '{0}'.format(col)
            if col+1 > num0:
                colstr = 'col+{0}'.format(col-num0)
            out.write('{0}[{1},{2}] = {3}\n'.format(name, row, colstr,
                                                    value.subs(subs)))
