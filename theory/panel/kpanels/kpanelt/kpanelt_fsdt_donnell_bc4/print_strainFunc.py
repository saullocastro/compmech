import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

var('i1, j1, k1, l1', integer=True)
var('x, t, xa, xb, tmin, tmax, L, r, r1, sina, cosa')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('wx, wt, w0, w0x, w0t')
var('c0ij, c1ij, c2ij, c3ij, c4ij')

def List(*e):
    return list(e)

filenames = [r'.\nonlinear_mathematica\fortran_strainFunc.txt',
             r'.\nonlinear_mathematica\fortran_e0.txt',
             r'.\nonlinear_mathematica\fortran_eL.txt']

outputs = ['output_strainFunc.txt',
           'output_e0.txt',
           'output_eL.txt']
strains = [['exx', 'ett', 'gxt', 'kxx', 'ktt', 'kxt', 'gtz', 'gxz'],
           ['exx0', 'ett0', 'gxt0', 'kxx0', 'ktt0', 'kxt0', 'gtz0', 'gxz0'],
           ['exxL', 'ettL', 'gxtL', 'kxxL', 'kttL', 'kxtL', 'gtzL', 'gxzL']]
for i, filename in enumerate(filenames):
    outname = outputs[i]
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

    filename = outname

    with open(filename, 'w') as out:
        out.write('subs\n\n')
        for k, value in subs.items():
            out.write('{0} = {1}\n'.format(k, value))
        out.write('\nexpressions\n\n')
        for (row, col), value in np.ndenumerate(matrix):
            out.write('{0} = \n'.format(strains[i][col]))
            if isinstance(value, sympy.Add):
                for arg in value.args:
                    s = arg.subs(subs)
                    out.write('    {}\n'.format(s))
            else:
                s = value.subs(subs)
                out.write('    {}\n'.format(s))
