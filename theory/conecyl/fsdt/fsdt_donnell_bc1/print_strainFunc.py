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
var('wx, wt, w0, w0x, w0t, tLA, castro')
var('c00, c01, c02')
var('c0i, c1i, c2i, c3i, c4i')
var('c0ij, c1ij, c2ij, c3ij, c4ij, c5ij, c6ij, c7ij, c8ij, c9ij')

def List(*e):
    return list(e)

filenames = [r'.\nonlinear_mathematica\fortran_strainFunc.txt',
             r'.\nonlinear_mathematica\fortran_e0.txt',
             r'.\nonlinear_mathematica\fortran_eL.txt']

outputs = ['output_strainFunc.txt',
           'output_e0.txt',
           'output_eL.txt']
strains = [['exx', 'ett', 'gxt', 'kxx', 'ktt', 'kxt', 'ktz', 'kxz'],
           ['exx0', 'ett0', 'gxt0', 'kxx0', 'ktt0', 'kxt0', 'ktz0', 'kxz0'],
           ['exxL', 'ettL', 'gxtL', 'kxxL', 'kttL', 'kxtL', 'ktzL', 'kxzL']]
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

    var('sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t')
    subs = {
            sin(pi*i1*x/L):sini1x,
            cos(pi*i1*x/L):cosi1x,
            sin(pi*i2*x/L):sini2x,
            cos(pi*i2*x/L):cosi2x,
            sin(j2*t):sinj2t,
            cos(j2*t):cosj2t,
            }

    filename = outname

    with open(filename, 'w') as out:
        out.write('subs\n\n')
        for k, value in subs.items():
            out.write('{0} = {1}\n'.format(k, value))
        out.write('\nexpressions\n\n')
        for (row,col),value in np.ndenumerate(matrix):
            out.write('{0} = \n'.format(strains[i][col]))
            for arg in value.args:
                s = arg.subs(subs)
                out.write('    {}\n'.format(s))
