import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse

var('i2, k2, j3, l3, i4, j4, k4, l4', integer=True)
var('x, t, tmin, tmax, xa, xb, L, r, r1, r2, sina, cosa')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('E11, nu, h, Fx, Ft, Fxt, Ftx')
var('kphixBot, kphitBot')
var('kphixTop, kphitTop')
var('kphixLeft, kphitLeft')
var('kphixRight, kphitRight')

subs = {
       }

def List(*e):
    return list(e)

for i, filepath in enumerate(
        glob.glob(r'.\linear_mathematica\fortran_*.txt')):
    print filepath
    with open(filepath) as f:
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_')
        lines = [line.strip() for line in f.readlines()]
        string = ''.join(lines)
        string = string.replace('Pi','pi')
        string = string.replace('Sin','sin')
        string = string.replace('Cos','cos')
        string = string.replace('{','(')
        string = string.replace('}',')')
        string = string.replace('^','**')
        string = string.replace('\\','')
        tmp = eval(string)
        matrix = sympy.Matrix(np.atleast_2d(tmp))
        printstr = mprint_as_sparse(matrix, names[2], names[3],
                                    print_file=False, collect_for=None,
                                    subs=subs)
    with open('.\\linear_sparse\\' + filename, 'w') as f:
        f.write(printstr)
