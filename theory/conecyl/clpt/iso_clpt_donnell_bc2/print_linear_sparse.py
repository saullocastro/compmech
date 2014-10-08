import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse

var('i1, k1, i2, j2, k2, l2', integer=True)
var('i1a, i1b, i1c, i2a, i2b, i2c, j2a, j2b, j2c', integer=True)
var('x, t, xa, xb, L, r, r1, r2, sina, cosa')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('E11, nu, h, Fc, P, T')
var('kuBot, kuTop')
var('kphixBot, kphixTop, kphitBot, kphitTop')
var('sini1xa, cosi1xa, sini1xb, cosi1xb,')
var('sini1xa_xb, sini1xaxb, cosi1xaxb,')
var('sink1xa, sink1xb, cosk1xa, cosk1xb, sini2xa, sini2xb,')
var('sin2i2xa, sin2i2xb, sini2xa_xb, sini2xaxb, cosi2xaxb')
var('cosi2xa, cosi2xb, cos2i2xa, cos2i2xb,')
var('cosk2xa, cosk2xb, sink2xa, sink2xb')
var('sin2i1xa, cos2i1xa, sin2i1xb, cos2i1xb')

subs = {sin(pi*i1*xa/L): sini1xa,
        cos(pi*i1*xa/L): cosi1xa,
        sin(pi*i1*xb/L): sini1xb,
        cos(pi*i1*xb/L): cosi1xb,
        sin(pi*i1*(xa - xb)/L): sini1xa_xb,
        sin(pi*i1*(xa + xb)/L): sini1xaxb,
        cos(pi*i1*(xa + xb)/L): cosi1xaxb,
        sin(pi*k1*xa/L): sink1xa,
        sin(pi*k1*xb/L): sink1xb,
        cos(pi*k1*xa/L): cosk1xa,
        cos(pi*k1*xb/L): cosk1xb,
        sin(pi*i2*xa/L): sini2xa,
        sin(pi*i2*xb/L): sini2xb,
        sin(2*pi*i2*xa/L): sin2i2xa,
        sin(2*pi*i2*xb/L): sin2i2xb,
        sin(pi*i2*(xa - xb)/L): sini2xa_xb,
        sin(pi*i2*(xa + xb)/L): sini2xaxb,
        cos(pi*i2*xa/L): cosi2xa,
        cos(pi*i2*xb/L): cosi2xb,
        cos(2*pi*i2*xa/L): cos2i2xa,
        cos(2*pi*i2*xb/L): cos2i2xb,
        cos(pi*i2*(xa + xb)/L): cosi2xaxb,
        cos(pi*k2*xa/L): cosk2xa,
        cos(pi*k2*xb/L): cosk2xb,
        sin(pi*k2*xa/L): sink2xa,
        sin(pi*k2*xb/L): sink2xb,
        sin(2*pi*i1*xa/L): sin2i1xa,
        cos(2*pi*i1*xa/L): cos2i1xa,
        sin(2*pi*i1*xb/L): sin2i1xb,
        cos(2*pi*i1*xb/L): cos2i1xb,
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
        printstr = mprint_as_sparse(matrix, names[3], names[4],
                                    print_file=False, collect_for=None,
                                    subs=subs)
    with open('.\\linear_sparse\\' + filename, 'w') as f:
        f.write(printstr)
