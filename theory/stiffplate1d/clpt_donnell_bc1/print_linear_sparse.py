import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, sqrt

from compmech.conecyl.sympytools import mprint_as_sparse

var('i1, j1, k1, l1', integer=True)
var('db, df, bb, bf, ys, Asf, mus, hb, Iyy, Jxx')
var('x, y, xa, xb, ya, yb, a, b, beta, gamma, aeromu')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('A11sb, A12sb, A16sb, A22sb, A26sb, A66sb')
var('B11sb, B12sb, B16sb, B22sb, B26sb, B66sb')
var('D11sb, D12sb, D16sb, D22sb, D26sb, D66sb')
var('mu')
var('E11, nu, h, Nxx, Nyy, Nxy')
var('kuBot, kvBot, kwBot, kphixBot, kphiyBot')
var('kuTop, kvTop, kwTop, kphixTop, kphiyTop')
var('kuLeft, kvLeft, kwLeft, kphixLeft, kphiyLeft')
var('kuRight, kvRight, kwRight, kphixRight, kphiyRight')
var('E1, S1, E2, F1')

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
        string = string.replace('Sqrt','sqrt')
        string = string.replace('{','(')
        string = string.replace('}',')')
        string = string.replace('^','**')
        string = string.replace('\\','')
        tmp = eval(string)
        matrix = sympy.Matrix(np.atleast_2d(tmp))
        printstr = mprint_as_sparse(matrix, names[1], names[2],
                                    print_file=False, collect_for=None,
                                    subs=subs)
    with open('.\\linear_sparse\\' + filename, 'w') as f:
        f.write(printstr)
