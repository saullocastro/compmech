import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos

from compmech.conecyl.sympytools import mprint_as_sparse

sympy.var('i1, k1, i2, j2, k2, l2', integer=True)
sympy.var('i1a, i1b, i1c, i2a, i2b, i2c, j2a, j2b, j2c', integer=True)
sympy.var('x, t, xa, xb, L, r, r1, r2, sina, cosa')
sympy.var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
sympy.var('B11, B12, B16, B22, B26, B66')
sympy.var('D11, D12, D16, D22, D26, D66')
sympy.var('E11, nu, h, Fc, P, T')
sympy.var('kuBot, kuTop, kphixBot, kphixTop')

def List(*e):
    return list(e)

for i, filepath in enumerate(
        glob.glob(r'.\outputs\fortran_*.txt')):
    print filepath
    with open(filepath) as f:
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_')
        offset = 0
        if '_la_' in filename:
            offset = 1
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
        printstr = mprint_as_sparse(matrix,
                names[2 + offset], names[3 + offset],
                       use_cse=False, print_file=False,
                       collect_for=None)
    with open('.\\sparse_printing\\' + filename, 'w') as f:
        f.write(printstr)
