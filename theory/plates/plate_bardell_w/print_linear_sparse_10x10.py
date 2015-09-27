import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse

var('bc1x, bc2x, bc3x, bc4x, bc1y, bc2y, bc3y, bc4y')
var('xi, eta, tmin, tmax, xa, xb, a, b')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('E11, nu, h, Fx, Ft, Fxt, Ftx')
var('kuBot, kvBot, kwBot, kphixBot, kphitBot')
var('kuTop, kvTop, kwTop, kphixTop, kphitTop')
var('kuLeft, kvLeft, kwLeft, kphixLeft, kphitLeft')
var('kuRight, kvRight, kwRight, kphixRight, kphitRight')

subs = {
       }

for i, filepath in enumerate(
        glob.glob(r'.\linear_mathematica_10x10\k*.txt')):
    print(filepath)
    with open(filepath) as f:
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_')
        string = f.read()
        string = string.replace('\r', '')
        string = string.replace('\n', '')
        string = string.replace('\n', '')
        string = string.replace('Pi','pi')
        string = string.replace('Sin','sin')
        string = string.replace('Cos','cos')
        string = string.replace('{','')
        string = string.replace('}','')
        string = string.replace('^','**')
        string = string.replace('\\','')
        tmp = map(eval, string.split(','))
        size = len(tmp)**0.5
        print size
        assert size % 1 == 0
        size = int(size)
        print('Estimated size: {0}'.format(size))
        matrix = sympy.Matrix(np.atleast_2d(tmp).reshape(size, size))
        printstr = mprint_as_sparse(matrix, names[0], names[1],
                                    print_file=False, collect_for=None,
                                    subs=subs, full_symmetric=True)
    with open('.\\linear_sparse_10x10\\' + filename, 'w') as f:
        f.write(printstr)
