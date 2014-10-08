import os
import glob

import numpy as np
from sympy import var, Matrix

from compmech.conecyl.sympytools import mprint_as_array

nnodes = 12

var('r, j11, j12, j21, j22, wi, xi, eta, Nxx0, Ntt0, Nxt0')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var(','.join(['x'+str(i) for i in range(1, 1+nnodes)]))
var(','.join(['y'+str(i) for i in range(1, 1+nnodes)]))

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
        string = string.replace('\\','')
        tmp = eval(string)
        matrix = Matrix(np.atleast_2d(tmp))
        printstr = mprint_as_array(matrix, names[1], '',
                                   print_file=False, collect_for=None,
                                   use_cse=False, op='=')
    with open('.\\linear_sparse\\' + filename, 'w') as f:
        f.write(printstr)
