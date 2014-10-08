import os
import glob

import numpy as np
from sympy import var, Matrix

from compmech.conecyl.sympytools import mprint_as_array

var('sina, cosa, s, wi, detJ, Nxxi, Nyyi, Nxyi')
var('h1, h2, h3, h4, h5, h6, h7, h8')
var('h1x, h2x, h3x, h4x, h5x, h6x, h7x, h8x')
var('h1y, h2y, h3y, h4y, h5y, h6y, h7y, h8y')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')

def List(*e):
    return list(e)

for i, filepath in enumerate(
        glob.glob('./mathematica/k*.txt')):
    filename = os.path.basename(filepath)
    name = filename.split('.')[0]
    with open(filepath) as f:
        lines = [line.strip() for line in f.readlines()]
        string = ''.join(lines)
        string = string.replace('\\','')
        tmp = eval(string)
        matrix = Matrix(np.atleast_2d(tmp))
        printstr = mprint_as_array(matrix, name, '',
                                   print_file=False, collect_for=None,
                                   use_cse=False, op='+=')
    with open('./code/' + filename, 'w') as f:
        f.write(printstr)

    code = '../../../compmech/fem/elements/fsdt_donnell_kquad8_{0}.pxi'.format(
           name)
    with open(code, 'w') as f:
        f.write(printstr)

