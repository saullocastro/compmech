import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse

var('xi1t, xi1r, xi2t, xi2r')

subs = {
       }

def List(*e):
    return list(e)

printstr = ''
for i, filepath in enumerate(
        glob.glob(r'.\bardell_mathematica\fortran_*.txt')):
    print(filepath)
    with open(filepath) as f:
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_')
        lines = [line.strip() for line in f.readlines()]
        string = ''.join(lines)
        string = string.replace('\\','')
        tmp = eval(string)
        matrix = sympy.Matrix(np.atleast_2d(tmp))
        printstr += '\n\ncdef void calc_%s(double[:, ::1] v, double xi1t, double xi1r, double xi2t, double xi2r) nogil:\n' % names[1]
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i,j] == 0:
                    continue
                else:
                    printstr += '    v[%d, %d] = %s\n' % (i, j, str(matrix[i, j]))

with open('.\\bardell_python\\bardell_python.txt', 'w') as f:
    f.write(printstr)
