import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse

var('x1t, x1r, x2t, x2r')
var('y1t, y1r, y2t, y2r')

subs = {
       }

def List(*e):
    return list(e)

printstr = ''
for i, filepath in enumerate(
        glob.glob(r'.\bardell_integrals_mathematica\fortran_*.txt')):
    print(filepath)
    with open(filepath) as f:
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_')
        lines = [line.strip() for line in f.readlines()]
        string = ''.join(lines)
        string = string.replace('\\','')
        tmp = eval(string)
        matrix = sympy.Matrix(np.atleast_2d(tmp))
        printstr += '\n\ncdef double %s(int i, int j, double x1t, double x1r, double x2t, double x2r,\n' % names[1]
        printstr += '                   double y1t, double y1r, double y2t, double y2r) nogil:\n'
        for i in range(matrix.shape[0]):
            activerow = False
            for j in range(matrix.shape[1]):
                if matrix[i,j] == 0:
                    continue
                if not activerow:
                    activerow = True
                    printstr += '    if i == %d:\n' % i
                printstr += '        if j == %d:\n' % j
                printstr += '            return %s\n' % str(matrix[i, j])

with open('.\\bardell_integrals_python\\bardell_integrals_python.txt', 'w') as f:
    f.write(printstr)
