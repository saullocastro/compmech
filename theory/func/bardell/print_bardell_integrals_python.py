import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse, pow2mult

var('x1t, x1r, x2t, x2r')
var('y1t, y1r, y2t, y2r')
var('xi1, xi2')

subs = {
       }

def List(*e):
    return list(e)

printstr_full = ''
printstr_12 = ''
for i, filepath in enumerate(
        glob.glob(r'.\bardell_integrals_mathematica\fortran_*.txt')):
    print(filepath)
    with open(filepath) as f:
        #if filepath != r'.\bardell_integrals_mathematica\fortran_ff_12.txt':
            #continue
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_')
        lines = [line.strip() for line in f.readlines()]
        string = ''.join(lines)
        string = string.replace('\\','')
        tmp = eval(string)
        print '\tfinished eval'
        printstr = ''
        if '_12' in filepath:
            name = '_'.join(names[1:3])
            printstr += 'cdef double integral_%s(double xi1, double xi2, int i, int j,\n' % name
            printstr += '                       double x1t, double x1r, double x2t, double x2r,\n'
            printstr += '                       double y1t, double y1r, double y2t, double y2r) nogil:\n'
        else:
            name = names[1]
            printstr += 'cdef double integral_%s(int i, int j, double x1t, double x1r, double x2t, double x2r,\n' % name
            printstr += '                        double y1t, double y1r, double y2t, double y2r) nogil:\n'
        matrix = sympy.Matrix(np.atleast_2d(tmp))
        for i in range(matrix.shape[0]):
            activerow = False
            for j in range(matrix.shape[1]):
                if matrix[i,j] == 0:
                    continue
                if not activerow:
                    activerow = True
                    if i == 0:
                        printstr += '    if i == %d:\n' % i
                    else:
                        printstr += '    elif i == %d:\n' % i
                    printstr += '        if j == %d:\n' % j
                else:
                    printstr += '        elif j == %d:\n' % j
                printstr += '            return %s\n' % str(matrix[i, j])
        printstr += '    return 0\n\n\n'

        if '_12' in filepath:
            printstr_12 += printstr
        else:
            printstr_full += printstr


with open('.\\bardell_integrals_python\\bardell_integrals_python_12.txt', 'w') as f:
    f.write(printstr_12)
with open('.\\bardell_integrals_python\\bardell_integrals_python_full.txt', 'w') as f:
    f.write(printstr_full)
