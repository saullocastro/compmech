import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

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
        printstr += 'SUBROUTINE %s(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)\n' % names[1]
        printstr += '\n'
        printstr += '    INTEGER, INTENT(IN) :: i, j\n'
        printstr += '    REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r\n'
        printstr += '    REAL*8, INTENT(OUT) :: out\n'
        for i in range(matrix.shape[0]):
            activerow = False
            for j in range(matrix.shape[1]):
                if matrix[i,j] == 0:
                    continue
                if not activerow:
                    activerow = True
                    printstr += '    IF (i == %d) THEN\n' % i
                printstr += '        IF (j == %d) THEN\n' % j
                printstr += '            out = %s\n' % str(matrix[i, j])
                printstr += '            RETURN\n'
                printstr += '        END IF\n'
            printstr += '    END IF\n'
        printstr += 'END SUBROUTINE %s\n\n\n' % names[1]

with open('.\\bardell_integrals_fortran\\bardell_integrals_fortran.txt', 'w') as f:
    f.write(printstr)
