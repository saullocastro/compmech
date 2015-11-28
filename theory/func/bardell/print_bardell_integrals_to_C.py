import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse, pow2mult, star2Cpow

var('x1t, x1r, x2t, x2r')
var('y1t, y1r, y2t, y2r')
var('xi1, xi2')

subs = {
       }

def List(*e):
    return list(e)

printstr_full = ''
printstr_full_h = ''
printstr_12 = ''
printstr_12_h = ''
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
            printstr += 'double integral_%s(double xi1, double xi2, int i, int j,\n' % name
            printstr += '                   double x1t, double x1r, double x2t, double x2r,\n'
            printstr += '                   double y1t, double y1r, double y2t, double y2r) {\n'

        else:
            name = names[1]
            printstr += 'double integral_%s(int i, int j, double x1t, double x1r, double x2t, double x2r,\n' % name
            printstr += '                   double y1t, double y1r, double y2t, double y2r) {\n'

        printstr_h = '\n'
        printstr_h += '#ifndef BARDELL_%s_H\n' % name.upper()
        printstr_h += '#define BARDELL_%s_H\n' % name.upper()
        printstr_h += printstr.replace(' {', ';')
        printstr_h += '#endif /** BARDELL_%s_H */\n' % name.upper()
        printstr_h += '\n'

        matrix = sympy.Matrix(np.atleast_2d(tmp))
        for i in range(matrix.shape[0]):
            activerow = False
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 0:
                    continue
                if not activerow:
                    activerow = True
                    if i == 0:
                        printstr += '    switch(i) {\n'
                    else:
                        printstr += '                  }\n'
                    printstr += '    case %d:\n' % i
                    printstr += '        switch(j) {\n'
                printstr += '        case %d:\n' % j
                printstr += '            return %s;\n' % star2Cpow(str(matrix[i, j]))
        printstr += '              }\n'
        printstr += '    }\n'
        printstr += '    return 0\n'
        printstr += '}\n'

        if '_12' in filepath:
            printstr_12 += printstr
            printstr_12_h += printstr_h
        else:
            printstr_full += printstr
            printstr_full_h += printstr_h


with open('.\\bardell_integrals_C\\bardell.h', 'w') as g:
    g.write(printstr_full_h)

with open('.\\bardell_integrals_C\\bardell.c', 'w') as g:
    g.write(printstr_full)

with open('.\\bardell_integrals_C\\bardell_12.h', 'w') as g:
    g.write(printstr_12_h)

with open('.\\bardell_integrals_C\\bardell_12.c', 'w') as g:
    g.write(printstr_12)
