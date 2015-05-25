import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

var('i1, k1, i2, j2, k2, l2', integer=True)
var('x, t, xa, xb, L, r, r2, sina, cosa')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('wx, wt, w0x, w0t, Nxx, Ntt, Nxt, tLA')

var('sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t')
var('sink1x, cosk1x, sink2x, cosk2x, sinl2t, cosl2t')
subs = {
        sin(pi*i1*x/L):sini1x,
        cos(pi*i1*x/L):cosi1x,
        sin(pi*k1*x/L):sink1x,
        cos(pi*k1*x/L):cosk1x,
        sin(pi*i2*x/L):sini2x,
        cos(pi*i2*x/L):cosi2x,
        sin(pi*k2*x/L):sink2x,
        cos(pi*k2*x/L):cosk2x,
        sin(j2*t):sinj2t,
        cos(j2*t):cosj2t,
        sin(l2*t):sinl2t,
        cos(l2*t):cosl2t,
        }


def List(*e):
    return list(e)

# define the values for pij and qij
# print these values
# print the formulas for k0L etc for each case based on pij and qij

files1 = glob.glob(r'.\nonlinear_mathematica_new\fortran*cone_k0L_*.txt')
files2 = glob.glob(r'.\nonlinear_mathematica_new\fortran*cone_kG_*.txt')
files3 = glob.glob(r'.\nonlinear_mathematica_new\fortran*cone_kLL_*.txt')

for filepath in (files1 + files2 + files3):
    print filepath
    with open(filepath) as f:
        print_str = ''
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_') # removing .txt

        # k0L, kG or kLL
        matrix_name = names[2]

        sub_matrix = names[3] # 00, 01, 02, 11, 12 or 22
        left, right = sub_matrix
        key = matrix_name + '_' + sub_matrix
        lines = [line.strip() for line in f.readlines()]
        string = ''.join(lines)
        #string = string.replace('List','')
        string = string.replace('Pi','pi')
        string = string.replace('Sin','sin')
        string = string.replace('Cos','cos')
        string = string.replace('\\','')
        tmp = eval(string)
        matrix = Matrix(tmp)
        tmp = ''
        count = 0
        for (row, col), value in np.ndenumerate(matrix):
            if value:
                value = value.subs(subs)
                count += 1
                tmp += 'c += 1\n'

                if int(left)==0:
                    tmp += 'rows[c] = {}\n'.format(row)
                else:
                    tmp += 'rows[c] = row+{}\n'.format(row)

                if int(right)==0:
                    tmp += 'cols[c] = {}\n'.format(col)
                else:
                    tmp += 'cols[c] = col+{}\n'.format(col)

                tmp += 'out[c] = beta*out[c] + alpha*({1})\n'.format(
                        matrix_name, value)

        print_str += '{0}_{1} with {2} non-null terms\n'.format(
                matrix_name, sub_matrix, count)
        print_str += tmp
        names[3] = sub_matrix
        filename = '_'.join(names) + '.txt'
    with open('.\\nonlinear_sparse_new\\' + filename, 'w') as f:
        f.write(print_str)


