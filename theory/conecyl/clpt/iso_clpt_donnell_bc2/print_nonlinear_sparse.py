import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

var('i1, k1, i2, j2, k2, l2', integer=True)
var('x, t, xa, xb, L, r, r2, sina, cosa')
var('E11, h, nu')
var('wx, wt, w0x, w0t, Nxx, Ntt, Nxt, tLA')

var('p00, p01, p02, p03, p04, p05')
var('p10, p11, p12, p13, p14, p15')
var('p20, p21, p22, p23, p24, p25')
var('p30, p31, p32, p33, p34, p35')
var('p40, p41, p42, p43, p44, p45')
var('p50, p51, p52, p53, p54, p55')

var('q00, q01, q02, q03, q04, q05')
var('q10, q11, q12, q13, q14, q15')
var('q20, q21, q22, q23, q24, q25')
var('q30, q31, q32, q33, q34, q35')
var('q40, q41, q42, q43, q44, q45')
var('q50, q51, q52, q53, q54, q55')

p = Matrix([[p00, p01, p02, p03, p04, p05],
            [p10, p11, p12, p13, p14, p15],
            [p20, p21, p22, p23, p24, p25],
            [p30, p31, p32, p33, p34, p35],
            [p40, p41, p42, p43, p44, p45],
            [p50, p51, p52, p53, p54, p55]])
pstr = np.array([str(i) for i in p]).reshape(6,6)
q = Matrix([[q00, q01, q02, q03, q04, q05],
            [q10, q11, q12, q13, q14, q15],
            [q20, q21, q22, q23, q24, q25],
            [q30, q31, q32, q33, q34, q35],
            [q40, q41, q42, q43, q44, q45],
            [q50, q51, q52, q53, q54, q55]])
qstr = np.array([str(i) for i in q]).reshape(6,6)

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

valid = {}
for i, sufix in enumerate(['p', 'q']):
    if sufix=='p':
        pqstr = pstr
        pq = p
    elif sufix=='q':
        pqstr = qstr
        pq = q
    for filepath in glob.glob(r'.\nonlinear_mathematica\fortran*cone*'+sufix+r'*.txt'):
        print filepath
        with open(filepath) as f:
            print_str = ''
            filename = os.path.basename(filepath)
            names = filename[:-4].split('_') # removing .txt

            # k0Lp, k0Lq, kGp, kGq, kLLp or kLLq
            matrix_name = names[3]

            sub_matrix = names[4][i] # 0, 1, or 2
            key = matrix_name[:-1] + '_' + names[4]
            if not key in valid.keys():
                valid[key] = set()
            lines = [line.strip() for line in f.readlines()]
            string = ''.join(lines)
            string = string.replace('Pi','pi')
            string = string.replace('Sin','sin')
            string = string.replace('Cos','cos')
            string = string.replace('\\','')
            tmp = eval(string)
            matrix = Matrix(tmp)
            print_str += '{0}_{1}\n'.format(matrix_name, sub_matrix)
            for (row, col), value in np.ndenumerate(matrix):
                if value:
                    value = value.subs(subs)
                    valid[key].add(pq[row, col])
                    print_str += '{0} = {1}\n'.format(pqstr[row,col], value)
            print_str += '\n# create buffer\n'
            print_str += '{0}_{1}\n'.format(matrix_name, sub_matrix)
            for (row,col), value in np.ndenumerate(matrix):
                if value:
                    value = value.subs(subs)
                    valid[key].add(pq[row, col])
                    print_str += '{0}_{1}_{2}[pos] = {2}\n'.format(
                            matrix_name, sub_matrix, pqstr[row,col])
            print_str += '\n# access buffer\n'
            for (row,col), value in np.ndenumerate(matrix):
                if value:
                    value = value.subs(subs)
                    valid[key].add(pq[row, col])
                    print_str += '{2} = {0}_{1}_{2}[pos]\n'.format(
                            matrix_name, sub_matrix, pqstr[row,col])
            print_str += '\nsubs\n\n'
            for k, value in subs.items():
                print_str += '{0} = {1}\n'.format(k, value)
            names[4] = sub_matrix
            filename = '_'.join(names) + '.txt'
        with open('.\\nonlinear_sparse\\' + filename, 'w') as f:
            f.write(print_str)

l1 = glob.glob(r'.\nonlinear_mathematica\fortran*iso_cone_k0L_*.txt')
l2 = glob.glob(r'.\nonlinear_mathematica\fortran*iso_cone_kG_*.txt')
l3 = glob.glob(r'.\nonlinear_mathematica\fortran*iso_cone_kLL_*.txt')

for filepath in (l1 + l2 + l3):
    print filepath
    with open(filepath) as f:
        print_str = ''
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_') # removing .txt

        # k0L, kG or kLL
        matrix_name = names[3]

        sub_matrix = names[4] # 00, 01, 02, 11, 12 or 22
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
            for s in value.free_symbols:
                if not s in valid[key]:
                    value = value.subs({s:0})
            if value:
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
        names[4] = sub_matrix
        filename = '_'.join(names) + '.txt'
    with open('.\\nonlinear_sparse\\' + filename, 'w') as f:
        f.write(print_str)


