import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

var('i1, j1, k1, l1', integer=True)
var('x, t, xa, xb, tmin, tmax, L, r, sina, cosa')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('wx, wt, w0x, w0t, Nxx, Ntt, Nxt')

var('p00, p01, p02, p03, p04, p05, p06, p07')
var('p10, p11, p12, p13, p14, p15, p16, p17')
var('p20, p21, p22, p23, p24, p25, p26, p27')
var('p30, p31, p32, p33, p34, p35, p36, p37')
var('p40, p41, p42, p43, p44, p45, p46, p47')
var('p50, p51, p52, p53, p54, p55, p56, p57')
var('p60, p61, p62, p63, p64, p65, p66, p67')
var('p70, p71, p72, p73, p74, p75, p76, p77')

var('q00, q01, q02, q03, q04, q05, q06, q07')
var('q10, q11, q12, q13, q14, q15, q16, q17')
var('q20, q21, q22, q23, q24, q25, q26, q27')
var('q30, q31, q32, q33, q34, q35, q36, q37')
var('q40, q41, q42, q43, q44, q45, q46, q47')
var('q50, q51, q52, q53, q54, q55, q56, q57')
var('q60, q61, q62, q63, q64, q65, q66, q67')
var('q70, q71, q72, q73, q74, q75, q76, q77')

p = Matrix([[p00, p01, p02, p03, p04, p05, p06, p07],
            [p10, p11, p12, p13, p14, p15, p16, p17],
            [p20, p21, p22, p23, p24, p25, p26, p27],
            [p30, p31, p32, p33, p34, p35, p36, p37],
            [p40, p41, p42, p43, p44, p45, p46, p47],
            [p50, p51, p52, p53, p54, p55, p56, p57],
            [p60, p61, p62, p63, p64, p65, p66, p67],
            [p70, p71, p72, p73, p74, p75, p76, p77]])
pstr = np.array([str(i) for i in p]).reshape(8, 8)
q = Matrix([[q00, q01, q02, q03, q04, q05, q06, q07],
            [q10, q11, q12, q13, q14, q15, q16, q17],
            [q20, q21, q22, q23, q24, q25, q26, q27],
            [q30, q31, q32, q33, q34, q35, q36, q37],
            [q40, q41, q42, q43, q44, q45, q46, q47],
            [q50, q51, q52, q53, q54, q55, q56, q57],
            [q60, q61, q62, q63, q64, q65, q66, q67],
            [q70, q71, q72, q73, q74, q75, q76, q77]])
qstr = np.array([str(i) for i in q]).reshape(8, 8)

var('sinj1_bt, cosj1_bt')
var('cosl1_bt')
var('sini1bx, cosi1bx')
var('sinj1bt, cosj1bt')
var('sink1bx, cosk1bx')
var('sinl1bt, cosl1bt')
subs = {
        sin(pi*j1*(t-tmin)/(tmax-tmin)): sinj1bt,
        cos(pi*j1*(t-tmin)/(tmax-tmin)): cosj1bt,
        sin(pi*l1*(t-tmin)/(tmax-tmin)): sinl1bt,
        cos(pi*l1*(t-tmin)/(tmax-tmin)): cosl1bt,

        sin(pi*j1*(t-tmin)/(-tmax+tmin)): sinj1_bt,
        cos(pi*j1*(t-tmin)/(-tmax+tmin)): cosj1_bt,
        cos(pi*l1*(t-tmin)/(-tmax+tmin)): cosl1_bt,

        sin(pi*k1*(0.5*L+x)/L): sink1bx,
        cos(pi*k1*(0.5*L+x)/L): cosk1bx,

        sin(pi*i1*(0.5*L+x)/L): sini1bx,
        cos(pi*i1*(0.5*L+x)/L): cosi1bx,
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
            matrix_name = names[2]

            sub_matrix = names[3][i] # 0, 1, or 2
            key = matrix_name[:-1] + '_' + names[3]
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
            names[3] = sub_matrix
            filename = '_'.join(names) + '.txt'
        with open('.\\nonlinear_sparse\\' + filename, 'w') as f:
            f.write(print_str)

l1 = glob.glob(r'.\nonlinear_mathematica\fortran*cone_k0L_*.txt')
l2 = glob.glob(r'.\nonlinear_mathematica\fortran*cone_kG_*.txt')
l3 = glob.glob(r'.\nonlinear_mathematica\fortran*cone_kLL_*.txt')

for filepath in (l1 + l2 + l3):
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
        names[3] = sub_matrix
        filename = '_'.join(names) + '.txt'
    with open('.\\nonlinear_sparse\\' + filename, 'w') as f:
        f.write(print_str)


