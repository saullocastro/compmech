import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

from mapy.sympytools.matrixtools import mprint_as_sparse

var('i1, k1, i2, j2, k2, l2', integer=True)
var('x, t, xa, xb, L, r, r2, sina, cosa')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('phix, phit, Nxx, Ntt, Nxt')

var('p00, p01, p02, p03, p04, p05, p06, p07, p08, p09')
var('p10, p11, p12, p13, p14, p15, p16, p17, p18, p19')
var('p20, p21, p22, p23, p24, p25, p26, p27, p28, p29')
var('p30, p31, p32, p33, p34, p35, p36, p37, p38, p39')
var('p40, p41, p42, p43, p44, p45, p46, p47, p48, p49')
var('p50, p51, p52, p53, p54, p55, p56, p57, p58, p59')
var('p60, p61, p62, p63, p64, p65, p66, p67, p68, p69')
var('p70, p71, p72, p73, p74, p75, p76, p77, p78, p79')
var('p80, p81, p82, p83, p84, p85, p86, p87, p88, p89')
var('p90, p91, p92, p93, p94, p95, p96, p97, p98, p99')

var('q00, q01, q02, q03, q04, q05, q06, q07, q08, q09')
var('q10, q11, q12, q13, q14, q15, q16, q17, q18, q19')
var('q20, q21, q22, q23, q24, q25, q26, q27, q28, q29')
var('q30, q31, q32, q33, q34, q35, q36, q37, q38, q39')
var('q40, q41, q42, q43, q44, q45, q46, q47, q48, q49')
var('q50, q51, q52, q53, q54, q55, q56, q57, q58, q59')
var('q60, q61, q62, q63, q64, q65, q66, q67, q68, q69')
var('q70, q71, q72, q73, q74, q75, q76, q77, q78, q79')
var('q80, q81, q82, q83, q84, q85, q86, q87, q88, q89')
var('q90, q91, q92, q93, q94, q95, q96, q97, q98, q99')

p = Matrix([[p00, p01, p02, p03, p04, p05, p06, p07, p08, p09],
            [p10, p11, p12, p13, p14, p15, p16, p17, p18, p19],
            [p20, p21, p22, p23, p24, p25, p26, p27, p28, p29],
            [p30, p31, p32, p33, p34, p35, p36, p37, p38, p39],
            [p40, p41, p42, p43, p44, p45, p46, p47, p48, p49],
            [p50, p51, p52, p53, p54, p55, p56, p57, p58, p59],
            [p60, p61, p62, p63, p64, p65, p66, p67, p68, p69],
            [p70, p71, p72, p73, p74, p75, p76, p77, p78, p79],
            [p80, p81, p82, p83, p84, p85, p86, p87, p88, p89],
            [p90, p91, p92, p93, p94, p95, p96, p97, p98, p99]])
pstr = np.array([str(i) for i in p]).reshape(10,10)

q = Matrix([[q00, q01, q02, q03, q04, q05, q06, q07, q08, q09],
            [q10, q11, q12, q13, q14, q15, q16, q17, q18, q19],
            [q20, q21, q22, q23, q24, q25, q26, q27, q28, q29],
            [q30, q31, q32, q33, q34, q35, q36, q37, q38, q39],
            [q40, q41, q42, q43, q44, q45, q46, q47, q48, q49],
            [q50, q51, q52, q53, q54, q55, q56, q57, q58, q59],
            [q60, q61, q62, q63, q64, q65, q66, q67, q68, q69],
            [q70, q71, q72, q73, q74, q75, q76, q77, q78, q79],
            [q80, q81, q82, q83, q84, q85, q86, q87, q88, q89],
            [q90, q91, q92, q93, q94, q95, q96, q97, q98, q99]])
qstr = np.array([str(i) for i in q]).reshape(10,10)

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
    for filepath in glob.glob(r'.\numerical\fortran_cone*'+sufix+r'*.txt'):
        print filepath
        with open(filepath) as f:
            print_str = ''
            filename = os.path.basename(filepath)
            names = filename[:-4].split('_') # removing .txt
            matrix_name = names[2] # k0Lp, k0Lq, kGNLp, kGNLq, kLLp or kLLq
            sub_matrix = names[3][i] # 0, 1, or 2
            key = matrix_name[:-1] + '_' + names[3]
            if not key in valid.keys():
                valid[key] = set()
            lines = [line.strip() for line in f.readlines()]
            string = ''.join(lines)
            #string = string.replace('List','')
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
        with open('.\\numerical_printing\\' + filename, 'w') as f:
            f.write(print_str)

for filepath in glob.glob(r'.\numerical\fortran_cone*L_*.txt'):
    print filepath
    with open(filepath) as f:
        print_str = ''
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_') # removing .txt
        matrix_name = names[2] # k0L, kGNL or kLL
        sub_matrix = names[3] # 00, 01, 02, 11, 12 or 22
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
                tmp += 'rows[c] = row+{}\n'.format(row)
                tmp += 'cols[c] = col+{}\n'.format(col)
                tmp += 'out[c] = beta*out[c] + alpha*({1})\n'.format(
                        matrix_name, value)
        print_str += '{0}_{1} with {2} non-null terms\n'.format(
                matrix_name, sub_matrix, count)
        print_str += tmp
        names[3] = sub_matrix
        filename = '_'.join(names) + '.txt'
    with open('.\\numerical_printing\\' + filename, 'w') as f:
        f.write(print_str)


