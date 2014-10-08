import os
import glob

import numpy as np
import sympy
from sympy import pi, sin, cos, var, Matrix

var('i1, k1, i2, j2, k2, l2', integer=True)
var('x, t, xa, xb, L, r, r2, sina, cosa')
var('A11, A12, A16, A22, A26, A66, A44, A45, A55')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('wx, wt, w0x, w0t, Nxx, Ntt, Nxt, tLA')

var('p0000, p0001, p0002, p0003, p0004, p0005, p0006, p0007, p0008, p0009, p0010, p0011, p0012, p0013')
var('p0100, p0101, p0102, p0103, p0104, p0105, p0106, p0107, p0108, p0109, p0110, p0111, p0112, p0113')
var('p0200, p0201, p0202, p0203, p0204, p0205, p0206, p0207, p0208, p0209, p0210, p0211, p0212, p0213')
var('p0300, p0301, p0302, p0303, p0304, p0305, p0306, p0307, p0308, p0309, p0310, p0311, p0312, p0313')
var('p0400, p0401, p0402, p0403, p0404, p0405, p0406, p0407, p0408, p0409, p0410, p0411, p0412, p0413')
var('p0500, p0501, p0502, p0503, p0504, p0505, p0506, p0507, p0508, p0509, p0510, p0511, p0512, p0513')
var('p0600, p0601, p0602, p0603, p0604, p0605, p0606, p0607, p0608, p0609, p0610, p0611, p0612, p0613')
var('p0700, p0701, p0702, p0703, p0704, p0705, p0706, p0707, p0708, p0709, p0710, p0711, p0712, p0713')
var('p0800, p0801, p0802, p0803, p0804, p0805, p0806, p0807, p0808, p0809, p0810, p0811, p0812, p0813')
var('p0900, p0901, p0902, p0903, p0904, p0905, p0906, p0907, p0908, p0909, p0910, p0911, p0912, p0913')
var('p1000, p1001, p1002, p1003, p1004, p1005, p1006, p1007, p1008, p1009, p1010, p1011, p1012, p1013')
var('p1100, p1101, p1102, p1103, p1104, p1105, p1106, p1107, p1108, p1109, p1110, p1111, p1112, p1113')
var('p1200, p1201, p1202, p1203, p1204, p1205, p1206, p1207, p1208, p1209, p1210, p1211, p1212, p1213')
var('p1300, p1301, p1302, p1303, p1304, p1305, p1306, p1307, p1308, p1309, p1310, p1311, p1312, p1313')

var('q0000, q0001, q0002, q0003, q0004, q0005, q0006, q0007, q0008, q0009, q0010, q0011, q0012, q0013')
var('q0100, q0101, q0102, q0103, q0104, q0105, q0106, q0107, q0108, q0109, q0110, q0111, q0112, q0113')
var('q0200, q0201, q0202, q0203, q0204, q0205, q0206, q0207, q0208, q0209, q0210, q0211, q0212, q0213')
var('q0300, q0301, q0302, q0303, q0304, q0305, q0306, q0307, q0308, q0309, q0310, q0311, q0312, q0313')
var('q0400, q0401, q0402, q0403, q0404, q0405, q0406, q0407, q0408, q0409, q0410, q0411, q0412, q0413')
var('q0500, q0501, q0502, q0503, q0504, q0505, q0506, q0507, q0508, q0509, q0510, q0511, q0512, q0513')
var('q0600, q0601, q0602, q0603, q0604, q0605, q0606, q0607, q0608, q0609, q0610, q0611, q0612, q0613')
var('q0700, q0701, q0702, q0703, q0704, q0705, q0706, q0707, q0708, q0709, q0710, q0711, q0712, q0713')
var('q0800, q0801, q0802, q0803, q0804, q0805, q0806, q0807, q0808, q0809, q0810, q0811, q0812, q0813')
var('q0900, q0901, q0902, q0903, q0904, q0905, q0906, q0907, q0908, q0909, q0910, q0911, q0912, q0913')
var('q1000, q1001, q1002, q1003, q1004, q1005, q1006, q1007, q1008, q1009, q1010, q1011, q1012, q1013')
var('q1100, q1101, q1102, q1103, q1104, q1105, q1106, q1107, q1108, q1109, q1110, q1111, q1112, q1113')
var('q1200, q1201, q1202, q1203, q1204, q1205, q1206, q1207, q1208, q1209, q1210, q1211, q1212, q1213')
var('q1300, q1301, q1302, q1303, q1304, q1305, q1306, q1307, q1308, q1309, q1310, q1311, q1312, q1313')

p = Matrix([[p0000, p0001, p0002, p0003, p0004, p0005, p0006, p0007, p0008, p0009, p0010, p0011, p0012, p0013],
            [p0100, p0101, p0102, p0103, p0104, p0105, p0106, p0107, p0108, p0109, p0110, p0111, p0112, p0113],
            [p0200, p0201, p0202, p0203, p0204, p0205, p0206, p0207, p0208, p0209, p0210, p0211, p0212, p0213],
            [p0300, p0301, p0302, p0303, p0304, p0305, p0306, p0307, p0308, p0309, p0310, p0311, p0312, p0313],
            [p0400, p0401, p0402, p0403, p0404, p0405, p0406, p0407, p0408, p0409, p0410, p0411, p0412, p0413],
            [p0500, p0501, p0502, p0503, p0504, p0505, p0506, p0507, p0508, p0509, p0510, p0511, p0512, p0513],
            [p0600, p0601, p0602, p0603, p0604, p0605, p0606, p0607, p0608, p0609, p0610, p0611, p0612, p0613],
            [p0700, p0701, p0702, p0703, p0704, p0705, p0706, p0707, p0708, p0709, p0710, p0711, p0712, p0713],
            [p0800, p0801, p0802, p0803, p0804, p0805, p0806, p0807, p0808, p0809, p0810, p0811, p0812, p0813],
            [p0900, p0901, p0902, p0903, p0904, p0905, p0906, p0907, p0908, p0909, p0910, p0911, p0912, p0913],
            [p1000, p1001, p1002, p1003, p1004, p1005, p1006, p1007, p1008, p1009, p1010, p1011, p1012, p1013],
            [p1100, p1101, p1102, p1103, p1104, p1105, p1106, p1107, p1108, p1109, p1110, p1111, p1112, p1113],
            [p1200, p1201, p1202, p1203, p1204, p1205, p1206, p1207, p1208, p1209, p1210, p1211, p1212, p1213],
            [p1300, p1301, p1302, p1303, p1304, p1305, p1306, p1307, p1308, p1309, p1310, p1311, p1312, p1313]])

pstr = np.array([str(i) for i in p]).reshape(14, 14)

q = Matrix([[q0000, q0001, q0002, q0003, q0004, q0005, q0006, q0007, q0008, q0009, q0010, q0011, q0012, q0013],
            [q0100, q0101, q0102, q0103, q0104, q0105, q0106, q0107, q0108, q0109, q0110, q0111, q0112, q0113],
            [q0200, q0201, q0202, q0203, q0204, q0205, q0206, q0207, q0208, q0209, q0210, q0211, q0212, q0213],
            [q0300, q0301, q0302, q0303, q0304, q0305, q0306, q0307, q0308, q0309, q0310, q0311, q0312, q0313],
            [q0400, q0401, q0402, q0403, q0404, q0405, q0406, q0407, q0408, q0409, q0410, q0411, q0412, q0413],
            [q0500, q0501, q0502, q0503, q0504, q0505, q0506, q0507, q0508, q0509, q0510, q0511, q0512, q0513],
            [q0600, q0601, q0602, q0603, q0604, q0605, q0606, q0607, q0608, q0609, q0610, q0611, q0612, q0613],
            [q0700, q0701, q0702, q0703, q0704, q0705, q0706, q0707, q0708, q0709, q0710, q0711, q0712, q0713],
            [q0800, q0801, q0802, q0803, q0804, q0805, q0806, q0807, q0808, q0809, q0810, q0811, q0812, q0813],
            [q0900, q0901, q0902, q0903, q0904, q0905, q0906, q0907, q0908, q0909, q0910, q0911, q0912, q0913],
            [q1000, q1001, q1002, q1003, q1004, q1005, q1006, q1007, q1008, q1009, q1010, q1011, q1012, q1013],
            [q1100, q1101, q1102, q1103, q1104, q1105, q1106, q1107, q1108, q1109, q1110, q1111, q1112, q1113],
            [q1200, q1201, q1202, q1203, q1204, q1205, q1206, q1207, q1208, q1209, q1210, q1211, q1212, q1213],
            [q1300, q1301, q1302, q1303, q1304, q1305, q1306, q1307, q1308, q1309, q1310, q1311, q1312, q1313]])

qstr = np.array([str(i) for i in q]).reshape(14, 14)

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
    for filepath in glob.glob(r'.\nonlinear_mathematica\fortran_cone*'+sufix+r'*.txt'):
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
        matrix_name = names[2] # k0L, kGNL or kLL
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

                if left=='0':
                    tmp += 'rows[c] = {}\n'.format(row)
                else:
                    tmp += 'rows[c] = row+{}\n'.format(row)

                if right=='0':
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


