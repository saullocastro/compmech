from __future__ import division
import numpy as np
from sympy import var, factorial, factorial2, sympify, diff

from compmech.conecyl.sympytools import star2Cpow

nmax = 30

xi = var('xi')

u = map(sympify, ['1./2. - 3./4.*xi + 1./4.*xi**3',
                  '1./8. - 1./8.*xi - 1./8.*xi**2 + 1./8.*xi**3',
                  '1./2. + 3./4.*xi - 1./4.*xi**3',
                  '-1./8. - 1./8.*xi + 1./8.*xi**2 + 1./8.*xi**3'])

for r in range(5, nmax+1):
    utmp = []
    for n in range(0, r//2+1):
        den = 2**n*factorial(n)*factorial(r-2*n-1)
        utmp.append((-1)**n*factorial2(2*r - 2*n - 7)/den * xi**(r-2*n-1)/1.)
    u.append(sum(utmp))

with open('../../../C/bardell/bardell_functions.c', 'w') as f:
    f.write("// Bardell's hierarchical functions\n\n")
    f.write('// Number of terms: {0}\n\n'.format(len(u)))
    f.write('#include <stdlib.h>\n')
    f.write('#include <math.h>\n\n')
    f.write('void calc_vec_f(double *f, double xi, double xi1t, double xi1r,\n' +
            '                double xi2t, double xi2r) {\n')
    consts = {0:'xi1t', 1:'xi1r', 2:'xi2t', 3:'xi2r'}
    for i in range(len(u)):
        const = consts.get(i)
        if const is None:
            f.write('    f[%d] = %s;\n' % (i, str(u[i])))
        else:
            f.write('    f[%d] = %s*(%s);\n' % (i, const, str(u[i])))
    f.write('}\n')

    f.write('\n\n')
    f.write('void calc_vec_fxi(double *fxi, double xi, double xi1t, double xi1r,\n' +
            '                  double xi2t, double xi2r) {\n')
    for i in range(len(u)):
        const = consts.get(i)
        if const is None:
            f.write('    fxi[%d] = %s;\n' % (i, str(diff(u[i], xi))))
        else:
            f.write('    fxi[%d] = %s*(%s);\n' % (i, const, star2Cpow(str(diff(u[i], xi)))))
    f.write('}\n')

    f.write('\n\n')

    f.write('double calc_f(int i, double xi, double xi1t, double xi1r,\n' +
            '              double xi2t, double xi2r) {\n')
    f.write('    switch(i) {\n')
    for i in range(len(u)):
        const = consts.get(i)
        f.write('    case %d:\n' % i)
        if const is None:
            f.write('        return %s;\n' % str(u[i]))
        else:
            f.write('        return %s*(%s);\n' % (const, star2Cpow(str(u[i]))))
    f.write('    }\n')
    f.write('}\n')

    f.write('\n\n')
    f.write('double calc_fxi(int i, double xi, double xi1t, double xi1r,\n' +
            '                double xi2t, double xi2r) {\n')
    f.write('    switch(i) {\n')
    for i in range(len(u)):
        const = consts.get(i)
        f.write('    case %d:\n' % i)
        if const is None:
            f.write('        return %s;\n' % star2Cpow(str(diff(u[i], xi))))
        else:
            f.write('        return %s*(%s);\n' % (const, star2Cpow(str(diff(u[i], xi)))))
    f.write('    }\n')
    f.write('}\n')

with open('../../../C/bardell/bardell_functions.h', 'w') as g:
    g.write('void calc_vec_f(double *f, double xi, double xi1t, double xi1r,\n' +
            '                double xi2t, double xi2r);\n')
    g.write('\n')
    g.write('void calc_vec_fxi(double *fxi, double xi, double xi1t, double xi1r,\n' +
            '                  double xi2t, double xi2r);\n')
    g.write('\n')
    g.write('double calc_f(int i, double xi, double xi1t, double xi1r,\n' +
            '              double xi2t, double xi2r);\n')
    g.write('\n')
    g.write('double calc_fxi(int i, double xi, double xi1t, double xi1r,\n' +
            '                double xi2t, double xi2r);\n')
