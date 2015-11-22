from __future__ import division
import numpy as np
from sympy import var, factorial, factorial2, sympify

nmax = 30

xi = var('xi')

u = map(sympify, ['1./2. - 3./4.*xi + 1./4.*xi**3',
                  '1./8. - 1./8.*xi - 1./8.*xi**2 + 1./8.*xi**3',
                  '1./2. + 3./4.*xi - 1./4.*xi**3',
                  '-1./8. - 1./8.*xi + 1./8.*xi**2 + 1./8.*xi**3'])

for r in range(5, nmax+1):
    utmp = []
    for n in range(0, r//2+1):
        den = 2.**n*factorial(n)*factorial(r-2*n-1)
        utmp.append((-1)**n*factorial2(2*r - 2*n - 7)/den * xi**(r-2*n-1))
    u.append(sum(utmp))

with open('bardell_floating_point.txt', 'w') as f:
    f.write("Bardell's hierarchical functions\n\n")
    f.write('Number of terms: {0}\n\n'.format(len(u)))
    for i in range(len(u)):
         f.write('fxi[%d] = %s\n' % (i, str(u[i])))

