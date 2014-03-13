from numpy import hstack, vstack
import sympy
from sympy import sin, cos, pi, Matrix

from desicos.theories.clpt.matrices_chebyshev import calc_matrices

sympy.var('x, t, cosa, r', commutative=False)
cgen = sympy.numbered_symbols(prefix='c', start=0, commutative=False)

m1 = 10
m2 = 10
n2 = 10

# g0
g0 = Matrix([[(1/cosa)*(2-x)/2],
             [0],
             [0]])

# g1
fx0 = 2*(x-1)*(x+1)
fx1 = 2*x*(2*(x-1)*(x+1))
g1 = Matrix([[fx0,0,0, fx1,0,0],
             [0,fx0,0, 0,fx1,0],
             [0,0,fx0, 0,0,fx1]])
for i in range(3, m1+1):
    pn0 = g1[:,-6:-3]
    pn1 = g1[:,-3:]
    pn2 = sympy.expand(2*x*pn1 - pn0)
    g1 = Matrix(hstack((g1, pn2)))

# g2
ft0 = 1
ft1 = t
g2 = Matrix([[],[],[]])
for j in range(1, 2*(n2+1)):
    if j == 1:
        ft2 = ft0
    elif j == 2:
        ft2 = ft1
    else:
        ft2 = 2*t*ft1 - ft0
        ft0, ft1 = ft1, ft2
    fx0 = 2*(x-1)*(x+1)
    fx1 = 2*x*(2*(x-1)*(x+1))
    for i in range(1, m2+1):
        if i == 1:
            fx2 = fx0
        elif i == 2:
            fx2 = fx1
        else:
            fx2 = 2*x*fx1 - fx0
            fx0, fx1 = fx1, fx2
        if ((j+1) % 2)==0:
            g2 = Matrix(hstack((g2, Matrix([[fx2*ft2,0,0],
                                            [0,fx2*ft2,0],
                                            [0,0,fx2*ft2]]))))
g2 = sympy.expand(g2)

g = Matrix(hstack([g0, g1, g2]))
print 'g.shape', g.shape
c = Matrix(vstack([cgen.next() for i in range(g.shape[1])]))

matrices = calc_matrices(c, g,
        prefix='print_derivations', NL_kinematics='donnell', analytical=True)

