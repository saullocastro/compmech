import numpy as np
from sympy import simplify, Matrix, var

xi = var('xi')
eta = var('eta')

print('Chebyshev Polynomials of the Second Kind')
u = [1, 2*xi]
for i in range(2, 15):
    ui = simplify(2*xi*u[i-1] - u[i-2])
    u.append(ui)
for i, ui in enumerate(u):
    print('u({0}) = {1}'.format(i, ui))

with open('cheb.txt', 'w') as f:
    f.write('Number of terms: {0}\n\n'.format(len(u)))
    f.write(','.join(map(str, u)).replace('**','^') + '\n\n')
    f.write(','.join(map(str, u)).replace('xi', 'eta').replace('**','^'))

print np.outer(u, [ui.subs({xi:eta}) for ui in u])

if False:
    m = Matrix([[2*xi, 1, 0, 0],
                [1, 2*xi, 1, 0],
                [0, 1, 2*xi, 1],
                [0, 0, 1, 2*xi]])

    print('m.det() {0}'.format(simplify(m.det())))

