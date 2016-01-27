#!/nfs/cae/Ferramentas/EXEC/PYTHON/default/bin/python
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


M = 25
N = 15

E = 71e9
nu = 0.33
t = 1.e-3
Nxx = 0.
Nyy = 0.
Nxy = 100.
Zs = np.logspace(0, 3, 50)
for a_b in [1., 1.5, 2., 3., 5.]:
    kss = []
    for Z in Zs:
        r = 5.
        b = (Z*r*t/(1 - nu**2)**0.5)**0.5
        a = a_b*b
        print('a = {0}'.format(a))
        print('b = {0}'.format(b))

        text = """
M
{M:d}
N
{N:d}

a
{a:f}
b
{b:f}
r
{r:f}

Nxx
{Nxx:f}
Nyy
{Nyy:f}
Nxy
{Nxy:f}

ISOTROPIC
t
{t:f}
E
{E:f}
nu
{nu:f}

u1tx
0.
u1rx
1.
u2tx
0.
u2rx
1.

u1ty
0.
u1ry
1.
u2ty
0.
u2ry
1.

v1tx
0.
v1rx
1.
v2tx
0.
v2rx
1.

v1ty
0.
v1ry
1.
v2ty
0.
v2ry
1.

w1tx
0.
w1rx
1.
w2tx
0.
w2rx
1.

w1ty
0.
w1ry
1.
w2ty
0.
w2ry
1.

END
""".format(M=M, N=N, a=a, b=b, r=r, Nxx=Nxx, Nyy=Nyy, Nxy=Nxy, t=t, E=E, nu=nu)

        with open('input.txt', 'wb') as f:
            f.write(text)
        os.system('buckling_cpanel_bardell input.txt out.txt')
        with open('out.txt', 'rb') as f:
            Nxycr = float(f.readline().strip())*abs(Nxy)
            Fscr = Nxycr/t
        ks = Fscr/np.pi**2/E*b**2/t**2*12*(1-nu**2)
        kss.append(ks)
    plt.plot(Zs, kss, label='$a/b = {a_b:1.1f}$'.format(a_b=a_b))
plt.legend(loc='lower right')
plt.ylim(1., 200.)
plt.xscale('log')
plt.yscale('log')
plt.savefig(filename='BruhnFigC9.2.png', bbox_inches='tight')


