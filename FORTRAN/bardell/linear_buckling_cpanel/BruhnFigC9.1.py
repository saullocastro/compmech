#!/nfs/cae/Ferramentas/EXEC/PYTHON/default/bin/python
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


M = 25
N = 15

E = 71.e9
nu = 0.33
Nxx = -1.
Nyy = 0.
Nxy = 0.
r = 10.
Zs = np.logspace(0, 5, num=50)
for r_t in [500, 3000]:
    kcs = []
    for Z in Zs:
        t = r / r_t
        b = (Z*r*t/(1 - nu**2)**0.5)**0.5
        a = 10*b
        print('a = {0}'.format(a))
        print('b = {0}'.format(b))
        print('t = {0}'.format(t))

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
            Nxxcr = float(f.readline().strip())*abs(Nxx)
            Fccr = Nxxcr/t
        kc = Fccr/np.pi**2/E*b**2/t**2*12*(1-nu**2)
        kcs.append(kc)
    plt.plot(Zs, kcs, label='$R/t = {r_t:d}$'.format(r_t=r_t))
plt.legend(loc='lower right')
plt.ylim(1., 1000.)
plt.xscale('log')
plt.yscale('log')
plt.savefig(filename='BruhnFigC9.1.png', bbox_inches='tight')


