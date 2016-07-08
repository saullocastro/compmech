import numpy as np

from compmech.panel import Panel
from compmech.analysis import Analysis
from compmech.sparse import solve


def test_kT():
    mns = [[4, 4], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6],
           [8, 9], [9, 8]]
    for m, n in mns:
        for model in ['plate_clt_donnell_bardell',
                      'cpanel_clt_donnell_bardell']:
            p = Panel()
            p.model = model
            p.bc_ssss()
            p.w1tx = 0
            p.u1tx = 1
            p.u1ty = 1
            p.u2ty = 1
            p.a = 2.
            p.b = 0.5
            p.r = 10
            p.stack = [0, 90, -45, +45]
            p.plyt = 1e-3*0.125
            p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
            p.m = m
            p.n = n
            p.nx = m
            p.ny = n

            P = 1000.
            npts = 5
            p.forces_inc = []
            for y in np.linspace(0, p.b, npts):
                p.forces_inc.append([0., y, P/(npts-1.), 0, 0])
            p.forces_inc[0][2] /= 2.
            p.forces_inc[-1][2] /= 2.

            k0 = p.calc_k0(silent=True)
            kT = p.calc_kT(c=np.zeros(p.get_size()), silent=True)

            error = np.abs(kT-k0).sum()

            assert error < 1.e-7


def test_fint():
    m = 6
    n = 6
    for model in ['plate_clt_donnell_bardell',
                  #'cpanel_clt_donnell_bardell'
                  ]:
        p = Panel()
        p.model = model
        p.bc_ssss()
        p.w1tx = 0
        p.w1rx = 1
        p.u1tx = 1
        p.u1ty = 1
        p.u2ty = 1
        p.a = 2.
        p.b = 0.5
        p.r = 1.e9
        p.stack = [0, 90, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.nx = m
        p.ny = n
        p.m = m
        p.n = n

        P = 1000.
        npts = 100
        p.forces_inc = []
        for y in np.linspace(0, p.b, npts):
            p.forces_inc.append([0., y, P/(npts-1.), 0, 0])
        p.forces_inc[0][2] /= 2.
        p.forces_inc[-1][2] /= 2.

        if False:
            fext = p.calc_fext(silent=True)
            k0 = p.calc_k0(silent=True)
            c = solve(k0, fext)
            fint = p.calc_fint(c=c)
            kT = p.calc_kT(c=c, silent=True)

            np.savetxt('debug_fint_from_kT.txt', kT*c)
            np.savetxt('debug_fint.txt', fint)
            np.savetxt('debug_fext.txt', fext)

        else:
            p.analysis.line_search = False
            p.static(NLgeom=True)






def test():
    #p.plot(cs[0], filename='linear_static.png', colorbar=True)

    fext = p.calc_fext()

    c = solve(k0, fext)
    kL = p.calc_k0(c=c)
    kG = p.calc_kG0(c=c)
    kT = kL + kG

    print('k0.sum()', k0.sum())
    print('kT.sum()', kT.sum())

    print('fext.sum()', fext.sum())

    fint = p.calc_fint(c=c)
    #fint = p.calc_kT(c=c)*c
    print('fint.sum()', fint.sum())

    np.savetxt('debug_fint.txt', fint)
    np.savetxt('debug_fext.txt', fext)

    R = -fint + fext
    print('error.sum()', R.sum())
    dc = solve(kT, R)
    c += dc

    #fint = p.calc_kT(c=c)*c
    fint = p.calc_fint(c=c)
    print('dc.sum()', dc.sum())
    print('fint.sum()', fint.sum())

    R = -fint + fext

    print('\nerror.sum()', R.sum())
    dc = solve(kT, R)
    c += dc
    print('c.sum()', c.sum())
    print('dc.sum()', dc.sum())

    #fint = p.calc_kT(c=c)*c
    fint = p.calc_fint(c=c)
    print('fint.sum()', fint.sum())
    R = -fint + fext
    print('\nerror.sum()', R.sum())
    dc = solve(kT, R)
    c += dc
    print('c.sum()', c.sum())
    print('dc.sum()', dc.sum())

    #fint = p.calc_kT(c=c)*c
    fint = p.calc_fint(c=c)
    print('fint.sum()', fint.sum())
    R = -fint + fext
    print('\nerror.sum()', R.sum())
    dc = solve(kT, R)
    c += dc
    print('c.sum()', c.sum())
    print('dc.sum()', dc.sum())

    #fint = p.calc_kT(c=c)*c
    fint = p.calc_fint(c=c)
    print('fint.sum()', fint.sum())
    R = -fint + fext
    print('\nerror.sum()', R.sum())


    if False:
        p.analysis.initialInc = 0.5
        p.analysis.line_search = False
        c = p.static(NLgeom=True, silent=True)

    print(c)

    p.plot(c, filename='non_linear_analysis.png', colorbar=True)


if __name__ == '__main__':
    #test_kT()
    test_fint()
