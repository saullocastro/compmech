import numpy as np

from compmech.panel import Panel


def test_kT():
    mns = [[4, 4], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6],
           [8, 9], [9, 8]]
    for m, n in mns:
        for model in ['plate_clt_donnell_bardell',
                      'cpanel_clt_donnell_bardell']:
            p = Panel()
            p.model = model
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
                  'cpanel_clt_donnell_bardell'
                  ]:
        p = Panel()
        p.model = model
        p.w1tx = 0
        p.w1rx = 1
        p.u1tx = 1
        p.u1ty = 1
        p.u2ty = 1
        p.a = 2.
        p.b = 1.
        p.r = 1.e5
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
        p.forces.append([p.a/2., p.b/2., 0, 0, 0.001])

        p.static(NLgeom=True, silent=True)
        c = p.analysis.cs[0]

        p.uvw(c)
        assert np.isclose(p.w.max(), 0.000144768080125, rtol=0.001)


if __name__ == '__main__':
    test_kT()
    test_fint()
