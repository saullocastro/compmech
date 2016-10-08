import numpy as np

from compmech.panel import Panel
from compmech.analysis import Analysis
from compmech.sparse import solve

def test_panel_field_outputs():
    m = 7
    n = 6
    #TODO implement for conical panels
    strain_field = dict(exx=None, eyy=None, gxy=None, kxx=None, kyy=None, kxy=None)
    stress_field = dict(Nxx=None, Nyy=None, Nxy=None, Mxx=None, Myy=None, Mxy=None)
    for model in ['plate_clt_donnell_bardell',
                  'cpanel_clt_donnell_bardell']:
        p = Panel()
        p.model = model
        p.u1tx = 1
        p.u1ty = 1
        p.u2ty = 1
        p.v1tx = 0
        p.v2tx = 0
        p.v1ty = 0
        p.v2ty = 0

        p.a = 2.
        p.b = 1.
        p.r = 1.e5
        p.stack = [0, -45, +45, 90, +45, -45, 0, 0]
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

        p.static()
        c = p.analysis.cs[0]
        Ns = p.stress(c, gridx=50, gridy=50)
        es = p.strain(c, gridx=50, gridy=50)
        for k, v in strain_field.items():
            if v is None:
                strain_field[k] = es.get(k).min()
            else:
                assert np.isclose(strain_field[k], es.get(k).min(), rtol=0.05)
            p.plot(c, vec=k, filename='tmp_test_panel_strain_field_%s.png' % k)
        for k, v in stress_field.items():
            if v is None:
                stress_field[k] = Ns.get(k).min()
            else:
                assert np.isclose(stress_field[k], Ns.get(k).min(), rtol=0.05)
            p.plot(c, vec=k, filename='tmp_test_panel_stress_field_%s.png' % k)
