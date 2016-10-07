import numpy as np

from compmech.panel import Panel
from compmech.analysis import Analysis
from compmech.sparse import solve

def test_panel_field_outputs():
    m = 7
    n = 6
    #TODO implement for conical panels
    store = dict(exx=None, eyy=None, gxy=None, kxx=None, kyy=None, kxy=None,
            Nxx=None, Nyy=None, Nxy=None, Mxx=None, Myy=None, Mxy=None)
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
        Ns = p.stress(c)
        store.get('exx') = es[0]
        store.get('eyy') = es[1]
        store.get('gxy') = es[2]
        store.get('kxx') = es[3]
        store.get('kyy') = es[4]
        store.get('kxy') = es[5]
        store.get('Nxx') = Ns[..., 0]
        store.get('Nyy') = Ns[..., 1]
        store.get('Nxy') = Ns[..., 2]
        store.get('Mxx') = Ns[..., 3]
        store.get('Myy') = Ns[..., 4]
        store.get('Mxy') = Ns[..., 5]
        es = p.strain(c)
        for k, v in store.items():
            p.plot(c, vec=k, filename='tmp_test_panel_field_%s.png' % k)
        #TODO include assertions
