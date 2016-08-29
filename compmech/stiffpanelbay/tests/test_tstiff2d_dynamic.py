import numpy as np

from compmech.stiffpanelbay import StiffPanelBay
from compmech.analysis import freq


def test_dynamic_with_TStiff2D():
    print('Testing dynamic analysis with TStiff2D')
    spb = StiffPanelBay()
    spb.a = 2.
    spb.b = 1.
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'plate_clt_donnell_bardell'
    spb.m = 12
    spb.n = 13
    spb.mu = 1.3e3

    spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt)
    spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt)

    bb = spb.b/5.
    bf = bb
    stiff = spb.add_tstiff2d(ys=spb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*2,
                     fplyt=spb.plyt*1., flaminaprop=spb.laminaprop,
                     bstack=[0, 90, 90, 0]*1,
                     bplyt=spb.plyt*1., blaminaprop=spb.laminaprop,
                     mb=12, nb=13, mf=12, nf=13)

    k0 = spb.calc_k0()
    M = spb.calc_kM()
    eigvals, eigvecs = freq(k0, M, silent=True)

    #assert np.isclose(eigvals[0], 10.19522215600389, atol=0.01)
