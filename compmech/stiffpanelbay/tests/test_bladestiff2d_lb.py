import numpy as np

from compmech.stiffpanelbay import StiffPanelBay
from compmech.analysis import lb


def test_bladestiff2d_lb():
    print('Testing linear buckling with BladeStiff2D')

    spb = StiffPanelBay()
    spb.a = 2.
    spb.b = 1.
    spb.stack = [0, +45, -45, 90, -45, +45]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'plate_clt_donnell_bardell'
    spb.m = 17
    spb.n = 16

    Nxx = -50.
    spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt, Nxx=Nxx)
    spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt, Nxx=Nxx)

    bb = spb.b/5.
    bf = bb
    stiff = spb.add_bladestiff2d(ys=spb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*8,
                     fplyt=spb.plyt*1., flaminaprop=spb.laminaprop,
                     bstack=[0, 90, 90, 0]*4,
                     bplyt=spb.plyt*1., blaminaprop=spb.laminaprop,
                     mf=17, nf=16)

    spb.calc_k0()
    spb.calc_kG0()
    eigvals, eigvecs = lb(spb.k0, spb.kG0, silent=True)

    spb.plot_skin(eigvecs[:, 0], filename='tmp_test_bladestiff2d_lb_skin.png',
            colorbar=True)
    spb.plot_stiffener(eigvecs[:, 0], si=0, region='flange',
            filename='tmp_test_bladestiff2d_lb_stiff_flange.png', colorbar=True)

    calc = eigvals[0]*Nxx

    spb.plot_skin(eigvecs[:, 0], filename='tmp_test_bladestiff2d_lb_skin.png', colorbar=True, vec='w', clean=False)
    assert np.isclose(calc, -759.05689868085778, atol=0.0001, rtol=0.001)

