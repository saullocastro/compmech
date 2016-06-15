import numpy as np

from compmech.stiffpanelbay import StiffPanelBay
from compmech.analysis import lb


def test_lb_without_defect():
    print('Testing linear buckling')

    spb = StiffPanelBay()
    spb.a = 2
    spb.b = 1
    spb.stack = [0, +45 -45, 90, -45, +45, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'plate_clt_donnell_bardell'
    spb.m = 16
    spb.n = 16

    Nxx = -50.
    spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt, Nxx=Nxx)
    spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt, Nxx=Nxx)

    bb = spb.b/10.
    stiff = spb.add_bladestiff2d(ys=spb.b/2., bf=bb/2, bb=bb,
                     fstack=[0, 90, 90, 0]*8,
                     fplyt=spb.plyt*1., flaminaprop=spb.laminaprop,
                     bstack=[0, 90, 90, 0]*4,
                     bplyt=spb.plyt*1., blaminaprop=spb.laminaprop,
                     m1=16, n1=16, m2=16, n2=16)

    spb.calc_k0()
    spb.calc_kG0()
    eigvals, eigvecs = lb(spb.k0, spb.kG0, silent=True)

    calc = eigvals[0]*Nxx

    assert np.isclose(calc, -400.939130626, atol=0.001)

if __name__ == '__main__':
    test_lb_without_defect()
