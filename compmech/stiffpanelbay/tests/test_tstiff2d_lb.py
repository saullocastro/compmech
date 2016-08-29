import numpy as np

from compmech.stiffpanelbay import StiffPanelBay
from compmech.analysis import lb


def test_tstiff2d_lb():
    #NOTE reference values taken from
    # compmech.panel.assembly.tstiff2d_1stiff_compression
    a_b_list = [0.5, 2., 10.]
    ref_values = [-152.607275551, -80.15391295315146, -79.39137361313862]
    for a_b, ref_value in zip(a_b_list, ref_values):
        print('Testing linear buckling')
        spb = StiffPanelBay()
        spb.b = 1.
        spb.a = spb.b * a_b
        spb.stack = [0, +45, -45, 90, -45, +45]
        spb.plyt = 1e-3*0.125
        spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        spb.model = 'plate_clt_donnell_bardell'
        spb.m = 12
        spb.n = 13

        Nxx = -10.
        spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt, Nxx=Nxx)
        spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt, Nxx=Nxx)

        bb = spb.b/5.
        if False:
            stiff = spb.add_tstiff2d(ys=spb.b/2., bf=bb/2, bb=bb,
                             fstack=[0, 90, 90, 0]*8,
                             fplyt=spb.plyt*1., flaminaprop=spb.laminaprop,
                             bstack=[0, 90, 90, 0]*4,
                             bplyt=spb.plyt*1., blaminaprop=spb.laminaprop,
                             mb=13, nb=12, mf=13, nf=12, Nxxf=Nxx)

        spb.calc_k0()
        spb.calc_kG0()
        eigvals, eigvecs = lb(spb.k0, spb.kG0, silent=True)

        calc = eigvals[0]*Nxx

        assert np.isclose(calc, ref_value, rtol=0.05)

