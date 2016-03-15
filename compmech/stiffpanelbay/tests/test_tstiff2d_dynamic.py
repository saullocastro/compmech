import numpy as np

from compmech.stiffpanelbay import StiffPanelBay


def test_dynamic_with_TStiff2D():
    print('Testing dynamic analysis with TStiff2D')
    spb = StiffPanelBay()
    spb.a = 2.
    spb.b = 1.
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'plate_clt_donnell_bardell'
    spb.m = 10
    spb.n = 10
    spb.mu = 1.3e3

    spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt)
    spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt)

    bb = spb.b/5.
    bf = bb
    stiff = spb.add_tstiff2d(ys=spb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*1,
                     fplyt=spb.plyt*1., flaminaprop=spb.laminaprop,
                     bstack=[0, 90, 90, 0]*1,
                     bplyt=spb.plyt*1., blaminaprop=spb.laminaprop,
                     m1=10, n1=10, m2=10, n2=10)

    spb.freq(atype=4, silent=False, sparse_solver=False)

    if True:
        spb.plot_skin(spb.eigvecs[:, 0], filename='skin.png', colorbar=True,
                vec='w')
        spb.plot_stiffener(spb.eigvecs[:, 0], si=0, region='base',
                filename='stiff_base.png', colorbar=True, vec='w')
        spb.plot_stiffener(spb.eigvecs[:, 0], si=0, region='flange',
                filename='stiff_flange.png', colorbar=True)


def __test_dynamic_with_damaged_TStiff2D():
    print('Testing dynamic analysis with damaged TStiff2D')
    spb = StiffPanelBay()
    spb.a = 2.
    spb.b = 1.
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'cpanel_clt_donnell_bardell'
    spb.m = 11
    spb.n = 11
    spb.r = 25.

    spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt)
    spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt)

    bb = spb.b/5.
    bf = bb
    stiff = spb.add_tstiff2d(ys=spb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*2,
                     fplyt=spb.plyt*1., flaminaprop=spb.laminaprop,
                     bstack=[0, 90, 90, 0]*1,
                     bplyt=spb.plyt*1., blaminaprop=spb.laminaprop,
                     m1=13, n1=13, m2=13, n2=12,
                     x1=0.2*spb.a, x2=0.5*spb.a)

    stiff.forces_flange.append([spb.a/2., bf, 0., 0., 1000.])

    spb.freq(silent=True)

    if True:
        spb.plot_skin(cs[0], filename='skin.png', colorbar=True, vec='w')
        spb.plot_stiffener(cs[0], si=0, region='base', filename='stiff_base.png',
                colorbar=True, vec='w')
        spb.plot_stiffener(cs[0], si=0, region='flange',
                filename='stiff_flange.png', colorbar=True)


if __name__ == '__main__':
    test_dynamic_with_TStiff2D()
    #test_dynamic_with_damaged_TStiff2D()
