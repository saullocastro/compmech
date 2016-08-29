import numpy as np

from compmech.stiffpanelbay import StiffPanelBay
from compmech.analysis import static


def test_static_with_BladeStiff2D():
    print('Testing linear statics with BladeStiff2D')

    spb = StiffPanelBay()
    spb.a = 2.
    spb.b = 1.
    spb.m = 12
    spb.n = 13
    spb.stack = [0, +45, -45, 90, -45, +45, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)

    spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt)
    spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt)

    bb = spb.b/5.
    bf = bb
    stiff = spb.add_bladestiff2d(ys=spb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*8,
                     fplyt=spb.plyt*1., flaminaprop=spb.laminaprop,
                     bstack=[0, 90, 90, 0]*4,
                     bplyt=spb.plyt*1., blaminaprop=spb.laminaprop,
                     mf=17, nf=16)
    stiff.flange.forces.append([stiff.flange.a/2., stiff.flange.b, 0., 0., 1000.])

    spb.calc_k0()
    fext = spb.calc_fext()
    inc, cs = static(spb.k0, fext, silent=True)

    wpanelmin = spb.uvw_skin(cs[0])[2].min()
    #NOTE repeated call on purpose to evaluate if cs[0] is being messed up
    #     somewhere
    wpanelmin = spb.uvw_skin(cs[0])[2].min()
    wflangemax = spb.uvw_stiffener(cs[0], 0, region='flange')[2].max()
    #NOTE repeated call on purpose
    wflangemax = spb.uvw_stiffener(cs[0], 0, region='flange')[2].max()
    assert np.isclose(wpanelmin, -0.30581458201781481, atol=1.e-4, rtol=0.001)
    assert np.isclose(wflangemax, 0.331155797371884, atol=1.e-4, rtol=0.001)
    spb.plot_skin(cs[0], filename='tmp_test_bladestiff2d_static_skin.png', colorbar=True, vec='w', clean=False)
    spb.plot_stiffener(cs[0], si=0, region='flange',
            filename='tmp_test_bladestiff2d_stiff_static_flange.png', colorbar=True, clean=False)
