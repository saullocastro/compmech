import numpy as np

from compmech.stiffpanelbay import StiffPanelBay
from compmech.analysis import static


def test_static_with_TStiff2D():
    print('Testing static analysis with TStiff2D')
    spb = StiffPanelBay()
    spb.a = 2.
    spb.b = 1.
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'cpanel_clt_donnell_bardell'
    spb.m = 17
    spb.n = 16
    spb.r = 25.

    spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt)
    spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt)

    bb = spb.b/5.
    bf = bb
    stiff = spb.add_tstiff2d(ys=spb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*4,
                     fplyt=spb.plyt*1., flaminaprop=spb.laminaprop,
                     bstack=[0, 90, 90, 0]*8,
                     bplyt=spb.plyt*1., blaminaprop=spb.laminaprop,
                     m1=13, n1=13, m2=13, n2=12)

    stiff.forces_flange.append([spb.a/2., bf, 0., 0., 1000.])

    k0 = spb.calc_k0()
    fext = spb.calc_fext()
    inc, cs = static(spb.k0, fext, silent=True)

    wpanelmin = spb.uvw_skin(cs[0])[2].min()
    wbasemin = spb.uvw_stiffener(cs[0], 0, region='base')[2].min()
    wflangemax = spb.uvw_stiffener(cs[0], 0, region='flange')[2].max()
    assert np.isclose(wpanelmin, -0.119762615184, atol=1.e-6)
    assert np.isclose(wbasemin, -0.0993165706363, atol=1.e-6)
    assert np.isclose(wflangemax, 4.29278723868, atol=1.e-2)
    if False:
        spb.plot_skin(cs[0], filename='skin.png', colorbar=True, vec='w')
        spb.plot_stiffener(cs[0], si=0, region='base', filename='stiff_base.png',
                colorbar=True, vec='w')
        spb.plot_stiffener(cs[0], si=0, region='flange',
                filename='stiff_flange.png', colorbar=True)


if __name__ == '__main__':
    test_static_with_TStiff2D()
