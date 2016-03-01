import numpy as np

from compmech.stiffpanelbay import StiffPanelBay


def test_lb():
    print('Testing linear buckling')
    spb = StiffPanelBay()
    spb.a = 1.
    spb.b = 0.5
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'cpanel_clt_donnell_bardell'
    spb.m = 14
    spb.n = 12
    spb.r = 100.

    spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt, Nxx=-1000.)
    spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt, Nxx=-1000.)

    spb.add_tstiff2d(ys=spb.b/2., bf=0.05, bb=0.05,
                     fstack=[0, 90, 90, 0],
                     fplyt=spb.plyt, flaminaprop=spb.laminaprop,
                     bstack=[0, 90, 90, 0],
                     bplyt=spb.plyt, blaminaprop=spb.laminaprop)
                     #x1=spb.a/2., x2=3*spb.a/4.)

    spb.lb(silent=True)

    spb.plot_skin(spb.eigvecs[:, 0], filename='tmp.png')

    print spb.eigvals[0]
    #assert np.isclose(spb.eigvals[0].real, 301.0825234, atol=0.1, rtol=0)


#def test_freq_Stiffener1D():
    #print('Testing frequency analysis')
    #spb = StiffPanelBay()
    #spb.a = 1.
    #spb.b = 0.5
    #spb.stack = [0, 90, 90, 0]
    #spb.plyt = 1e-3*0.125
    #spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    #spb.model = 'cpanel_clt_donnell_bardell'
    #spb.mu = 1.3e3
    #spb.m = 15
    #spb.n = 16
#
    #spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt)
    #spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt)
#
    #spb.add_bladestiff1d(ys=spb.b/2., Fx=0., bf=0.08, fstack=[0, 90, 90, 0]*5,
            #fplyt=spb.plyt, flaminaprop=spb.laminaprop)
#
    #spb.freq(silent=True, atype=4)
#
    #assert np.isclose(spb.eigvals[0].real, 81.9342, atol=0.1, rtol=0)
#

if __name__ == '__main__':
    test_lb()
