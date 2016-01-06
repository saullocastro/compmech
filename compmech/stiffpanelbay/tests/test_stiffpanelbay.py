import numpy as np

from compmech.stiffpanelbay import StiffPanelBay

def test_freq():
    print('Testing frequency analysis for StiffPanelBay with 2 panels')
    sp = StiffPanelBay()
    sp.a = 1.
    sp.b = 0.5
    sp.r = sp.b*10
    sp.stack = [0, 90, -45, +45]
    sp.plyt = 1e-3*0.125
    sp.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    sp.model = 'cpanel_clt_donnell_bardell'
    sp.mu = 1.3e3
    sp.m = 11
    sp.n = 12

    sp.add_panel(0, sp.b/2., plyt=sp.plyt, Nxx=-1)
    sp.add_panel(sp.b/2., sp.b, plyt=sp.plyt, Nxx=-1)

    sp.freq(sparse_solver=False, silent=True)

    assert np.allclose(sp.eigvals[0], 155.648180838)


def test_lb_Stiffener1D():
    print('Testing linear buckling for StiffPanelBay with a 1D Stiffener')
    sp = StiffPanelBay()
    sp.a = 1.
    sp.b = 0.5
    sp.stack = [0, 90, 90, 0]
    sp.plyt = 1e-3*0.125
    sp.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    sp.model = 'plate_clt_donnell_bardell'
    sp.mu = 1.3e3
    sp.m = 15
    sp.n = 16

    sp.add_panel(y1=0, y2=sp.b/2., plyt=sp.plyt, Nxx=-1.)
    sp.add_panel(y1=sp.b/2., y2=sp.b, plyt=sp.plyt, Nxx_cte=1000.)

    sp.add_bladestiff1d(ys=sp.b/2., Fx=0., bf=0.05, fstack=[0, 90, 90, 0],
            fplyt=sp.plyt, flaminaprop=sp.laminaprop)

    sp.lb(silent=True)

    assert np.isclose(sp.eigvals[0].real, 1929.50283797)


def test_lb_Stiffener2D():
    print('Testing linear buckling for StiffPanelBay with a 2D Stiffener')
    sp = StiffPanelBay()
    sp.a = 1.
    sp.b = 0.5
    sp.stack = [0, 90, 90, 0]
    sp.plyt = 1e-3*0.125
    sp.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    sp.model = 'plate_clt_donnell_bardell'
    sp.mu = 1.3e3
    sp.m = 15
    sp.n = 16

    sp.add_panel(y1=0, y2=sp.b/2., plyt=sp.plyt, Nxx=-1.)
    sp.add_panel(y1=sp.b/2., y2=sp.b, plyt=sp.plyt, Nxx_cte=1000.)

    sp.add_bladestiff2d(ys=sp.b/2., m1=14, n1=11, bf=0.05,
                        fstack=[0, 90, 90, 0],
                        fplyt=sp.plyt, flaminaprop=sp.laminaprop)

    sp.lb(silent=True)

    assert np.isclose(sp.eigvals[0].real, 299.162436099)


def test_freq_Stiffener1D():
    print('Testing frequency analysis for StiffPanelBay with a 1D Stiffener')
    sp = StiffPanelBay()
    sp.a = 1.
    sp.b = 0.5
    sp.stack = [0, 90, 90, 0]
    sp.plyt = 1e-3*0.125
    sp.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    sp.model = 'plate_clt_donnell_bardell'
    sp.mu = 1.3e3
    sp.m = 15
    sp.n = 16

    sp.add_panel(y1=0, y2=sp.b/2., plyt=sp.plyt)
    sp.add_panel(y1=sp.b/2., y2=sp.b, plyt=sp.plyt)

    sp.add_bladestiff1d(ys=sp.b/2., Fx=0., bf=0.08, fstack=[0, 90, 90, 0]*5,
            fplyt=sp.plyt, flaminaprop=sp.laminaprop)

    sp.freq(silent=True, atype=4)

    assert np.isclose(sp.eigvals[0].real, 850.402645325)


def test_freq_Stiffener2D():
    print('Testing frequency analysis for StiffPanelBay with a 2D Stiffener')
    sp = StiffPanelBay()
    sp.a = 1.
    sp.b = 0.5
    sp.stack = [0, 90, 90, 0]
    sp.plyt = 1e-3*0.125
    sp.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    sp.model = 'plate_clt_donnell_bardell'
    sp.mu = 1.3e3
    sp.m = 15
    sp.n = 16

    sp.add_panel(y1=0, y2=sp.b/2., plyt=sp.plyt)
    sp.add_panel(y1=sp.b/2., y2=sp.b, plyt=sp.plyt)

    sp.add_bladestiff2d(ys=sp.b/2., m1=14, n1=11, bf=0.08,
                        fstack=[0, 90, 90, 0]*5, fplyt=sp.plyt,
                        flaminaprop=sp.laminaprop)

    sp.freq(silent=True, atype=4)

    assert np.isclose(sp.eigvals[0].real, 161.159295942)


if __name__ == '__main__':
    test_freq()
    test_lb_Stiffener1D()
    test_lb_Stiffener2D()
    test_freq_Stiffener1D()
    test_freq_Stiffener2D()

