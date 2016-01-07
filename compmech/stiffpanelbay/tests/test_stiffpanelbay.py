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

    assert np.isclose(sp.eigvals[0].real, 295.35629419)


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

    assert np.isclose(sp.eigvals[0].real, 81.9342050889)


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
    sp.m = 11
    sp.n = 12

    sp.add_panel(y1=0, y2=sp.b/2., plyt=sp.plyt)
    sp.add_panel(y1=sp.b/2., y2=sp.b, plyt=sp.plyt)

    sp.add_bladestiff2d(ys=sp.b/2., m1=14, n1=11, bf=0.08,
                        fstack=[0, 90, 90, 0]*5, fplyt=sp.plyt,
                        flaminaprop=sp.laminaprop)

    sp.freq(silent=True, atype=4)

    assert np.isclose(sp.eigvals[0].real, 138.51917530043477)


def test_Lee_and_Lee_table4():
    print('Testing Lee and Lee Table 4')
    # Lee and Lee. "Vibration analysis of anisotropic plates with eccentric
    #    stiffeners". Computers & Structures, Vol. 57, No. 1, pp. 99-105,
    #    1995.
    models = (
        ('model4', 0.00208, 0.0060, 138.801067988),
        ('model5', 0.00260, 0.0075, 174.624343202),
        ('model7', 0.00364, 0.0105, 205.433509024))
    for model, hf, bf, value in models:
        spb = StiffPanelBay()
        spb.model = 'plate_clt_donnell_bardell'
        spb.mu = 1.500e3 # plate material density in kg / m^3
        spb.laminaprop = (128.e9, 11.e9, 0.25, 4.48e9, 1.53e9, 1.53e9)
        spb.stack = [0, -45, +45, 90, 90, +45, -45, 0]
        plyt = 0.00013
        spb.plyt = plyt
        spb.a = 0.5
        spb.b = 0.250
        spb.m = 14
        spb.n = 15
        hf = hf
        bf = bf
        n = int(hf/plyt)
        fstack = [0]*(n//4) + [90]*(n//4) + [90]*(n//4) + [0]*(n//4)
        spb.w1rx = 0.
        spb.w2rx = 0.
        spb.w1ry = 0.
        spb.w2ry = 0.

        spb.add_panel(y1=0, y2=spb.b/2.)
        spb.add_panel(y1=spb.b/2., y2=spb.b)
        spb.add_bladestiff1d(mu=spb.mu, ys=spb.b/2., bb=0., bf=bf,
                      fstack=fstack, fplyt=plyt, flaminaprop=spb.laminaprop)
        spb.freq(atype=4, silent=True, reduced_dof=False)

        assert np.isclose(spb.eigvals[0].real/2/np.pi, value)


if __name__ == '__main__':
    test_freq()
    test_lb_Stiffener1D()
    test_lb_Stiffener2D()
    test_freq_Stiffener1D()
    test_freq_Stiffener2D()
    test_Lee_and_Lee_table4()
