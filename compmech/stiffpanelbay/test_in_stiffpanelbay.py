import numpy as np

from compmech.stiffpanelbay import StiffPanelBay

def test_freq():
    print('Testing frequency analysis for StiffPanelBay')
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


if __name__ == '__main__':
    test_freq()




