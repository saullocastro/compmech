import numpy as np

from compmech.panel import Panel


def test_lb():
    for model in ['plate_clt_donnell_bardell',
                  'plate_clt_donnell_bardell_w',
                  'cpanel_clt_donnell_bardell']:
        # ssss
        p = Panel()
        p.bc_ssss()
        p.m = 12
        p.n = 13
        p.stack = [0, 90, -45, +45]
        p.plyt = 0.125e-3
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.model = model
        p.a = 1.
        p.b = 0.5
        p.r = 1.e8
        p.Nxx = -1
        p.lb(silent=True)
        assert np.allclose(p.eigvals[0], 85.2911727144)

        p.Nxx = 0
        p.Nyy = -1
        p.lb(silent=True)
        assert np.allclose(p.eigvals[0], 25.1756170679)

        # ssfs
        p = Panel()
        p.bc_ssfs()
        p.m = 12
        p.n = 13
        p.stack = [0, 90, -45, +45]
        p.plyt = 0.125e-3
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)

        p.model = model
        p.a = 1.
        p.b = 0.5
        p.Nxx = -1
        p.lb(silent=True)
        assert np.allclose(p.eigvals[0], 15.8423562314)

        p.bc_sfss()
        p.Nxx = 0
        p.Nyy = -1
        p.lb(silent=True)
        assert np.allclose(p.eigvals[0], 13.9421987614)


if __name__ == '__main__':
    test_lb()
