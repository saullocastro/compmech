import numpy as np

from compmech.panel import Panel


def test_lb():
    for model in ['plate_clt_donnell_bardell',
                  'plate_clt_donnell_bardell_w',
                  'cpanel_clt_donnell_bardell']:
        print('Linear buckling for model {0}'.format(model))
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
        if '_w' in model:
            assert np.allclose(p.eigvals[0], 88.4769618837)
        else:
            assert np.allclose(p.eigvals[0], 85.2911727144)

        p.Nxx = 0
        p.Nyy = -1
        p.lb(silent=True)
        if '_w' in model:
            assert np.allclose(p.eigvals[0], 26.4588171556)
        else:
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
        p.r = 1.e8
        p.Nxx = -1
        p.lb(silent=True)
        if '_w' in model:
            assert np.allclose(p.eigvals[0], 17.1442703121)
        else:
            assert np.allclose(p.eigvals[0], 15.8423562314)

        p.bc_sfss()
        p.Nxx = 0
        p.Nyy = -1
        p.lb(silent=True)
        if '_w' in model:
            assert np.allclose(p.eigvals[0], 15.8099861083)
        else:
            assert np.allclose(p.eigvals[0], 13.9421987614)


def test_freq():
    for model in ['plate_clt_donnell_bardell',
                  'plate_clt_donnell_bardell_w',
                  'cpanel_clt_donnell_bardell']:
        print('Frequency Analysis for model {0}'.format(model))
        p = Panel()
        p.model = model
        p.bc_ssss()
        p.a = 1.
        p.b = 0.5
        p.r = 1.e8#p.b*10
        p.stack = [0, 90, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.mu = 1.3e3
        p.m = 11
        p.n = 12
        p.freq(sparse_solver=True, reduced_dof=False, silent=False)
        p.plot(p.eigvecs[:, 0], filename='test_freq_{0}.png'.format(model))
        #assert np.allclose(p.eigvals[0], 281.9019611355)


if __name__ == '__main__':
    test_lb()
    test_freq()
