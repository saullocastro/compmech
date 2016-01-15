import numpy as np

from compmech.panel import Panel


def test_panel_freq():
    for model in ['plate_clt_donnell_bardell',
                  'plate_clt_donnell_bardell_w',
                  'cpanel_clt_donnell_bardell',
                  'kpanel_clt_donnell_bardell']:
        for atype in [3, 4]:
            print('Frequency Analysis, atype={0}, model={1}'.format(
                  atype, model))
            p = Panel()
            p.model = model
            p.bc_ssss()
            p.a = 1.
            p.b = 0.5
            p.r = 1.e8
            p.alphadeg = 0.
            p.stack = [0, 90, -45, +45]
            p.plyt = 1e-3*0.125
            p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
            p.mu = 1.3e3
            p.m = 11
            p.n = 12
            p.Nxx = -60.
            p.Nyy = -5.
            p.freq(sparse_solver=True, reduced_dof=False, silent=True,
                   atype=atype)
            if atype == 3:
                if '_w' in model:
                    assert np.isclose(p.eigvals[0], 19.9272, atol=0.1, rtol=0)
                else:
                    assert np.isclose(p.eigvals[0], 17.85875, atol=0.1, rtol=0)
            elif atype == 4:
                if '_w' in model:
                    assert np.isclose(p.eigvals[0], 40.37281, atol=0.1, rtol=0)
                else:
                    assert np.isclose(p.eigvals[0], 39.31476, atol=0.1, rtol=0)


def test_reduced_dof_freq_plate():
    models = ['plate_clt_donnell_bardell',
              'cpanel_clt_donnell_bardell']
    for model in models:
        atype = 3
        print('Test reduced_dof solver, atype={0}, model={1}'.format(atype, model))
        p = Panel()
        p.model = model
        p.bc_ssss()
        p.a = 1.
        p.b = 0.5
        p.r = 100.
        p.alphadeg = 0.
        p.stack = [0, 90, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.mu = 1.3e3
        p.m = 11
        p.n = 12
        p.Nxx = -60.
        p.Nyy = -5.
        p.freq(sparse_solver=True, reduced_dof=False, silent=True,
               atype=atype)
        reduced_false = p.eigvals[0]
        p.freq(sparse_solver=True, reduced_dof=True, silent=True,
               atype=atype)
        reduced_true = p.eigvals[0]
        assert np.isclose(reduced_false, reduced_true, atol=0.0001)


if __name__ == '__main__':
    test_panel_freq()
    test_reduced_dof_freq_plate()
