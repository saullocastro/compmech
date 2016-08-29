import numpy as np

from compmech.panel import Panel
from compmech.analysis import freq


def test_panel_freq():
    for model in ['plate_clt_donnell_bardell',
                  'plate_clt_donnell_bardell_w',
                  'cpanel_clt_donnell_bardell',
                  'kpanel_clt_donnell_bardell']:
        for prestress in [True, False]:
            print('Frequency Analysis, prestress={0}, model={1}'.format(
                  prestress, model))
            p = Panel()
            p.model = model
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
            k0 = p.calc_k0(silent=True)
            M = p.calc_kM(silent=True)
            if prestress:
                kG0 = p.calc_kG0(silent=True)
                k0 += kG0
            eigvals, eigvecs = freq(k0, M, sparse_solver=True, reduced_dof=False, silent=True)
            if prestress:
                if '_w' in model:
                    assert np.isclose(eigvals[0], 19.9272, rtol=0.001)
                else:
                    assert np.isclose(eigvals[0], 17.85875, rtol=0.001)
            else:
                if '_w' in model:
                    assert np.isclose(eigvals[0], 40.37281, rtol=0.001)
                else:
                    assert np.isclose(eigvals[0], 39.31476, rtol=0.001)


def test_reduced_dof_freq_plate():
    models = ['plate_clt_donnell_bardell',
              'cpanel_clt_donnell_bardell']
    for model in models:
        print('Test reduced_dof solver, prestress=True, model={0}'.format(model))
        p = Panel()
        p.model = model
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
        k0 = p.calc_k0(silent=True)
        M = p.calc_kM(silent=True)
        kG0 = p.calc_kG0(silent=True)
        k0 += kG0
        eigvals, eigvecs = freq(k0, M, sparse_solver=True, reduced_dof=False, silent=True)
        reduced_false = eigvals[0]
        freq(k0, M, sparse_solver=True, reduced_dof=True, silent=True)
        reduced_true = eigvals[0]
        assert np.isclose(reduced_false, reduced_true, rtol=0.001)


if __name__ == '__main__':
    test_panel_freq()
    test_reduced_dof_freq_plate()
