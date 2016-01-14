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
                    assert np.allclose(p.eigvals[0], 19.9271684726)
                else:
                    assert np.allclose(p.eigvals[0], 17.8587479369)
            elif atype == 4:
                if '_w' in model:
                    assert np.allclose(p.eigvals[0], 40.3728103572)
                else:
                    assert np.allclose(p.eigvals[0], 39.3147553173)

if __name__ == '__main__':
    test_panel_freq()
