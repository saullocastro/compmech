import numpy as np

from compmech.panel import Panel


def test_panel_aero():
    for flow in ['x', 'y']:
        for model in ['plate_clt_donnell_bardell',
                      'plate_clt_donnell_bardell_w',
                      'cpanel_clt_donnell_bardell']:
            for atype in [1, 2]:
                print('Flutter Analysis Piston Theory, flow={0}, atype={1}, model={2}'.
                      format(flow, atype, model))
                p = Panel()
                p.model = model
                if flow == 'x':
                    p.a = 1.
                    p.b = 0.5
                else:
                    p.a = 0.5
                    p.b = 1.
                p.r = 1.e8
                p.alphadeg = 0.
                p.stack = [0, 90, -45, +45]
                p.plyt = 1e-3*0.125
                E2 = 8.7e9
                p.laminaprop = (142.5e9, E2, 0.28, 5.1e9, 5.1e9, 5.1e9)
                p.mu = 1.5e3
                p.m = 8
                p.n = 9
                p.flow = flow

                # pre-stress applied when atype == 1
                if flow == 'x':
                    p.Nxx = -60.
                    p.Nyy = -5.
                else:
                    p.Nxx = -5.
                    p.Nyy = -60.

                # testing commong methodology based on betastar
                if atype == 1:
                    betasstar = np.linspace(150, 350, 40)
                elif atype == 2:
                    betasstar = np.linspace(670, 690, 40)
                betas = betasstar/(p.a**3/E2/(len(p.stack)*p.plyt)**3)
                p.beta = betas[0]
                p.freq(atype=1, reduced_dof=False, sparse_solver=False,
                       silent=True)
                out = np.zeros((len(betasstar), p.eigvals.shape[0]),
                        dtype=p.eigvals.dtype)
                for i, beta in enumerate(betas):
                    p.beta = beta
                    p.freq(atype=2, reduced_dof=False, sparse_solver=False,
                           silent=True)
                    eigvals = p.eigvals*p.a**2/(np.pi**2*sum(p.plyts))*np.sqrt(p.mu/E2)
                    out[i, :] = eigvals

                ind = np.where(np.any(out.imag != 0, axis=1))[0][0]
                if flow == 'x':
                    if atype == 1:
                        if not model.endswith('_w'):
                            assert np.isclose(betas[ind], 347.16346, atol=0.1, rtol=0)
                        else:
                            assert np.isclose(betas[ind], 174.27885, atol=0.1, rtol=0)
                    elif atype == 2:
                        if not model.endswith('_w'):
                            assert np.isclose(betas[ind], 728.625, atol=0.1, rtol=0)
                        else:
                            assert np.isclose(betas[ind], 728.625, atol=0.1, rtol=0)
                else:
                    if atype == 1:
                        if not model.endswith('_w'):
                            assert np.isclose(betas[ind], 1305.0, atol=0.1, rtol=0)
                        else:
                            assert np.isclose(betas[ind], 1305.0, atol=0.1, rtol=0)
                    elif atype == 2:
                        if not model.endswith('_w'):
                            assert np.isclose(betas[ind], 5829.0, atol=0.1, rtol=0)
                        else:
                            assert np.isclose(betas[ind], 5829.0, atol=0.1, rtol=0)

if __name__ == '__main__':
    out = test_panel_aero()
