import numpy as np

from compmech.stiffpanelbay import StiffPanelBay
from compmech.analysis import freq

def test_bladestiff1d_dynamic():
    m = 9
    n = 10
    for flow in ['x', 'y']:
        for model in ['plate_clt_donnell_bardell',
                      'cpanel_clt_donnell_bardell',
                      'kpanel_clt_donnell_bardell',
                      ]:
            sb = StiffPanelBay()
            sb.flow = 'y'
            sb.model = model

            sb.mu = 1.500e3 # plate material density in kg / m^3
            E1 = 31*6.89475729e9 # Pa
            E2 = 2.7*6.89475729e9 # Pa
            G = 0.75*6.89475729e9 # Pa
            nu = 0.28
            laminaprop = (E1, E2, nu, G, G, G)
            stack = [-45, +45, 0, 0, +45, -45]
            a = 2.
            b = 2.
            sb.a = a
            sb.r = 1.e5
            sb.alphadeg = 0.
            sb.b = b
            sb.m = m
            sb.n = n
            h = a/20.
            plyt = h/len(stack)
            sb.add_panel(0, b/2., stack=stack, plyt=plyt, laminaprop=laminaprop,
                    model=sb.model, mu=sb.mu, m=m, n=n)
            sb.add_panel(b/2., b, stack=stack, plyt=plyt, laminaprop=laminaprop,
                    model=sb.model, mu=sb.mu, m=m, n=n)
            bf = 6.*h
            fstack = [-45, +45, 0, 0, +45, -45]
            fplyts = [plyt for i in fstack]
            stifflaminaprop = laminaprop
            flaminaprops = [stifflaminaprop for i in fstack]
            sb.add_bladestiff1d(mu=sb.mu, ys=0.5*b, bb=0., bf=bf,
                           bstack=None, bplyts=None, blaminaprops=None,
                           fstack=fstack, fplyts=fplyts, flaminaprops=flaminaprops)

            beta = 0
            sb.Mach = 2.
            sb.speed_sound = 343.
            sb.rho_air = 0.3
            sb.V = ((sb.Mach**2 - 1)**0.5 * beta / sb.rho_air)**0.5
            k0 = sb.calc_k0(silent=True)
            #NOTE no piston theory for conical panels
            if not 'kpanel' in model:
                kA = sb.calc_kA(silent=True)
            else:
                kA = 0
            kM = sb.calc_kM(silent=True)
            eigvals, eigvecs = freq((k0 + kA), kM, silent=True,
                    sparse_solver=False)

            filename = 'tmp_test_bladestiff1d_dynamic_%s_%s.png' % (model, flow)
            sb.plot_skin(eigvecs[:, 0], filename=filename)
            assert np.isclose(eigvals[0], 1873.45244832, rtol=0.001)
