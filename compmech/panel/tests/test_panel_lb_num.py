import numpy as np

from compmech.panel import Panel


def test_panel_fkG0_num():
    for model in ['plate_clt_donnell_bardell']:
        print('Checking fkG0_num for model {0}'.format(model))
        # ssss
        p = Panel()
        p.bc_ssss()
        p.a = 8.
        p.b = 4.
        p.stack = [0, 90, 90, 0, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.model = 'plate_clt_donnell_bardell'

        p.Nxx = -1.

        p.m = 11
        p.n = 12
        p.nx = 9
        p.ny = 9

        p.u1ty = 1
        p.u2ty = 1
        p.u2tx = 1

        p.v1tx = 0
        p.v2tx = 1
        p.v1ty = 1
        p.v2ty = 1

        num = 1000.
        f = 0.
        for i in range(int(num)):
            if i == 0 or i == num - 2:
                fx = p.Nxx*p.b/(num-1.)/2.
            else:
                fx = p.Nxx*p.b/(num-1.)
            p.add_force(x=p.a, y=i*p.b/(num-1.), fx=fx, fy=0., fz=0.,
                        cte=True)
            f += fx

        p.static(silent=True)

        p.lb(silent=True)
        assert np.isclose(p.eigvals[0], 4.506284, atol=0.1, rtol=0)

        p.lb(silent=True, c=p.analysis.cs[0])
        assert np.isclose(p.eigvals[0], 4.4375669, atol=0.1, rtol=0)


if __name__ == '__main__':
    test_panel_fkG0_num()
