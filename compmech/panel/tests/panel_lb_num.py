import numpy as np

from compmech.panel import Panel


def test_panel_fkG0y1y2_num():
    for model in ['plate_clt_donnell_bardell']:
        print('Checking fkG0y1y2_num for model {0}'.format(model))
        # ssss
        p = Panel()
        p.bc_ssss()
        p.a = 5.
        p.b = 3.
        p.stack = [0, 90, 90, 0, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.model = 'plate_clt_donnell_bardell'

        p.Nxx = -1.

        p.m = 10
        p.n = 11
        p.nx = 40
        p.ny = 40

        p.u1ty = 1
        p.u2ty = 1
        p.u2tx = 1

        num = 100.
        for i in range(int(num)):
            if i == 0 or i == num - 2:
                fx = p.Nxx*p.b/(num-1.)/2.
            else:
                fx = p.Nxx*p.b/(num-1.)
            p.add_force(x=p.a, y=i*p.b/(num-1.), fx=fx, fy=0., fz=0.,
                        cte=True)
        p.static()
        p.plot(p.analysis.cs[0], vec='w', filename='test_w.png', colorbar=True)
        p.plot(p.analysis.cs[0], vec='u', filename='test_u.png', colorbar=True)

        p.lb(silent=False)
        p.plot(p.eigvecs[:, 0], filename='test_lb.png')

        p.lb(silent=False, c=p.analysis.cs[0])
        p.plot(p.eigvecs[:, 0], filename='test_lb2.png')


if __name__ == '__main__':
    test_panel_fkG0y1y2_num()
