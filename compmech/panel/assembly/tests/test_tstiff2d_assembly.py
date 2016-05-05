import numpy as np

from compmech.panel.assembly.tstiff2d_assembly import tstiff2d_1stiff


def test_tstiff2d_1stiff():
    b = 1.
    bb = b/5.
    bf = bb/2.
    ys = b/2.
    assy, eigvals, eigvecs = tstiff2d_1stiff(
        b=b,
        bb=bb,
        bf=bf,
        a=3.,
        ys=ys,
        deffect_a=0.1,
        mu=1.3e3,
        plyt=0.125e-3,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        stack_skin=[0, 45, -45, 90, -45, 45, 0],
        stack_base=[0, 90, 0]*4,
        stack_flange=[0, 90, 0]*8,
        m=6, n=6,
        )
    assert np.isclose(eigvals[0], 48.55740345+0.j)
