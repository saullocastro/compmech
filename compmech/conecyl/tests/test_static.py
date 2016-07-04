import numpy as np

from compmech.conecyl import ConeCyl


def test_static():
    wmin_ref = [
            ['clpt_donnell_bc1', -0.0512249327106],
            ['clpt_donnell_bc2', -0.0104494223857],
            ['clpt_donnell_bc3', -0.0509280039584],
            ['clpt_donnell_bc4', -0.0104423789402],
            ['fsdt_donnell_bc1', -0.0516948646563],
            ['fsdt_donnell_bc2', -0.01048277484],
            ['fsdt_donnell_bc3', -0.0513959737413],
            ['fsdt_donnell_bc4', -0.0104756892015],
            ]

    for model, wmin in wmin_ref:
        cc = ConeCyl()
        cc.model = model
        cc.m1 = 20
        cc.m2 = 10
        cc.n2 = 11
        cc.name = 'Z33'
        cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
        cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
        cc.plyt = 0.125
        cc.r2 = 250.
        cc.H = 510.
        cc.add_SPL(10)
        for thetadeg in np.linspace(0, 360, 300, endpoint=False):
            cc.add_force(0., thetadeg, -15., 0, 0, increment=True)
        cs = cc.static()
        cc.uvw(cs[0])
        assert np.isclose(cc.w.min(), wmin, rtol=0.01)


def test_NL_static():
    wmin_ref = [
            ['clpt_donnell_bc1', -0.0477604792756],
            ['clpt_donnell_bc2', -0.0466197517352],
            ['clpt_donnell_bc3', -0.047461024091],
            ['clpt_donnell_bc4', -0.0463451053441],
            ['fsdt_donnell_bc1', -0.0463340957349],
            ]

    for model, wmin in wmin_ref:
        cc = ConeCyl()
        cc.model = model
        cc.m1 = 20
        cc.m2 = 10
        cc.n2 = 11
        cc.name = 'Z33'
        cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
        cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
        cc.plyt = 0.125
        cc.r2 = 250.
        cc.H = 510.
        cc.add_SPL(10, increment=False)
        for thetadeg in np.linspace(0, 360, 300, endpoint=False):
            cc.add_force(0., thetadeg, -15., 0, 0, increment=True)
        cc.analysis.initialInc = 0.5
        cs = cc.static(NLgeom=True)
        cc.uvw(cs[0])
        assert np.isclose(cc.w.min(), wmin, rtol=0.01)


if __name__ == '__main__':
    test_static()
    test_NL_static()
