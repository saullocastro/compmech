import numpy as np

import modelDB
import compmech.panel.modelDB as panmodelDB


class BladeStiff1d(object):
    r"""Blade Stiffener using 1D Formulation for Flange

    Blade-type of stiffener model using a 1D formulation for the flange and a
    2D formulation for the padup (base)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panels
         Panel1      Panel2

    Both the flange and the padup are optional.

    Each stiffener has a constant `y` coordinate.

    """
    def __init__(self, bay, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.model = 'bladestiff1d_clt_donnell_bardell'
        self.mu = None
        self.ys = ys
        self.bb = bb
        self.hb = 0.
        self.bf = bf
        self.hf = 0.

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops
        self.fstack = fstack
        self.fplyts = fplyts
        self.flaminaprops = flaminaprops
        self.blam = None
        self.flam = None

        self.As = None
        self.Asf = None
        self.Jxx = None
        self.Iyy = None

        self._rebuild()


    def _rebuild(self):
        if self.fstack != []:
            self.hf = sum(self.fplyts)
            self.flam = laminate.read_stack(self.fstack, plyts=self.fplyts,
                                             laminaprops=self.flaminaprops)
            self.flam.calc_equivalent_modulus()

        h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
        if self.bstack != []:
            hb = sum(self.bplyts)
            self.blam = laminate.read_stack(self.bstack, plyts=self.bplyts,
                                            laminaprops=self.blaminaprops,
                                            offset=(-h/2.-hb/2.))
            self.hb = hb

        #TODO check offset effect on curved panels
        self.df = self.bf/2. + self.hb + h/2.
        self.Iyy = self.hf*self.bf**3/12.
        #self.Iyy = self.hf*self.bf**3/12. + self.hf*self.bf*self.df**2
        self.Jxx = self.hf*self.bf**3/12. + self.bf*self.hf**3/12.

        self.Asb = self.bb*self.hb
        self.Asf = self.bf*self.hf
        self.As = self.Asb + self.Asf

        if self.fstack != []:
            self.E1 = 0
            #E3 = 0
            self.S1 = 0
            yply = self.flam.plies[0].t/2.
            for i, ply in enumerate(self.flam.plies):
                if i > 0:
                    yply += self.flam.plies[i-1].t/2. + self.flam.plies[i].t/2.
                q = ply.QL
                self.E1 += ply.t*(q[0,0] - q[0,1]**2/q[1,1])
                #E3 += ply.t*(q[2,2] - q[1,2]**2/q[1,1])
                self.S1 += -yply*ply.t*(q[0,2] - q[0,1]*q[1,2]/q[1,1])

            self.F1 = self.bf**2/12.*self.E1

        assert self.panel1.model == self.panel2.model
        assert self.panel1.m1 == self.panel2.m1
        assert self.panel1.n1 == self.panel2.n1


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)

        panmod = panmodelDB.db[self.panel1.model]['matrices']
        mod = modelDB.db[self.model]['matrices']

        bay = self.bay
        a = bay.a
        b = bay.b
        m1 = self.panel1.m1
        n1 = self.panel1.n1

        k0 = 0.
        if self.blam is not None:
            Fsb = self.blam.ABD
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            k0 += panmod.fk0y1y2(y1, y2, a, b, r, alpharad, Fsb, m1, n1,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                                 size=size, row0=row0, col0=col0)

        if self.flam is not None:
            k0 += mod.fk0sf(self.bf, self.df, self.ys, a, b, r, m1, n1,
                    self.E1, self.F1, self.S1, self.Jxx)

        if finalize:
            assert np.any(np.isnan(k0.data)) == False
            assert np.any(np.isinf(k0.data)) == False
            k0 = csr_matrix(make_symmetric(k0))

        self.k0 = k0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kG0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        self._rebuild()
        msg('Calculating kG0... ', level=2, silent=silent)

        panmod = panmodelDB.db[self.panel1.model]['matrices']
        mod = modelDB.db[self.model]['matrices']

        m1 = self.m1
        n1 = self.n1
        mu = self.mu

        kG0 = 0.

        if self.blam is not None:
            Fsb = self.blam.ABD
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            # TODO include kG0 for padup

        if self.flam is not None:
            #TODO where is the pre-load going???

            kG0 += mod.fkG0f(self.bf, self.df, self.ys, a, b, r, m1, n1,
                    self.E1, self.F1, self.S1, self.Jxx)

        if finalize:
            assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
            kG0 = csr_matrix(make_symmetric(kG0))

        self.kG0 = kG0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kM(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)

        panmod = panmodelDB.db[self.panel1.model]['matrices']
        mod = modelDB.db[self.model]['matrices']

        bay = self.bay
        a = bay.a
        b = bay.b
        m1 = self.m1
        n1 = self.n1
        mu = self.mu

        kM = 0.
        if self.blam is not None:
            Fsb = self.blam.ABD
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            kM += panmod.fkMy1y2(y1, y2, self.mu, self.db, self.hb, a, b,
                          m1, n1,
                          bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                          bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                          bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                          bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                          bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                          bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                          size=size, row0=row0, col0=col0)

        if self.flam is not None:
            kM += mod.fkMsf(self.mu, self.ys, self.df, self.Asf, a, b,
                    self.Iyy, self.Jxx, m1, n1)

        if finalize:
            assert np.any(np.isnan(kM.data)) == False
            assert np.any(np.isinf(kM.data)) == False
            kM = csr_matrix(make_symmetric(kM))

        self.kM = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)



class BladeStiff2D(object):
    r"""Blade Stiffener using 2D Formulation for Flange

    Blade-type of stiffener model using a 2D formulation for the flange and a
    2D formulation for the padup (base)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panels
         Panel1      Panel2

    Both the flange and the padup are optional.

    Each stiffener has a constant `y` coordinate.

    """
    def __init__(self, mu, bay, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.model = 'bladestiff2d_clt_donnell_bardell'
        self.m1 = None
        self.n1 = None
        self.mu = mu
        self.ys = ys
        self.bb = bb
        self.hb = 0.
        self.bf = bf
        self.hf = 0.
        self.forces = []

        self.inf = 1.e8

        self.Nxx = None
        self.Nxy = None

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops
        self.fstack = fstack
        self.fplyts = fplyts
        self.flaminaprops = flaminaprops
        self.blam = None
        self.flam = None

        self.k0 = None
        self.kM = None
        self.kG0 = None
        self.kG0_Nxx = None
        self.kG0_Nxy = None

        self.u1txf = 0.
        self.u1rxf = 1.
        self.u2txf = 0.
        self.u2rxf = 1.
        self.v1txf = 0.
        self.v1rxf = 1.
        self.v2txf = 0.
        self.v2rxf = 1.
        self.w1txf = 0.
        self.w1rxf = 1.
        self.w2txf = 0.
        self.w2rxf = 1.

        self.u1tyf = 1.
        self.u1ryf = 1.
        self.u2tyf = 1.
        self.u2ryf = 1.
        self.v1tyf = 1.
        self.v1ryf = 1.
        self.v2tyf = 1.
        self.v2ryf = 1.
        self.w1tyf = 1.
        self.w1ryf = 1.
        self.w2tyf = 1.
        self.w2ryf = 1.

        self._rebuild()


    def _rebuild(self):
        if self.fstack != []:
            self.hf = sum(self.fplyts)
            self.flam = laminate.read_stack(self.fstack, plyts=self.fplyts,
                                            laminaprops=self.flaminaprops)
            self.flam.calc_equivalent_modulus()

        h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
        if self.bstack != []:
            hb = sum(self.bplyts)
            self.db = abs(-h/2.-hb/2.)
            self.blam = laminate.read_stack(self.bstack, plyts=self.bplyts,
                                            laminaprops=self.blaminaprops,
                                            offset=(-h/2.-hb/2.))
            self.hb = hb

        assert self.panel1.model == self.panel2.model


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)

        panmod = panmodelDB.db[self.panel1.model]['matrices']
        mod = modelDB.db[self.model]['matrices']

        bay = self.bay
        a = bay.a
        b = bay.b
        r = bay.r
        m = bay.m
        n = bay.n

        m1 = self.m1
        n1 = self.n1
        bf = self.bf

        k0 = 0.
        if self.blam is not None:
            # stiffener pad-up
            Fsb = self.blam.ABD
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            k0 += panmod.fk0y1y2(y1, y2, a, b, r, Fsb, m, n,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                                 size, 0, 0)
        #TODO add contribution from Nxx_cte from flange and padup
        if self.flam is not None:
            kt = self.inf
            kr = self.inf
            F = self.flam.ABD
            k0 += mod.fk0f(a, bf, F, m1, n1,
                           self.u1txf, self.u1rxf, self.u2txf, self.u2rxf,
                           self.v1txf, self.v1rxf, self.v2txf, self.v2rxf,
                           self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                           self.u1tyf, self.u1ryf, self.u2tyf, self.u2ryf,
                           self.v1tyf, self.v1ryf, self.v2tyf, self.v2ryf,
                           self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                           size, row0, col0)

            # connectivity between skin-stiffener flange
            k0 += mod.fkCff(kt, kr, a, bf, m1, n1,
                            self.u1txf, self.u1rxf, self.u2txf, self.u2rxf,
                            self.v1txf, self.v1rxf, self.v2txf, self.v2rxf,
                            self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                            self.u1tyf, self.u1ryf, self.u2tyf, self.u2ryf,
                            self.v1tyf, self.v1ryf, self.v2tyf, self.v2ryf,
                            self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                            size, row0, col0)
            k0 += mod.fkCsf(kt, kr, self.ys, a, b, bf, m, n, m1, n1,
                            bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                            bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                            bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                            bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                            bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                            bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                            self.u1txf, self.u1rxf, self.u2txf, self.u2rxf,
                            self.v1txf, self.v1rxf, self.v2txf, self.v2rxf,
                            self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                            self.u1tyf, self.u1ryf, self.u2tyf, self.u2ryf,
                            self.v1tyf, self.v1ryf, self.v2tyf, self.v2ryf,
                            self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                            size, 0, col0)
            k0 += mod.fkCss(kt, kr, self.ys, a, b, m, n,
                            bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                            bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                            bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                            bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                            bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                            bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                            size, 0, 0)

        if finalize:
            assert np.any(np.isnan(k0.data)) == False
            assert np.any(np.isinf(k0.data)) == False
            k0 = csr_matrix(make_symmetric(k0))

        self.k0 = k0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kG0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        self._rebuild()
        msg('Calculating kG0... ', level=2, silent=silent)

        mod = modelDB.db[self.model]['matrices']

        bay = self.bay
        a = bay.a

        kG0 = 0.

        if self.blam is not None:
            # stiffener pad-up
            #TODO include kG0 for pad-up (Nxx load that arrives there)
            pass

        if self.flam is not None:
            F = self.flam.ABD
            # stiffener flange

            Nxx = self.Nxx if self.Nxx is not None else 0.
            Nxy = self.Nxy if self.Nxy is not None else 0.
            kG0 += mod.fkG0f(Nxx, 0., Nxy, a, self.bf, self.m1, self.n1,
                             self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                             self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                             size, row0, col0)

        if finalize:
            assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
            kG0 = csr_matrix(make_symmetric(kG0))

        self.kG0 = kG0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kM(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)

        panmod = panmodelDB.db[self.panel1.model]['matrices']
        mod = modelDB.db[self.model]['matrices']

        bay = self.bay
        a = bay.a
        b = bay.b
        m = bay.m
        n = bay.n

        m1 = self.m1
        n1 = self.n1
        bf = self.bf
        if self.blam is not None:
            # stiffener pad-up
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            kM += panmod.fkMy1y2(y1, y2, self.mu, self.db, self.hb, a, b, m, n,
                          bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                          bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                          bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                          bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                          bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                          bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                          size, 0, 0)

        if self.flam is not None:
            kM += mod.fkMf(self.mu, self.hf, a, bf, 0., m1, n1,
                           self.u1txf, self.u1rxf, self.u2txf, self.u2rxf,
                           self.v1txf, self.v1rxf, self.v2txf, self.v2rxf,
                           self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                           self.u1tyf, self.u1ryf, self.u2tyf, self.u2ryf,
                           self.v1tyf, self.v1ryf, self.v2tyf, self.v2ryf,
                           self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                           size, row0, col0)

        if finalize:
            assert np.any(np.isnan(kM.data)) == False
            assert np.any(np.isinf(kM.data)) == False
            kM = csr_matrix(make_symmetric(kM))

        self.kM = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


