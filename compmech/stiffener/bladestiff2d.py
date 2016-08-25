from __future__ import division, absolute_import
import gc

import numpy as np
from numpy import deg2rad

import compmech.panel.modelDB as panmodelDB
from compmech.panel import Panel
from compmech.logger import msg, warn
from compmech.composite import laminate
from compmech.panel.connections import fkCBFycte11, fkCBFycte12, fkCBFycte22
from .modelDB import db

class BladeStiff2D(object):
    r"""Blade Stiffener using 2D Formulation for Flange

    Blade-type of stiffener model using a 2D formulation for the flange and a
    2D formulation for the base (padup)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panels
         Panel1      Panel2

    Both the flange and the base are optional. The stiffener's base is modeled
    using the same approximation functions as the skin, with the proper
    offset.

    Each stiffener has a constant `y_s` coordinate.

    """
    def __init__(self, bay, mu, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops, m1=14, n1=11):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.mu = mu
        self.ys = ys
        self.bb = bb
        self.hb = 0.
        self.hf = 0.
        self.forces_flange = []

        self.Nxx = None
        self.Nxy = None

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops
        self.blam = None

        self.k0 = None
        self.kM = None
        self.kG0 = None

        self.flange = Panel(m=m1, n=n1, a=bay.a, b=bf, stack=fstack, plyts=fplyts, laminaprops=flaminaprops,
                model='plate_clt_donnell_bardell')
        self.flange.u1tx = 0.
        self.flange.u1rx = 1.
        self.flange.u2tx = 0.
        self.flange.u2rx = 1.
        self.flange.v1tx = 0.
        self.flange.v1rx = 1.
        self.flange.v2tx = 0.
        self.flange.v2rx = 1.
        self.flange.w1tx = 0.
        self.flange.w1rx = 1.
        self.flange.w2tx = 0.
        self.flange.w2rx = 1.

        self.flange.u1ty = 1.
        self.flange.u1ry = 1.
        self.flange.u2ty = 1.
        self.flange.u2ry = 1.
        self.flange.v1ty = 1.
        self.flange.v1ry = 1.
        self.flange.v2ty = 1.
        self.flange.v2ry = 1.
        self.flange.w1ty = 1.
        self.flange.w1ry = 1.
        self.flange.w2ty = 1.
        self.flange.w2ry = 1.

        self._rebuild()


    def _rebuild(self):
        if self.flange.stack is not None:
            self.hf = sum(self.flange.plyts)
            self.flange.lam = laminate.read_stack(self.flange.stack, plyts=self.flange.plyts,
                                            laminaprops=self.flange.laminaprops)
            self.flange.lam.calc_equivalent_modulus()

        h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
        if self.bstack is not None:
            hb = sum(self.bplyts)
            self.dpb = h/2. + hb/2.
            self.blam = laminate.read_stack(self.bstack, plyts=self.bplyts,
                                            laminaprops=self.blaminaprops,
                                            offset=(-h/2.-hb/2.))
            self.hb = hb

        assert self.panel1.model == self.panel2.model
        assert self.panel1.m == self.panel2.m
        assert self.panel1.n == self.panel2.n
        assert self.panel1.r == self.panel2.r
        assert self.panel1.alphadeg == self.panel2.alphadeg


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the linear constitutive stiffness matrix
        """
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)

        basemod = panmodelDB.db[self.panel1.model]['matrices']
        flangemod = panmodelDB.db[self.flange.model]['matrices']

        bay = self.bay
        a = bay.a
        b = bay.b
        r = bay.r if bay.r is not None else 0.
        alphadeg = self.panel1.alphadeg
        alphadeg = alphadeg if alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)

        k0 = 0.
        if self.blam is not None:
            # stiffener pad-up
            Fsb = self.blam.ABD
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            k0 += basemod.fk0y1y2(y1, y2, a, b, r, alpharad, Fsb, bay.m, bay.n,
                                 1, 1, 1, 1,
                                 1, 1, 1, 1,
                                 1, 1, 1, 1,
                                 1, 1, 1, 1,
                                 1, 1, 1, 1,
                                 1, 1, 1, 1,
                                 size, 0, 0)

        #TODO add contribution from Nxx_cte from flange and padup
        if self.flange.lam is not None:
            F = self.flange.lam.ABD
            bf = self.flange.b
            k0 += flangemod.fk0(a, bf, 0., 0., F, self.flange.m, self.flange.n,
                           self.flange.u1tx, self.flange.u1rx, self.flange.u2tx, self.flange.u2rx,
                           self.flange.v1tx, self.flange.v1rx, self.flange.v2tx, self.flange.v2rx,
                           self.flange.w1tx, self.flange.w1rx, self.flange.w2tx, self.flange.w2rx,
                           self.flange.u1ty, self.flange.u1ry, self.flange.u2ty, self.flange.u2ry,
                           self.flange.v1ty, self.flange.v1ry, self.flange.v2ty, self.flange.v2ry,
                           self.flange.w1ty, self.flange.w1ry, self.flange.w2ty, self.flange.w2ry,
                           size, row0, col0)

            # connectivity between skin-stiffener flange
            mod = db['bladestiff2d_clt_donnell_bardell']['connections']
            k0 += mod.fkCss(kt, kr, self.ys, a, b, bay.m, bay.n,
                            bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                            bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                            bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                            bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                            bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                            bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                            size, 0, 0)
            k0 += mod.fkCsf(kt, kr, self.ys, a, b, bf, bay.m, bay.n, self.flange.m, self.flange.n,
                            bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                            bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                            bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                            bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                            bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                            bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                            self.flange.u1tx, self.flange.u1rx, self.flange.u2tx, self.flange.u2rx,
                            self.flange.v1tx, self.flange.v1rx, self.flange.v2tx, self.flange.v2rx,
                            self.flange.w1tx, self.flange.w1rx, self.flange.w2tx, self.flange.w2rx,
                            self.flange.u1ty, self.flange.u1ry, self.flange.u2ty, self.flange.u2ry,
                            self.flange.v1ty, self.flange.v1ry, self.flange.v2ty, self.flange.v2ry,
                            self.flange.w1ty, self.flange.w1ry, self.flange.w2ty, self.flange.w2ry,
                            size, 0, col0)
            k0 += mod.fkCff(kt, kr, a, bf, self.flange.m, self.flange.n,
                            self.flange.u1tx, self.flange.u1rx, self.flange.u2tx, self.flange.u2rx,
                            self.flange.v1tx, self.flange.v1rx, self.flange.v2tx, self.flange.v2rx,
                            self.flange.w1tx, self.flange.w1rx, self.flange.w2tx, self.flange.w2rx,
                            self.flange.u1ty, self.flange.u1ry, self.flange.u2ty, self.flange.u2ry,
                            self.flange.v1ty, self.flange.v1ry, self.flange.v2ty, self.flange.v2ry,
                            self.flange.w1ty, self.flange.w1ry, self.flange.w2ty, self.flange.w2ry,
                            size, row0, col0)

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


    def calc_kG0(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None):
        """Calculate the linear geometric stiffness matrix
        """
        #TODO
        if c is not None:
            raise NotImplementedError('numerical kG0 not implemented')

        self._rebuild()
        msg('Calculating kG0... ', level=2, silent=silent)

        flangemod = panmodelDB.db[self.flange.model]['matrices']

        bay = self.bay
        a = bay.a

        kG0 = 0.

        if self.blam is not None:
            # stiffener pad-up
            #TODO include kG0 for pad-up (Nxx load that arrives there)
            pass

        if self.flange.lam is not None:
            F = self.flange.lam.ABD
            # stiffener flange

            Nxx = self.Nxx if self.Nxx is not None else 0.
            Nxy = self.Nxy if self.Nxy is not None else 0.
            kG0 += flangemod.fkG0(Nxx, 0., Nxy, a, self.flange.b, 0., 0., self.flange.m, self.flange.n,
                             self.flange.w1tx, self.flange.w1rx, self.flange.w2tx, self.flange.w2rx,
                             self.flange.w1ty, self.flange.w1ry, self.flange.w2ty, self.flange.w2ry,
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
        """Calculate the mass matrix
        """
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)

        basemod = panmodelDB.db[self.panel1.model]['matrices']
        flangemod = panmodelDB.db[self.flange.model]['matrices']

        bay = self.bay
        a = bay.a
        b = bay.b
        r = bay.r if bay.r is not None else 0.
        m = bay.m
        n = bay.n
        alphadeg = self.panel1.alphadeg
        alphadeg = alphadeg if alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)

        m1 = self.flange.m
        n1 = self.flange.n
        bf = self.flange.b

        kM = 0.

        if self.blam is not None:
            # stiffener pad-up
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            kM += basemod.fkMy1y2(y1, y2, self.mu, self.dpb, self.hb,
                          a, b, r, alpharad, m, n,
                          bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                          bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                          bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                          bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                          bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                          bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                          size, 0, 0)

        if self.flange.lam is not None:
            kM += flangemod.fkM(self.mu, 0., self.hf, a, bf, 0., 0., m1, n1,
                           self.flange.u1tx, self.flange.u1rx, self.flange.u2tx, self.flange.u2rx,
                           self.flange.v1tx, self.flange.v1rx, self.flange.v2tx, self.flange.v2rx,
                           self.flange.w1tx, self.flange.w1rx, self.flange.w2tx, self.flange.w2rx,
                           self.flange.u1ty, self.flange.u1ry, self.flange.u2ty, self.flange.u2ry,
                           self.flange.v1ty, self.flange.v1ry, self.flange.v2ty, self.flange.v2ry,
                           self.flange.w1ty, self.flange.w1ry, self.flange.w2ty, self.flange.w2ry,
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


