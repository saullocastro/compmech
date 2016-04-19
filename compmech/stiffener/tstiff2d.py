from __future__ import division
import gc

import numpy as np
from numpy import deg2rad

import modelDB as stiffmDB
import compmech.panel.modelDB as panelmDB
from compmech.logger import msg, warn
from compmech.composite import laminate


class TStiff2D(object):
    r"""T Stiffener using 2D Formulation for the Base and Flange

    T-type of stiffener model using a 2D formulation for the flange and a
    2D formulation for the base::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> base      |
      =========================  --> panels
         Panel1      Panel2

    The difference between this model and :class:'.BladeStiff2D' is that here
    the stiffener's base has independent field variables allowing the
    simulation of skin-stiffener debounding effects.

    Each stiffener has a constant `y_s` coordinate.

    """
    def __init__(self, bay, mu, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.model = 'tstiff2d_clt_donnell_bardell'
        self.m1 = 12
        self.n1 = 9
        self.m2 = 11
        self.n2 = 10
        self.mu = mu
        self.ys = ys
        self.bb = bb
        self.bf = bf
        self.forces_base = []
        self.forces_flange = []

        self.x1 = None
        self.x2 = None

        self.kt = 1.e8
        self.kr = 1.e8

        self.Nxxb = None
        self.Nxyb = None
        self.Nxxf = None
        self.Nxyf = None

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

        self.u1txb = 1.
        self.u1rxb = 1.
        self.u2txb = 1.
        self.u2rxb = 1.
        self.v1txb = 1.
        self.v1rxb = 1.
        self.v2txb = 1.
        self.v2rxb = 1.
        self.w1txb = 1.
        self.w1rxb = 1.
        self.w2txb = 1.
        self.w2rxb = 1.

        self.u1tyb = 1.
        self.u1ryb = 1.
        self.u2tyb = 1.
        self.u2ryb = 1.
        self.v1tyb = 1.
        self.v1ryb = 1.
        self.v2tyb = 1.
        self.v2ryb = 1.
        self.w1tyb = 1.
        self.w1ryb = 1.
        self.w2tyb = 1.
        self.w2ryb = 1.

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
        if self.fstack is None:
            raise ValueError('Flange laminate must be defined!')
        if self.bstack is None:
            raise ValueError('Base laminate must be defined!')

        self.flam = laminate.read_stack(self.fstack, plyts=self.fplyts,
                                        laminaprops=self.flaminaprops)
        self.flam.calc_equivalent_modulus()
        self.blam = laminate.read_stack(self.bstack, plyts=self.bplyts,
                                        laminaprops=self.blaminaprops)

        assert self.panel1.model == self.panel2.model
        assert self.panel1.r == self.panel2.r
        assert self.panel1.alphadeg == self.panel2.alphadeg


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the linear constitutive stiffness matrix
        """
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)

        modelb = panelmDB.db[self.panel1.model]['matrices']
        modelf = stiffmDB.db[self.model]['matrices_flange']
        conn = stiffmDB.db[self.model]['connections']
        num1 = stiffmDB.db[self.model]['num1']

        bay = self.bay
        a = bay.a
        b = bay.b
        bb = self.bb
        bf = self.bf
        ys = self.ys
        r = bay.r if bay.r is not None else 0.
        m = bay.m
        n = bay.n
        m1 = self.m1
        n1 = self.n1
        m2 = self.m2
        n2 = self.n2
        alphadeg = bay.alphadeg
        alphadeg = alphadeg if alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)

        m1 = self.m1
        n1 = self.n1

        # NOTE
        #     row0 and col0 define where the stiffener's base matrix starts
        #     row1 and col1 define where the stiffener's flange matrix starts

        row1 = row0 + num1*self.m1*self.n1
        col1 = col0 + num1*self.m1*self.n1

        k0 = 0.

        #TODO add contribution from Nxx_cte from flange and base

        # stiffener base
        Fsb = self.blam.ABD
        # default is to have no unbouded region
        x1 = self.x1 if self.x1 is not None else a/2.
        x2 = self.x2 if self.x2 is not None else a/2.
        y1 = ys - self.bb/2.
        y2 = ys + self.bb/2.
        k0 += modelb.fk0y1y2(y1, y2, a, b, r, alpharad, Fsb, m1, n1,
                             self.u1txb, self.u1rxb, self.u2txb, self.u2rxb,
                             self.v1txb, self.v1rxb, self.v2txb, self.v2rxb,
                             self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                             self.u1tyb, self.u1ryb, self.u2tyb, self.u2ryb,
                             self.v1tyb, self.v1ryb, self.v2tyb, self.v2ryb,
                             self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                             size, row0, col0)
        Fsf = self.flam.ABD

        h = sum(self.panel1.plyts)/2.
        hb = sum(self.bplyts)/2.

        # stiffener flange
        k0 += modelf.fk0(a, b, r, alpharad, Fsf, m2, n2,
                         self.u1txf, self.u1rxf, self.u2txf, self.u2rxf,
                         self.v1txf, self.v1rxf, self.v2txf, self.v2rxf,
                         self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                         self.u1tyf, self.u1ryf, self.u2tyf, self.u2ryf,
                         self.v1tyf, self.v1ryf, self.v2tyf, self.v2ryf,
                         self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                         size, row1, col1)

        # connectivity panel-base
        ktpb = self.kt/((a - x2 + x1)*(y2 - y1))
        krpb = self.kr/((a - x2 + x1)*(y2 - y1))
        k0 += conn.fkCppx1x2y1y2(0, x1, y1, y2,
                                 ktpb, a, b, h, m, n,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                                 size, 0, 0)
        k0 += conn.fkCppx1x2y1y2(x2, a, y1, y2,
                                 ktpb, a, b, h, m, n,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                                 size, 0, 0)

        k0 += conn.fkCpbx1x2y1y2(0, x1, y1, y2,
                                 ktpb, a, b, bb, h, hb,
                                 m, n, m1, n1,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                             self.u1txb, self.u1rxb, self.u2txb, self.u2rxb,
                             self.v1txb, self.v1rxb, self.v2txb, self.v2rxb,
                             self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                             self.u1tyb, self.u1ryb, self.u2tyb, self.u2ryb,
                             self.v1tyb, self.v1ryb, self.v2tyb, self.v2ryb,
                             self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                                 size, 0, col0)
        k0 += conn.fkCpbx1x2y1y2(x2, a, y1, y2,
                                 ktpb, a, b, bb, h, hb,
                                 m, n, m1, n1,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                             self.u1txb, self.u1rxb, self.u2txb, self.u2rxb,
                             self.v1txb, self.v1rxb, self.v2txb, self.v2rxb,
                             self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                             self.u1tyb, self.u1ryb, self.u2tyb, self.u2ryb,
                             self.v1tyb, self.v1ryb, self.v2tyb, self.v2ryb,
                             self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                                 size, 0, col0)

        k0 += conn.fkCbbpbx1x2(0, x1, y1, y2,
                               ktpb, krpb, a, b, bb, hb, m1, n1,
                               self.u1txb, self.u1rxb, self.u2txb, self.u2rxb,
                               self.v1txb, self.v1rxb, self.v2txb, self.v2rxb,
                               self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                               self.u1tyb, self.u1ryb, self.u2tyb, self.u2ryb,
                               self.v1tyb, self.v1ryb, self.v2tyb, self.v2ryb,
                               self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                               size, row0, col0)
        k0 += conn.fkCbbpbx1x2(x2, a, y1, y2,
                               ktpb, krpb, a, b, bb, hb, m1, n1,
                               self.u1txb, self.u1rxb, self.u2txb, self.u2rxb,
                               self.v1txb, self.v1rxb, self.v2txb, self.v2rxb,
                               self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                               self.u1tyb, self.u1ryb, self.u2tyb, self.u2ryb,
                               self.v1tyb, self.v1ryb, self.v2tyb, self.v2ryb,
                               self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                               size, row0, col0)

        # connectivity base-flange
        ktbf = self.kt/a
        krbf = self.kr/a
        eta_conn_base = 0.
        k0 += conn.fkCbbbf(ktbf, krbf, a, bb,
                           m1, n1, eta_conn_base,
                           self.u1txb, self.u1rxb, self.u2txb, self.u2rxb,
                           self.v1txb, self.v1rxb, self.v2txb, self.v2rxb,
                           self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                           self.u1tyb, self.u1ryb, self.u2tyb, self.u2ryb,
                           self.v1tyb, self.v1ryb, self.v2tyb, self.v2ryb,
                           self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                           size, row0, col0)

        eta_conn_flange = -1.
        k0 += conn.fkCbf(ktbf, krbf, a, bb, bf,
                         m1, n1, m2, n2,
                         eta_conn_base, eta_conn_flange,
                         self.u1txb, self.u1rxb, self.u2txb, self.u2rxb,
                         self.v1txb, self.v1rxb, self.v2txb, self.v2rxb,
                         self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                         self.u1tyb, self.u1ryb, self.u2tyb, self.u2ryb,
                         self.v1tyb, self.v1ryb, self.v2tyb, self.v2ryb,
                         self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                         self.u1txf, self.u1rxf, self.u2txf, self.u2rxf,
                         self.v1txf, self.v1rxf, self.v2txf, self.v2rxf,
                         self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                         self.u1tyf, self.u1ryf, self.u2tyf, self.u2ryf,
                         self.v1tyf, self.v1ryf, self.v2tyf, self.v2ryf,
                         self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                         size, row0, col1)

        k0 += conn.fkCff(ktbf, krbf, a, bf, m2, n2,
                         eta_conn_flange,
                         self.u1txf, self.u1rxf, self.u2txf, self.u2rxf,
                         self.v1txf, self.v1rxf, self.v2txf, self.v2rxf,
                         self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                         self.u1tyf, self.u1ryf, self.u2tyf, self.u2ryf,
                         self.v1tyf, self.v1ryf, self.v2tyf, self.v2ryf,
                         self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                         size, row1, col1)

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

        modelb = panelmDB.db[self.panel1.model]['matrices']
        modelf = stiffmDB.db[self.model]['matrices_flange']
        num1 = stiffmDB.db[self.model]['num1']

        bay = self.bay
        a = bay.a
        r = bay.r if bay.r is not None else 0.
        alphadeg = bay.alphadeg if bay.alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)

        # NOTE
        #     row0 and col0 define where the stiffener's base matrix starts
        #     row1 and col1 define where the stiffener's flange matrix starts

        row1 = row0 + num1*self.m1*self.n1
        col1 = col0 + num1*self.m1*self.n1

        kG0 = 0.

        # stiffener base
        Nxxb = self.Nxxb if self.Nxxb is not None else 0.
        Nxyb = self.Nxyb if self.Nxyb is not None else 0.
        kG0 += modelb.fkG0(Nxxb, 0., Nxyb, a, self.bb, r, alpharad,
                         self.m1, self.n1,
                         self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                         self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                         size, row0, col0)

        # stiffener flange
        Nxxf = self.Nxxf if self.Nxxf is not None else 0.
        Nxyf = self.Nxyf if self.Nxyf is not None else 0.
        kG0 += modelf.fkG0(Nxxf, 0., Nxyf, a, self.bf, r, alpharad,
                         self.m2, self.n2,
                         self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                         self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                         size, row1, col1)

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

        modelb = panelmDB.db[self.panel1.model]['matrices']
        modelf = stiffmDB.db[self.model]['matrices_flange']
        num1 = stiffmDB.db[self.model]['num1']

        bay = self.bay

        m1 = self.m1
        n1 = self.n1
        m2 = self.m2
        n2 = self.n2
        a = bay.a
        bb = self.bb
        bf = self.bf

        r = bay.r if bay.r is not None else 0.
        alphadeg = bay.alphadeg
        alphadeg = alphadeg if alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)

        row1 = row0 + num1*self.m1*self.n1
        col1 = col0 + num1*self.m1*self.n1

        kM = 0.

        hb = sum(self.bplyts)
        hf = sum(self.fplyts)

        # stiffener base
        kM += modelb.fkM(self.mu, 0., hb, a, bb, r, alpharad, m1, n1,
                      self.u1txb, self.u1rxb, self.u2txb, self.u2rxb,
                      self.v1txb, self.v1rxb, self.v2txb, self.v2rxb,
                      self.w1txb, self.w1rxb, self.w2txb, self.w2rxb,
                      self.u1tyb, self.u1ryb, self.u2tyb, self.u2ryb,
                      self.v1tyb, self.v1ryb, self.v2tyb, self.v2ryb,
                      self.w1tyb, self.w1ryb, self.w2tyb, self.w2ryb,
                      size, row0, col0)

        # stiffener flange
        kM += modelf.fkM(self.mu, 0., hf, a, bf, r, alpharad, m2, n2,
                       self.u1txf, self.u1rxf, self.u2txf, self.u2rxf,
                       self.v1txf, self.v1rxf, self.v2txf, self.v2rxf,
                       self.w1txf, self.w1rxf, self.w2txf, self.w2rxf,
                       self.u1tyf, self.u1ryf, self.u2tyf, self.u2ryf,
                       self.v1tyf, self.v1ryf, self.v2tyf, self.v2ryf,
                       self.w1tyf, self.w1ryf, self.w2tyf, self.w2ryf,
                       size, row1, col1)

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


