from __future__ import division, absolute_import
import gc

import numpy as np
from numpy import deg2rad

from . import modelDB as stiffmDB
from compmech.panel import Panel
import compmech.panel.modelDB as panelmDB
from compmech.logger import msg, warn
from compmech.composite import laminate
from compmech.panel.connections import fkCBFycte11, fkCBFycte12, fkCBFycte22


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
            blaminaprops, fstack, fplyts, flaminaprops,
            model='tstiff2d_clt_donnell_bardell', m1=15, n1=12, m2=15, n2=12):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.model = model
        self.base = Panel(m=m1, n=n1, b=bb, a=bay.a, model=bay.model)
        self.flange = Panel(m=m2, n=n2, b=bf, a=bay.a, model='plate_clt_donnell_bardell')
        self.ys = ys
        self.mu = mu
        self.forces_base = []
        self.forces_flange = []

        self.eta_conn_base = 0.
        self.eta_conn_flange = -1.

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

        self.base.u1tx = 1.
        self.base.u1rx = 1.
        self.base.u2tx = 1.
        self.base.u2rx = 1.
        self.base.v1tx = 1.
        self.base.v1rx = 1.
        self.base.v2tx = 1.
        self.base.v2rx = 1.
        self.base.w1tx = 1.
        self.base.w1rx = 1.
        self.base.w2tx = 1.
        self.base.w2rx = 1.

        self.base.u1ty = 1.
        self.base.u1ry = 1.
        self.base.u2ty = 1.
        self.base.u2ry = 1.
        self.base.v1ty = 1.
        self.base.v1ry = 1.
        self.base.v2ty = 1.
        self.base.v2ry = 1.
        self.base.w1ty = 1.
        self.base.w1ry = 1.
        self.base.w2ty = 1.
        self.base.w2ry = 1.

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
        a = None
        b = None
        if self.panel1 is not None:
            a = self.panel1.a
            b = self.panel1.b
        elif self.panel2 is not None:
            a = self.panel2.a
            b = self.panel2.b
        if a is not None and b is not None:
            if a / b > 10.:
                if self.base.m <= 15 and self.flange.m <= 15:
                    raise RuntimeError('For a/b > 10. use base.m and flange.m > 15')
                else:
                    warn('For a/b > 10. be sure to check convergence for base.m and flange.m')
        if self.fstack is None:
            raise ValueError('Flange laminate must be defined!')
        if self.bstack is None:
            raise ValueError('Base laminate must be defined!')

        self.flam = laminate.read_stack(self.fstack, plyts=self.fplyts,
                                        laminaprops=self.flaminaprops)
        self.flam.calc_equivalent_modulus()
        h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
        hb = sum(self.bplyts)
        self.dpb = h/2. + hb/2.
        self.blam = laminate.read_stack(self.bstack, plyts=self.bplyts,
                                        laminaprops=self.blaminaprops,
                                        offset=0.)

        assert self.panel1.model == self.panel2.model
        assert self.panel1.r == self.panel2.r
        assert self.panel1.alphadeg == self.panel2.alphadeg


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the linear constitutive stiffness matrix
        """
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)

        modelb = panelmDB.db[self.base.model]['matrices']
        modelf = panelmDB.db[self.flange.model]['matrices']
        conn = stiffmDB.db[self.model]['connections']
        num1 = panelmDB.db[self.base.model]['num']

        bay = self.bay
        a = bay.a
        b = bay.b
        ys = self.ys
        r = bay.r if bay.r is not None else 0.
        alphadeg = bay.alphadeg
        alphadeg = alphadeg if alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)


        # NOTE
        #     row0 and col0 define where the stiffener's base matrix starts
        #     row1 and col1 define where the stiffener's flange matrix starts
        row1 = row0 + num1*self.base.m*self.base.n
        col1 = col0 + num1*self.base.m*self.base.n

        k0 = 0.

        #TODO add contribution from Nxx_cte from flange and base

        # stiffener base
        Fsb = self.blam.ABD
        # default is to have no unbouded region

        #TODO remove from Cython the capability to run with debonding defect
        x1 = a/2.
        x2 = a/2.

        y1 = ys - self.base.b/2.
        y2 = ys + self.base.b/2.
        k0 += modelb.fk0y1y2(y1, y2, a, b, r, alpharad, Fsb, self.base.m, self.base.n,
                             self.base.u1tx, self.base.u1rx, self.base.u2tx, self.base.u2rx,
                             self.base.v1tx, self.base.v1rx, self.base.v2tx, self.base.v2rx,
                             self.base.w1tx, self.base.w1rx, self.base.w2tx, self.base.w2rx,
                             self.base.u1ty, self.base.u1ry, self.base.u2ty, self.base.u2ry,
                             self.base.v1ty, self.base.v1ry, self.base.v2ty, self.base.v2ry,
                             self.base.w1ty, self.base.w1ry, self.base.w2ty, self.base.w2ry,
                             size, row0, col0)

        # stiffener flange
        Fsf = self.flam.ABD
        k0 += modelf.fk0(a, b, r, alpharad, Fsf, self.flange.m, self.flange.n,
                         self.flange.u1tx, self.flange.u1rx, self.flange.u2tx, self.flange.u2rx,
                         self.flange.v1tx, self.flange.v1rx, self.flange.v2tx, self.flange.v2rx,
                         self.flange.w1tx, self.flange.w1rx, self.flange.w2tx, self.flange.w2rx,
                         self.flange.u1ty, self.flange.u1ry, self.flange.u2ty, self.flange.u2ry,
                         self.flange.v1ty, self.flange.v1ry, self.flange.v2ty, self.flange.v2ry,
                         self.flange.w1ty, self.flange.w1ry, self.flange.w2ty, self.flange.w2ry,
                         size, row1, col1)

        # connectivity panel-base
        dpb = self.dpb
        den = min(self.panel1.lam.t, self.panel2.lam.t, self.bplyts[0]) * min(a, b)
        ktpb = max(self.panel1.lam.ABD[0, 0], self.blam.ABD[0, 0])/den

        k0 += conn.fkCppx1x2y1y2(0, x1, y1, y2,
                                 ktpb, a, b, dpb, bay.m, bay.n,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                                 size, 0, 0)
        k0 += conn.fkCppx1x2y1y2(x2, a, y1, y2,
                                 ktpb, a, b, dpb, bay.m, bay.n,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                                 size, 0, 0)

        k0 += conn.fkCpbx1x2y1y2(0, x1, y1, y2,
                                 ktpb, a, b, dpb,
                                 bay.m, bay.n, self.base.m, self.base.n,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                                 self.base.u1tx, self.base.u1rx, self.base.u2tx, self.base.u2rx,
                                 self.base.v1tx, self.base.v1rx, self.base.v2tx, self.base.v2rx,
                                 self.base.w1tx, self.base.w1rx, self.base.w2tx, self.base.w2rx,
                                 self.base.u1ty, self.base.u1ry, self.base.u2ty, self.base.u2ry,
                                 self.base.v1ty, self.base.v1ry, self.base.v2ty, self.base.v2ry,
                                 self.base.w1ty, self.base.w1ry, self.base.w2ty, self.base.w2ry,
                                 size, 0, col0)
        k0 += conn.fkCpbx1x2y1y2(x2, a, y1, y2,
                                 ktpb, a, b, dpb,
                                 bay.m, bay.n, self.base.m, self.base.n,
                                 bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                                 bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                                 bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                                 bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                                 bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                                 bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                                 self.base.u1tx, self.base.u1rx, self.base.u2tx, self.base.u2rx,
                                 self.base.v1tx, self.base.v1rx, self.base.v2tx, self.base.v2rx,
                                 self.base.w1tx, self.base.w1rx, self.base.w2tx, self.base.w2rx,
                                 self.base.u1ty, self.base.u1ry, self.base.u2ty, self.base.u2ry,
                                 self.base.v1ty, self.base.v1ry, self.base.v2ty, self.base.v2ry,
                                 self.base.w1ty, self.base.w1ry, self.base.w2ty, self.base.w2ry,
                                 size, 0, col0)

        k0 += conn.fkCbbpbx1x2(0, x1, y1, y2,
                               ktpb, a, b, self.base.m, self.base.n,
                               self.base.u1tx, self.base.u1rx, self.base.u2tx, self.base.u2rx,
                               self.base.v1tx, self.base.v1rx, self.base.v2tx, self.base.v2rx,
                               self.base.w1tx, self.base.w1rx, self.base.w2tx, self.base.w2rx,
                               self.base.u1ty, self.base.u1ry, self.base.u2ty, self.base.u2ry,
                               self.base.v1ty, self.base.v1ry, self.base.v2ty, self.base.v2ry,
                               self.base.w1ty, self.base.w1ry, self.base.w2ty, self.base.w2ry,
                               size, row0, col0)
        k0 += conn.fkCbbpbx1x2(x2, a, y1, y2,
                               ktpb, a, b, self.base.m, self.base.n,
                               self.base.u1tx, self.base.u1rx, self.base.u2tx, self.base.u2rx,
                               self.base.v1tx, self.base.v1rx, self.base.v2tx, self.base.v2rx,
                               self.base.w1tx, self.base.w1rx, self.base.w2tx, self.base.w2rx,
                               self.base.u1ty, self.base.u1ry, self.base.u2ty, self.base.u2ry,
                               self.base.v1ty, self.base.v1ry, self.base.v2ty, self.base.v2ry,
                               self.base.w1ty, self.base.w1ry, self.base.w2ty, self.base.w2ry,
                               size, row0, col0)

        # connectivity base-flange
        ktbf = (self.blam.ABD[1, 1] + self.flam.ABD[1, 1])/(self.blam.t + self.flam.t)
        krbf = (self.blam.ABD[4, 4] + self.flam.ABD[4, 4])/(self.blam.t + self.flam.t)
        ycte1 = (self.eta_conn_base+1)/2.*self.base.b
        ycte2 = (self.eta_conn_flange+1)/2.*self.flange.b
        k0 += fkCBFycte11(ktbf, krbf, self.base, ycte1, size, row0, col0)
        k0 += fkCBFycte12(ktbf, krbf, self.base, self.flange, ycte1, ycte2, size, row0, col1)
        k0 += fkCBFycte22(ktbf, krbf, self.base, self.flange, ycte2, size, row1, col1)

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

        modelb = panelmDB.db[self.base.model]['matrices']
        modelf = panelmDB.db[self.flange.model]['matrices']
        num1 = panelmDB.db[self.base.model]['num']

        bay = self.bay
        a = bay.a
        r = bay.r if bay.r is not None else 0.
        alphadeg = bay.alphadeg if bay.alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)

        # NOTE
        #     row0 and col0 define where the stiffener's base matrix starts
        #     row1 and col1 define where the stiffener's flange matrix starts

        row1 = row0 + num1*self.base.m*self.base.n
        col1 = col0 + num1*self.base.m*self.base.n

        kG0 = 0.

        # stiffener base
        Nxxb = self.Nxxb if self.Nxxb is not None else 0.
        Nxyb = self.Nxyb if self.Nxyb is not None else 0.
        kG0 += modelb.fkG0(Nxxb, 0., Nxyb, a, self.base.b, r, alpharad,
                         self.base.m, self.base.n,
                         self.base.w1tx, self.base.w1rx, self.base.w2tx, self.base.w2rx,
                         self.base.w1ty, self.base.w1ry, self.base.w2ty, self.base.w2ry,
                         size, row0, col0)

        # stiffener flange
        Nxxf = self.Nxxf if self.Nxxf is not None else 0.
        Nxyf = self.Nxyf if self.Nxyf is not None else 0.
        kG0 += modelf.fkG0(Nxxf, 0., Nxyf, a, self.flange.b, r, alpharad,
                         self.flange.m, self.flange.n,
                         self.flange.w1tx, self.flange.w1rx, self.flange.w2tx, self.flange.w2rx,
                         self.flange.w1ty, self.flange.w1ry, self.flange.w2ty, self.flange.w2ry,
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

        modelb = panelmDB.db[self.base.model]['matrices']
        modelf = panelmDB.db[self.flange.model]['matrices']
        num1 = panelmDB.db[self.base.model]['num']

        bay = self.bay
        a = bay.a

        r = bay.r if bay.r is not None else 0.
        alphadeg = bay.alphadeg
        alphadeg = alphadeg if alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)

        row1 = row0 + num1*self.base.m*self.base.n
        col1 = col0 + num1*self.base.m*self.base.n

        kM = 0.

        hb = sum(self.bplyts)
        hf = sum(self.fplyts)

        # stiffener base
        kM += modelb.fkM(self.mu, 0., hb, a, self.base.b, r, alpharad, self.base.m, self.base.n,
                      self.base.u1tx, self.base.u1rx, self.base.u2tx, self.base.u2rx,
                      self.base.v1tx, self.base.v1rx, self.base.v2tx, self.base.v2rx,
                      self.base.w1tx, self.base.w1rx, self.base.w2tx, self.base.w2rx,
                      self.base.u1ty, self.base.u1ry, self.base.u2ty, self.base.u2ry,
                      self.base.v1ty, self.base.v1ry, self.base.v2ty, self.base.v2ry,
                      self.base.w1ty, self.base.w1ry, self.base.w2ty, self.base.w2ry,
                      size, row0, col0)

        # stiffener flange
        kM += modelf.fkM(self.mu, 0., hf, a, self.flange.b, r, alpharad, self.flange.m, self.flange.n,
                      self.flange.u1tx, self.flange.u1rx, self.flange.u2tx, self.flange.u2rx,
                      self.flange.v1tx, self.flange.v1rx, self.flange.v2tx, self.flange.v2rx,
                      self.flange.w1tx, self.flange.w1rx, self.flange.w2tx, self.flange.w2rx,
                      self.flange.u1ty, self.flange.u1ry, self.flange.u2ty, self.flange.u2ry,
                      self.flange.v1ty, self.flange.v1ry, self.flange.v2ty, self.flange.v2ry,
                      self.flange.w1ty, self.flange.w1ry, self.flange.w2ty, self.flange.w2ry,
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


