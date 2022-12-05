import gc

from . import modelDB as stiffmDB
from compmech.panel import Panel
from compmech.logger import msg, warn
from compmech.composite import laminate
from compmech.panel.connections import (fkCBFycte11, fkCBFycte12, fkCBFycte22,
        calc_kt_kr)
from compmech.sparse import finalize_symmetric_matrix

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
            model='tstiff2d_clt_donnell_bardell', mb=15, nb=12, mf=15, nf=12):

        print('\n\nWARNING - TStiff2D no longer recommended!\n'
              '          There is a numerically unstable way to compute the\n'
              '          connection between the skin and stiffener\'s base, it is\n'
              '          recommended to use panel.assembly instead.\n'
              '          Module panel.assembly allows stable ways to perform\n'
              '          connection among panels, and avoids using sliced\n'
              '          integration intervals that were causing trouble in\n'
              '          TStiff2D\n\n')

        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.model = model

        if bstack is None:
            raise ValueError('Base laminate must be defined!')
        self.ys = ys
        y1 = ys - bb/2.
        y2 = ys + bb/2.
        self.base = Panel(a=bay.a, b=bb, r=bay.r, alphadeg=bay.alphadeg,
                stack=bstack, plyts=bplyts, laminaprops=blaminaprops,
                mu=mu, m=mb, n=nb, offset=0.,
                u1tx=1, u1rx=0, u2tx=1, u2rx=0,
                v1tx=1, v1rx=0, v2tx=1, v2rx=0,
                w1tx=1, w1rx=1, w2tx=1, w2rx=1,
                u1ty=1, u1ry=0, u2ty=1, u2ry=0,
                v1ty=1, v1ry=0, v2ty=1, v2ry=0,
                w1ty=1, w1ry=1, w2ty=1, w2ry=1,
                y1=y1, y2=y2)
        self.base._rebuild()

        if fstack is None:
            raise ValueError('Flange laminate must be defined!')
        self.flange = Panel(a=bay.a, b=bf, model='plate_clt_donnell_bardell',
                stack=fstack, plyts=fplyts, laminaprops=flaminaprops,
                mu=mu, m=mf, n=nf, offset=0.,
                u1tx=0, u1rx=0, u2tx=0, u2rx=0,
                v1tx=0, v1rx=0, v2tx=0, v2rx=0,
                w1tx=0, w1rx=1, w2tx=0, w2rx=1,
                u1ty=1, u1ry=0, u2ty=1, u2ry=0,
                v1ty=1, v1ry=0, v2ty=1, v2ry=0,
                w1ty=1, w1ry=1, w2ty=1, w2ry=1)
        self.flange._rebuild()

        self.eta_conn_base = 0.
        self.eta_conn_flange = -1.

        self.k0 = None
        self.kM = None
        self.kG0 = None

        self._rebuild()


    def _rebuild(self):
        assert self.panel1.model == self.panel2.model
        assert self.panel1.r == self.panel2.r
        assert self.panel1.alphadeg == self.panel2.alphadeg
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

        self.flange.lam = laminate.read_stack(self.flange.stack,
                plyts=self.flange.plyts, laminaprops=self.flange.laminaprops)
        #NOTE below offset is not needed since it is already considered in the
        #     connectivity matrices, using dpb
        h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
        hb = sum(self.base.plyts)
        self.dpb = h/2. + hb/2.
        self.base.lam = laminate.read_stack(self.base.stack,
                plyts=self.base.plyts, laminaprops=self.base.laminaprops,
                offset=0.)


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the linear constitutive stiffness matrix
        """
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)

        # NOTE
        #     row0 and col0 define where the stiffener's base matrix starts
        #     rowf and colf define where the stiffener's flange matrix starts
        rowf = row0 + self.base.get_size()
        colf = col0 + self.base.get_size()

        k0 = 0.
        k0 += self.base.calc_k0(size=size, row0=row0, col0=col0, silent=True, finalize=False)
        k0 += self.flange.calc_k0(size=size, row0=rowf, col0=colf, silent=True, finalize=False)

        # connectivity panel-base
        conn = stiffmDB.db[self.model]['connections']

        ktpb, krpb = calc_kt_kr(self.panel1, self.base, 'bot-top')
        ktpb = min(1.e7, ktpb)


        #TODO remove from Cython the capability to run with debonding defect
        y1 = self.ys - self.base.b/2.
        y2 = self.ys + self.base.b/2.
        bay = self.bay

        k0 += conn.fkCppy1y2(y1, y2,
                             ktpb, bay.a, bay.b, self.dpb, bay.m, bay.n,
                             bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                             bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                             bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                             bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                             bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                             bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                             size, 0, 0)

        k0 += conn.fkCpby1y2(y1, y2,
                             ktpb, bay.a, bay.b, self.dpb,
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

        k0 += conn.fkCbbpby1y2(y1, y2,
                           ktpb, bay.a, bay.b, self.base.m, self.base.n,
                           self.base.u1tx, self.base.u1rx, self.base.u2tx, self.base.u2rx,
                           self.base.v1tx, self.base.v1rx, self.base.v2tx, self.base.v2rx,
                           self.base.w1tx, self.base.w1rx, self.base.w2tx, self.base.w2rx,
                           self.base.u1ty, self.base.u1ry, self.base.u2ty, self.base.u2ry,
                           self.base.v1ty, self.base.v1ry, self.base.v2ty, self.base.v2ry,
                           self.base.w1ty, self.base.w1ry, self.base.w2ty, self.base.w2ry,
                           size, row0, col0)

        # connectivity base-flange
        ktbf, krbf = calc_kt_kr(self.base, self.flange, 'ycte')
        ycte1 = (self.eta_conn_base + 1)/2. * self.base.b
        ycte2 = (self.eta_conn_flange + 1)/2. * self.flange.b
        k0 += fkCBFycte11(ktbf, krbf, self.base, ycte1, size, row0, col0)
        k0 += fkCBFycte12(ktbf, krbf, self.base, self.flange, ycte1, ycte2, size, row0, colf)
        k0 += fkCBFycte22(ktbf, krbf, self.base, self.flange, ycte2, size, rowf, colf)

        if finalize:
            k0 = finalize_symmetric_matrix(k0)
        self.k0 = k0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kG0(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None, NLgeom=False):
        """Calculate the linear geometric stiffness matrix

        See :meth:`.Panel.calc_k0` for details on each parameter.

        """
        self._rebuild()
        if c is None:
            msg('Calculating kG0... ', level=2, silent=silent)
        else:
            msg('Calculating kG... ', level=2, silent=silent)

        # NOTE:
        # - row0 and col0 define where the stiffener's base matrix starts
        # - rowf and colf define where the stiffener's flange matrix starts

        rowf = row0 + self.base.get_size()
        colf = col0 + self.base.get_size()

        kG0 = 0.
        kG0 += self.base.calc_kG0(c=c, size=size, row0=row0, col0=col0, silent=True,
                finalize=False, NLgeom=NLgeom)
        kG0 += self.flange.calc_kG0(c=c, size=size, row0=rowf, col0=colf,
                silent=True, finalize=False, NLgeom=NLgeom)

        if finalize:
            kG0 = finalize_symmetric_matrix(kG0)
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

        rowf = row0 + self.base.get_size()
        colf = col0 + self.base.get_size()

        kM = 0.
        kM += self.base.calc_kM(size=size, row0=row0, col0=col0, silent=True,
                finalize=False)
        kM += self.flange.calc_kM(size=size, row0=rowf, col0=colf, silent=True,
                finalize=False)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)
