import gc

from . import modelDB
from compmech.panel import Panel
from compmech.logger import msg
from compmech.composite import laminate
from compmech.sparse import finalize_symmetric_matrix


class BladeStiff1D(object):
    r"""Blade Stiffener using 1D Formulation for Flange

    Blade-type of stiffener model using a 1D formulation for the flange and a
    2D formulation for the padup (base)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panels
         Panel1      Panel2

    Both the flange and the padup are optional, but one must exist.

    Each stiffener has a constant `y` coordinate.

    """
    def __init__(self, bay, mu, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.model = 'bladestiff1d_clt_donnell_bardell'
        self.mu = mu
        self.ys = ys
        self.bb = bb
        self.hb = 0.
        self.bf = bf
        self.hf = 0.

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops
        self.base = None
        self.fstack = fstack
        self.fplyts = fplyts
        self.flaminaprops = flaminaprops
        self.flam = None

        self.As = None
        self.Asb = None
        self.Asf = None
        self.Jxx = None
        self.Iyy = None

        self.Fx = None

        self._rebuild()


    def _rebuild(self):
        assert self.panel1.model == self.panel2.model
        assert self.panel1.m == self.panel2.m
        assert self.panel1.n == self.panel2.n
        assert self.panel1.r == self.panel2.r
        assert self.panel1.alphadeg == self.panel2.alphadeg

        if self.fstack is not None:
            self.hf = sum(self.fplyts)
            self.Asf = self.bf*self.hf
            self.flam = laminate.read_stack(self.fstack, plyts=self.fplyts,
                                             laminaprops=self.flaminaprops)
            self.flam.calc_equivalent_modulus()

        h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
        hb = 0.
        bay = self.bay
        if self.bstack is not None:
            hb = sum(self.bplyts)
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            self.base = Panel(a=bay.a, b=bay.b, r=bay.r, alphadeg=bay.alphadeg,
                    stack=self.bstack, plyts=self.bplyts,
                    mu=self.mu, m=bay.m, n=bay.n,
                    laminaprops=self.blaminaprops, offset=(-h/2.-hb/2.),
                    u1tx=bay.u1tx, u1rx=bay.u1rx, u2tx=bay.u2tx, u2rx=bay.u2rx,
                    v1tx=bay.v1tx, v1rx=bay.v1rx, v2tx=bay.v2tx, v2rx=bay.v2rx,
                    w1tx=bay.w1tx, w1rx=bay.w1rx, w2tx=bay.w2tx, w2rx=bay.w2rx,
                    u1ty=bay.u1ty, u1ry=bay.u1ry, u2ty=bay.u2ty, u2ry=bay.u2ry,
                    v1ty=bay.v1ty, v1ry=bay.v1ry, v2ty=bay.v2ty, v2ry=bay.v2ry,
                    w1ty=bay.w1ty, w1ry=bay.w1ry, w2ty=bay.w2ty, w2ry=bay.w2ry,
                    y1=y1, y2=y2)
            self.Asb = self.bb*hb

        #TODO check offset effect on curved panels
        self.dbf = self.bf/2. + hb + h/2.
        self.Iyy = self.hf*self.bf**3/12.
        self.Jxx = self.hf*self.bf**3/12. + self.bf*self.hf**3/12.

        Asb = self.Asb if self.Asb is not None else 0.
        Asf = self.Asf if self.Asf is not None else 0.
        self.As = Asb + Asf

        if self.fstack is not None:
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


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the linear constitutive stiffness matrix
        """
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)

        k0 = 0.
        if self.base is not None:
            k0 += self.base.calc_k0(size=size, row0=row0, col0=col0,
                    silent=True, finalize=False)
        if self.flam is not None:
            mod = modelDB.db[self.model]['matrices']
            bay = self.bay
            k0 += mod.fk0f(self.ys, bay.a, bay.b, self.bf, self.dbf, self.E1, self.F1,
                           self.S1, self.Jxx, bay.m, bay.n,
                           bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                           bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                           bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                           bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                           size=size, row0=row0, col0=col0)

        if finalize:
            k0 = finalize_symmetric_matrix(k0)
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

        kG0 = 0.
        if self.base is not None:
            # TODO include kG0 for padup
            #      now it is assumed that all the load goes to the flange
            pass
        if self.flam is not None:
            Fx = self.Fx if self.Fx is not None else 0.
            mod = modelDB.db[self.model]['matrices']
            bay = self.bay
            kG0 += mod.fkG0f(self.ys, Fx, bay.a, bay.b, self.bf, bay.m, bay.n,
                             bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                             bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                             size, row0, col0)

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

        mod = modelDB.db[self.model]['matrices']

        kM = 0.
        if self.base is not None:
            kM += self.base.calc_kM(size=size, row0=row0, col0=col0, silent=silent, finalize=False)
        if self.flam is not None:
            bay = self.bay
            h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
            kM += mod.fkMf(self.ys, self.mu, h, self.hb, self.hf, bay.a, bay.b,
                           self.bf, self.dbf, bay.m, bay.n,
                           bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                           bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                           bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                           bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                           bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                           bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                           size=size, row0=row0, col0=col0)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

