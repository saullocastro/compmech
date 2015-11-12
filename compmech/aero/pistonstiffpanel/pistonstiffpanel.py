from __future__ import division
import gc
import os
import sys
import traceback
from collections import Iterable
import time
import cPickle
from multiprocessing import cpu_count
import __main__

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig, eigh
from numpy import linspace

import compmech.composite.laminate as laminate
from compmech.analysis import Analysis
from compmech.logger import msg, warn
from compmech.constants import DOUBLE
from compmech.sparse import (make_symmetric, make_skew_symmetric,
                             remove_null_cols)

import modelDB


def load(name):
    if '.AeroPistonStiffPanel' in name:
        return cPickle.load(open(name, 'rb'))
    else:
        return cPickle.load(open(name + '.AeroPistonStiffPanel', 'rb'))


class Stiffener(object):
    r"""Stiffener

    Blade-type of stiffener model using a 1D formulation for the flange and a
    2D formulation for the padup (base)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panel

    Both the flange and the padup are optional. The
    :class:`.AeroPistonStiffPanel` object may have any number of stiffeners.

    Each stiffener has a constant `y` coordinate.


    """
    def __init__(self, mu, panel, ys, bb, bf, bstack, bplyts, blaminaprops,
                 fstack, fplyts, flaminaprops):
        self.panel = panel
        self.mu = mu
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

        self.rebuild()


    def rebuild(self):
        if self.fstack != []:
            self.hf = sum(self.fplyts)
            self.flam = laminate.read_stack(self.fstack, plyts=self.fplyts,
                                             laminaprops=self.flaminaprops)
            self.flam.calc_equivalent_modulus()

        h = sum(self.panel.plyts)
        if self.bstack != []:
            hb = sum(self.bplyts)
            self.db = h/2.+hb/2.
            self.blam = laminate.read_stack(self.bstack, plyts=self.bplyts,
                                            laminaprops=self.blaminaprops,
                                            offset=-self.db)
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


class AeroPistonStiffPanel(object):
    r"""Stiffened Panel for Aeroelastic Studies using the Piston Theory

    Main characteristics:
        - Supports both airflows along x (axis) or y (circumferential).
          Controlled by the parameter ``flow``
        - Can have any number of :class:`.Stiffener` objects along the axial
          direction (`x`), with a fixed `y` position

    """
    def __init__(self):
        self.name = ''

        # boundary conditions
        # "inf" is used to define the high stiffnesses (removed dofs)
        #       a high value will cause numerical instabilities
        #TODO use a marker number for self.inf and self.maxinf if the
        #     normalization of edge stiffnesses is adopted
        #     now it is already independent of self.inf and more robust
        self.flow = 'x'
        self.inf = 1.e+8
        self.maxinf = 1.e+8
        self.zero = 0. # used to define zero stiffnesses
        self.bc = None
        self.kuBot = self.inf
        self.kvBot = self.inf
        self.kwBot = self.inf
        self.kphixBot = 0.
        self.kphiyBot = 0.
        self.kuTop = self.inf
        self.kvTop = self.inf
        self.kwTop = self.inf
        self.kphixTop = 0.
        self.kphiyTop = 0.
        self.kuLeft = self.inf
        self.kvLeft = self.inf
        self.kwLeft = self.inf
        self.kphixLeft = 0.
        self.kphiyLeft = 0.
        self.kuRight = self.inf
        self.kvRight = self.inf
        self.kwRight = self.inf
        self.kphixRight = 0.
        self.kphiyRight = 0.

        # default equations
        self.model = 'clpt_donnell_bc1'

        # approximation series
        self.m1 = 11
        self.n1 = 11

        # numerical integration
        self.nx = 160
        self.ny = 160
        self.ni_num_cores = cpu_count()//2
        self.ni_method = 'trapz2d'

        # loads
        self.Nxx = None
        self.Nyy = None
        self.Nxy = None

        # shear correction factor (FSDT only)
        self.K = 5/6.

        # stiffeners
        self.stiffeners = []

        # geometry
        self.a = None
        self.b = None
        self.r = None

        # material
        self.mu = None # laminate material density
        self.laminaprop = None
        self.plyt = None
        self.laminaprops = []
        self.stack = []
        self.plyts = []

        # aerodynamic properties for the Piston theory
        self.beta = None
        self.gamma = None
        self.aeromu = None
        self.rho_air = None
        self.speed_sound = None
        self.Mach = None
        self.V = None

        # constitutive law
        self.F = None

        # eigenvalue analysis
        self.num_eigvalues = 25
        self.num_eigvalues_print = 5

        # output queries
        self.out_num_cores = cpu_count()

        # analysis
        self.analysis = Analysis()

        # outputs
        self.eigvecs = None
        self.eigvals = None

        self._clear_matrices()


    def _clear_matrices(self):
        self.k0 = None
        self.kT = None
        self.kG0 = None
        self.kG0_Nxx = None
        self.kG0_Nyy = None
        self.kG0_Nxy = None
        self.kM = None
        self.kA = None
        self.cA = None
        self.lam = None
        self.u = None
        self.v = None
        self.w = None
        self.phix = None
        self.phiy = None
        self.Xs = None
        self.Ys = None

        gc.collect()


    def _rebuild(self):
        if not self.name:
            try:
                self.name = os.path.basename(__main__.__file__).split('.py')[0]
            except AttributeError:
                warn('AeroPistonStiffPanel name unchanged')

        self.model = self.model.lower()

        valid_models = sorted(modelDB.db.keys())

        if not self.model in valid_models:
            raise ValueError('ERROR - valid models are:\n    ' +
                     '\n    '.join(valid_models))

        # boundary conditions
        inf = self.inf
        zero = self.zero

        if inf > self.maxinf:
            warn('inf reduced to {0:1.1e4} due to the verified'.format(
                 self.maxinf) +
                 ' numerical instability for higher values', level=2)
            inf = self.maxinf

        if self.bc is not None:
            bc = self.bc.lower()

            if '_' in bc:
                # different bc for Bot, Top, Left and Right
                bc_Bot, bc_Top, bc_Left, bc_Right = self.bc.split('_')
            elif '-' in bc:
                # different bc for Bot, Top, Left and Right
                bc_Bot, bc_Top, bc_Left, bc_Right = self.bc.split('-')
            else:
                bc_Bot = bc_Top = bc_Left = bc_Right = bc

            bcs = dict(bc_Bot=bc_Bot, bc_Top=bc_Top,
                       bc_Left=bc_Left, bc_Right=bc_Right)
            for k in bcs.keys():
                sufix = k.split('_')[1] # Bot or Top
                if bcs[k] == 'ss1':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphiy' + sufix, zero)
                elif bcs[k] == 'ss2':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphiy' + sufix, zero)
                elif bcs[k] == 'ss3':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphiy' + sufix, zero)
                elif bcs[k] == 'ss4':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphiy' + sufix, zero)

                elif bcs[k] == 'cc1':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphiy' + sufix, inf)
                elif bcs[k] == 'cc2':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphiy' + sufix, inf)
                elif bcs[k] == 'cc3':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphiy' + sufix, inf)
                elif bcs[k] == 'cc4':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphiy' + sufix, inf)

                elif bcs[k] == 'free':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, zero)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphiy' + sufix, zero)

                else:
                    txt = '"{0}" is not a valid boundary condition!'.format(bc)
                    raise ValueError(txt)

        if self.a is None:
            raise ValueError('The length a must be specified')

        if self.b is None:
            raise ValueError('The width b must be specified')

        if not self.laminaprops:
            self.laminaprops = [self.laminaprop for i in self.stack]
        if not self.plyts:
            self.plyts = [self.plyt for i in self.stack]

        # defining load components from force vectors
        if self.laminaprop is None:
            raise ValueError('laminaprop must be defined')


    def get_size(self):
        r"""Calculate the size of the stiffness matrices

        The size of the stiffness matrices can be interpreted as the number of
        rows or columns, recalling that this will be the size of the Ritz
        constants' vector `\{c\}`, the internal force vector `\{F_{int}\}` and
        the external force vector `\{F_{ext}\}`.

        Returns
        -------
        size : int
            The size of the stiffness matrices.

        """
        num0 = modelDB.db[self.model]['num0']
        num1 = modelDB.db[self.model]['num1']
        self.size = num0 + num1*self.m1*self.n1
        return self.size


    def _default_field(self, xs, ys, gridx, gridy):
        if xs is None or ys is None:
            xs = linspace(0., self.a, gridx)
            ys = linspace(0., self.b, gridy)
            xs, ys = np.meshgrid(xs, ys, copy=False)
        xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
        ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
        xshape = xs.shape
        tshape = ys.shape
        if xshape != tshape:
            raise ValueError('Arrays xs and ys must have the same shape')
        self.Xs = xs
        self.Ys = ys
        xs = xs.ravel()
        ys = ys.ravel()

        return xs, ys, xshape, tshape


    def calc_linear_matrices(self, combined_load_case=None, silent=False,
            calc_kG0=True, calc_kA=True, calc_kM=True):
        self._rebuild()
        msg('Calculating linear matrices... ', level=2, silent=silent)

        fk0, fkG0, fkAx, fkAy, fcA, fkM, fk0edges, fk0sb, fk0sf, fkMsb, fkMsf = \
                modelDB.get_linear_matrices(self)
        model = self.model
        a = self.a
        b = self.b
        r = self.r
        m1 = self.m1
        n1 = self.n1
        laminaprops = self.laminaprops
        plyts = self.plyts
        h = sum(plyts)
        stack = self.stack
        mu = self.mu
        if calc_kA and self.beta is None:
            if self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                self.Mach = 1.0001
            M = self.Mach
            beta = self.rho_air * self.V**2 / (M**2 - 1)**0.5
            if self.gamma is None:
                gamma = beta*1./(2.*r*(M**2 - 1)**0.5)
            else:
                gamma = self.gamma
            ainf = self.speed_sound
            aeromu = beta/(M*ainf)*(M**2 - 2)/(M**2 - 1)
        elif calc_kA and self.beta is not None:
            beta = self.beta
            if self.gamma is None:
                M = self.Mach
                gamma = beta*1./(2.*r*(M**2 - 1)**0.5)
            else:
                gamma = self.gamma
            aeromu = self.aeromu if self.aeromu is not None else 0.
        elif not calc_kA:
            pass
        else:
            raise NotImplementedError('check here')

        if stack != []:
            lam = laminate.read_stack(stack, plyts=plyts,
                                             laminaprops=laminaprops)

        if 'clpt' in model:
            if lam is not None:
                F = lam.ABD

        elif 'fsdt' in model:
            if lam is not None:
                F = lam.ABDE
                F[6:, 6:] *= self.K

        self.lam = lam
        self.F = F

        k0 = fk0(a, b, r, F, m1, n1)

        if (self.model == 'clpt_donnell_bc1'
        or  self.model == 'clpt_sanders_bc1'):
            k0edges = fk0edges(m1, n1, a, b,
                               self.kphixBot, self.kphixTop,
                               self.kphiyLeft, self.kphiyRight)
        elif self.model == 'fsdt_donnell_bc1':
            k0edges = fk0edges(m1, n1, a, b,
                               self.kphixBot, self.kphixTop,
                               self.kphiyBot, self.kphiyTop,
                               self.kphixLeft, self.kphixRight,
                               self.kphiyLeft, self.kphiyRight)
        else:
            raise

        if calc_kA:
            if self.flow == 'x':
                kA = fkAx(beta, gamma, a, b, m1, n1)
            elif self.flow == 'y':
                kA = fkAy(beta, a, b, m1, n1)
            if fcA is None:
                cA = None
            else:
                cA = fcA(aeromu, a, b, m1, n1)
                cA = cA*(0+1j)
        if calc_kM:
            if self.model == 'fsdt_donnell_bc1':
                raise NotImplementedError('There is a bug with kM for model %s'
                        % self.model)
            kM = fkM(mu, h, a, b, m1, n1)

        if calc_kG0:
            Nxx = self.Nxx if self.Nxx is not None else 0.
            Nyy = self.Nyy if self.Nyy is not None else 0.
            Nxy = self.Nxy if self.Nxy is not None else 0.

            if not combined_load_case:
                kG0 = fkG0(Nxx, Nyy, Nxy, a, b, r, m1, n1)
            else:
                kG0_Nxx = fkG0(Nxx, 0, 0, a, b, r, m1, n1)
                kG0_Nyy = fkG0(0, Nyy, 0, a, b, r, m1, n1)
                kG0_Nxy = fkG0(0, 0, Nxy, a, b, r, m1, n1)

        # contributions from stiffeners
        #TODO summing up coo_matrix objects may be very slow!
        h = sum(self.plyts)
        for s in self.stiffeners:
            if s.blam is not None:
                Fsb = s.blam.ABD
                s.k0sb = fk0sb(s.ys, s.bb, a, b, r, m1, n1, Fsb)
                s.kMsb = fkMsb(s.mu, s.ys, s.bb, s.db, s.hb, h, a, b, m1, n1)
                k0 += s.k0sb
                if calc_kM:
                    kM += s.kMsb

            if s.flam is not None:
                s.k0sf = fk0sf(s.bf, s.df, s.ys, a, b, r, m1, n1, s.E1, s.F1,
                               s.S1, s.Jxx)
                s.kMsf = fkMsf(s.mu, s.ys, s.df, s.hf, s.bf, h, s.hb,
                               a, b, m1, n1)
                k0 += s.k0sf
                if calc_kM:
                    kM += s.kMsf

        # performing checks for the linear stiffness matrices

        assert np.any(np.isnan(k0.data)) == False
        assert np.any(np.isinf(k0.data)) == False

        if calc_kA:
            assert np.any(np.isnan(kA.data)) == False
            assert np.any(np.isinf(kA.data)) == False
            if cA is not None:
                assert np.any(np.isnan(cA.data)) == False
                assert np.any(np.isinf(cA.data)) == False

        if calc_kM:
            assert np.any(np.isnan(kM.data)) == False
            assert np.any(np.isinf(kM.data)) == False

        k0 = csr_matrix(make_symmetric(k0))
        if calc_kA:
            kA = csr_matrix(make_skew_symmetric(kA))
            if cA is not None:
                cA = csr_matrix(make_symmetric(cA))
        if calc_kM:
            kM = csr_matrix(make_symmetric(kM))

        assert np.any(np.isnan(k0edges.data)) == False
        assert np.any(np.isinf(k0edges.data)) == False
        k0edges = csr_matrix(make_symmetric(k0edges))

        k0 = k0 + k0edges

        self.k0 = k0
        if calc_kA:
            self.kA = kA
            self.cA = cA
        if calc_kM:
            self.kM = kM

        if calc_kG0:
            if not combined_load_case:
                assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
                kG0 = csr_matrix(make_symmetric(kG0))
                self.kG0 = kG0

            else:
                assert np.any((np.isnan(kG0_Nxx.data)
                               | np.isinf(kG0_Nxx.data))) == False
                assert np.any((np.isnan(kG0_Nyy.data)
                               | np.isinf(kG0_Nyy.data))) == False
                assert np.any((np.isnan(kG0_Nxy.data)
                               | np.isinf(kG0_Nxy.data))) == False

                kG0_Nxx = csr_matrix(make_symmetric(kG0_Nxx))
                kG0_Nyy = csr_matrix(make_symmetric(kG0_Nyy))
                kG0_Nxy = csr_matrix(make_symmetric(kG0_Nxy))

                self.kG0_Nxx = kG0_Nxx
                self.kG0_Nyy = kG0_Nyy
                self.kG0_Nxy = kG0_Nxy

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process

        gc.collect()

        msg('finished!', level=2, silent=silent)


    def lb(self, tol=0, combined_load_case=None, sparse_solver=True,
            calc_kA=False):
        """Performs a linear buckling analysis

        The following parameters of the ``AeroPistonStiffPanel`` object will
        affect the linear buckling analysis:

        =======================    =====================================
        parameter                  description
        =======================    =====================================
        ``num_eigenvalues``        Number of eigenvalues to be extracted
        ``num_eigvalues_print``    Number of eigenvalues to print after
                                   the analysis is completed
        =======================    =====================================

        Parameters
        ----------
        combined_load_case : int, optional
            It tells whether the linear buckling analysis must be computed
            considering combined load cases, each value will tell
            the algorithm to rearrange the linear matrices in a different
            way. The valid values are ``1``, or ``2``, where:

            - ``1`` : find the critical Nxx for a fixed Nxy
            - ``2`` : find the critical Nxx for a fixed Nyy
            - ``3`` : find the critical Nyy for a fixed Nxx
        sparse_solver : bool, optional
            Tells if solver :func:`scipy.linalg.eigh` or
            :func:`scipy.sparse.linalg.eigs` should be used.

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``AeroPistonStiffPanel`` object and the `i^{th}` eigenvector in
        the ``eigvecs[:, i-1]`` parameter.

        """
        if not modelDB.db[self.model]['linear buckling']:
            msg('________________________________________________')
            msg('')
            warn('Model {0} cannot be used in linear buckling analysis!'.
                 format(self.model))
            msg('________________________________________________')

        msg('Running linear buckling analysis...')

        self.calc_linear_matrices(combined_load_case=combined_load_case,
                calc_kM=False, calc_kA=calc_kA)

        msg('Eigenvalue solver... ', level=2)

        if calc_kA:
            kA = self.kA
        else:
            kA = self.k0*0

        if combined_load_case is None:
            M = self.k0 + kA
            A = self.kG0
        elif combined_load_case == 1:
            M = self.k0 - kA + self.kG0_Nxy
            A = self.kG0_Nxx
        elif combined_load_case == 2:
            M = self.k0 - kA + self.kG0_Nyy
            A = self.kG0_Nxx
        elif combined_load_case == 3:
            M = self.k0 - kA + self.kG0_Nxx
            A = self.kG0_Nyy

        Amin = abs(A.min())
        # Normalizing A to improve numerical stability
        A /= Amin

        if sparse_solver:
            try:
                msg('eigs() solver...', level=3)
                eigvals, eigvecs = eigs(A=A, k=self.num_eigvalues, which='SM',
                                        M=M, tol=tol, sigma=1.)
                msg('finished!', level=3)
            except Exception, e:
                warn(str(e), level=4)
                msg('aborted!', level=3)
                sizebkp = A.shape[0]
                M, A, used_cols = remove_null_cols(M, A)
                msg('eigs() solver...', level=3)
                eigvals, peigvecs = eigs(A=A, k=self.num_eigvalues,
                        which='SM', M=M, tol=tol, sigma=1.)
                msg('finished!', level=3)
                eigvecs = np.zeros((sizebkp, self.num_eigvalues),
                                   dtype=DOUBLE)
                eigvecs[used_cols, :] = peigvecs

            # Un-normalizing eigvals
            eigvals *= Amin

        else:
            from scipy.linalg import eig

            size22 = A.shape[0]
            M, A, used_cols = remove_null_cols(M, A)
            M = M.toarray()
            A = A.toarray()
            msg('eig() solver...', level=3)
            eigvals, peigvecs = eig(a=A, b=M)
            msg('finished!', level=3)
            eigvecs = np.zeros((size22, self.num_eigvalues), dtype=DOUBLE)
            eigvecs[used_cols, :] = peigvecs[:, :self.num_eigvalues]

        eigvals = -1./eigvals

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        msg('finished!', level=2)

        msg('first {0} eigenvalues:'.format(self.num_eigvalues_print), level=1)
        for eig in eigvals[:self.num_eigvalues_print]:
            msg('{0}'.format(eig), level=2)
        self.analysis.last_analysis = 'lb'


    def freq(self, atype=4, tol=0, sparse_solver=False, silent=False,
            sort=True, damping=False, reduced_dof=False):
        """Performs a frequency analysis

        The following parameters of the ``AeroPistonStiffPanel`` object will
        affect the linear buckling analysis:

        =======================    =====================================
        parameter                  description
        =======================    =====================================
        ``num_eigenvalues``        Number of eigenvalues to be extracted
        ``num_eigvalues_print``    Number of eigenvalues to print after
                                   the analysis is completed
        =======================    =====================================

        Parameters
        ----------
        atype : int, optional
            Tells which analysis type should be performed:

            - ``1`` : considers k0, kA and kG0
            - ``2`` : considers k0 and kA
            - ``3`` : considers k0 and kG0
            - ``4`` : considers k0 only

        tol : float, optional
            A tolerance value passed to ``scipy.sparse.linalg.eigs``.
        sparse_solver : bool, optional
            Tells if solver :func:`scipy.linalg.eig` or
            :func:`scipy.sparse.linalg.eigs` should be used.

            .. note:: It is recommended ``sparse_solver=False``, because it
                      was verified that the sparse solver becomes unstable
                      for some cases, though the sparse solver is faster.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        sort : bool, optional
            Sort the output eigenvalues and eigenmodes.
        damping : bool, optinal
            If aerodynamic damping should be taken into account.
        reduced_dof : bool, optional
            Considers only the contributions of `v` and `w` to the stiffness
            matrix and accelerates the run. Only effective when
            ``sparse_solver=False``.

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``AeroPistonStiffPanel`` object and the `i^{th}` eigenvector in
        the ``eigvecs[:, i-1]`` parameter.

        """
        if not modelDB.db[self.model]['linear buckling']:
            msg('________________________________________________')
            msg('')
            warn('Model {0} cannot be used in linear buckling analysis!'.
                 format(self.model))
            msg('________________________________________________')

        msg('Running frequency analysis...', silent=silent)

        if atype == 1:
            self.calc_linear_matrices(silent=silent)
        elif atype == 2:
            self.calc_linear_matrices(silent=silent, calc_kG0=False)
        elif atype == 3:
            self.calc_linear_matrices(silent=silent, calc_kA=False)
        elif atype == 4:
            self.calc_linear_matrices(silent=silent, calc_kA=False,
                                      calc_kG0=False)

        msg('Eigenvalue solver... ', level=2, silent=silent)

        if atype == 1:
            K = self.k0 + self.kA + self.kG0
        elif atype == 2:
            K = self.k0 + self.kA
        elif atype == 3:
            K = self.k0 + self.kG0
        elif atype == 4:
            K = self.k0
        M = self.kM

        if damping and self.cA is None:
            warn('Aerodynamic damping could not be calculated!', level=3,
                    silent=silent)
            damping = False
        elif damping and self.cA is not None:
            if self.cA.sum() == 0j:
                warn('Aerodynamic damping is null!', level=3, silent=silent)
                damping = False

        msg('eigs() solver...', level=3, silent=silent)
        k = min(self.num_eigvalues, M.shape[0]-2)
        if sparse_solver:
            eigvals, eigvecs = eigs(A=M, M=K, k=k, tol=tol, which='SM',
                                    sigma=-1.)
            eigvals = np.sqrt(1./eigvals) # omega^2 to omega, in rad/s
        else:
            M = M.toarray()
            K = K.toarray()
            if reduced_dof:
                i = np.arange(M.shape[0])
                take = np.column_stack((i[1::3], i[2::3])).flatten()
                M = M[:, take][take, :]
                K = K[:, take][take, :]
            if not damping:
                M = -M
            else:
                size = M.shape[0]
                cA = self.cA.toarray()
                if reduced_dof:
                    cA = cA[:, take][take, :]
                I = np.identity(M.shape[0])
                Z = np.zeros_like(M)
                M = np.row_stack((np.column_stack((I, Z)),
                                  np.column_stack((Z, -M))))
                K = np.row_stack((np.column_stack((Z, -I)),
                                  np.column_stack((K, cA))))

            eigvals, eigvecs = eig(a=M, b=K)

            if not damping:
                eigvals = np.sqrt(-1./eigvals) # -1/omega^2 to omega, in rad/s
                eigvals = eigvals
            else:
                eigvals = -1./eigvals # -1/omega to omega, in rad/s
                shape = eigvals.shape
                eigvals = eigvals[:shape[0]//2]
                eigvecs = eigvecs[:eigvecs.shape[0]//2, :shape[0]//2]

        msg('finished!', level=3, silent=silent)

        if sort:
            if damping:
                higher_zero = eigvals.real > 1e-6

                eigvals = eigvals[higher_zero]
                eigvecs = eigvecs[:, higher_zero]

                sort_ind = np.lexsort((np.round(eigvals.imag, 1),
                                       np.round(eigvals.real, 0)))
                eigvals = eigvals[sort_ind]
                eigvecs = eigvecs[:, sort_ind]

            else:
                sort_ind = np.lexsort((np.round(eigvals.imag, 1),
                                       np.round(eigvals.real, 1)))
                eigvals = eigvals[sort_ind]
                eigvecs = eigvecs[:, sort_ind]

                higher_zero = eigvals.real > 1e-6

                eigvals = eigvals[higher_zero]
                eigvecs = eigvecs[:, higher_zero]

        if not sparse_solver and reduced_dof:
            new_eigvecs = np.zeros((3*eigvecs.shape[0]//2, eigvecs.shape[1]),
                    dtype=eigvecs.dtype)
            new_eigvecs[take, :] = eigvecs
            eigvecs = new_eigvecs

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        msg('finished!', level=2, silent=silent)

        msg('first {0} eigenvalues:'.format(self.num_eigvalues_print), level=1,
                silent=silent)
        for eigval in eigvals[:self.num_eigvalues_print]:
            msg('{0} rad/s'.format(eigval), level=2, silent=silent)
        self.analysis.last_analysis = 'freq'


    def calc_betacr(self, beta1=1.e4, beta2=1.e5, rho_air=0.3, Mach=2.,
                    modes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                    num=5, silent=False, TOL=0.001, reduced_dof=False):
        r"""Calculate the critical aerodynamic pressure coefficient

        Parameters
        ----------
        rho_air : float, optional
            Air density.
        Mach : float, optional
            Mach number.
        modes : tuple, optional
            The modes that should be monitored.
        num : int, optional
            Number of points to search for each iteration.
        TOL: float, optional
            Convergence criterion.
        reduced_dof : bool, optional
            Considers only the contributions of `v` and `w` to the stiffness
            matrix and accelerates the run. Only effective when
            ``sparse_solver=False``.

        Returns
        -------
        betacr : float
            The critical ``beta``.

        """
        #TODO
        # - use a linear or parabolic interpolation to estimate new_beta1
        msg('Flutter calculation...', level=1, silent=silent)
        if self.speed_sound is None:
            self.speed_sound = 1.
        new_beta1 = 1.e6
        new_beta2 = -1e6
        eigvals_imag = np.zeros((num, len(modes)))
        if max(modes) > self.num_eigvalues-1:
            self.num_eigvalues = max(modes)+1

        count = 0

        # storing original values
        beta_bkp = self.beta
        gamma_bkp = self.gamma
        Mach_bkp = self.Mach
        V_bkp = self.V
        rho_air_bkp = self.rho_air

        self.beta = None
        self.gamma = None
        self.Mach = Mach
        self.rho_air = rho_air

        while True:
            count += 1
            betas = np.linspace(beta1, beta2, num)
            msg('iteration %d:' % count, level=2, silent=silent)
            msg('beta_min: %1.3f' % beta1, level=3, silent=silent)
            msg('beta_max: %1.3f' % beta2, level=3, silent=silent)

            for i, beta in enumerate(betas):
                self.V = ((Mach**2 - 1)**0.5 * beta / rho_air)**0.5
                self.freq(atype=1, sparse_solver=False, silent=True,
                          reduced_dof=reduced_dof)
                for j, mode in enumerate(modes):
                    eigvals_imag[i, j] = self.eigvals[mode].imag

            check = np.where(eigvals_imag != 0.)
            if not np.any(check):
                beta1 = beta1 / 2.
                beta2 = 2 * beta2
                continue
            if np.abs(eigvals_imag[check]).min() < TOL:
                break
            if 0 in check[0]:
                new_beta1 = min(new_beta1, 0.5*betas[check[0][0]])
                new_beta2 = max(new_beta2, 1.5*betas[check[0][-1]])
            elif check[0].min() > 0:
                new_beta1 = betas[check[0][0]-1]
                new_beta2 = betas[check[0][0]]
            else:
                new_beta1 = min(new_beta1, beta1/2.)
                new_beta2 = max(new_beta2, 2*beta2)

            beta1 = new_beta1
            beta2 = new_beta2

        # recovering original values
        self.beta = beta_bkp
        self.gamma = gamma_bkp
        self.Mach = Mach_bkp
        self.V = V_bkp
        self.rho_air = rho_air_bkp

        msg('finished!', level=1)
        msg('Number of analyses = %d' % (count*num), level=1)
        return beta1


    def uvw(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the
        ``AeroPistonStiffPanel`` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        xs : np.ndarray
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int
            Number of points along the `y` where to calculate the
            displacement field.

        Returns
        -------
        out : tuple
            A tuple of ``np.ndarrays`` containing
            ``(u, v, w, phix, phiy)``.

        Notes
        -----
        The returned values ``u```, ``v``, ``w``, ``phix``, ``phiy`` are
        stored as parameters with the same name in the
        ``AeroPistonStiffPanel`` object.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ys, xshape, tshape = self._default_field(xs, ys, gridx, gridy)
        m1 = self.m1
        n1 = self.n1
        a = self.a
        b = self.b
        model = self.model

        fuvw = modelDB.db[model]['commons'].fuvw
        us, vs, ws, phixs, phiys = fuvw(c, m1, n1, a, b, xs, ys,
                self.out_num_cores)

        self.u = us.reshape(xshape)
        self.v = vs.reshape(xshape)
        self.w = ws.reshape(xshape)
        self.phix = phixs.reshape(xshape)
        self.phiy = phiys.reshape(xshape)

        return self.u, self.v, self.w, self.phix, self.phiy


    def plot(self, c, invert_y=False, plot_type=1, vec='w',
             deform_u=False, deform_u_sf=100.,
             filename='',
             ax=None, figsize=(3.5, 2.), save=True,
             add_title=False, title='',
             colorbar=False, cbar_nticks=2, cbar_format=None,
             cbar_title='', cbar_fontsize=10,
             aspect='equal', clean=True, dpi=400,
             texts=[], xs=None, ys=None, gridx=300, gridy=300,
             num_levels=400, vecmin=None, vecmax=None):
        r"""Contour plot for a Ritz constants vector.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``
        deform_u : bool, optional
            If ``True`` the contour plot will look deformed.
        deform_u_sf : float, optional
            The scaling factor used to deform the contour.
        invert_y : bool, optional
            Inverts the `y` axis of the plot. It may be used to match
            the coordinate system of the finite element models created
            using the ``desicos.abaqus`` module.
        plot_type : int, optional
            For cylinders only ``4`` and ``5`` are valid.
            For cones all the following types can be used:

            - ``1``: concave up (with ``invert_y=False``) (default)
            - ``2``: concave down (with ``invert_y=False``)
            - ``3``: stretched closed
            - ``4``: stretched opened (`r \times y` vs. `a`)
            - ``5``: stretched opened (`y` vs. `a`)

        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
        dpi : int, optional
            Resolution of the saved file in dots per inch.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            the `name` parameter of the ``AeroPistonStiffPanel`` object will
            be used.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it.
        figsize : tuple, optional
            The figure size given by ``(width, height)``.
        add_title : bool, optional
            If a title should be added to the figure.
        title : str, optional
            If any string is given ``add_title`` will be ignored and the given
            title added to the contour plot.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method.
        clean : bool, optional
            Clean axes ticks, grids, spines etc.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
        vecmin : float, optional
            Minimum value for the contour scale (useful to compare with other
            results). If not specified it will be taken from the calculated
            field.
        vecmax : float, optional
            Maximum value for the contour scale.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.

        """
        msg('Plotting contour...')

        ubkp, vbkp, wbkp, phixbkp, phiybkp = (self.u, self.v, self.w,
                                              self.phix, self.phiy)

        import matplotlib.pyplot as plt
        import matplotlib

        msg('Computing field variables...', level=1)
        displs = ['u', 'v', 'w', 'phix', 'phiy']
        if vec in displs:
            self.uvw(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
            field = getattr(self, vec)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))
        msg('Finished!', level=1)

        Xs = self.Xs
        Ys = self.Ys

        if vecmin is None:
            vecmin = field.min()
        if vecmax is None:
            vecmax = field.max()

        levels = linspace(vecmin, vecmax, num_levels)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            if isinstance(ax, matplotlib.axes.Axes):
                ax = ax
                fig = ax.figure
                save = False
            else:
                raise ValueError('ax must be an Axes object')

        x = Ys
        y = Xs

        if deform_u:
            if vec in displs:
                pass
            else:
                self.uvw(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
            field_u = self.u
            field_v = self.v
            y -= deform_u_sf*field_u
            x += deform_u_sf*field_v
        contour = ax.contourf(x, y, field, levels=levels)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbarticks = linspace(vecmin, vecmax, cbar_nticks)
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            cbar.outline.remove()
            cbar.ax.tick_params(labelsize=fsize, pad=0., tick2On=False)

        if invert_y == True:
            ax.invert_yaxis()
        ax.invert_xaxis()

        if title != '':
            ax.set_title(str(title))

        elif add_title:
            if self.analysis.last_analysis == 'static':
                ax.set_title('$m_1, n_1={0}, {1}$'.format(self.m1, self.n1))

            elif self.analysis.last_analysis == 'lb':
                ax.set_title(
       r'$m_1, n_1={0}, {1}$, $\lambda_{{CR}}={4:1.3e}$'.format(self.m1,
           self.n1, self.eigvals[0]))

        fig.tight_layout()
        ax.set_aspect(aspect)

        ax.grid(False)
        ax.set_frame_on(False)
        if clean:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        else:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        for kwargs in texts:
            ax.text(transform=ax.transAxes, **kwargs)

        if save:
            if not filename:
                filename = 'test.png'
            fig.savefig(filename, transparent=True,
                        bbox_inches='tight', pad_inches=0.05, dpi=dpi)
            plt.close()

        if ubkp is not None:
            self.u = ubkp
        if vbkp is not None:
            self.v = vbkp
        if wbkp is not None:
            self.w = wbkp
        if phixbkp is not None:
            self.phix = phixbkp
        if phiybkp is not None:
            self.phiy = phiybkp

        msg('finished!')

        return ax


    def save(self):
        """Save the ``AeroPistonStiffPanel`` object using ``cPickle``

        Notes
        -----
        The pickled file will have the name stored in
        ``AeroPistonStiffPanel.name`` followed by a
        ``'.AeroPistonStiffPanel'`` extension.

        """
        name = self.name + '.AeroPistonStiffPanel'
        msg('Saving AeroPistonStiffPanel to {0}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)

