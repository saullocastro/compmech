from __future__ import division
import gc
import os
import sys
import traceback
from collections import Iterable
import time
import cPickle
import __main__

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigsh
from scipy.optimize import leastsq
from numpy import linspace, pi, cos, sin, tan, deg2rad

import compmech.composite.laminate as laminate
from compmech.logger import msg, warn
from compmech.constants import DOUBLE
import non_linear
import modelDB


def load(name):
    if '.KPanelT' in name:
        return cPickle.load(open(name, 'rb'))
    else:
        return cPickle.load(open(name + '.KPanelT', 'rb'))


class KPanelT(object):
    """
    """

    def __init__(self):
        self.name = ''
        self.forces = []
        self.alphadeg = 0.
        self.alpharad = 0.
        self.is_cylinder = None
        self.last_analysis = None

        # boundary conditions
        self.inf = 1.e8 # used to define high stiffnesses
        self.zero = 0. # used to define zero stiffnesses
        self.bc = None
        self.kuBot = self.inf
        self.kvBot = self.inf
        self.kwBot = self.inf
        self.kphixBot = 0.
        self.kphitBot = 0.
        self.kuTop = self.inf
        self.kvTop = self.inf
        self.kwTop = self.inf
        self.kphixTop = 0.
        self.kphitTop = 0.
        self.kuLeft = self.inf
        self.kvLeft = self.inf
        self.kwLeft = self.inf
        self.kphixLeft = 0.
        self.kphitLeft = 0.
        self.kuRight = self.inf
        self.kvRight = self.inf
        self.kwRight = self.inf
        self.kphixRight = 0.
        self.kphitRight = 0.

        # default equations
        self.model = 'fsdt_donnell_bc1'

        # approximation series
        self.m2 = 100
        self.n3 = 100
        self.m4 = 40
        self.n4 = 40

        # analytical integration for cones
        self.s = 79

        # numerical integration
        self.nx = 160
        self.nt = 160

        # internal pressure measured in force/area
        self.P = 0.

        # loads
        self.Fx = None
        self.Ft = None
        self.NxxTop = None
        self.NxtTop = None
        self.NttLeft = None
        self.NtxLeft = None

        # initial imperfection
        self.c0 = None
        self.m0 = 0
        self.n0 = 0
        self.funcnum = 2

        self.r1 = None
        self.r2 = None
        self.L = None
        self.tmindeg = None
        self.tmaxdeg = None
        self.tminrad = None
        self.tmaxrad = None
        self.K = 5/6.
        self.sina = None
        self.cosa = None

        # material
        self.laminaprop = None
        self.plyt = None
        self.laminaprops = []
        self.stack = []
        self.plyts = []

        # constitutive law
        self.F = None
        self.force_orthotropic_laminate = False

        # eigenvalue analysis
        self.num_eigvalues = 50
        self.num_eigvalues_print = 5

        # non-linear algorithm
        self.NL_method = 'NR' # Newton-Raphson
        self.modified_NR = True # modified Newton-Raphson
        self.line_search = True
        self.compute_every_n = 6 # for modified Newton-Raphson

        # incrementation
        self.initialInc = 0.1
        self.minInc = 1.e-3
        self.maxInc = 1.

        # convergence criteria
        self.absTOL = 1.e-3
        self.relTOL = 1.e-3

        self.cs = []
        self.increments = []

        #self.maxNumInc = 100
        self.maxNumIter = 30

        # output queries
        self.out_num_cores = 4

        # numerical integration
        self.ni_num_cores = 4 # showed to scale well up to 4
        self.ni_method = 'trapz2d'

        # outputs
        self.outputs = {}

        self._clear_matrices()


    def _clear_matrices(self):
        self.k0 = None
        self.kT = None
        self.kG0 = None
        self.kG0_Fx = None
        self.kG0_Ft = None
        self.kG0_Fxt = None
        self.kG0_Ftx = None
        self.kG = None
        self.kL = None
        self.lam = None
        self.u = None
        self.v = None
        self.w = None
        self.phix = None
        self.phit = None
        self.Xs = None
        self.Ts = None

        gc.collect()


    def _rebuild(self):
        if not self.name:
            try:
                self.name = os.path.basename(__main__.__file__).split('.py')[0]
            except AttributeError:
                warn('KPanelT name unchanged')

        self.model = self.model.lower()

        valid_models = sorted(modelDB.db.keys())

        if not self.model in valid_models:
            raise ValueError('ERROR - valid models are:\n    ' +
                     '\n    '.join(valid_models))

        # boundary conditions
        inf = self.inf
        zero = self.zero

        if inf > 1.e8:
            warn('inf reduced to 1.e8 due to the verified ' +
                 'numerical instability for higher values', level=2)
            inf = 1.e8

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
                    setattr(self, 'kphit' + sufix, inf)
                elif bcs[k] == 'ss2':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphit' + sufix, inf)
                elif bcs[k] == 'ss3':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphit' + sufix, inf)
                elif bcs[k] == 'ss4':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphit' + sufix, inf)

                elif bcs[k] == 'cc1':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphit' + sufix, inf)
                elif bcs[k] == 'cc2':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphit' + sufix, inf)
                elif bcs[k] == 'cc3':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphit' + sufix, inf)
                elif bcs[k] == 'cc4':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphit' + sufix, inf)

                elif bcs[k] == 'free':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, zero)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphit' + sufix, zero)

                else:
                    txt = '"{}" is not a valid boundary condition!'.format(bc)
                    raise ValueError(txt)

        self.tminrad = deg2rad(self.tmindeg)
        self.tmaxrad = deg2rad(self.tmaxdeg)
        self.alpharad = deg2rad(self.alphadeg)
        self.sina = sin(self.alpharad)
        self.cosa = cos(self.alpharad)

        if self.L is None:
            raise ValueError('The length L must be specified')

        if not self.r2:
            if not self.r1:
                raise ValueError('Radius r1 or r2 must be specified')
            else:
                self.r2 = self.r1 - self.L*self.sina
        else:
            self.r1 = self.r2 + self.L*self.sina

        if not self.laminaprops:
            self.laminaprops = [self.laminaprop for i in self.stack]
        if not self.plyts:
            self.plyts = [self.plyt for i in self.stack]

        if self.alpharad == 0:
            self.is_cylinder = True
        else:
            self.is_cylinder = False

        self.maxInc = max(self.initialInc, self.maxInc)

        # axial load
        size = max(self.n3, self.n4)+1
        if self.NxxTop is not None:
            check = False
            if isinstance(self.NxxTop, np.ndarray):
                if self.NxxTop.ndim == 1:
                    assert self.NxxTop.shape[0] == size
                    check=True
            if not check:
                raise ValueError('Invalid NxxTop input')
        else:
            self.NxxTop = np.zeros(size, dtype=DOUBLE)

        if self.NxtTop is not None:
            check = False
            if isinstance(self.NxtTop, np.ndarray):
                if self.NxtTop.ndim == 1:
                    assert self.NxtTop.shape[0] == size
                    check=True
            if not check:
                raise ValueError('Invalid NxtTop input')
        else:
            self.NxtTop = np.zeros(size, dtype=DOUBLE)

        size = 2*max(self.m2, self.m4)+1
        if self.NttLeft is not None:
            check = False
            if isinstance(self.NttLeft, np.ndarray):
                if self.NttLeft.ndim == 1:
                    assert self.NttLeft.shape[0] == size
                    check=True
            if not check:
                raise ValueError('Invalid NttLeft input')
        else:
            self.NttLeft = np.zeros(size, dtype=DOUBLE)

        if self.NtxLeft is not None:
            check = False
            if isinstance(self.NtxLeft, np.ndarray):
                if self.NtxLeft.ndim == 1:
                    assert self.NtxLeft.shape[0] == size
                    check=True
            if not check:
                raise ValueError('Invalid NtxLeft input')
        else:
            self.NtxLeft = np.zeros(size, dtype=DOUBLE)

        if self.Fx is not None:
            self.NxxTop[0] = self.Fx/((self.tmaxrad-self.tminrad)*self.r2)
            msg('NxxTop[0] calculated from Fx', level=2)

        if self.Ft is not None:
            self.NttLeft[0] = self.Ft/self.L
            msg('NttLeft[0] calculated from Ft', level=2)

        if self.laminaprop is None:
            raise ValueError('laminaprop must be defined')


    def get_size(self):
        r"""Calculates the size of the stiffness matrices

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
        num2 = modelDB.db[self.model]['num2']
        num3 = modelDB.db[self.model]['num3']
        num4 = modelDB.db[self.model]['num4']
        self.size = (num0 + num1 + num2*self.m2 + num3*self.n3 +
                num4*self.m4*self.n4)
        return self.size


    def from_DB(self, name):
        """Load cone / cylinder data from the local database

        Parameters
        ----------
        name : str
            A key contained in the ``ccs`` dictionary of module
            :mod:`compmech.conecyl.conecylDB`.

        """
        try:
            attrs = ['r1', 'r2', 'L', 'alphadeg', 'plyt', 'stack']
            cc = ccs[name]
            self.laminaprop = laminaprops[cc['laminapropKey']]
            for attr in attrs:
                setattr(self, attr, cc.get(attr, getattr(self, attr)))
        except:
            raise ValueError('Invalid data-base entry!')


    def _default_field(self, xs, ts, gridx, gridt):
        if xs is None or ts is None:
            xs = linspace(-self.L/2., +self.L/2, gridx)
            ts = linspace(self.tminrad, self.tmaxrad, gridt)
            xs, ts = np.meshgrid(xs, ts, copy=False)
        xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
        ts = np.atleast_1d(np.array(ts, dtype=DOUBLE))
        xshape = xs.shape
        tshape = ts.shape
        if xshape != tshape:
            raise ValueError('Arrays xs and ts must have the same shape')
        self.Xs = xs
        self.Ts = ts
        xs = xs.ravel()
        ts = ts.ravel()

        return xs, ts, xshape, tshape


    def _calc_linear_matrices(self, combined_load_case=None):
        self._rebuild()
        msg('Calculating linear matrices... ', level=2)

        fk0, fk0_cyl, fkG0, fkG0_cyl, k0edges = modelDB.get_linear_matrices(
                                                    self, combined_load_case)
        model = self.model
        alpharad = self.alpharad
        cosa = self.cosa
        r1 = self.r1
        r2 = self.r2
        L = self.L
        tminrad = self.tminrad
        tmaxrad = self.tmaxrad
        m2 = self.m2
        n3 = self.n3
        m4 = self.m4
        n4 = self.n4
        s = self.s
        laminaprops = self.laminaprops
        plyts = self.plyts
        stack = self.stack

        Fx = self.NxxTop[0]*((tmaxrad-tminrad)*r2)
        Ft = self.NttLeft[0]*L
        Fxt = self.NxtTop[0]*((tmaxrad-tminrad)*r2)
        Ftx = self.NtxLeft[0]*L

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

        if self.force_orthotropic_laminate:
            msg('')
            msg('Forcing orthotropic laminate...', level=2)
            F[0, 2] = 0. # A16
            F[1, 2] = 0. # A26
            F[2, 0] = 0. # A61
            F[2, 1] = 0. # A62

            F[0, 5] = 0. # B16
            F[5, 0] = 0. # B61
            F[1, 5] = 0. # B26
            F[5, 1] = 0. # B62

            F[3, 2] = 0. # B16
            F[2, 3] = 0. # B61
            F[4, 2] = 0. # B26
            F[2, 4] = 0. # B62

            F[3, 5] = 0. # D16
            F[4, 5] = 0. # D26
            F[5, 3] = 0. # D61
            F[5, 4] = 0. # D62

            if F.shape[0] == 8:
                F[6, 7] = 0. # A45
                F[7, 6] = 0. # A54

        self.lam = lam
        self.F = F

        if self.is_cylinder:
            k0 = fk0_cyl(r1, L, tminrad, tmaxrad, F, m2, n3, m4, n4)

            if not combined_load_case:
                kG0 = fkG0_cyl(Fx, Ft, Fxt, Ftx, r1, L, tminrad, tmaxrad,
                        m2, n3, m4, n4)
            else:
                kG0_Fx = fkG0_cyl(Fx, 0, 0, 0, r1, L, tminrad, tmaxrad,
                        m2, n3, m4, n4)
                kG0_Ft = fkG0_cyl(0, Ft, 0, 0, r1, L, tminrad, tmaxrad,
                        m2, n3, m4, n4)
                kG0_Fxt = fkG0_cyl(0, 0, Fxt, 0, r1, L, tminrad, tmaxrad,
                        m2, n3, m4, n4)
                kG0_Ftx = fkG0_cyl(0, 0, 0, Ftx, r1, L, tminrad, tmaxrad,
                        m2, n3, m4, n4)
        else:
            k0 = fk0(r1, L, tminrad, tmaxrad, F, m2, n3, m4, n4, alpharad, s)

            if not combined_load_case:
                kG0 = fkG0(Fx, Ft, Fxt, Ftx, r1, L, tminrad, tmaxrad, m2, n3,
                        m4, n4, alpharad, s)
            else:
                kG0_Fx = fkG0(Fx, 0, 0, 0, r1, L, tminrad, tmaxrad, m2,
                        n3, m4, n4, alpharad, s)
                kG0_Ft = fkG0(0, Ft, 0, 0, r1, L, tminrad, tmaxrad, m2,
                        n3, m4, n4, alpharad, s)
                kG0_Fxt = fkG0(0, 0, Fxt, 0, r1, L, tminrad, tmaxrad, m2,
                        n3, m4, n4, alpharad, s)
                kG0_Ftx = fkG0(0, 0, 0, Ftx, r1, L, tminrad, tmaxrad, m2,
                        n3, m4, n4, alpharad, s)

        if k0edges is not None:
            msg('Applying elastic constraints!', level=3)
            k0 = k0 + k0edges

        assert np.any((np.isnan(k0.data) | np.isinf(k0.data))) == False

        self.k0 = k0
        if not combined_load_case:
            assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
            self.kG0 = kG0
        else:
            assert np.any((np.isnan(kG0_Fx.data)
                           | np.isinf(kG0_Fx.data))) == False
            assert np.any((np.isnan(kG0_Ft.data)
                           | np.isinf(kG0_Ft.data))) == False
            assert np.any((np.isnan(kG0_Fxt.data)
                           | np.isinf(kG0_Fxt.data))) == False
            assert np.any((np.isnan(kG0_Ftx.data)
                           | np.isinf(kG0_Ftx.data))) == False
            self.kG0_Fx = kG0_Fx
            self.kG0_Ft = kG0_Ft
            self.kG0_Fxt = kG0_Fxt
            self.kG0_Ftx = kG0_Ftx

        self.k0 = k0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process

        gc.collect()

        msg('finished!', level=2)


    def lb(self, c=None, tol=0, combined_load_case=None):
        """Performs a linear buckling analysis

        The following parameters of the ``KPanelT`` object will affect the
        linear buckling analysis:

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

            - ``1`` : find the critical Fx for a fixed Fxt
            - ``2`` : find the critical Fx for a fixed Ft
            - ``3`` : find the critical Ft for a fixed Ftx
            - ``4`` : find the critical Ft for a fixed Fx

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``KPanelT`` object and the `i^{th}` eigenvector in the
        ``eigvecs[i-1, :]`` parameter.

        """
        if not modelDB.db[self.model]['linear buckling']:
            msg('________________________________________________')
            msg('')
            warn('Model {} cannot be used in linear buckling analysis!'.
                 format(self.model))
            msg('________________________________________________')

        msg('Running linear buckling analysis...')

        self._calc_linear_matrices(combined_load_case=combined_load_case)

        msg('Eigenvalue solver... ', level=2)

        if not combined_load_case:
            A = self.k0
            M = -self.kG0
        elif combined_load_case == 1:
            A = self.k0 + self.kG0_Fxt
            M = -self.kG0_Fx
        elif combined_load_case == 2:
            A = self.k0 + self.kG0_Ft
            M = -self.kG0_Fx
        elif combined_load_case == 3:
            A = self.k0 + self.kG0_Ftx
            M = -self.kG0_Ft
        elif combined_load_case == 4:
            A = self.k0 + self.kG0_Fx
            M = -self.kG0_Ft

        db = modelDB.db
        num0 = db[self.model]['num0']
        num1 = db[self.model]['num1']
        num2 = db[self.model]['num2']
        num3 = db[self.model]['num3']
        num4 = db[self.model]['num4']

        print A.data.shape
        print M.data.shape
        print self.get_size()
        if True:
            A, M = M, A
            eigvals, eigvecs = eigsh(A=A, k=self.num_eigvalues, which='SM',
                                     M=M, tol=tol, sigma=-1.,
                                     mode='cayley')
            eigvals = 1./eigvals
        else:
            from scipy.linalg import eigh

            A = A.toarray()
            M = M.toarray()
            A, M = M, A
            eigvals, eigvecs = eigh(a=A, b=M, lower=False)
            eigvals = eigvals

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        msg('finished!', level=2)

        msg('first {} eigenvalues:'.format(self.num_eigvalues_print), level=1)
        for eig in eigvals[:self.num_eigvalues_print]:
            msg('{}'.format(eig), level=2)
        self.last_analysis = 'lb'


    def calc_NL_matrices(self, c, num_cores=None):
        r"""Calculates the non-linear stiffness matrices

        Parameters
        ----------
        c : np.ndarray
            Ritz constants representing the current state to calculate the
            stiffness matrices.
        num_cores : int, optional
            Number of CPU cores used by the algorithm.

        Notes
        -----
        Nothing is returned, the calculated matrices

        """
        if not num_cores:
            num_cores=self.ni_num_cores

        if self.k0 is None:
            self._calc_linear_matrices()

        msg('Calculating non-linear matrices...', level=2)
        alpharad = self.alpharad
        r2 = self.r2
        L = self.L
        F = self.F
        m2 = self.m2
        n3 = self.n3
        m4 = self.m4
        n4 = self.n4
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0
        funcnum = self.funcnum

        nlmodule = modelDB.db[self.model]['non-linear']
        if nlmodule:
            calc_k0L = nlmodule.calc_k0L
            calc_kG = nlmodule.calc_kG
            calc_kLL = nlmodule.calc_kLL

            kG = calc_kG(c, alpharad, r1, L, F, m2, n3, m4, n4,
                         nx=self.nx, nt=self.nt,
                         num_cores=num_cores,
                         method=self.ni_method,
                         c0=c0, m0=m0, n0=n0)
            k0L = calc_k0L(c, alpharad, r1, L, F, m2, n3, m4, n4,
                           nx=self.nx, nt=self.nt,
                           num_cores=num_cores,
                           method=self.ni_method,
                           c0=c0, m0=m0, n0=n0)
            kLL = calc_kLL(c, alpharad, r1, L, F, m2, n3, m4, n4,
                           nx=self.nx, nt=self.nt,
                           num_cores=num_cores,
                           method=self.ni_method,
                           c0=c0, m0=m0, n0=n0)

        else:
            raise ValueError(
            'Non-Linear analysis not implemented for model {0}'.format(
                self.model))

        kL0 = k0L.T

        #TODO maybe slow...
        self.kT = self.k0 + k0L + kL0 + kLL + kG

        #NOTE intended for non-linear eigenvalue analyses
        self.kL = self.k0 + k0L + kL0 + kLL
        self.kG = kG

        msg('finished!', level=2)


    def uvw(self, c, xs=None, ts=None, gridx=300, gridt=300):
        r"""Calculates the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phit`` of the ``KPanelT`` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        xs : np.ndarray
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ts : np.ndarray
            The ``theta`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridt : int
            Number of points along the `theta` where to calculate the
            displacement field.

        Returns
        -------
        out : tuple
            A tuple of ``np.ndarrays`` containing
            ``(u, v, w, phix, phit)``.

        Notes
        -----
        The returned values ``u```, ``v``, ``w``, ``phix``, ``phit`` are
        stored as parameters with the same name in the ``KPanelT`` object.

        """
        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)
        alpharad = self.alpharad
        m2 = self.m2
        n3 = self.n3
        m4 = self.m4
        n4 = self.n4
        r2 = self.r2
        L = self.L
        tminrad = self.tminrad
        tmaxrad = self.tmaxrad
        model = self.model

        fuvw = modelDB.db[model]['commons'].fuvw
        us, vs, ws, phixs, phits = fuvw(c, m2, n3, m4, n4, L, tminrad,
                tmaxrad, xs, ts, alpharad, self.out_num_cores)

        self.u = us.reshape(xshape)
        self.v = vs.reshape(xshape)
        self.w = ws.reshape(xshape)
        self.phix = phixs.reshape(xshape)
        self.phit = phits.reshape(xshape)

        return self.u, self.v, self.w, self.phix, self.phit


    def strain(self, c, xs=None, ts=None, gridx=300, gridt=300):
        r"""Calculates the strain field

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation.
        xs : np.ndarray, optional
            The `x` coordinates where to calculate the strains.
        ts : np.ndarray, optional
            The `\theta` coordinates where to calculate the strains, must
            have the same shape as ``xs``.
        gridx : int, optional
            When ``xs`` and ``ts`` are not supplied, ``gridx`` and ``gridt``
            are used.
        gridt : int, optional
            When ``xs`` and ``ts`` are not supplied, ``gridx`` and ``gridt``
            are used.

        """
        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)

        alpharad = self.alpharad
        L = self.L
        r2 = self.r2
        sina = self.sina
        cosa = self.cosa
        m2 = self.m2
        n3 = self.n3
        m4 = self.m4
        n4 = self.n4
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0
        funcnum = self.funcnum
        model = self.model
        NL_kinematics = model.split('_')[1]
        fstrain = modelDB.db[model]['commons'].fstrain
        e_num = modelDB.db[model]['e_num']

        if 'donnell' in NL_kinematics:
            int_NL_kinematics = 0
        elif 'sanders' in NL_kinematics:
            int_NL_kinematics = 1
        else:
            raise NotImplementedError(
                '{} is not a valid NL_kinematics option'.format(NL_kinematics))

        es = fstrain(c, sina, cosa, xs, ts, r1, L,
                     m2, n3, m4, n4, c0, m0, n0, funcnum, int_NL_kinematics,
                     self.out_num_cores)

        return es.reshape((xshape + (e_num,)))


    def stress(self, c, xs=None, ts=None, gridx=300, gridt=300):
        r"""Calculates the stress field

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation.
        xs : np.ndarray, optional
            The `x` coordinates where to calculate the strains.
        ts : np.ndarray, optional
            The `\theta` coordinates where to calculate the strains, must
            have the same shape as ``xs``.
        gridx : int, optional
            When ``xs`` and ``ts`` are not supplied, ``gridx`` and ``gridt``
            are used.
        gridt : int, optional
            When ``xs`` and ``ts`` are not supplied, ``gridx`` and ``gridt``
            are used.

        """
        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)

        F = self.F
        alpharad = self.alpharad
        L = self.L
        r2 = self.r2
        sina = self.sina
        cosa = self.cosa
        m2 = self.m2
        n3 = self.n3
        m4 = self.m4
        n4 = self.n4
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0
        funcnum = self.funcnum
        model = self.model
        NL_kinematics = model.split('_')[1]
        fstress = modelDB.db[model]['commons'].fstress
        e_num = modelDB.db[model]['e_num']

        if 'donnell' in NL_kinematics:
            int_NL_kinematics = 0
        elif 'sanders' in NL_kinematics:
            int_NL_kinematics = 1
        else:
            raise NotImplementedError(
                    '{} is not a valid NL_kinematics option'.format(
                    NL_kinematics))

        Ns = fstress(c, F, sina, cosa, xs, ts, r1, L,
                     m2, n3, m4, n4, c0, m0, n0, funcnum, int_NL_kinematics,
                     self.out_num_cores)
        return Ns.reshape((xshape + (e_num,)))


    def calc_fint(self, c, m=1):
        r"""Calculates the internal force vector `\{F_{int}\}`

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the internal
            forces.
        m : integer, optional
            A multiplier to be applied to ``nx`` and ``nt``, if one
            whishes to use more integration points.

        Returns
        -------
        fint : np.ndarray
            The internal force vector.

        """
        nlmodule = modelDB.db[self.model]['non-linear']
        fint = nlmodule.calc_fint_0L_L0_LL(c, self.alpharad, self.r1, self.L,
                                  self.F,
                                  self.m2, self.n3, self.m4, self.n4,
                                  self.nx*m, self.nt*m, self.ni_num_cores,
                                  self.ni_method, self.c0, self.m0, self.n0)
        fint += self.k0*c

        return fint


    def _calc_kT(self, c):
        nlmodule = modelDB.db[self.model]['non-linear']
        kT = nlmodule.calc_kT(c, self.alpharad, self.r1, self.L,
                                self.F,
                                self.m2, self.n3, self.m4, self.n4,
                                self.nx, self.nt, self.ni_num_cores,
                                self.ni_method, self.c0, self.m0, self.n0)
        return kT


    def add_SPL(self, PL, pt=0.5, theta=0.):
        """Add a Single Perturbation Load `\{{F_{PL}}_i\}`

        Adds a perturbation load to the ``KPanelT`` object, the perturbation
        load is a particular case of the punctual load with only a normal
        component.

        Parameters
        ----------
        PL : float
            The perturbation load value.
        pt : float
            The normalized position along the `x` axis in which the new SPL
            will be included.
        theta : float
            The angular position in radians of the new SPL.

        Notes
        -----
        Each single perturbation load is added to the ``forces`` parameter
        of the ``KPanelT`` object, which may be changed by the analyst at
        any time.

        """
        self._rebuild()
        self.forces.append([pt*self.L, theta, 0., 0., PL])


    def add_force(self, x, theta, fx, ftheta, fz):
        r"""Add a punctual force

        Adds a force vector `\{f_x, f_\theta, f_z\}^T` to the ``forces``
        parameter of the ``KPanelT`` object.

        Parameters
        ----------
        x : float
            The `x` position.
        theta : float
            The `\theta` position in radians.
        fx : float
            The `x` component of the force vector.
        ftheta : float
            The `\theta` component of the force vector.
        fz : float
            The `z` component of the force vector.

        """
        self.forces.append([x, theta, fx, ftheta, fz])


    def calc_fext(self, inc=None, silent=False):
        """Calculates the external force vector `\{F_{ext}\}`

        Recall that:

        .. math::

            \{F_{ext}\}=\{{F_{ext}}_0\} + \{{F_{ext}}_\lambda\}

        such that the terms in `\{{F_{ext}}_0\}` are constant and the terms in
        `\{{F_{ext}}_\lambda\}` will be scaled by the parameter ``inc``.

        Parameters
        ----------
        inc : float, optional
            Since this function is called during the non-linear analysis,
            ``inc`` will multiply the terms `\{{F_{ext}}_\lambda\}`.

        silent : bool, optional
            A boolean to tell whether the log messages should be printed.

        Returns
        -------

        fext : np.ndarray
            The external force vector

        """
        msg('Calculating external forces...', level=2, silent=silent)
        if inc is None:
            NxxTop = self.NxxTop
            NttLeft = self.NttLeft
        else:
            NxxTop = inc*self.NxxTop
            NttLeft = inc*self.NttLeft
        sina = self.sina
        cosa = self.cosa
        r2 = self.r2
        L = self.L
        tminrad = self.tminrad
        tmaxrad = self.tmaxrad
        m2 = self.m2
        n3 = self.n3
        m4 = self.m4
        n4 = self.n4
        model = self.model

        if not model in modelDB.db.keys():
            raise ValueError(
                    '{} is not a valid model option'.format(model))

        db = modelDB.db
        num0 = db[model]['num0']
        num1 = db[model]['num1']
        num2 = db[model]['num2']
        num3 = db[model]['num3']
        num4 = db[model]['num4']
        dofs = db[model]['dofs']
        fg = db[model]['commons'].fg

        size = self.get_size()

        g = np.zeros((dofs, size), dtype=DOUBLE)
        fext = np.zeros(size, dtype=DOUBLE)

        # punctual forces
        for i, force in enumerate(self.forces):
            x, theta, fx, ftheta, fz = force
            fg(g, m2, n3, m4, n4, x, theta, L, tminrad, tmaxrad)

            if dofs == 3:
                fpt = np.array([[fx, ftheta, fz]])
            elif dofs == 5:
                fpt = np.array([[fx, ftheta, fz, 0, 0]])
            fext += fpt.dot(g).ravel()

        # axial load
        fext[0] += NxxTop[0]*(tmaxrad-tminrad)*r2
        #fext[1] += NttLeft[0]*L
        if 'bc2' in model or 'bc4' in model:
            for i4 in range(1, m4+1):
                for j4 in range(1, n4+1):
                    row = (num0 + num1 + num2*m2 + num3*n3 + (j4-1)*num4*m4 +
                            (i4-1)*num4)
                    rowNxx = 1+2*(j4-1)
                    #FIXME check this
                    raise
                    fext[row+0]+=(NxxTop[rowNxx+0]*pi*r2)
                    fext[row+1]+=(NxxTop[rowNxx+1]*pi*r2)

        msg('finished!', level=2, silent=silent)

        return fext


    def static(self, NLgeom=False, silent=False):
        """Static analysis for cones and cylinders

        The analysis can be linear or non-linear. In case of a non-linear
        analysis the following parameters of the ``KPanelT`` object will
        affect the non-linear analysis:

        ====================    ==========================================
        non-linear algorithm    description
        ====================    ==========================================
        ``NL_method``           ``'NR'`` for the Newton-Raphson
                                ``'arc_length'`` for the Arc-Length method
        ``line_search``         activate line_search (for Newton-Raphson
                                methods only)
        ``modified_NR``         activate the modified Newton-Raphson
        ``compute_every_n``     if ``modified_NR=True``, the non-linear
                                matrices will be updated at every `n`
                                iterations
        ====================    ==========================================

        ==============     =================================================
        incrementation     description
        ==============     =================================================
        ``initialInc``     initial load increment size. In the arc-length
                           method it will be the initial value for
                           `\lambda`
        ``minInc``         minimum increment size; if achieved the analysis
                           is terminated. The arc-length method will use
                           this parameter to terminate when the minimum
                           arc-length increment is smaller than ``minInc``
        ``maxInc``         maximum increment size
        ==============     =================================================

        ====================    ============================================
        convergence criteria    description
        ====================    ============================================
        ``absTOL``              the convergence is achieved when the maximum
                                residual force is smaller than this value
        ``maxNumIter``          maximum number of iteration; if achieved the
                                load increment is bisected
        ====================    ============================================

        =====================    ===========================================
        numerical integration    description
        =====================    ===========================================
        ``ni_num_cores``         number of cores used for the numerical
                                 integration
        ``ni_method``            ``'trapz2d'`` for 2-D Trapezoidal's
                                 ``'simps2d'`` for 2-D Simpsons' integration
        =====================    ===========================================

        Parameters
        ----------
        NLgeom : bool
            Flag to indicate whether a linear or a non-linear analysis is to
            be performed.

        silent : bool, optional
            A boolean to tell whether the log messages should be printed.

        Returns
        -------
        cs : list
            A list containing the Ritz constants for each load increment of
            the static analysis. The list will have only one entry in case
            of a linear analysis.

        Notes
        -----
        The returned ``cs`` is stored in the ``cs`` parameter of the
        ``KPanelT`` object. The actual increments used in the non-linear
        analysis are stored in the ``increments`` parameter.

        """
        self.cs = []
        self.increments = []
        if NLgeom:
            if not modelDB.db[self.model]['non-linear static']:
                msg('________________________________________________',
                    silent=silent)
                msg('', silent=silent)
                warn(
            'Model {} cannot be used in non-linear static analysis!'.
            format(self.model), silent=silent)
                msg('________________________________________________',
                    silent=silent)

            msg('Started Non-Linear Static Analysis', silent=silent)
            self._calc_linear_matrices()
            if self.NL_method == 'NR':
                non_linear.NR(self)
            if self.NL_method == 'NR_lebofsky':
                non_linear.NR_lebofsky(self)
            elif self.NL_method == 'NR_Broyden':
                non_linear.NR_Broyden(self)
            elif self.NL_method == 'arc_length':
                non_linear.arc_length(self)
        else:
            if not modelDB.db[self.model]['linear static']:
                msg('________________________________________________',
                    level=1, silent=silent)
                msg('', level=1, silent=silent)
                warn('Model {} cannot be used in linear static analysis!'.
                     format(self.model), level=1, silent=silent)
                lob('________________________________________________',
                    level=1, silent=silent)

            msg('Started Linear Static Analysis', silent=silent)
            self._calc_linear_matrices()
            fext = self.calc_fext()
            c = spsolve(self.k0, fext)
            self.cs.append(c)
            self.increments.append(1.)
            msg('Finished Linear Static Analysis', silent=silent)

        self.last_analysis = 'static'

        return self.cs


    def plot(self, c, invert_x=False, plot_type=1, vec='w',
             deform_u=False, deform_u_sf=100.,
             filename='',
             ax=None, figsize=(3.5, 2.), save=True,
             add_title=True, title='',
             colorbar=False, cbar_nticks=2, cbar_format=None,
             cbar_title='', cbar_fontsize=10,
             aspect='equal', clean=True, dpi=400,
             texts=[], xs=None, ts=None, gridx=300, gridt=300,
             num_levels=400):
        r"""Contour plot for a Ritz constants vector.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phit'``
            - Strain: ``'exx'``, ``'ett'``, ``'gxt'``, ``'kxx'``, ``'ktt'``,
              ``'kxt'``, ``'gtz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Ntt'``, ``'Nxt'``, ``'Mxx'``, ``'Mtt'``,
              ``'Mxt'``, ``'Qt'``, ``'Qx'``
        deform_u : bool, optional
            If ``True`` the contour plot will look deformed.
        deform_u_sf : float, optional
            The scaling factor used to deform the contour.
        invert_x : bool, optional
            Inverts the `x` axis of the plot. It may be used to match
            the coordinate system of the finite element models created
            using the ``desicos.abaqus`` module.
        plot_type : int, optional
            For cylinders only ``4`` and ``5`` are valid.
            For cones all the following types can be used:

            - ``1``: concave up (with ``invert_x=False``) (default)
            - ``2``: concave down (with ``invert_x=False``)
            - ``3``: stretched closed
            - ``4``: stretched opened (`r \times \theta` vs. `L`)
            - ``5``: stretched opened (`\theta` vs. `L`)

        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
        dpi : int, optional
            Resolution of the saved file in dots per inch.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            the `name` parameter of the ``KPanelT`` object will be used.
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
        ts : np.ndarray, optional
            The ``theta`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridt : int, optional
            Number of points along the `theta` where to calculate the
            displacement field.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.

        """
        msg('Plotting contour...')

        ubkp, vbkp, wbkp, phixbkp, phitbkp = (self.u, self.v, self.w,
                                              self.phix, self.phit)

        import matplotlib.pyplot as plt
        import matplotlib

        msg('Computing field variables...', level=1)
        displs = ['u', 'v', 'w', 'phix', 'phit']
        strains = ['exx', 'ett', 'gxt', 'kxx', 'ktt', 'kxt', 'gtz', 'gxz']
        stresses = ['Nxx', 'Ntt', 'Nxt', 'Mxx', 'Mtt', 'Mxt', 'Qt', 'Qx']
        if vec in displs:
            self.uvw(c, xs=xs, ts=ts, gridx=gridx, gridt=gridt)
            field = getattr(self, vec)
        elif vec in strains:
            es = self.strain(c, xs=xs, ts=ts,
                             gridx=gridx, gridt=gridt)
            field = es[..., strains.index(vec)]
        elif vec in stresses:
            Ns = self.stress(c, xs=xs, ts=ts,
                             gridx=gridx, gridt=gridt)
            field = Ns[..., stresses.index(vec)]
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))
        msg('Finished!', level=1)

        Xs = self.Xs
        Ts = self.Ts

        vecmin = field.min()
        vecmax = field.max()

        levels = np.linspace(vecmin, vecmax, num_levels)

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

        def r(x):
            return self.r1 - self.sina*(x + self.L/2.)

        if self.is_cylinder:
            plot_type = 4
        if plot_type == 1:
            r_plot = self.r2/self.sina + self.L/2.-Xs
            r_plot_max = self.r2/self.sina + self.L
            y = r_plot_max - r_plot*cos(Ts*self.sina)
            x = r_plot*sin(Ts*self.sina)
        elif plot_type == 2:
            r_plot = self.r2/self.sina + self.L/2.-Xs
            y = r_plot*cos(Ts*self.sina)
            x = r_plot*sin(Ts*self.sina)
        elif plot_type == 3:
            r_plot = self.r2/self.sina + self.L/2.-Xs
            r_plot_max = self.r2/self.sina + self.L
            y = r_plot_max - r_plot*cos(Ts)
            x = r_plot*sin(Ts)
        elif plot_type == 4:
            x = r(Xs)*Ts
            y = Xs
        elif plot_type == 5:
            x = Ts
            y = Xs
        if deform_u:
            if vec in displs:
                pass
            else:
                self.uvw(c, xs=xs, ts=ts, gridx=gridx, gridt=gridt)
            field_u = self.u
            y -= deform_u_sf*field_u
        contour = ax.contourf(x, y, field, levels=levels)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbarticks = np.linspace(vecmin, vecmax, cbar_nticks)
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            cbar.outline.remove()
            cbar.ax.tick_params(labelsize=fsize, pad=0., tick2On=False)

        if invert_x == True:
            ax.invert_yaxis()
        ax.invert_xaxis()

        if title != '':
            ax.set_title(str(title))

        elif add_title:
            if self.last_analysis == 'static':
                ax.set_title('$m_2, n_3, m_4, n_4={0}, {1}, {2}, {3}$'.
                             format(self.m2, self.n3, self.m4, self.n4))

            elif self.last_analysis == 'lb':
                ax.set_title(
       r'$m_2, n_3, m_4, n_4={0}, {1}, {2}, {3}$, $\lambda_{{CR}}={4:1.3e}$'.
       format(self.m2, self.n3, self.m4, self.n4, self.eigvals[0]))

        fig.tight_layout()
        ax.set_aspect(aspect)

        if clean:
            ax.grid(False)
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.set_frame_on(False)

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
        if phitbkp is not None:
            self.phit = phitbkp

        msg('finished!')

        return ax


    def save(self):
        """Save the ``KPanelT`` object using ``cPickle``

        Notes
        -----
        The pickled file will have the name stored in ``KPanelT.name``
        followed by a ``'.KPanelT'`` extension.

        """
        name = self.name + '.KPanelT'
        msg('Saving KPanelT to {}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)

