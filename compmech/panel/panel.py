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
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig
from numpy import linspace, deg2rad

import compmech.composite.laminate as laminate
from compmech.analysis import Analysis
from compmech.logger import msg, warn
from compmech.constants import DOUBLE
from compmech.sparse import make_symmetric, remove_null_cols

import modelDB


def load(name):
    if '.Panel' in name:
        return cPickle.load(open(name, 'rb'))
    else:
        return cPickle.load(open(name + '.Panel', 'rb'))


class Panel(object):
    r"""General Panel class

    The approximation functions for the displacement fields are built using
    :ref:`Bardell's functions <theory_func_bardell>`.

    It will select the proper model according to the `r` (radius) and
    `alphadeg` (semi-vertex angle).


    """
    def __init__(self, a=None, b=None, y1=None, y2=None, r=None, alphadeg=None,
            stack=[], plyt=None, laminaprop=[]):
        self.a = a
        self.b = b
        self.y1 = y1
        self.y2 = y2
        self.r = r
        self.alphadeg = alphadeg
        self.stack = stack
        self.plyt = plyt
        self.laminaprop = laminaprop

        self.name = ''
        self.bay = None

        # model
        self.model = None
        self.K = 5/6. # in case of First-order Shear Deformation Theory

        # approximation series
        self.m = 11
        self.n = 11

        # numerical integration
        self.nx = 160
        self.ny = 160
        self.ni_num_cores = cpu_count()//2
        self.ni_method = 'trapz2d'

        # loads
        self.Nxx = None
        self.Nyy = None
        self.Nxy = None
        self.Nxx_cte = None
        self.Nyy_cte = None
        self.Nxy_cte = None
        self.forces = []
        self.forces_inc = []

        # boundary conditions

        # displacement at 4 edges is zero
        # free to rotate at 4 edges (simply supported by default)
        self.u1tx = 0.
        self.u1rx = 1.
        self.u2tx = 0.
        self.u2rx = 1.
        self.v1tx = 0.
        self.v1rx = 1.
        self.v2tx = 0.
        self.v2rx = 1.
        self.w1tx = 0.
        self.w1rx = 1.
        self.w2tx = 0.
        self.w2rx = 1.
        self.u1ty = 0.
        self.u1ry = 1.
        self.u2ty = 0.
        self.u2ry = 1.
        self.v1ty = 0.
        self.v1ry = 1.
        self.v2ty = 0.
        self.v2ry = 1.
        self.w1ty = 0.
        self.w1ry = 1.
        self.w2ty = 0.
        self.w2ry = 1.

        # material
        self.mu = None
        self.plyts = []
        self.laminaprops = []

        # aeroelastic parameters
        self.flow = 'x'
        self.beta = None
        self.gamma = None
        self.aeromu = None
        self.rho_air = None
        self.speed_sound = None
        self.Mach = None
        self.V = None

        # constitutive law
        self.F = None
        self.force_orthotropic_laminate = False

        # eigenvalue analysis
        self.num_eigvalues = 5
        self.num_eigvalues_print = 5

        # output queries
        self.out_num_cores = cpu_count()

        # analysis
        self.analysis = Analysis(self.calc_fext, self.calc_k0, None, None)

        # outputs
        self.increments = None
        self.eigvecs = None
        self.eigvals = None

        self._clear_matrices()


    def _clear_matrices(self):
        self.k0 = None
        self.kG0 = None
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

        #NOTE memory cleanup
        gc.collect()


    def _rebuild(self):
        if not self.name:
            try:
                self.name = os.path.basename(__main__.__file__).split('.py')[0]
            except AttributeError:
                warn('Panel name unchanged')

        if self.model is None:
            if self.r is None and self.alphadeg is None:
                self.model = 'plate_clt_donnell_bardell'
            elif self.r is not None and self.alphadeg is None:
                self.model = 'cpanel_clt_donnell_bardell'
            elif self.r is not None and self.alphadeg is not None:
                self.model = 'kpanel_clt_donnell_bardell'

        valid_models = sorted(modelDB.db.keys())

        if not self.model in valid_models:
            raise ValueError('ERROR - valid models are:\n    ' +
                     '\n    '.join(valid_models))

        if not self.stack:
            raise ValueError('stack must be defined')

        if not self.laminaprops:
            if not self.laminaprop:
                raise ValueError('laminaprop must be defined')
            self.laminaprops = [self.laminaprop for i in self.stack]

        if not self.plyts:
            if self.plyt is None:
                raise ValueError('plyt must be defined')
            self.plyts = [self.plyt for i in self.stack]

        if False:
            def check_load(load, size):
                if load is not None:
                    check = False
                    if isinstance(load, np.ndarray):
                        if load.ndim == 1:
                            assert load.shape[0] == size

                            return load
                    elif type(load) in (int, float):
                        newload = np.zeros(size, dtype=DOUBLE)
                        newload[0] = load

                        return newload
                    if not check:
                        raise ValueError('Invalid Nxx_cte input')
                else:
                    return np.zeros(size, dtype=DOUBLE)


            # axial load
            size = self.n+1
            self.Nxx_cte = check_load(self.Nxx_cte, size)
            self.Nxx = check_load(self.Nxx, size)
            # shear xt
            self.Nxy_cte = check_load(self.Nxy_cte, size)
            self.Nxy = check_load(self.Nxy, size)
            # circumferential load
            size = self.m+1
            self.Nyy_cte = check_load(self.Nyy_cte, size)
            self.Nyy = check_load(self.Nyy, size)


    def bc_sfss(self):
        self.u1tx = 0.
        self.u1rx = 0.
        self.u2tx = 1.
        self.u2rx = 1.
        self.v1tx = 0.
        self.v1rx = 0.
        self.v2tx = 1.
        self.v2rx = 1.
        self.w1tx = 0.
        self.w1rx = 1.
        self.w2tx = 1.
        self.w2rx = 1.
        self.u1ty = 0.
        self.u1ry = 1.
        self.u2ty = 0.
        self.u2ry = 0.
        self.v1ty = 0.
        self.v1ry = 1.
        self.v2ty = 0.
        self.v2ry = 0.
        self.w1ty = 0.
        self.w1ry = 1.
        self.w2ty = 0.
        self.w2ry = 1.

    def bc_ssfs(self):
        self.u1tx = 0.
        self.u1rx = 0.
        self.u2tx = 0.
        self.u2rx = 0.
        self.v1tx = 0.
        self.v1rx = 0.
        self.v2tx = 0.
        self.v2rx = 0.
        self.w1tx = 0.
        self.w1rx = 1.
        self.w2tx = 0.
        self.w2rx = 1.
        self.u1ty = 1.
        self.u1ry = 1.
        self.u2ty = 0.
        self.u2ry = 0.
        self.v1ty = 1.
        self.v1ry = 1.
        self.v2ty = 0.
        self.v2ry = 0.
        self.w1ty = 1.
        self.w1ry = 1.
        self.w2ty = 0.
        self.w2ry = 1.


    def bc_ssss(self):
        self.u1tx = 0.
        self.u1rx = 0.
        self.u2tx = 0.
        self.u2rx = 0.
        self.v1tx = 0.
        self.v1rx = 0.
        self.v2tx = 0.
        self.v2rx = 0.
        self.w1tx = 0.
        self.w1rx = 1.
        self.w2tx = 0.
        self.w2rx = 1.
        self.u1ty = 0.
        self.u1ry = 0.
        self.u2ty = 0.
        self.u2ry = 0.
        self.v1ty = 0.
        self.v1ry = 0.
        self.v2ty = 0.
        self.v2ry = 0.
        self.w1ty = 0.
        self.w1ry = 1.
        self.w2ty = 0.
        self.w2ry = 1.


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
        num = modelDB.db[self.model]['num']
        self.size = num*self.m*self.n
        return self.size


    def _default_field(self, xs, ys, gridx, gridy):
        if xs is None or ys is None:
            xs = linspace(0, self.a, gridx)
            ys = linspace(0, self.b, gridy)
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


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)

        if size is None:
            size = self.get_size()

        matrices = modelDB.db[self.model]['matrices']

        a = self.a
        b = self.b
        y1 = self.y1
        y2 = self.y2
        r = self.r if self.r is not None else 0.
        alphadeg = self.alphadeg if self.alphadeg is not None else 0.

        m = self.m
        n = self.n

        stack = self.stack
        plyts = self.plyts
        laminaprops = self.laminaprops

        if stack != []:
            lam = laminate.read_stack(stack, plyts=plyts,
                                             laminaprops=laminaprops)

        if 'clt' in self.model:
            if lam is not None:
                F = lam.ABD

        elif 'fsdt' in self.model:
            if lam is not None:
                F = lam.ABDE
                F[6:, 6:] *= self.K

        if self.force_orthotropic_laminate:
            msg('', silent=silent)
            msg('Forcing orthotropic laminate...', level=2, silent=silent)
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

        alpharad = np.deg2rad(alphadeg)

        if y1 is not None and y2 is not None:
            k0 = matrices.fk0y1y2(y1, y2, a, b, r, alpharad, F,
                     self.m, self.n,
                     self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                     self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                     self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                     self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                     self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                     self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                     size, row0, col0)
        else:
            k0 = matrices.fk0(a, b, r, alpharad, F,
                     self.m, self.n,
                     self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                     self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                     self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                     self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                     self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                     self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                     size, row0, col0)

        Nxx_cte = self.Nxx_cte if self.Nxx_cte is not None else 0.
        Nyy_cte = self.Nyy_cte if self.Nyy_cte is not None else 0.
        Nxy_cte = self.Nxy_cte if self.Nxy_cte is not None else 0.

        if Nxx_cte != 0. or Nyy_cte != 0. or Nxy_cte != 0.:
            if y1 is not None and y2 is not None:
                k0 += fkG0y1y2(y1, y2, Nxx_cte, Nyy_cte, Nxy_cte,
                               a, b, r, alpharad, self.m, self.n,
                               self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                               self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                               size, row0, col0)
            else:
                k0 += fkG0(Nxx_cte, Nyy_cte, Nxy_cte,
                           a, b, r, alpharad, self.m, self.n,
                           self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                           self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                           size, row0, col0)

        if finalize:
            # performing final checks for k0 and making symmetry
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
        msg('Calculating kG0... ', level=2, silent=silent)

        if size is None:
            size = self.get_size()

        matrices = modelDB.db[self.model]['matrices']

        a = self.a
        b = self.b
        r = self.r if self.r is not None else 0.
        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        alpharad = deg2rad(alphadeg)

        m = self.m
        n = self.n

        Nxx = self.Nxx if self.Nxx is not None else 0.
        Nyy = self.Nyy if self.Nyy is not None else 0.
        Nxy = self.Nxy if self.Nxy is not None else 0.

        kG0 = matrices.fkG0(Nxx, Nyy, Nxy, a, b, r, alpharad, self.m, self.n,
                   self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                   self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                   size, row0, col0)

        if finalize:
            assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
            kG0 = csr_matrix(make_symmetric(kG0))

        self.kG0 = kG0

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kM(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        msg('Calculating kM... ', level=2, silent=silent)

        y1 = self.y1
        y2 = self.y2
        matrices = modelDB.db[self.model]['matrices']

        if size is None:
            size = self.get_size()

        if y1 is not None and y2 is not None:
            h = sum(self.plyts)
            kM = matrices.fkMy1y2(y1, y2, self.mu, self.d, h,
                                  self.a, self.b, self.m, self.n,
                                  self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                                  self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                                  self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                  self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                                  self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                                  self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                  size, row0, col0)
        else:
            kM = fkM(mu, d, h, a, b, m, n,
                self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                size, row0, col0)

        if finalize:
            assert np.any(np.isnan(kM.data)) == False
            assert np.any(np.isinf(kM.data)) == False
            kM = csr_matrix(make_symmetric(kM))

        self.kM = kM

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kA(self, silent=False, finalize=True):
        msg('Calculating kA... ', level=2, silent=silent)

        if 'kpanel' in self.model:
            raise NotImplementedError('Conical panels not supported')

        r = self.r if self.r is not None else 0.

        if self.beta is None:
            if self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                self.Mach = 1.0001
            M = self.Mach
            beta = self.rho_air * self.V**2 / (M**2 - 1)**0.5
            gamma = beta*1./(2.*self.r*(M**2 - 1)**0.5)
            ainf = self.speed_sound
            aeromu = beta/(M*ainf)*(M**2 - 2)/(M**2 - 1)
        else:
            beta = self.beta
            gamma = self.gamma if self.gamma is not None else 0.
            aeromu = self.aeromu if self.aeromu is not None else 0.

        if self.flow.lower() == 'x':
            kA = matrices.fkAx(beta, gamma, a, b, m, n, self.size, 0, 0)
        elif self.flow.lower() == 'y':
            kA = mod.fkAy(beta, a, b, m, n, size, 0, 0)
        else:
            raise ValueError('Invalid flow value, must be x or y')

        if finalize:
            assert np.any(np.isnan(kA.data)) == False
            assert np.any(np.isinf(kA.data)) == False
            kA = csr_matrix(make_skew_symmetric(kA))

        self.kA = kA

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_cA(self, aeromu, silent=False, finalize=True):
        msg('Calculating cA... ', level=2, silent=silent)

        cA = matrices.fcA(aeromu, self.a, self.b, self.m, self.n,
                          self.size, 0, 0)
        cA = cA*(0+1j)

        if finalize:
            assert np.any(np.isnan(cA.data)) == False
            assert np.any(np.isinf(cA.data)) == False
            cA = csr_matrix(make_symmetric(cA))

        self.cA = cA

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def lb(self, tol=0, sparse_solver=True, calc_kA=False, silent=False):
        """Performs a linear buckling analysis

        The following parameters will affect the linear buckling analysis:

        =======================    =====================================
        parameter                  description
        =======================    =====================================
        ``num_eigenvalues``        Number of eigenvalues to be extracted
        ``num_eigvalues_print``    Number of eigenvalues to print after
                                   the analysis is completed
        =======================    =====================================

        Parameters
        ----------
        tol : float, optional
            A float tolerance passsed to the eigenvalue solver.
        sparse_solver : bool, optional
            Tells if solver :func:`scipy.linalg.eigh` or
            :func:`scipy.sparse.linalg.eigsh` should be used.
        calc_kA : bool, optional
            If the Aerodynamic matrix should be considered.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``Panel`` object and the `i^{th}` eigenvector in the
        ``eigvecs[:, i-1]`` parameter.

        """
        msg('Running linear buckling analysis...', silent=silent)

        msg('Eigenvalue solver... ', level=2, silent=silent)

        self.calc_k0(silent=silent)
        self.calc_kG0(silent=silent)

        if calc_kA:
            self.calc_kA(silent=silent)
            kA = self.kA
        else:
            kA = self.k0*0

        M = self.k0 + kA
        A = self.kG0

        if sparse_solver:
            mode = 'cayley'
            try:
                msg('eigsh() solver...', level=3, silent=silent)
                eigvals, eigvecs = eigsh(A=A, k=self.num_eigvalues,
                        which='SM', M=M, tol=tol, sigma=1., mode=mode)
                msg('finished!', level=3, silent=silent)
            except Exception, e:
                warn(str(e), level=4, silent=silent)
                msg('aborted!', level=3, silent=silent)
                sizebkp = A.shape[0]
                M, A, used_cols = remove_null_cols(M, A, silent=silent)
                msg('eigsh() solver...', level=3, silent=silent)
                eigvals, peigvecs = eigsh(A=A, k=self.num_eigvalues,
                        which='SM', M=M, tol=tol, sigma=1., mode=mode)
                msg('finished!', level=3, silent=silent)
                eigvecs = np.zeros((sizebkp, self.num_eigvalues),
                                   dtype=DOUBLE)
                eigvecs[used_cols, :] = peigvecs

        else:
            from scipy.linalg import eigh

            size = A.shape[0]
            M, A, used_cols = remove_null_cols(M, A, silent=silent)
            M = M.toarray()
            A = A.toarray()
            msg('eigh() solver...', level=3, silent=silent)
            eigvals, peigvecs = eigh(a=A, b=M)
            msg('finished!', level=3, silent=silent)
            eigvecs = np.zeros((size, self.num_eigvalues), dtype=DOUBLE)
            eigvecs[used_cols, :] = peigvecs[:, :self.num_eigvalues]

        eigvals = -1./eigvals

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        msg('finished!', level=2, silent=silent)

        msg('first {0} eigenvalues:'.format(self.num_eigvalues_print), level=1,
            silent=silent)
        for eig in eigvals[:self.num_eigvalues_print]:
            msg('{0}'.format(eig), level=2, silent=silent)
        self.analysis.last_analysis = 'lb'


    def freq(self, atype=4, tol=0, sparse_solver=False, silent=False,
             sort=True, damping=False, reduced_dof=False):
        """Performs a natural frequency analysis

        The following parameters of the will affect the linear buckling
        analysis:

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
        The extracted eigenvalues are stored in the ``eigvals`` parameter and
        the `i^{th}` eigenvector in the ``eigvecs[:, i-1]`` parameter.

        """
        msg('Running frequency analysis...', silent=silent)

        self.calc_k0(silent=silent)
        self.calc_kM(silent=silent)
        if atype == 1:
            self.calc_kG0(silent=silent)
            self.calc_kA(silent=silent)
            if damping:
                self.calc_cA(silent=silent)
                K = self.k0 + self.kA + self.kG0 + self.cA
            else:
                K = self.k0 + self.kA + self.kG0

        elif atype == 2:
            self.calc_kA(silent=silent)
            K = self.k0 + self.kA
            if damping:
                self.calc_cA(silent=silent)
                K = self.k0 + self.kA + self.cA
            else:
                K = self.k0 + self.kA

        elif atype == 3:
            self.calc_kG0(silent=silent)
            K = self.k0 + self.kG0

        elif atype == 4:
            K = self.k0

        M = self.kM

        msg('Eigenvalue solver... ', level=2, silent=silent)
        msg('eigs() solver...', level=3, silent=silent)
        k = min(self.num_eigvalues, M.shape[0]-2)
        if sparse_solver:
            try:
                eigvals, eigvecs = eigs(A=M, M=K, k=k, tol=tol, which='SM',
                                        sigma=-1.)
            except:
                sizebkp = M.shape[0]
                M, K, used_cols = remove_null_cols(M, K, silent=silent,
                        level=3)
                eigvals, peigvecs = eigs(A=M, k=k, which='SM', M=K, tol=tol,
                                         sigma=-1.)
                eigvecs = np.zeros((sizebkp, self.num_eigvalues),
                                   dtype=DOUBLE)
                eigvecs[used_cols, :] = peigvecs

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
        r"""Calculates the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the ``Panel`` object.

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
        stored as parameters with the same name in the ``Panel`` object.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ys, xshape, tshape = self._default_field(xs, ys, gridx, gridy)
        m = self.m
        n = self.n
        a = self.a
        b = self.b
        model = self.model

        fuvw = modelDB.db[model]['field'].fuvw
        us, vs, ws, phixs, phiys = fuvw(c, m, n, a, b, xs, ys,
                self.out_num_cores)

        self.u = us.reshape(xshape)
        self.v = vs.reshape(xshape)
        self.w = ws.reshape(xshape)
        self.phix = phixs.reshape(xshape)
        self.phiy = phiys.reshape(xshape)

        return self.u, self.v, self.w, self.phix, self.phiy


    def strain(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculates the strain field

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation.
        xs : np.ndarray, optional
            The `x` coordinates where to calculate the strains.
        ys : np.ndarray, optional
            The `y` coordinates where to calculate the strains, must
            have the same shape as ``xs``.
        gridx : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used.
        gridy : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ys, xshape, tshape = self._default_field(xs, ys, gridx, gridy)

        a = self.a
        b = self.b
        m = self.m
        n = self.n
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0
        funcnum = self.funcnum
        model = self.model
        NL_kinematics = model.split('_')[1]
        fstrain = modelDB.db[model]['field'].fstrain
        e_num = modelDB.db[model]['e_num']

        if 'donnell' in NL_kinematics:
            int_NL_kinematics = 0
        elif 'sanders' in NL_kinematics:
            int_NL_kinematics = 1
        else:
            raise NotImplementedError(
                '{} is not a valid NL_kinematics option'.format(NL_kinematics))

        es = fstrain(c, xs, ys, a, b, m, n,
                c0, m0, n0, funcnum, int_NL_kinematics, self.out_num_cores)

        return es.reshape((xshape + (e_num,)))


    def stress(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculates the stress field

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation.
        xs : np.ndarray, optional
            The `x` coordinates where to calculate the strains.
        ys : np.ndarray, optional
            The `y` coordinates where to calculate the strains, must
            have the same shape as ``xs``.
        gridx : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used.
        gridy : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ys, xshape, tshape = self._default_field(xs, ys, gridx, gridy)

        F = self.F
        a = self.a
        b = self.b
        m = self.m
        n = self.n
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0
        funcnum = self.funcnum
        model = self.model
        NL_kinematics = model.split('_')[1]
        fstress = modelDB.db[model]['field'].fstress
        e_num = modelDB.db[model]['e_num']

        if 'donnell' in NL_kinematics:
            int_NL_kinematics = 0
        elif 'sanders' in NL_kinematics:
            int_NL_kinematics = 1
        else:
            raise NotImplementedError(
                    '{} is not a valid NL_kinematics option'.format(
                    NL_kinematics))

        Ns = fstress(c, F, xs, ys, a, b, m, n, c0, m0, n0,
                funcnum, int_NL_kinematics, self.out_num_cores)
        return Ns.reshape((xshape + (e_num,)))


    def add_force(self, x, y, fx, fy, fz, cte=True):
        r"""Add a punctual force with three components

        Parameters
        ----------
        x : float
            The `x` position.
        y : float
            The `y` position in radians.
        fx : float
            The `x` component of the force vector.
        fy : float
            The `y` component of the force vector.
        fz : float
            The `z` component of the force vector.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis.

        """
        if cte:
            self.forces.append([x, y, fx, fy, fz])
        else:
            self.forces_inc.append([x, y, fx, fy, fz])


    def calc_fext(self, inc=1., silent=False):
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
        self._rebuild()
        msg('Calculating external forces...', level=2, silent=silent)
        a = self.a
        b = self.b
        m = self.m
        n = self.n
        model = self.model

        if not model in modelDB.db.keys():
            raise ValueError(
                    '{} is not a valid model option'.format(model))

        db = modelDB.db
        num = db[model]['num']
        dofs = db[model]['dofs']
        fg = db[model]['field'].fg

        size = self.get_size()

        g = np.zeros((dofs, size), dtype=DOUBLE)
        fext = np.zeros(size, dtype=DOUBLE)

        # non-incrementable punctual forces
        for i, force in enumerate(self.forces):
            x, y, fx, fy, fz = force
            fg(g, m, n, x, y, a, b)
            if dofs == 3:
                fpt = np.array([[fx, fy, fz]])
            elif dofs == 5:
                fpt = np.array([[fx, fy, fz, 0, 0]])
            fext += fpt.dot(g).ravel()

        # incrementable punctual forces
        for i, force in enumerate(self.forces_inc):
            x, y, fx, fy, fz = force
            fg(g, m, n, x, y, a, b)
            if dofs == 3:
                fpt = np.array([[fx, fy, fz]])*inc
            elif dofs == 5:
                fpt = np.array([[fx, fy, fz, 0, 0]])*inc
            fext += fpt.dot(g).ravel()

        # Nxx_cte

        Nxx_cte = self.Nxx_cte
        Nyy_cte = self.Nyy_cte
        Nxx_cte += inc*self.Nxx
        Nyy_cte += inc*self.Nyy

        for j1 in range(1, n+1):
            Nxxj = Nxx_cte[j1]
            for i1 in range(1, m+1):
                col = num*((j1-1)*m + (i1-1))
                fext[col+0] += 1/2.*(-1)**i1*Nxxj*b

        msg('finished!', level=2, silent=silent)

        if np.all(fext==0):
            raise ValueError('No load was applied!')

        return fext


    def static(self, NLgeom=False, silent=False):
        """Static analysis for cones and cylinders

        The analysis can be linear or geometrically non-linear. See
        :class:`.Analysis` for further details about the parameters
        controlling the non-linear analysis.

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
        The returned ``cs`` is stored in ``self.analysis.cs``. The actual
        increments used in the non-linear analysis are stored in the
        ``self.analysis.increments`` parameter.

        """
        if self.c0 is not None:
            self.analysis.kT_initial_state = True
        else:
            self.analysis.kT_initial_state = False

        if NLgeom and not modelDB.db[self.model]['non-linear static']:
            msg('________________________________________________',
                silent=silent)
            msg('', silent=silent)
            warn('Model {} cannot be used in non-linear static analysis!'.
                 format(self.model), silent=silent)
            msg('________________________________________________',
                silent=silent)
            raise
        elif not NLgeom and not modelDB.db[self.model]['linear static']:
            msg('________________________________________________',
                level=1, silent=silent)
            msg('', level=1, silent=silent)
            warn('Model {} cannot be used in linear static analysis!'.
                 format(self.model), level=1, silent=silent)
            msg('________________________________________________',
                level=1, silent=silent)
            raise
        self.analysis.static(NLgeom=NLgeom, silent=silent)
        self.increments = self.analysis.increments

        return self.analysis.cs


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
            the `name` parameter of the ``Panel`` object will be used.
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
        strains = ['exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz']
        stresses = ['Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx']
        if vec in displs:
            self.uvw(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
            field = getattr(self, vec)
        elif vec in strains:
            es = self.strain(c, xs=xs, ys=ys,
                             gridx=gridx, gridy=gridy)
            field = es[..., strains.index(vec)]
        elif vec in stresses:
            Ns = self.stress(c, xs=xs, ys=ys,
                             gridx=gridx, gridy=gridy)
            field = Ns[..., stresses.index(vec)]
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
                ax.set_title('$m_1, n_1={0}, {1}$'.format(self.m, self.n))

            elif self.analysis.last_analysis == 'lb':
                ax.set_title(
       r'$m_1, n_1={0}, {1}$, $\lambda_{{CR}}={4:1.3e}$'.format(self.m,
           self.n, self.eigvals[0]))

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
        """Save the ``Panel`` object using ``cPickle``

        Notes
        -----
        The pickled file will have the name stored in ``Panel.name``
        followed by a ``'.Panel'`` extension.

        """
        name = self.name + '.Panel'
        msg('Saving Panel to {}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)

