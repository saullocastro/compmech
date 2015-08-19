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
    if '.AeroPistonPlate' in name:
        return cPickle.load(open(name, 'rb'))
    else:
        return cPickle.load(open(name + '.AeroPistonPlate', 'rb'))


class AeroPistonPlate(object):
    r"""Conical (Konus) panel using trigonometric series

    The approximation functions for the displacement field are:

        .. math::
            \begin{tabular}{l c r}
            CLPT & FSDT \\
            \hline
            $u$ & $u$ \\
            $v$ & $v$ \\
            $w$ & $w$ \\
            $NA$ & $\phi_x$ \\
            $NA$ & $\phi_y$ \\
            \end{tabular}

    with:

        .. math::
            u = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_{cos}}}
            \\
            v = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_{cos}}}
            \\
            w = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_{sim}}} +
                \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_{cos}}}
            \\
            \phi_x = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_{cos}}}
            \\
            \phi_y = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_{cos}}}
            \\
            f_{sim} = sin(i_1 \pi b_x)cos(j_1 \pi b_y)
            f_{cos} = cos(i_1 \pi b_x)cos(j_1 \pi b_y)
            \\
            b_x = \frac{x + \frac{a}{2}}{a}
            \\
            b_y = \frac{y + \frac{b}{2}}{b}

    """
    def __init__(self):
        self.name = ''

        # boundary conditions
        # "inf" is used to define the high stiffnesses (removed dofs)
        #       a high value will cause numerical instabilities
        #TODO use a marker number for self.inf and self.maxinf if the
        #     normalization of edge stiffenesses is adopted
        #     now it is already independent of self.inf and more robust
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
        self.model = 'clpt_donnell_free'

        # approximation series
        self.m1 = 11
        self.n1 = 11

        # numerical integration
        self.nx = 160
        self.ny = 160
        self.ni_num_cores = cpu_count()//2
        self.ni_method = 'trapz2d'

        # loads
        self.Fx = None
        self.Fy = None
        self.Fxy = None
        self.Fyx = None

        # shear correction factor (FSDT only)
        self.K = 5/6.

        # geometry
        self.a = None
        self.b = None

        # material
        self.mu = None # laminate material density
        self.laminaprop = None
        self.plyt = None
        self.laminaprops = []
        self.stack = []
        self.plyts = []

        # aerodynamic properties for the Piston theory
        self.lambdap = None
        self.rho = None
        self.M = None
        self.V = None

        # constitutive law
        self.F = None
        self.force_orthotropic_laminate = False

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
        self.kG0_Fx = None
        self.kG0_Fy = None
        self.kG0_Fxy = None
        self.kG0_Fyx = None
        self.kM = None
        self.kA = None
        self.kG = None
        self.kL = None
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
                warn('AeroPistonPlate name unchanged')

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
                    txt = '"{}" is not a valid boundary condition!'.format(bc)
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

        fk0, fkG0, fkA, fkM, k0edges = modelDB.get_linear_matrices(self)
        model = self.model
        a = self.a
        b = self.b
        m1 = self.m1
        n1 = self.n1
        laminaprops = self.laminaprops
        plyts = self.plyts
        h = sum(plyts)
        stack = self.stack
        mu = self.mu
        if calc_kA and self.lambdap is None:
            if self.M < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.M == 1:
                self.M = 1.0001
            self.lambdap = self.rho * self.V**2 / (self.M**2 - 1)**0.5
        lambdap = self.lambdap

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

        k0 = fk0(a, b, F, m1, n1)
        if calc_kA:
            kA = fkA(lambdap, a, b, m1, n1)
        if calc_kM:
            kM = fkM(mu, h, a, b, m1, n1)

        if calc_kG0:
            Fx = self.Fx if self.Fx is not None else 0.
            Fy = self.Fy if self.Fy is not None else 0.
            Fxy = self.Fxy if self.Fxy is not None else 0.
            Fyx = self.Fyx if self.Fyx is not None else 0.

            if not combined_load_case:
                kG0 = fkG0(Fx, Fy, Fxy, Fyx, a, b, m1, n1)
            else:
                kG0_Fx = fkG0(Fx, 0, 0, 0, a, b, m1, n1)
                kG0_Fy = fkG0(0, Fy, 0, 0, a, b, m1, n1)
                kG0_Fxy = fkG0(0, 0, Fxy, 0, a, b, m1, n1)
                kG0_Fyx = fkG0(0, 0, 0, Fyx, a, b, m1, n1)

        # performing checks for the linear stiffness matrices

        assert np.any(np.isnan(k0.data)) == False
        assert np.any(np.isnan(k0.data)) == False

        if calc_kA:
            assert np.any(np.isinf(kA.data)) == False
            assert np.any(np.isinf(kA.data)) == False

        if calc_kM:
            assert np.any(np.isinf(kM.data)) == False
            assert np.any(np.isinf(kM.data)) == False

        k0 = csr_matrix(make_symmetric(k0))
        if calc_kA:
            kA = csr_matrix(make_skew_symmetric(kA))
        if calc_kM:
            kM = csr_matrix(make_symmetric(kM))

        if k0edges is not None:
            assert np.any((np.isnan(k0edges.data)
                           | np.isinf(k0edges.data))) == False
            k0edges = csr_matrix(make_symmetric(k0edges))

        if k0edges is not None:
            k0 = k0 + k0edges

        self.k0 = k0
        if calc_kA:
            self.kA = kA
        if calc_kM:
            self.kM = kM

        if calc_kG0:
            if not combined_load_case:
                assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
                kG0 = csr_matrix(make_symmetric(kG0))
                self.kG0 = kG0

            else:
                assert np.any((np.isnan(kG0_Fx.data)
                               | np.isinf(kG0_Fx.data))) == False
                assert np.any((np.isnan(kG0_Fy.data)
                               | np.isinf(kG0_Fy.data))) == False
                assert np.any((np.isnan(kG0_Fxy.data)
                               | np.isinf(kG0_Fxy.data))) == False
                assert np.any((np.isnan(kG0_Fyx.data)
                               | np.isinf(kG0_Fyx.data))) == False

                kG0_Fx = csr_matrix(make_symmetric(kG0_Fx))
                kG0_Fy = csr_matrix(make_symmetric(kG0_Fy))
                kG0_Fxy = csr_matrix(make_symmetric(kG0_Fxy))
                kG0_Fyx = csr_matrix(make_symmetric(kG0_Fyx))

                self.kG0_Fx = kG0_Fx
                self.kG0_Fy = kG0_Fy
                self.kG0_Fxy = kG0_Fxy
                self.kG0_Fyx = kG0_Fyx

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process

        gc.collect()

        msg('finished!', level=2, silent=silent)


    def lb(self, tol=0, combined_load_case=None, sparse_solver=True):
        """Performs a linear buckling analysis

        The following parameters of the ``AeroPistonPlate`` object will affect
        the linear buckling analysis:

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

            - ``1`` : find the critical Fx for a fixed Fxy
            - ``2`` : find the critical Fx for a fixed Fy
            - ``3`` : find the critical Fy for a fixed Fyx
            - ``4`` : find the critical Fy for a fixed Fx
        sparse_solver : bool, optional
            Tells if solver :func:`scipy.linalg.eigh` or
            :func:`scipy.sparse.linalg.eigs` should be used.

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``AeroPistonPlate`` object and the `i^{th}` eigenvector in the
        ``eigvecs[:, i-1]`` parameter.

        """
        if not modelDB.db[self.model]['linear buckling']:
            msg('________________________________________________')
            msg('')
            warn('Model {} cannot be used in linear buckling analysis!'.
                 format(self.model))
            msg('________________________________________________')

        msg('Running linear buckling analysis...')

        self.calc_linear_matrices(combined_load_case=combined_load_case)

        msg('Eigenvalue solver... ', level=2)

        if not combined_load_case:
            M = self.k0 + self.kA
            A = self.kG0
        elif combined_load_case == 1:
            M = self.k0 - self.kA + self.kG0_Fxy
            A = self.kG0_Fx
        elif combined_load_case == 2:
            M = self.k0 - self.kA + self.kG0_Fy
            A = self.kG0_Fx
        elif combined_load_case == 3:
            M = self.k0 - self.kA + self.kG0_Fyx
            A = self.kG0_Fy
        elif combined_load_case == 4:
            M = self.k0 - self.kA + self.kG0_Fx
            A = self.kG0_Fy

        #print M.max()
        #raise

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
            from scipy.linalg import eigh

            size22 = A.shape[0]
            M, A, used_cols = remove_null_cols(M, A)
            M = M.toarray()
            A = A.toarray()
            msg('eigh() solver...', level=3)
            eigvals, peigvecs = eigh(a=A, b=M)
            msg('finished!', level=3)
            eigvecs = np.zeros((size22, self.num_eigvalues), dtype=DOUBLE)
            eigvecs[used_cols, :] = peigvecs[:, :self.num_eigvalues]

        eigvals = -1./eigvals

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        msg('finished!', level=2)

        msg('first {} eigenvalues:'.format(self.num_eigvalues_print), level=1)
        for eig in eigvals[:self.num_eigvalues_print]:
            msg('{}'.format(eig), level=2)
        self.analysis.last_analysis = 'lb'


    def freq(self, atype=1, tol=0, sparse_solver=False, silent=False,
            sort=True):
        """Performs a frequency analysis

        The following parameters of the ``AeroPistonPlate`` object will affect
        the linear buckling analysis:

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

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``AeroPistonPlate`` object and the `i^{th}` eigenvector in the
        ``eigvecs[:, i-1]`` parameter.

        """
        if not modelDB.db[self.model]['linear buckling']:
            msg('________________________________________________')
            msg('')
            warn('Model {} cannot be used in linear buckling analysis!'.
                 format(self.model))
            msg('________________________________________________')

        msg('Running frequency analysis...', silent=silent)

        if atype == 1:
            self.calc_linear_matrices(silent=silent)
        elif atype == 2:
            self.calc_linear_matrices(silent=silent, calc_kG0=False)
        elif atype == 3:
            self.calc_linear_matrices(silent=silent, calc_kA=False)

        msg('Eigenvalue solver... ', level=2, silent=silent)

        if atype == 1:
            M = self.k0 - self.kA + self.kG0
        elif atype == 2:
            M = self.k0 - self.kA
        elif atype == 3:
            M = self.k0 + self.kG0
        A = self.kM

        msg('eigs() solver...', level=3, silent=silent)
        k = min(self.num_eigvalues, A.shape[0]-2)
        if sparse_solver:
            eigvals, eigvecs = eigs(A=A, M=M, k=k, tol=tol, which='SM', sigma=-1.)
        else:
            eigvals, eigvecs = eig(a=A.toarray(), b=M.toarray())
        msg('finished!', level=3, silent=silent)

        eigvals = np.sqrt(1./eigvals) # omega^2 to omega, in rad/s

        if sort:
            sort_ind = np.lexsort((np.round(eigvals.imag, 1),
                                   np.round(eigvals.real, 1)))
            eigvals = eigvals[sort_ind]
            eigvecs = eigvecs[:, sort_ind]

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        msg('finished!', level=2, silent=silent)

        msg('first {} eigenvalues:'.format(self.num_eigvalues_print), level=1,
                silent=silent)
        for eigval in eigvals[:self.num_eigvalues_print]:
            msg('{0} rad/s'.format(eigval), level=2, silent=silent)
        self.analysis.last_analysis = 'freq'


    def calc_Vf(self, rho=None, M=None, modes=(0, 1, 2, 3, 4, 5), num=10,
                silent=False):
        r"""Calculate the flutter speed

        If ``rho`` and ``M`` are not supplied, ``lambdap`` will be returned.

        Parameters
        ----------
        rho : float, optional
            Air density.
        M : float, optional
            Mach number.
        modes : tuple, optional
            The modes that should be monitored.
        num : int, optional
            Number of points to search for each iteration.

        Returns
        -------
        lambdacr : float
            The critical ``lambdap``.

        """
        #TODO
        # - use a linear or parabolic interpolation to estimate new_lim1
        msg('Flutter calculation...', level=1, silent=silent)
        lim1 = 0.1
        lim2 = 100.
        new_lim1 = 1e6
        new_lim2 = -1e6
        eigvals_imag = np.zeros((num, len(modes)))
        if max(modes) > self.num_eigvalues-1:
            self.num_eigvalues = max(modes)+1

        count = 0
        while True:
            count += 1
            lambdaps = np.linspace(lim1, lim2, num)
            msg('iteration %d:' % count, level=2, silent=silent)
            msg('lambda_min: %1.3f' % lim1, level=3, silent=silent)
            msg('lambda_max: %1.3f' % lim2, level=3, silent=silent)

            for i, lambdap in enumerate(lambdaps):
                self.lambdap = lambdap
                self.freq(atype=1, sparse_solver=False, silent=True)
                for j, mode in enumerate(modes):
                    eigvals_imag[i, j] = self.eigvals[mode].imag

            check = np.where(eigvals_imag != 0.)
            if not np.any(check):
                continue
            if np.abs(eigvals_imag[check]).min() < 0.01:
                break
            if 0 in check[0]:
                new_lim1 = min(new_lim1, 0.5*lambdaps[check[0][0]])
                new_lim2 = max(new_lim2, 1.5*lambdaps[check[0][-1]])
            elif check[0].min() > 0:
                new_lim1 = lambdaps[check[0][0]-1]
                new_lim2 = lambdaps[check[0][0]]
            else:
                new_lim1 = min(new_lim1, lim1/2.)
                new_lim2 = max(new_lim2, 2*lim2)

            lim1 = new_lim1
            lim2 = new_lim2
        msg('finished!', level=1)
        msg('Number of analyses = %d' % (count*num), level=1)
        return lim1


    def uvw(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the ``AeroPistonPlate``
        object.

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
        stored as parameters with the same name in the ``AeroPistonPlate``
        object.

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
            the `name` parameter of the ``AeroPistonPlate`` object will be
            used.
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
        """Save the ``AeroPistonPlate`` object using ``cPickle``

        Notes
        -----
        The pickled file will have the name stored in ``AeroPistonPlate.name``
        followed by a ``'.AeroPistonPlate'`` extension.

        """
        name = self.name + '.AeroPistonPlate'
        msg('Saving AeroPistonPlate to {}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
