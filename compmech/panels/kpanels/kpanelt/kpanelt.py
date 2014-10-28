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
from scipy.sparse.linalg import eigsh
from scipy.optimize import leastsq
from numpy import linspace, cos, sin, deg2rad

import compmech.composite.laminate as laminate
from compmech.analysis import Analysis
from compmech.logger import msg, warn
from compmech.constants import DOUBLE
from compmech.sparse import (make_symmetric, solve, remove_null_cols,
                             is_symmetric)
import modelDB


def load(name):
    if '.KPanelT' in name:
        return cPickle.load(open(name, 'rb'))
    else:
        return cPickle.load(open(name + '.KPanelT', 'rb'))


class KPanelT(object):
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
            $NA$ & $\phi_\theta$ \\
            \end{tabular}

    with:

        .. math::
            u = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_a}}
            \\
            v = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_a}}
            \\
            w = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_b}}
            \\
            \phi_x = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_a}}
            \\
            \phi_\theta = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f_a}}
            \\
            f_a = cos(i_1 \pi b_x)cos(j_1 \pi b_\theta)
            \\
            f_b = sin(i_1 \pi b_x)sin(j_1 \pi b_\theta)
            \\
            b_x = \frac{x + \frac{L}{2}}{L}
            \\
            b_\theta = \frac{\theta - \theta_{min}}{\theta_{max}-\theta_{min}}

    """
    def __init__(self):
        self.name = ''
        self.forces = []
        self.forces_inc = []
        self.alphadeg = 0.
        self.alpharad = 0.
        self.is_cylinder = None

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
        self.model = 'clpt_donnell_bc4'

        # approximation series
        self.m1 = 40
        self.n1 = 40

        # analytical integration for cones
        self.s = 79

        # loads
        self.Fx = None
        self.Ft = None
        self.Fxt = None
        self.Ftx = None
        self.NxxTop = None
        self.NxtTop = None
        self.NttLeft = None
        self.NtxLeft = None
        self.Fx_inc = None
        self.Ft_inc = None
        self.Fxt_inc = None
        self.Ftx_inc = None
        self.NxxTop_inc = None
        self.NxtTop_inc = None
        self.NttLeft_inc = None
        self.NtxLeft_inc = None

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

        #self.maxNumInc = 100
        self.maxNumIter = 30

        # output queries
        self.out_num_cores = 4

        # analysis
        self.analysis = Analysis(self.calc_fext, self.calc_k0, self.calc_fint,
                self.calc_kT)

        # outputs
        self.increments = None
        self.cs = None
        self.eigvecs = None
        self.eigvals = None

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
                    setattr(self, 'kphit' + sufix, zero)
                elif bcs[k] == 'ss2':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphit' + sufix, zero)
                elif bcs[k] == 'ss3':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphit' + sufix, zero)
                elif bcs[k] == 'ss4':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphit' + sufix, zero)

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
                    raise ValueError('Invalid NxxTop input')
            else:
                return np.zeros(size, dtype=DOUBLE)


        # axial load
        size = self.n1+1
        self.NxxTop = check_load(self.NxxTop, size)
        self.NxxTop_inc = check_load(self.NxxTop_inc, size)
        # shear xt
        self.NxtTop = check_load(self.NxtTop, size)
        self.NxtTop_inc = check_load(self.NxtTop_inc, size)
        # circumferential load
        size = self.m1+1
        self.NttLeft = check_load(self.NttLeft, size)
        self.NttLeft_inc = check_load(self.NttLeft_inc, size)
        # shear tx
        self.NtxLeft = check_load(self.NtxLeft, size)
        self.NtxLeft_inc = check_load(self.NtxLeft_inc, size)

        # defining load components from force vectors

        tmin = self.tminrad
        tmax = self.tmaxrad

        if self.Fx is not None:
            self.NxxTop[0] = self.Fx/((tmax - tmin)*self.r2)
            msg('NxxTop[0] calculated from Fx', level=2)

        if self.Fx_inc is not None:
            self.NxxTop_inc[0] = self.Fx_inc/((tmax - tmin)*self.r2)
            msg('NxxTop_inc[0] calculated from Fx_inc', level=2)

        if self.Fxt is not None:
            self.NxtTop[0] = self.Fxt/((tmax - tmin)*self.r2)
            msg('NxtTop[0] calculated from Fxt', level=2)

        if self.Fxt_inc is not None:
            self.NxtTop_inc[0] = self.Fxt_inc/((tmax - tmin)*self.r2)
            msg('NxtTop_inc[0] calculated from Fxt_inc', level=2)

        if self.Ft is not None:
            self.NttLeft[0] = self.Ft/self.L
            msg('NttLeft[0] calculated from Ft', level=2)

        if self.Ft_inc is not None:
            self.NttLeft_inc[0] = self.Ft_inc/self.L
            msg('NttLeft_inc[0] calculated from Ft_inc', level=2)

        if self.Ftx is not None:
            self.NtxLeft[0] = self.Ftx/self.L
            msg('NtxLeft[0] calculated from Ftx', level=2)

        if self.Ftx_inc is not None:
            self.NtxLeft_inc[0] = self.Ftx_inc/self.L
            msg('NtxLeft_inc[0] calculated from Ftx_inc', level=2)

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
        self.size = (num0 + num1*self.m1*self.n1)
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
            xs = linspace(-self.L/2., self.L/2, gridx)
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


    def calc_linear_matrices(self, combined_load_case=None):
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
        m1 = self.m1
        n1 = self.n1
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
            k0 = fk0_cyl(r1, L, tminrad, tmaxrad, F, m1, n1)

            if not combined_load_case:
                kG0 = fkG0_cyl(Fx, Ft, Fxt, Ftx, r1, L, tminrad, tmaxrad, m1,
                               n1)
            else:
                kG0_Fx = fkG0_cyl(Fx, 0, 0, 0, r1, L, tminrad, tmaxrad, m1, n1)
                kG0_Ft = fkG0_cyl(0, Ft, 0, 0, r1, L, tminrad, tmaxrad, m1, n1)
                kG0_Fxt = fkG0_cyl(0, 0, Fxt, 0, r1, L, tminrad, tmaxrad,
                                   m1, n1)
                kG0_Ftx = fkG0_cyl(0, 0, 0, Ftx, r1, L, tminrad, tmaxrad,
                                   m1, n1)
        else:
            k0 = fk0(r1, L, tminrad, tmaxrad, F, m1, n1, alpharad, s)

            if not combined_load_case:
                kG0 = fkG0(Fx, Ft, Fxt, Ftx, r1, L, tminrad, tmaxrad, m1, n1,
                        alpharad, s)
            else:
                kG0_Fx = fkG0(Fx, 0, 0, 0, r1, L, tminrad, tmaxrad, m1, n1,
                        alpharad, s)
                kG0_Ft = fkG0(0, Ft, 0, 0, r1, L, tminrad, tmaxrad, m1, n1,
                        alpharad, s)
                kG0_Fxt = fkG0(0, 0, Fxt, 0, r1, L, tminrad, tmaxrad, m1, n1,
                        alpharad, s)
                kG0_Ftx = fkG0(0, 0, 0, Ftx, r1, L, tminrad, tmaxrad, m1, n1,
                        alpharad, s)

        # performing checks for the linear stiffness matrices

        assert np.any(np.isnan(k0.data)) == False
        assert np.any(np.isinf(k0.data)) == False

        k0 = csr_matrix(make_symmetric(k0))

        if k0edges is not None:
            assert np.any((np.isnan(k0edges.data)
                           | np.isinf(k0edges.data))) == False
            k0edges = csr_matrix(make_symmetric(k0edges))

        if k0edges is not None:
            msg('Applying elastic constraints!', level=3)
            k0 = k0 + k0edges

        self.k0 = k0

        if not combined_load_case:
            assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
            kG0 = csr_matrix(make_symmetric(kG0))
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

            kG0_Fx = csr_matrix(make_symmetric(kG0_Fx))
            kG0_Ft = csr_matrix(make_symmetric(kG0_Ft))
            kG0_Fxt = csr_matrix(make_symmetric(kG0_Fxt))
            kG0_Ftx = csr_matrix(make_symmetric(kG0_Ftx))

            self.kG0_Fx = kG0_Fx
            self.kG0_Ft = kG0_Ft
            self.kG0_Fxt = kG0_Fxt
            self.kG0_Ftx = kG0_Ftx

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process

        gc.collect()

        msg('finished!', level=2)


    def lb(self, tol=0, combined_load_case=None, remove_null_i1_j1=False,
            sparse_solver=True):
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
        remove_null_i1_j1 : bool, optional
            It was observed that the eigenvectors can be described using only
            the homogeneous part of the approximation functions, which are
            obtained with `i_1 > 0` and `j_1 > 0`. Therefore, the terms with
            `i_1 = 0` and `j_1 = 0` can be ignored.
        sparse_solver : bool, optional
            Tells if solver :func:`scipy.linalg.eigh` or
            :func:`scipy.sparse.linalg.eigsh` should be used.

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

        self.calc_linear_matrices(combined_load_case=combined_load_case)

        msg('Eigenvalue solver... ', level=2)

        if not combined_load_case:
            M = self.k0
            A = self.kG0
        elif combined_load_case == 1:
            M = self.k0 + self.kG0_Fxt
            A = self.kG0_Fx
        elif combined_load_case == 2:
            M = self.k0 + self.kG0_Ft
            A = self.kG0_Fx
        elif combined_load_case == 3:
            M = self.k0 + self.kG0_Ftx
            A = self.kG0_Ft
        elif combined_load_case == 4:
            M = self.k0 + self.kG0_Fx
            A = self.kG0_Ft

        if remove_null_i1_j1:
            msg('removing rows and columns for i1=0 and j1=0 ...', level=3)
            db = modelDB.db
            num0 = db[self.model]['num0']
            num1 = db[self.model]['num1']
            dofs = db[self.model]['dofs']
            valid = []
            removed = []
            for i1 in range(self.m1):
                for j1 in range(self.n1):
                    col = num0 + num1*((j1)*self.m1 + (i1))
                    if i1 == 0 or j1 == 0:
                        for dof in range(dofs):
                            removed.append(col+dof)
                    else:
                        for dof in range(dofs):
                            valid.append(col+dof)
            valid.sort()
            removed.sort()

            A = A[:, valid][valid, :]
            M = M[:, valid][valid, :]
            msg('finished!', level=3)

        if sparse_solver:
            mode = 'cayley'
            try:
                msg('eigsh() solver...', level=3)
                eigvals, eigvecs = eigsh(A=A, k=self.num_eigvalues,
                        which='SM', M=M, tol=tol, sigma=1., mode=mode)
                msg('finished!', level=3)
            except Exception, e:
                warn(str(e), level=4)
                msg('aborted!', level=3)
                size22 = A.shape[0]
                M, A, used_cols = remove_null_cols(M, A)
                msg('eigsh() solver...', level=3)
                eigvals, peigvecs = eigsh(A=A, k=self.num_eigvalues,
                        which='SM', M=M, tol=tol, sigma=1., mode=mode)
                msg('finished!', level=3)
                eigvecs = np.zeros((size22, self.num_eigvalues), dtype=DOUBLE)
                eigvecs[used_cols, :] = peigvecs

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

        if remove_null_i1_j1:
            eigvecsALL = np.zeros((self.get_size(), self.num_eigvalues),
                    dtype=DOUBLE)
            eigvecsALL[valid, :] = eigvecs
        else:
            eigvecsALL = eigvecs

        self.eigvals = eigvals
        self.eigvecs = eigvecsALL

        msg('finished!', level=2)

        msg('first {} eigenvalues:'.format(self.num_eigvalues_print), level=1)
        for eig in eigvals[:self.num_eigvalues_print]:
            msg('{}'.format(eig), level=2)
        self.analysis.last_analysis = 'lb'


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
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        if num_cores is None:
            num_cores = self.analysis.ni_num_cores

        if self.k0 is None:
            self.calc_linear_matrices()

        msg('Calculating non-linear matrices...', level=2)
        alpharad = self.alpharad
        r1 = self.r1
        L = self.L
        tminrad = self.tminrad
        tmaxrad = self.tmaxrad
        F = self.F
        m1 = self.m1
        n1 = self.n1
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0
        funcnum = self.funcnum

        nlmodule = modelDB.db[self.model]['non-linear']
        if nlmodule:
            calc_k0L = nlmodule.calc_k0L
            calc_kG = nlmodule.calc_kG
            calc_kLL = nlmodule.calc_kLL

            ni_method = self.analysis.ni_method
            nx = self.analysis.nx
            nt = self.analysis.nt
            kG = calc_kG(c, alpharad, r1, L, tminrad, tmaxrad, F, m1, n1,
                    nx=nx, nt=nt, num_cores=num_cores, method=ni_method,
                    c0=c0, m0=m0, n0=n0)
            k0L = calc_k0L(c, alpharad, r1, L, tminrad, tmaxrad, F, m1, n1,
                    nx=nx, nt=nt, num_cores=num_cores, method=ni_method,
                    c0=c0, m0=m0, n0=n0)
            kLL = calc_kLL(c, alpharad, r1, L, tminrad, tmaxrad, F, m1, n1,
                    nx=nx, nt=nt, num_cores=num_cores, method=ni_method,
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
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)
        alpharad = self.alpharad
        m1 = self.m1
        n1 = self.n1
        r1 = self.r1
        L = self.L
        tminrad = self.tminrad
        tmaxrad = self.tmaxrad
        model = self.model

        fuvw = modelDB.db[model]['commons'].fuvw
        us, vs, ws, phixs, phits = fuvw(c, m1, n1, L, tminrad,
                tmaxrad, xs, ts, r1, alpharad, self.out_num_cores)

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
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)

        alpharad = self.alpharad
        r1 = self.r1
        L = self.L
        tminrad = self.tminrad
        tmaxrad = self.tmaxrad
        sina = self.sina
        cosa = self.cosa
        m1 = self.m1
        n1 = self.n1
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

        es = fstrain(c, sina, cosa, xs, ts, r1, L, tminrad, tmaxrad, m1, n1,
                c0, m0, n0, funcnum, int_NL_kinematics, self.out_num_cores)

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
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)

        F = self.F
        alpharad = self.alpharad
        r1 = self.r1
        L = self.L
        tminrad = self.tminrad
        tmaxrad = self.tmaxrad
        sina = self.sina
        cosa = self.cosa
        m1 = self.m1
        n1 = self.n1
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

        Ns = fstress(c, F, sina, cosa, xs, ts, r1, L, tminrad, tmaxrad, m1,
                n1, c0, m0, n0, funcnum, int_NL_kinematics,
                self.out_num_cores)
        return Ns.reshape((xshape + (e_num,)))


    def add_SPL(self, PL, pt=0.5, theta=0., cte=True):
        """Add a Single Perturbation Load `\{{F_{PL}}_i\}`

        The perturbation load is a particular case of the punctual load which
        as only the normal component (along the `z` axis).

        Parameters
        ----------
        PL : float
            The perturbation load value.
        pt : float, optional
            The normalized meridional in which the new SPL will be included.
        theta : float, optional
            The angular position in radians.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis.

        Notes
        -----
        Each single perturbation load is added to the ``forces`` parameter of
        the ``KPanelT`` object if ``cte=True``, or to the ``forces_inc``
        parameter if ``cte=False``, which may be changed by the analyst at any
        time.

        """
        self._rebuild()
        if cte:
            self.forces.append([pt*self.L, theta, 0., 0., PL])
        else:
            self.forces_inc.append([pt*self.L, theta, 0., 0., PL])


    def add_force(self, x, theta, fx, ftheta, fz, cte=True):
        r"""Add a punctual force with three components

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
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis.

        """
        if cte:
            self.forces.append([x, theta, fx, ftheta, fz])
        else:
            self.forces_inc.append([x, theta, fx, ftheta, fz])


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
        sina = self.sina
        cosa = self.cosa
        r2 = self.r2
        L = self.L
        tminrad = self.tminrad
        tmaxrad = self.tmaxrad
        m1 = self.m1
        n1 = self.n1
        model = self.model

        if not model in modelDB.db.keys():
            raise ValueError(
                    '{} is not a valid model option'.format(model))

        db = modelDB.db
        num0 = db[model]['num0']
        num1 = db[model]['num1']
        dofs = db[model]['dofs']
        fg = db[model]['commons'].fg

        size = self.get_size()

        g = np.zeros((dofs, size), dtype=DOUBLE)
        fext = np.zeros(size, dtype=DOUBLE)

        # non-incrementable punctual forces
        for i, force in enumerate(self.forces):
            x, theta, fx, ftheta, fz = force
            fg(g, m1, n1, x, theta, L, tminrad, tmaxrad)
            if dofs == 3:
                fpt = np.array([[fx, ftheta, fz]])
            elif dofs == 5:
                fpt = np.array([[fx, ftheta, fz, 0, 0]])
            fext += fpt.dot(g).ravel()

        # incrementable punctual forces
        for i, force in enumerate(self.forces_inc):
            x, theta, fx, ftheta, fz = force
            fg(g, m1, n1, x, theta, L, tminrad, tmaxrad)
            if dofs == 3:
                fpt = np.array([[fx, ftheta, fz]])*inc
            elif dofs == 5:
                fpt = np.array([[fx, ftheta, fz, 0, 0]])*inc
            fext += fpt.dot(g).ravel()

        # NxxTop

        NxxTop = self.NxxTop
        NttLeft = self.NttLeft
        NxxTop += inc*self.NxxTop_inc
        NttLeft += inc*self.NttLeft_inc

        Nxx0 = NxxTop[0]
        for j1 in range(n1):
            if j1 > 0:
                Nxxj = NxxTop[j1]
            for i1 in range(m1):
                col = num1*((j1)*m1 + (i1))
                if j1 == 0:
                    fext[col+0] += (-1)**i1*Nxx0*r2*(tmaxrad - tminrad)
                else:
                    fext[col+0] += 1/2.*(-1)**i1*Nxxj*r2*(tmaxrad - tminrad)

        msg('finished!', level=2, silent=silent)

        if np.all(fext==0):
            raise ValueError('No load was applied!')

        return fext


    def calc_k0(self):
        if self.k0 is None:
            self.calc_linear_matrices()
        return self.k0


    def calc_fint(self, c, m=1):
        r"""Calculates the internal force vector `\{F_{int}\}`

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the internal
            forces.
        m : integer, optional
            A multiplier to the number of integration points if one wishes to
            use more integration points to calculate `\{F_{int}\}` than to
            calculate `[K_T]`.

        Returns
        -------
        fint : np.ndarray
            The internal force vector.

        """
        ni_num_cores = self.analysis.ni_num_cores
        ni_method = self.analysis.ni_method
        nlmodule = modelDB.db[self.model]['non-linear']
        nx = self.analysis.nx*m
        nt = self.analysis.nt*m
        fint = nlmodule.calc_fint_0L_L0_LL(c, self.alpharad, self.r1, self.L,
                self.tminrad, self.tmaxrad, self.F, self.m1, self.n1, nx, nt,
                ni_num_cores, ni_method, self.c0, self.m0, self.n0)
        fint += self.k0*c

        return fint


    def calc_kT(self, c):
        self.calc_NL_matrices(c)
        return self.kT


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
        self.cs = self.analysis.cs
        self.increments = self.analysis.increments

        return self.analysis.cs


    def plot(self, c, invert_theta=False, plot_type=1, vec='w',
             deform_u=False, deform_u_sf=100.,
             filename='',
             ax=None, figsize=(3.5, 2.), save=True,
             add_title=False, title='',
             colorbar=False, cbar_nticks=2, cbar_format=None,
             cbar_title='', cbar_fontsize=10,
             aspect='equal', clean=True, dpi=400,
             texts=[], xs=None, ts=None, gridx=300, gridt=300,
             num_levels=400, vecmin=None, vecmax=None):
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
        invert_theta : bool, optional
            Inverts the `\theta` axis of the plot. It may be used to match
            the coordinate system of the finite element models created
            using the ``desicos.abaqus`` module.
        plot_type : int, optional
            For cylinders only ``4`` and ``5`` are valid.
            For cones all the following types can be used:

            - ``1``: concave up (with ``invert_theta=False``) (default)
            - ``2``: concave down (with ``invert_theta=False``)
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

        if invert_theta == True:
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

