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
from numpy import linspace

import compmech.composite.laminate as laminate
from compmech.analysis import Analysis
from compmech.logger import msg, warn
from compmech.constants import DOUBLE
from compmech.sparse import make_symmetric, remove_null_cols

import modelDB


def load(name):
    if '.Plate' in name:
        return cPickle.load(open(name, 'rb'))
    else:
        return cPickle.load(open(name + '.Plate', 'rb'))


class Plate(object):
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
            u = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f}}
            \\
            v = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f}}
            \\
            w = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f}}
            \\
            \phi_x = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f}}
            \\
            \phi_y = \sum_{i_1=0}^{m_1}{\sum_{j_1=0}^{n_1}{f}}
            \\
            f = cos(i_1 \pi b_x)cos(j_1 \pi b_y)
            \\
            b_x = \frac{x + \frac{a}{2}}{a}
            \\
            b_y = \frac{y + \frac{b}{2}}{b}

    """
    def __init__(self):
        self.name = ''

        # boundary conditions
        self.inf = 1.e8 # used to define high stiffnesses
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
        self.m1 = 10
        self.n1 = 10

        # numerical integration
        self.nx = 160
        self.ny = 160
        self.ni_num_cores = 4
        self.ni_method = 'trapz2d'

        # loads
        self.Fx = None
        self.Fy = None
        self.Fxy = None
        self.Fyx = None
        self.NxxTop = None
        self.NxyTop = None
        self.NyyLeft = None
        self.NyxLeft = None
        self.Fx_inc = None
        self.Fy_inc = None
        self.Fxy_inc = None
        self.Fyx_inc = None
        self.NxxTop_inc = None
        self.NxyTop_inc = None
        self.NyyLeft_inc = None
        self.NyxLeft_inc = None
        self.forces = []
        self.forces_inc = []

        # initial imperfection
        self.c0 = None
        self.m0 = 0
        self.n0 = 0
        self.funcnum = 2

        self.a = None
        self.b = None
        self.K = 5/6.

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

        # output queries
        self.out_num_cores = 4

        # analysis
        self.analysis = Analysis(self.calc_fext, self.calc_k0, self.calc_fint,
                self.calc_kT)

        # outputs
        self.increments = None
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
                warn('Plate name unchanged')

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
        self.NxyTop = check_load(self.NxyTop, size)
        self.NxyTop_inc = check_load(self.NxyTop_inc, size)
        # circumferential load
        size = self.m1+1
        self.NyyLeft = check_load(self.NyyLeft, size)
        self.NyyLeft_inc = check_load(self.NyyLeft_inc, size)
        # shear tx
        self.NyxLeft = check_load(self.NyxLeft, size)
        self.NyxLeft_inc = check_load(self.NyxLeft_inc, size)

        # defining load components from force vectors
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
        self.size = num0 + num1*self.m1*self.n1
        return self.size


    def _default_field(self, xs, ys, gridx, gridy):
        if xs is None or ys is None:
            xs = linspace(-self.a/2., self.a/2, gridx)
            ys = linspace(-self.b/2., self.b/2, gridy)
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


    def calc_linear_matrices(self, combined_load_case=None):
        self._rebuild()
        msg('Calculating linear matrices... ', level=2)

        fk0, fkG0, k0edges = modelDB.get_linear_matrices(self)
        model = self.model
        a = self.a
        b = self.b
        m1 = self.m1
        n1 = self.n1
        laminaprops = self.laminaprops
        plyts = self.plyts
        stack = self.stack

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

        msg('finished!', level=2)


    def lb(self, tol=0, combined_load_case=None, sparse_solver=True):
        """Performs a linear buckling analysis

        The following parameters of the ``Plate`` object will affect the
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

            - ``1`` : find the critical Fx for a fixed Fxy
            - ``2`` : find the critical Fx for a fixed Fy
            - ``3`` : find the critical Fy for a fixed Fyx
            - ``4`` : find the critical Fy for a fixed Fx
        sparse_solver : bool, optional
            Tells if solver :func:`scipy.linalg.eigh` or
            :func:`scipy.sparse.linalg.eigsh` should be used.

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``Plate`` object and the `i^{th}` eigenvector in the
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
            M = self.k0 + self.kG0_Fxy
            A = self.kG0_Fx
        elif combined_load_case == 2:
            M = self.k0 + self.kG0_Fy
            A = self.kG0_Fx
        elif combined_load_case == 3:
            M = self.k0 + self.kG0_Fyx
            A = self.kG0_Fy
        elif combined_load_case == 4:
            M = self.k0 + self.kG0_Fx
            A = self.kG0_Fy

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
                sizebkp = A.shape[0]
                M, A, used_cols = remove_null_cols(M, A)
                msg('eigsh() solver...', level=3)
                eigvals, peigvecs = eigsh(A=A, k=self.num_eigvalues,
                        which='SM', M=M, tol=tol, sigma=1., mode=mode)
                msg('finished!', level=3)
                eigvecs = np.zeros((sizebkp, self.num_eigvalues),
                                   dtype=DOUBLE)
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

        self.eigvals = eigvals
        self.eigvecs = eigvecs

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
            num_cores = self.ni_num_cores

        if self.k0 is None:
            self.calc_linear_matrices()

        msg('Calculating non-linear matrices...', level=2)
        a = self.a
        b = self.b
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

            ni_method = self.ni_method
            nx = self.nx
            ny = self.ny
            kG = calc_kG(c, a, b, F, m1, n1, nx=nx, ny=ny,
                    num_cores=num_cores, method=ni_method, c0=c0, m0=m0,
                    n0=n0)
            k0L = calc_k0L(c, a, b, F, m1, n1, nx=nx, ny=ny,
                    num_cores=num_cores, method=ni_method, c0=c0, m0=m0,
                    n0=n0)
            kLL = calc_kLL(c, a, b, F, m1, n1, nx=nx, ny=ny,
                    num_cores=num_cores, method=ni_method, c0=c0, m0=m0,
                    n0=n0)

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


    def uvw(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculates the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the ``Plate`` object.

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
        stored as parameters with the same name in the ``Plate`` object.

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

        es = fstrain(c, xs, ys, a, b, m1, n1,
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

        Ns = fstress(c, F, xs, ys, a, b, m1, n1, c0, m0, n0,
                funcnum, int_NL_kinematics, self.out_num_cores)
        return Ns.reshape((xshape + (e_num,)))


    def add_SPL(self, PL, pt=0.5, y=0., cte=True):
        """Add a Single Perturbation Load `\{{F_{PL}}_i\}`

        The perturbation load is a particular case of the punctual load which
        as only the normal component (along the `z` axis).

        Parameters
        ----------
        PL : float
            The perturbation load value.
        pt : float, optional
            The normalized meridional in which the new SPL will be included.
        y : float, optional
            The angular position in radians.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis.

        Notes
        -----
        Each single perturbation load is added to the ``forces`` parameter of
        the ``Plate`` object if ``cte=True``, or to the ``forces_inc``
        parameter if ``cte=False``, which may be changed by the analyst at any
        time.

        """
        self._rebuild()
        if cte:
            self.forces.append([pt*self.a, y, 0., 0., PL])
        else:
            self.forces_inc.append([pt*self.a, y, 0., 0., PL])


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
            x, y, fx, fy, fz = force
            fg(g, m1, n1, x, y, a, b)
            if dofs == 3:
                fpt = np.array([[fx, fy, fz]])
            elif dofs == 5:
                fpt = np.array([[fx, fy, fz, 0, 0]])
            fext += fpt.dot(g).ravel()

        # incrementable punctual forces
        for i, force in enumerate(self.forces_inc):
            x, y, fx, fy, fz = force
            fg(g, m1, n1, x, y, a, b)
            if dofs == 3:
                fpt = np.array([[fx, fy, fz]])*inc
            elif dofs == 5:
                fpt = np.array([[fx, fy, fz, 0, 0]])*inc
            fext += fpt.dot(g).ravel()

        # NxxTop

        NxxTop = self.NxxTop
        NyyLeft = self.NyyLeft
        NxxTop += inc*self.NxxTop_inc
        NyyLeft += inc*self.NyyLeft_inc

        for j1 in range(1, n1+1):
            Nxxj = NxxTop[j1]
            for i1 in range(1, m1+1):
                col = num1*((j1-1)*m1 + (i1-1))
                fext[col+0] += 1/2.*(-1)**i1*Nxxj*b

        msg('finished!', level=2, silent=silent)

        if np.all(fext==0):
            raise ValueError('No load was applied!')

        return fext


    def calc_k0(self):
        self.calc_linear_matrices()
        return self.k0


    def calc_fint(self, c, inc=1., m=1):
        r"""Calculates the internal force vector `\{F_{int}\}`

        The following attributes affect the numerical integration:

        =================    ================================================
        Attribute            Description
        =================    ================================================
        ``ni_num_cores``     ``int``, number of cores used for the numerical
                             integration
        ``ni_method``        ``str``, integration method:
                                 - ``'trapz2d'`` for 2-D Trapezoidal's rule
                                 - ``'simps2d'`` for 2-D Simpsons' rule
        ``nx``               ``int``, number of integration points along the
                             `x` coordinate
        ``ny``               ``int``, number of integration points along the
                             `y` coordinate
        =================    ================================================

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the internal
            forces.
        inc : float, optional
            A load multiplier only needed to fit the correct function
            signature.
        m : integer, optional
            A multiplier to the number of integration points if one wishes to
            use more integration points to calculate `\{F_{int}\}` than to
            calculate `[K_T]`.

        Returns
        -------
        fint : np.ndarray
            The internal force vector.

        """
        ni_num_cores = self.ni_num_cores
        ni_method = self.ni_method
        nlmodule = modelDB.db[self.model]['non-linear']
        nx = self.nx*m
        ny = self.ny*m
        fint = nlmodule.calc_fint_0L_L0_LL(c, self.a, self.b, self.F, self.m1,
                self.n1, nx, ny, ni_num_cores, ni_method, self.c0, self.m0,
                self.n0)
        fint += self.k0*c

        return fint


    def calc_kT(self, c, inc=1.):
        r"""Calculates the tangent stiffness matrix

        The following attributes affect the numerical integration:

        =================    ================================================
        Attribute            Description
        =================    ================================================
        ``ni_num_cores``     ``int``, number of cores used for the numerical
                             integration
        ``ni_method``        ``str``, integration method:
                                 - ``'trapz2d'`` for 2-D Trapezoidal's rule
                                 - ``'simps2d'`` for 2-D Simpsons' rule
        ``nx``               ``int``, number of integration points along the
                             `x` coordinate
        ``ny``               ``int``, number of integration points along the
                             `y` coordinate
        =================    ================================================

        Parameters
        ----------
        c : np.ndarray
            The Ritz constant vector of the current state.
        inc : float, optional
            A load multiplier only needed to fit the correct function
            signature.

        Returns
        -------
        kT : sparse matrix
            The tangent stiffness matrix.

        """
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
            the `name` parameter of the ``Plate`` object will be used.
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
        """Save the ``Plate`` object using ``cPickle``

        Notes
        -----
        The pickled file will have the name stored in ``Plate.name``
        followed by a ``'.Plate'`` extension.

        """
        name = self.name + '.Plate'
        msg('Saving Plate to {}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    p = Plate()
    p.a = 400. # mm
    p.b = 200. # mm

    for yi in linspace(-p.b/2., +p.b/2., 100):
        p.add_force(p.a/2., yi, 0., 1., 0.)

    p.laminaprop = (142.5e3, 8.7e3, 0.28, 5.1e3, 5.1e3, 5.1e3, 273.15)
    p.plyt = 0.125 # mm
    p.stack = [0, 0, 90]
    p.bc = 'ss1-ss4-ss4-ss4'
    p.m1 = 20
    p.n1 = 20

    lb = False
    if lb:
        p.lb()
        p.plot(p.eigvecs[:, 0])
    else:
        p.static()
        p.plot(p.analysis.cs[0], vec='v', colorbar=True, cbar_fontsize=6.)
