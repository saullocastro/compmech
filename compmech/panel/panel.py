import gc
import pickle
from multiprocessing import cpu_count

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig
from numpy import linspace, deg2rad
import matplotlib.cm as cm

import compmech.composite.laminate as laminate
from compmech.analysis import Analysis
from compmech.logger import msg, warn
from compmech.constants import DOUBLE
from compmech.sparse import (remove_null_cols, make_skew_symmetric,
        finalize_symmetric_matrix)

from . import modelDB


def load(name):
    if '.Panel' in name:
        return pickle.load(open(name, 'rb'))
    else:
        return pickle.load(open(name + '.Panel', 'rb'))


def check_c(c, size):
    if not isinstance(c, np.ndarray):
        raise TypeError('"c" must be a NumPy ndarray object')
    if c.ndim != 1:
        raise ValueError('"c" must be a 1-D ndarray object')
    if c.shape[0] != size:
        raise ValueError('"c" must have the same size as the global stiffness matrix')


class Panel(object):
    r"""General Panel class

    It works for both flat plates, cylindrical and conical panels. The right
    model is selected according to parameters ``r`` (radius) and ``alphadeg``
    (semi-vertex angle).

    The approximation functions for the displacement fields are built using
    :ref:`Bardell's functions <theory_func_bardell>`.

    Parameters
    ----------
    a : float, optional
        Length (along the `x` coordinate).
    b : float, optional
        Width (along the `y` coordinate).
    r : float, optional
        Radius for cylindrical panels.
    alphadeg : float, optional
        Semi-vertex angle for conical panels.
    stack : list or tuple, optional
        A sequence representing the angles for each ply.
    plyt : float, optional
        Ply thickness.
    laminaprop : list or tuple, optional
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`.
    mu : float, optional
        Material density.
    m, n : int, optional
        Number of terms for the approximation functions along `x` and `y`,
        respectively.
    offset : float, optional
        Laminate offset about panel mid-surface. The offset is measured along
        the normal (`z`) axis.
    y1, y2 : float, optional
        Define the lower and upper limit along `y` for panel with incomplete
        domains.

    """
    def __init__(self, a=None, b=None, r=None, alphadeg=None,
            stack=None, plyt=None, laminaprop=None, mu=None, m=11, n=11,
            offset=0., y1=None, y2=None, **kwargs):
        self.a = a
        self.b = b
        self.y1 = y1
        self.y2 = y2
        self.r = r
        self.alphadeg = alphadeg
        self.stack = stack
        self.plyt = plyt
        self.laminaprop = laminaprop
        self.offset = offset
        # assembly
        self.group = None
        self.x0 = 0
        self.y0 = 0
        self.row_start = None
        self.col_start = None
        self.row_end = None
        self.col_end = None

        self.name = 'panel'
        self.bay = None

        # model
        self.model = None
        self.fsdt_shear_correction = 5/6. # in case of First-order Shear Deformation Theory

        # approximation series
        self.m = m
        self.n = n

        # numerical integration
        self.nx = m
        self.ny = n
        self.ni_num_cores = cpu_count()//2
        self.ni_method = 'trapz2d'
        self.c0 = None

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

        # material
        self.mu = mu
        self.plyts = None
        self.laminaprops = None

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
        self.analysis = Analysis(self.calc_fext, self.calc_k0, self.calc_fint, self.calc_kT)

        # outputs
        self.increments = None
        self.eigvecs = None
        self.eigvals = None

        for k, v in kwargs.items():
            setattr(self, k, v)

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
        num = modelDB.db[self.model]['num']
        self.size = num*self.m*self.n
        return self.size


    def _default_field(self, xs, ys, gridx, gridy):
        if xs is None or ys is None:
            xs = linspace(0, self.a, gridx)
            ys = linspace(0, self.b, gridy)
            xs, ys = np.meshgrid(xs, ys, copy=True)
        xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
        ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
        xshape = xs.shape
        yshape = ys.shape
        if xshape != yshape:
            raise ValueError('Arrays xs and ys must have the same shape')
        self.Xs = xs
        self.Ys = ys
        xs = np.ascontiguousarray(xs.ravel(), dtype=DOUBLE)
        ys = np.ascontiguousarray(ys.ravel(), dtype=DOUBLE)

        return xs, ys, xshape, yshape


    def _get_lam_F(self, silent=False):
        if self.lam is None:
            raise RuntimeError('lam object is None!')
        if 'clt' in self.model:
            F = self.lam.ABD

        elif 'fsdt' in self.model:
            F = self.lam.ABDE
            F[6:, 6:] *= self.fsdt_shear_correction

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

        return F


    def calc_k0(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None, nx=None, ny=None, Fnxny=None, inc=None, NLgeom=False):
        r"""Calculate the constitutive stiffness matrix

        If ``c`` is not given it calculates the linear constitutive stiffness
        matrix, otherwise the large displacement linear constitutive stiffness
        matrix is calculated. When using ``c``the size of ``c`` must be the
        same as ``size``.

        In assemblies of semi-analytical models the sparse matrices that are
        calculated may have the ``size`` of the assembled global model, and the
        current constitutive matrix being calculated starts at position
        ``row0`` and ``col0``.

        Parameters
        ----------
        size : int
            The size of the calculated sparse matrices.
        row0, col0: int or None, optional
            Offset to populate the output sparse matrix (useful when
            assemblying panels).
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        finalize : bool, optional
            Asserts validity of output data and makes the output matrix
            symmetric, should be ``False`` when assemblying.
        c : array-like or None, optional
            This must be the result of a static analysis, used to compute
            non-linear terms based on the actual displacement field.
        nx, ny : int or None, optional
            Number of integration points along `x` and `y`, respectively, for
            the Legendre-Gauss quadrature rule applied in the numerical
            integration. Only used when ``c`` is given.
        Fnxny : 4-D array-like or None, optional
            The constitutive relations for the laminate at each integration
            point. Must be a 4-D array of shape ``(nx, ny, 6, 6)`` when using
            classical laminated plate theory models.
        NLgeom : bool, optional
            Flag to indicate if geometrically non-linearities should be
            considered.

        """
        self._rebuild()
        if size is None:
            size = self.get_size()
        if c is None:
            msg('Calculating k0... ', level=2, silent=silent)
        else:
            check_c(c, size)
            msg('Calculating kL... ', level=2, silent=silent)

        matrices = modelDB.db[self.model]['matrices']

        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        self.alpharad = deg2rad(alphadeg)
        self.r = self.r if self.r is not None else 0.

        if self.stack is not None:
            lam = laminate.read_stack(self.stack, plyts=self.plyts,
                                      laminaprops=self.laminaprops,
                                      offset=self.offset)
            self.lam = lam
            self.F = self._get_lam_F()

        if self.y1 is not None and self.y2 is not None:
            if c is not None or Fnxny is not None:
                raise NotImplementedError(
                'Partial domain from y1 to y2 not implemented for kL')
            k0 = matrices.fk0y1y2(self.y1, self.y2, self, size, row0, col0)
        else:
            if c is None and Fnxny is None:
                k0 = matrices.fk0(self, size, row0, col0)
            else:
                matrices_num = modelDB.db[self.model]['matrices_num']
                nx = self.nx if nx is None else nx
                ny = self.ny if ny is None else ny
                #NOTE the consistence checks for Fnxny are done within the .pyx
                #     files
                Fnxny = self.F if Fnxny is None else Fnxny
                if c is None:
                    # Empty c if the interest is only on the heterogeneous
                    # laminate properties
                    c = np.zeros(self.size, dtype=DOUBLE)
                c = np.ascontiguousarray(c, dtype=DOUBLE)
                k0 = matrices_num.fkL_num(c, Fnxny, self,
                         size, row0, col0, nx, ny, NLgeom=int(NLgeom))

        #TODO allow constant stress state to be obtained using static results,
        #     which would require just passing a 'c_cte' used to calculate kG0
        Nxx_cte = self.Nxx_cte if self.Nxx_cte is not None else 0.
        Nyy_cte = self.Nyy_cte if self.Nyy_cte is not None else 0.
        Nxy_cte = self.Nxy_cte if self.Nxy_cte is not None else 0.

        if Nxx_cte != 0. or Nyy_cte != 0. or Nxy_cte != 0.:
            if self.y1 is not None and self.y2 is not None:
                k0 += matrices.fkG0y1y2(self.y1, self.y2, Nxx_cte, Nyy_cte, Nxy_cte,
                           self, size, row0, col0)
            else:
                k0 += matrices.fkG0(Nxx_cte, Nyy_cte, Nxy_cte,
                           self, size, row0, col0)

        if finalize:
            k0 = finalize_symmetric_matrix(k0)
        self.k0 = k0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return k0


    def calc_kG0(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None, nx=None, ny=None, Fnxny=None, NLgeom=False):
        r"""Calculate the linear geometric stiffness matrix

        See :meth:`.Panel.calc_k0` for details on each parameter.

        """
        self._rebuild()
        if size is None:
            size = self.get_size()
        if c is None:
            msg('Calculating kG0... ', level=2, silent=silent)
            matrices = modelDB.db[self.model]['matrices']
        else:
            check_c(c, size)
            msg('Calculating kG... ', level=2, silent=silent)
            matrices = modelDB.db[self.model]['matrices_num']

        y1 = self.y1
        y2 = self.y2
        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        self.alpharad = deg2rad(alphadeg)
        self.r = self.r if self.r is not None else 0.

        Nxx = self.Nxx if self.Nxx is not None else 0.
        Nyy = self.Nyy if self.Nyy is not None else 0.
        Nxy = self.Nxy if self.Nxy is not None else 0.

        if c is None:
            if y1 is not None and y2 is not None:
                kG0 = matrices.fkG0y1y2(y1, y2, Nxx, Nyy, Nxy, self,
                           size, row0, col0)
            else:
                kG0 = matrices.fkG0(Nxx, Nyy, Nxy, self, size, row0, col0)
        else:
            if y1 is not None or y2 is not None:
                raise NotImplementedError('Only y1=0, y2=b is implemented!')
            c = np.ascontiguousarray(c, dtype=DOUBLE)
            nx = self.nx if nx is None else nx
            ny = self.ny if ny is None else ny
            if Fnxny is None:
                Fnxny = self._get_lam_F()
            kG0 = matrices.fkG_num(c, Fnxny, self,
                       size, row0, col0, nx, ny, NLgeom=int(NLgeom))

        if finalize:
            kG0 = finalize_symmetric_matrix(kG0)
        self.kG0 = kG0

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kG0


    def calc_kT(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None, nx=None, ny=None, Fnxny=None, inc=None):
        kL = self.calc_k0(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, Fnxny=Fnxny, inc=inc, NLgeom=True)
        kG = self.calc_kG0(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, Fnxny=Fnxny, NLgeom=True)
        kT = kL + kG
        self.kT = kT

        return kT


    def calc_kM(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        r"""Calculate the mass matrix
        """
        msg('Calculating kM... ', level=2, silent=silent)

        matrices = modelDB.db[self.model]['matrices']

        y1 = self.y1
        y2 = self.y2
        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        self.alpharad = deg2rad(alphadeg)
        self.r = self.r if self.r is not None else 0.

        if size is None:
            size = self.get_size()

        #TODO allow a distribution of mu instead of constant value, at least allow a mu for each ply
        if self.mu is None:
            raise ValueError('Attribute "mu" (density) must be defined')

        if y1 is not None and y2 is not None:
            kM = matrices.fkMy1y2(y1, y2, self.offset, self, size, row0, col0)
        else:
            kM = matrices.fkM(self.offset, self, size, row0, col0)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kM


    def calc_kA(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        r"""Calculate the aerodynamic matrix using the linear piston theory
        """
        msg('Calculating kA... ', level=2, silent=silent)

        if 'kpanel' in self.model:
            raise NotImplementedError('Conical panels not supported')

        matrices = modelDB.db[self.model]['matrices']

        if size is None:
            size = self.get_size()

        self.r = self.r if self.r is not None else 0.

        if self.beta is None:
            if self.Mach is None:
                raise ValueError('Mach number cannot be a NoneValue')
            elif self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                self.Mach = 1.0001
            Mach = self.Mach
            beta = self.rho_air * self.V**2 / (Mach**2 - 1)**0.5
            if self.r != 0.:
                gamma = beta*1./(2.*self.r*(Mach**2 - 1)**0.5)
            else:
                gamma = 0.
            ainf = self.speed_sound
            aeromu = beta/(Mach*ainf)*(Mach**2 - 2)/(Mach**2 - 1)
        else:
            beta = self.beta
            gamma = self.gamma if self.gamma is not None else 0.
            aeromu = self.aeromu if self.aeromu is not None else 0.

        if self.flow.lower() == 'x':
            kA = matrices.fkAx(beta, gamma, self, size, row0, col0)
        elif self.flow.lower() == 'y':
            kA = matrices.fkAy(beta, self, size, row0, col0)
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

        return kA


    def calc_cA(self, aeromu, silent=False, finalize=True):
        r"""Calculate the aerodynamic damping matrix using the piston theory
        """
        msg('Calculating cA... ', level=2, silent=silent)

        matrices = modelDB.db[self.model]['matrices']
        cA = matrices.fcA(aeromu, self, self.size, 0, 0)
        cA = cA*(0+1j)

        if finalize:
            cA = finalize_symmetric_matrix(cA)
        self.cA = cA

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def lb(self, tol=0, sparse_solver=True, calc_kA=False, silent=False,
           nx=10, ny=10, c=None, ckL=None, Fnxny=None):
        r"""Linear buckling analysis

        .. note:: This will be deprecated soon, use
                  :func:`.compmech.analysis.lb`.

        The following parameters will affect the linear buckling analysis:

        =======================    =====================================
        parameter                  description
        =======================    =====================================
        ``num_eigvalues``        Number of eigenvalues to be extracted
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
        c : array-like, optional
            A set of Ritz constants that will be use to compute KG.
        ckL : array-like, optional
            A set of Ritz constants that will be use to compute KL.
        nx and ny : int or None, optional
            Number of integration points along `x` and `y`, respectively, for
            the Legendre-Gauss quadrature rule applied in the numerical
            integration.
        Fnxny : 4-D array-like or None, optional
            The constitutive relations for the laminate at each integration
            point. Must be a 4-D array of shape ``(nx, ny, 6, 6)`` when using
            classical laminated plate theory models.

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``Panel`` object and the `i^{th}` eigenvector in the
        ``eigvecs[:, i-1]`` parameter.

        """
        msg('Running linear buckling analysis...', silent=silent)

        msg('Eigenvalue solver... ', level=2, silent=silent)

        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
        if ckL is None:
            self.calc_k0(silent=silent)
        else:
            self.calc_k0(silent=silent, c=ckL, nx=nx, ny=ny, Fnxny=Fnxny)
        self.calc_kG0(silent=silent, c=c, nx=nx, ny=ny, Fnxny=Fnxny)

        if calc_kA:
            raise NotImplementedError('kA requires non-Hermitian eigen solver')
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
            except Exception as e:
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
        for eigi in eigvals[:self.num_eigvalues_print]:
            msg('{0}'.format(eigi), level=2, silent=silent)
        self.analysis.last_analysis = 'lb'


    def freq(self, atype=4, tol=0, sparse_solver=True, silent=False,
             sort=True, damping=False, reduced_dof=False):
        r"""Natural frequency analysis

        .. note:: This will be deprecated soon, use
                  :func:`.compmech.analysis.freq`.

        The following parameters of the will affect the linear buckling
        analysis:

        =======================    =====================================
        parameter                  description
        =======================    =====================================
        ``num_eigvalues``        Number of eigenvalues to be extracted
        ``num_eigvalues_print``    Number of eigenvalues to print after
                                   the analysis is completed
        =======================    =====================================

        Parameters
        ----------
        atype : int, optional
            Tells which analysis type should be performed:

            - ``1`` : considers k0, kA and kG0 (and cA depending on 'damping')
            - ``2`` : considers k0 and kA (and cA depending on 'damping')
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
        k = min(self.num_eigvalues, M.shape[0]-2)
        if sparse_solver:
            if damping:
                raise NotImplementedError('Damping with sparse_solver not implemented!')

            msg('eigs() solver...', level=3, silent=silent)
            sizebkp = M.shape[0]
            K, M, used_cols = remove_null_cols(K, M, silent=silent,
                    level=3)
            eigvals, peigvecs = eigs(A=K, k=k, which='LM', M=M, tol=tol,
                                     sigma=-1.)
            eigvecs = np.zeros((sizebkp, self.num_eigvalues),
                               dtype=peigvecs.dtype)
            eigvecs[used_cols, :] = peigvecs

            eigvals = np.sqrt(eigvals) # omega^2 to omega, in rad/s

        else:
            msg('eig() solver...', level=3, silent=silent)
            M = M.toarray()
            K = K.toarray()
            sizebkp = M.shape[0]
            col_sum = M.sum(axis=0)
            check = col_sum != 0
            used_cols = np.arange(M.shape[0])[check]
            M = M[:, check][check, :]
            K = K[:, check][check, :]

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
                cA = cA[:, check][check, :]
                if reduced_dof:
                    cA = cA[:, take][take, :]
                I = np.identity(size)
                Z = np.zeros_like(M)
                M = np.row_stack((np.column_stack((I, Z)),
                                  np.column_stack((Z, -M))))
                K = np.row_stack((np.column_stack((Z, -I)),
                                  np.column_stack((K, cA))))

            eigvals, peigvecs = eig(a=M, b=K)

            eigvecs = np.zeros((sizebkp, K.shape[0]),
                               dtype=peigvecs.dtype)
            eigvecs[check, :] = peigvecs

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


    def uvw(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

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

        xs, ys, xshape, yshape = self._default_field(xs, ys, gridx, gridy)
        fuvw = modelDB.db[self.model]['field'].fuvw
        us, vs, ws, phixs, phiys = fuvw(c, self, xs, ys, self.out_num_cores)

        self.u = us.reshape(xshape)
        self.v = vs.reshape(xshape)
        self.w = ws.reshape(xshape)
        self.phix = phixs.reshape(xshape)
        self.phiy = phiys.reshape(xshape)

        return self.u, self.v, self.w, self.phix, self.phiy


    def strain(self, c, xs=None, ys=None, gridx=300, gridy=300, NLterms=True):
        r"""Calculate the strain field

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
        NLterms : bool
            Flag to indicate whether non-linear strain components should be considered.

        Returns
        -------
        res : dict
            A dictionary of ``np.ndarrays`` with the keys:
            ``(x, y, exx, eyy, gxy, kxx, kyy, kxy)``

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        xs, ys, xshape, yshape = self._default_field(xs, ys, gridx, gridy)
        fstrain = modelDB.db[self.model]['field'].fstrain
        exx, eyy, gxy, kxx, kyy, kxy = fstrain(c, self, xs, ys, self.out_num_cores, int(NLterms))
        res = {}
        res['x'] = xs.reshape(xshape)
        res['y'] = ys.reshape(yshape)
        res['exx'] = exx.reshape(xshape)
        res['eyy'] = eyy.reshape(xshape)
        res['gxy'] = gxy.reshape(xshape)
        res['kxx'] = kxx.reshape(xshape)
        res['kyy'] = kyy.reshape(xshape)
        res['kxy'] = kxy.reshape(xshape)
        return res


    def stress(self, c, F=None, xs=None, ys=None, gridx=300, gridy=300, NLterms=True):
        r"""Calculate the stress field

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation.
        F : np.ndarray, optional
            The laminate stiffness matrix. Can be a 6 x 6 (ABD) matrix for
            homogeneous laminates over the whole domain.
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
        NLterms : bool
            Flag to indicate whether non-linear strain components should be considered.

        Returns
        -------
        res : dict
            A dictionary of ``np.ndarrays`` with the keys:
            ``(x, y, Nxx, Nyy, Nxy, Mxx, Myy, Mxy)``

        """
        res_strain = self.strain(c, xs, ys, gridx, gridy)
        x = res_strain['x']
        y = res_strain['y']
        exx = res_strain['exx']
        eyy = res_strain['eyy']
        gxy = res_strain['gxy']
        kxx = res_strain['kxx']
        kyy = res_strain['kyy']
        kxy = res_strain['kxy']
        if F is None:
            F = self.F
        if F is None:
            raise ValueError('Laminate ABD matrix not defined for panel')
        res = {}
        res['x'] = x
        res['y'] = y
        res['Nxx'] = exx*F[0, 0] + eyy*F[0, 1] + gxy*F[0, 2] + kxx*F[0, 3] + kyy*F[0, 4] + kxy*F[0, 5]
        res['Nyy'] = exx*F[1, 0] + eyy*F[1, 1] + gxy*F[1, 2] + kxx*F[1, 3] + kyy*F[1, 4] + kxy*F[1, 5]
        res['Nxy'] = exx*F[2, 0] + eyy*F[2, 1] + gxy*F[2, 2] + kxx*F[2, 3] + kyy*F[2, 4] + kxy*F[2, 5]
        res['Mxx'] = exx*F[3, 0] + eyy*F[3, 1] + gxy*F[3, 2] + kxx*F[3, 3] + kyy*F[3, 4] + kxy*F[3, 5]
        res['Myy'] = exx*F[4, 0] + eyy*F[4, 1] + gxy*F[4, 2] + kxx*F[4, 3] + kyy*F[4, 4] + kxy*F[4, 5]
        res['Mxy'] = exx*F[5, 0] + eyy*F[5, 1] + gxy*F[5, 2] + kxx*F[5, 3] + kyy*F[5, 4] + kxy*F[5, 5]

        return res


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


    def calc_fext(self, inc=1., size=None, col0=0, silent=False):
        r"""Calculate the external force vector `\{F_{ext}\}`

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

        model = self.model
        if not model in modelDB.db.keys():
            raise ValueError(
                    '{} is not a valid model option'.format(model))
        db = modelDB.db
        dofs = db[model]['dofs']
        fg = db[model]['field'].fg

        if size is None:
            size = self.get_size()
        col1 = col0 + self.get_size()
        g = np.zeros((dofs, self.get_size()), dtype=DOUBLE)
        fext = np.zeros(size, dtype=DOUBLE)

        # non-incrementable punctual forces
        for i, force in enumerate(self.forces):
            x, y, fx, fy, fz = force
            fg(g, x, y, self)
            if dofs == 3:
                fpt = np.array([[fx, fy, fz]])
            elif dofs == 5:
                fpt = np.array([[fx, fy, fz, 0, 0]])
            fext[col0:col1] += fpt.dot(g).ravel()

        # incrementable punctual forces
        for i, force in enumerate(self.forces_inc):
            x, y, fx, fy, fz = force
            fg(g, x, y, self)
            if dofs == 3: #CLT
                fpt = np.array([[fx, fy, fz]])*inc
            elif dofs == 5: #FSDT
                fpt = np.array([[fx, fy, fz, 0, 0]])*inc
            fext[col0:col1] += fpt.dot(g).ravel()

        return fext


    def calc_fint(self, c, size=None, col0=0, silent=False, nx=None,
            ny=None, Fnxny=None, inc=None):
        r"""Calculate the internal force vector `\{F_{int}\}`


        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the internal forces
            calculation.
        size : int, optional
            The size of the internal force vector. Can be the size of a global
            internal force vector of an assembly.
        col0 : int, optional
            Offset in a global internal forcce vector of an assembly.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        nx : int, optional
            Number of integration points along `x`.
        ny : int, optional
            Number of integration points along `y`.
        Fnxny : np.ndarray, optional
            Laminate stiffness for each integration point, if not supplied it
            will assume constant properties over the panel domain.
        inc : float, optional
            Load increment.

        Returns
        -------
        fint : np.ndarray
            The internal force vector

        """
        #TODO inc not needed here; only with prescribed displacements
        msg('Calculating internal forces...', level=2, silent=silent)
        model = self.model
        if not model in modelDB.db.keys():
            raise ValueError(
                    '{0} is not a valid model option'.format(model))
        matrices_num = modelDB.db[model].get('matrices_num')
        if matrices_num is None:
            raise ValueError('matrices_num not implemented for model {0}'.
                    format(model))
        calc_fint = getattr(matrices_num, 'calc_fint', None)
        if calc_fint is None:
            raise ValueError('calc_fint not implemented for model {0}'.
                    format(model))

        if size is None:
            size = self.get_size()

        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        self.alpharad = deg2rad(alphadeg)
        self.r = self.r if self.r is not None else 0.
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
        Fnxny = self.F if Fnxny is None else Fnxny

        c = np.ascontiguousarray(c, dtype=DOUBLE)
        fint = calc_fint(c, Fnxny, self, size, col0, nx, ny)

        gc.collect()

        msg('finished!', level=2, silent=silent)

        return fint


    def static(self, NLgeom=False, silent=False):
        r"""Static analysis for cones and cylinders

        .. note:: This will be deprecated soon, use
                  :func:`.compmech.analysis.static`.

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
        self._rebuild()
        self.analysis.line_search = False
        self.analysis.kT_initial_state = False
        self.analysis.compute_every_n = 6

        if NLgeom and not modelDB.db[self.model]['non-linear static']:
            msg('________________________________________________',
                silent=silent)
            msg('', silent=silent)
            warn('Model {} cannot be used in non-linear static analysis!'.
                 format(self.model), silent=silent)
            msg('________________________________________________',
                silent=silent)
            raise
        elif not NLgeom and not modelDB.db[self.model]['matrices']:
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


    def plot(self, c, invert_y=False, vec='w', deform_u=False,
            deform_u_sf=100., filename='', ax=None, figsize=(3.5, 2.),
            save=True, title='', colorbar=False, cbar_nticks=2,
            cbar_format=None, cbar_title='', cbar_fontsize=10, colormap='jet',
            aspect='equal', clean=True, dpi=400, texts=[], xs=None, ys=None,
            gridx=300, gridy=300, num_levels=400, vecmin=None, vecmax=None,
            plotoffsetxs=0., plotoffsetys=0.):
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
            Inverts the `y` axis of the plot.
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
        title : str, optional
            If any string is given it is added as title to the contour plot.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels.
        colormap : string, optional
            Name of a matplotlib available colormap.
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

        import matplotlib
        import matplotlib.pyplot as plt

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
            field = es.get(vec)
        elif vec in stresses:
            Ns = self.stress(c, xs=xs, ys=ys,
                             gridx=gridx, gridy=gridy, NLterms=True)
            field = Ns.get(vec)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))
        msg('Finished!', level=1)

        Xs = self.Xs + plotoffsetxs
        Ys = self.Ys + plotoffsetys

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

        x = Ys # in matplotlib x goes vertically (axis=0)
        y = Xs # and y goes horizontally (axis=1)

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

            colormap_obj = getattr(cm, colormap, None)
            if colormap_obj is None:
                warn('Invalid colormap, using "jet"', level=1)
                colormap_obj = cm.jet

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbarticks = linspace(vecmin, vecmax, cbar_nticks)
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax, cmap=colormap_obj)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            try:
                cbar.outline.remove()
            except NotImplementedError:
                pass
            cbar.ax.tick_params(labelsize=fsize, pad=0., tick2On=False)

        if invert_y == True:
            ax.invert_yaxis()
        ax.invert_xaxis()

        if title != '':
            ax.set_title(str(title))

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
        r"""Save the ``Panel`` object using ``pickle``

        Notes
        -----
        The pickled file will have the name stored in ``Panel.name``
        followed by a ``'.Panel'`` extension.

        """
        name = self.name + '.Panel'
        msg('Saving Panel to {}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

