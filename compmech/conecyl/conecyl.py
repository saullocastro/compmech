import gc
import os
import traceback
import time
import pickle

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import leastsq
from numpy import linspace, pi, cos, sin, tan, deg2rad

from .conecylDB import ccs, laminaprops
import compmech.composite.laminate as laminate
from compmech.analysis import Analysis
from compmech.logger import msg, warn, error
from compmech.sparse import remove_null_cols, make_symmetric
from compmech.constants import DOUBLE
from . import modelDB
from .modelDB import get_model


def load(name):
    if '.ConeCyl' in name:
        cc = pickle.load(open(name, 'rb'))
    else:
        cc = pickle.load(open(name + '.ConeCyl', 'rb'))
    cc.analysis.calc_fext = cc.calc_fext
    cc.analysis.calc_k0 = cc.calc_k0
    cc.analysis.calc_fint = cc.calc_fint
    cc.analysis.calc_kT = cc.calc_kT
    return cc


class ConeCyl(object):
    r"""
    """
    __slots__ = ['_load_rebuilt', 'name', 'alphadeg', 'alpharad', 'r1', 'r2',
            'L', 'H', 'h', 'K', 'is_cylinder', 'inf', 'zero',
            'bc', 'kuBot', 'kvBot', 'kwBot', 'kphixBot', 'kphitBot', 'kuTop',
            'kvTop', 'kwTop', 'kphixTop', 'kphitTop', 'model', 'm1', 'm2',
            'size', 'n2', 's', 'nx', 'nt', 'ni_num_cores', 'ni_method',
            'forces', 'forces_inc', 'P',
            'P_inc', 'pdC', 'Fc', 'Nxxtop', 'uTM', 'c0', 'm0', 'n0',
            'funcnum', 'pdT', 'T', 'T_inc', 'thetaTdeg', 'thetaTrad', 'pdLA',
            'tLAdeg', 'tLArad', 'betadeg', 'betarad', 'xiLA', 'MLA', 'LA',
            'num0', 'excluded_dofs', 'excluded_dofs_ck', 'sina', 'cosa',
            'laminaprop', 'plyt', 'laminaprops', 'stack', 'plyts', 'F',
            'F_reuse', 'force_orthotropic_laminate', 'E11', 'nu',
            'num_eigvalues', 'num_eigvalues_print',

            'analysis', 'with_k0L', 'with_kLL',
            'cs', 'increments', 'outputs',
            'eigvals', 'eigvecs',

            'k0', 'k0uk', 'k0uu', 'kTuk', 'kTuu', 'kG0', 'kG0_Fc', 'kG0_P',
            'kG0_T', 'kG', 'kGuu', 'kL', 'kLuu', 'lam', 'u', 'v', 'w', 'phix',
            'phit', 'Xs', 'Ts',

            'out_num_cores',
        ]

    def __init__(self):
        self.name = 'no_name_defined'

        # geometry
        self.alphadeg = 0.
        self.alpharad = 0.
        self.r1 = None
        self.r2 = None
        self.L = None
        self.H = None
        self.h = None # total thickness, required for isotropic shells
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

        # default equations
        self.model = 'clpt_donnell_bc1'

        # approximation series
        self.m1 = 120
        self.m2 = 25
        self.n2 = 45

        # analytical integration for cones
        self.s = 79

        # numerical integration
        self.nx = 120
        self.nt = 180
        self.ni_num_cores = 4
        self.ni_method = 'trapz2d'

        # punctual loads
        self.forces = []
        self.forces_inc = []

        # internal pressure measured in force/area
        self.P = 0.
        self.P_inc = 0.

        # axial compression
        self.pdC = False
        self.Fc = None
        self.Nxxtop = None
        self.uTM = 0.
        self._load_rebuilt = False

        # initial imperfection
        self.c0 = None
        self.m0 = 0
        self.n0 = 0
        self.funcnum = 2

        # torsion
        self.pdT = True
        self.T = 0.
        self.T_inc = 0.
        self.thetaTdeg = 0.
        self.thetaTrad = 0.

        # load asymmetry (la)
        self.pdLA = True
        self.tLAdeg = 0.
        self.tLArad = 0.
        self.betadeg = 0.
        self.betarad = 0.
        self.xiLA = None
        self.MLA = None
        self.LA = None

        self.num0 = 3
        self.excluded_dofs = []
        self.excluded_dofs_ck = []

        self.sina = None
        self.cosa = None

        # material
        self.laminaprop = None
        self.plyt = None
        self.laminaprops = []
        self.stack = []
        self.plyts = []

        # constitutive law
        self.F_reuse = None
        self.F = None
        self.force_orthotropic_laminate = False
        self.E11 = None
        self.nu = None
        self.K = 5/6.

        # eigenvalue analysis
        self.num_eigvalues = 50
        self.num_eigvalues_print = 5

        # output queries
        self.out_num_cores = 4
        self.cs = []
        self.increments = []

        # analysis
        self.analysis = Analysis(self.calc_fext, self.calc_k0, self.calc_fint,
                                 self.calc_kT)
        self.with_k0L = True
        self.with_kLL = True

        # outputs
        self.outputs = {}

        self._clear_matrices()


    def _clear_matrices(self):
        self.k0 = None
        self.k0uk = None
        self.k0uu = None
        self.kTuk = None
        self.kTuu = None
        self.kG0 = None
        self.kG0_Fc = None
        self.kG0_P = None
        self.kG0_T = None
        self.kG = None
        self.kGuu = None
        self.kL = None
        self.kLuu = None
        self.lam = None
        self.u = None
        self.v = None
        self.w = None
        self.phix = None
        self.phit = None
        self.Xs = None
        self.Ts = None
        self.Nxxtop = None

        gc.collect()


    def _rebuild(self):
        if self.k0 is not None:
            if self.k0.shape[0] != self.get_size():
                self._clear_matrices()
                self._load_rebuilt = False
                self._rebuild()

        self.model = self.model.lower()

        # boundary conditions
        inf = self.inf
        zero = self.zero

        if inf > 1.e8:
            warn('"inf" parameter reduced to 1.e8 due to the verified ' +
                 'numerical instability for higher values', level=2)
            inf = 1.e8

        if self.bc is not None:
            bc = self.bc.lower()

            if '_' in bc:
                # different bc for Bot and Top
                bc_Bot, bc_Top = self.bc.split('_')
            elif '-' in bc:
                # different bc for Bot and Top
                bc_Bot, bc_Top = self.bc.split('-')
            else:
                bc_Bot = bc_Top = bc

            bcs = dict(bc_Bot=bc_Bot, bc_Top=bc_Top)
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
                    setattr(self, 'kphit' + sufix, zero)
                elif bcs[k] == 'cc2':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, inf)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphit' + sufix, zero)
                elif bcs[k] == 'cc3':
                    setattr(self, 'ku' + sufix, inf)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphit' + sufix, zero)
                elif bcs[k] == 'cc4':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, inf)
                    setattr(self, 'kphix' + sufix, inf)
                    setattr(self, 'kphit' + sufix, zero)

                elif bcs[k] == 'free':
                    setattr(self, 'ku' + sufix, zero)
                    setattr(self, 'kv' + sufix, zero)
                    setattr(self, 'kw' + sufix, zero)
                    setattr(self, 'kphix' + sufix, zero)
                    setattr(self, 'kphit' + sufix, zero)
                else:
                    text = '"{}" is not a valid boundary condition!'.format(bc)
                    raise ValueError(text)

        self.alpharad = deg2rad(self.alphadeg)
        self.sina = sin(self.alpharad)
        self.cosa = cos(self.alpharad)

        if not self.H and not self.L:
            self.H = (self.r1-self.r2)/tan(self.alpharad)
        if self.H and not self.L:
            self.L = self.H/self.cosa
        if self.L and not self.H:
            self.H = self.L*self.cosa

        if not self.r2:
            if not self.r1:
                raise ValueError('Radius "r1" or "r2" must be specified')
            else:
                self.r2 = self.r1 - self.L*self.sina
        else:
            self.r1 = self.r2 + self.L*self.sina

        self.thetaTrad = deg2rad(self.thetaTdeg)

        self.tLArad = deg2rad(self.tLAdeg)

        self.betarad = deg2rad(self.betadeg)
        self.LA = self.r2*tan(self.betarad)

        if not self.laminaprops:
            self.laminaprops = [self.laminaprop for i in self.stack]
        if not self.plyts:
            self.plyts = [self.plyt for i in self.stack]

        if self.alpharad == 0:
            self.is_cylinder = True
        else:
            self.is_cylinder = False

        self.excluded_dofs = []
        self.excluded_dofs_ck = []
        if self.pdC:
            self.excluded_dofs.append(0)
            self.excluded_dofs_ck.append(self.uTM)
        if self.pdT:
            self.excluded_dofs.append(1)
            self.excluded_dofs_ck.append(self.thetaTrad)
        if self.pdLA:
            self.excluded_dofs.append(2)
            self.excluded_dofs_ck.append(self.LA)
        else:
            raise NotImplementedError('pdLA == False is giving wrong results!')

        if self.nx < 4*self.m2:
            warn('Number of integration points along x too small')
        if self.nt < 4*self.n2:
            warn('Number of integration points along theta too small')

        if self.laminaprop is None:
            h = self.h
            E11 = self.E11
            nu = self.nu
            if h is None or E11 is None or nu is None:
                raise ValueError(
                        'laminaprop or (E11, nu and h) must be defined')

            G12 = E11/(2*(1 + nu))
            A11 = E11*h/(1 - nu**2)
            A12 = nu*E11*h/(1 - nu**2)
            A16 = 0
            A22 = E11*h/(1 - nu**2)
            A26 = 0
            A66 = G12*h
            D11 = E11*h**3/(12*(1 - nu**2))
            D12 = nu*E11*h**3/(12*(1 - nu**2))
            D16 = 0
            D22 = E11*h**3/(12*(1 - nu**2))
            D26 = 0
            D66 = G12*h**3/12

            # TODO, what if FSDT is used?
            if 'fsdt' in self.model:
                raise NotImplementedError(
                        'For FSDT laminaprop must be defined!')

            self.F = np.array([[A11, A12, A16, 0, 0, 0],
                               [A12, A22, A26, 0, 0, 0],
                               [A16, A26, A66, 0, 0, 0],
                               [0, 0, 0, D11, D12, D16],
                               [0, 0, 0, D12, D22, D26],
                               [0, 0, 0, D16, D26, D66]])

        if self.c0 is not None:
            self.analysis.kT_initial_state = True

        if self.Nxxtop is not None and self._load_rebuilt:
            return

        # axial load
        if self.Nxxtop is not None:
            if type(self.Nxxtop) in (int, float):
                Nxxtop0 = self.Nxxtop
                self.Nxxtop = np.zeros(2*self.n2+1, dtype=DOUBLE)
                self.Nxxtop[0] = Nxxtop0

            check = False
            if isinstance(self.Nxxtop, np.ndarray):
                if self.Nxxtop.ndim == 1:
                    assert self.Nxxtop.shape[0] == (2*self.n2+1)
                    check = True
            if not check:
                raise ValueError('Invalid Nxxtop input')

        else:
            self.Nxxtop = np.zeros(2*self.n2+1, dtype=DOUBLE)

        if self.Fc is not None:
            self.Nxxtop[0] = self.Fc/(2*pi*self.r2*self.cosa)
            msg('Nxxtop[0] calculated from Fc', level=2)
            if self.MLA is None:
                if self.xiLA is not None:
                    self.MLA = self.xiLA*self.Fc
                    msg('MLA calculated from xiLA', level=2)

        if self.MLA is not None:
            self.Nxxtop[2] = self.MLA/(pi*self.r2**2*self.cosa)
            msg('Nxxtop[2] calculated from MLA', level=2)

        self._load_rebuilt = True


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
        model_dict = get_model(self.model)
        num0 = model_dict['num0']
        num1 = model_dict['num1']
        num2 = model_dict['num2']
        self.size = num0 + num1*self.m1 + num2*self.m2*self.n2
        return self.size


    def from_DB(self, name):
        r"""Load cone/cylinder data from the local database

        Parameters
        ----------
        name : str
            A key contained in the ``ccs`` dictionary of module
            :mod:`compmech.conecyl.conecylDB`.

        """
        try:
            attrs = ['r1', 'r2', 'H', 'L', 'alphadeg', 'plyt', 'stack']
            cc = ccs[name]
            self.laminaprop = laminaprops[cc['laminapropKey']]
            for attr in attrs:
                setattr(self, attr, cc.get(attr, getattr(self, attr)))
        except:
            raise ValueError('Invalid data-base entry!')


    def exclude_dofs_matrix(self, k, return_kkk=False,
                                     return_kku=False,
                                     return_kuk=False):
        r"""Makes the partition of the dofs for prescribed displacements

        Makes the following partition of a given matrix::

            k = | kkk    kku |
                | kuk    kuu |

        Parameters
        ----------
        k : scipy.sparse.coo_matrix
            Matrix to be partitioned.
        return_kkk : bool, optional
            If the region `kkk` must be returned.
        return_kku : bool, optional
            If the region `kku` must be returned.
        return_kuk : bool, optional
            If the region `kuk` must be returned.

        Returns
        -------
        out : dict
            A ``dict`` object containing the keys for the
            corresponding sub-matrices ``kkk``, ``kku``, ``kuk``, ``kuu``.
            The sub-matrix ``out['kuu']`` is a ``scipy.sparse.csr_matrix``,
            while the others are 2-D ``np.ndarray`` objects.

        """
        if not isinstance(k, coo_matrix):
            k = coo_matrix(k)

        if return_kkk:
            kkk = coo_matrix(np.zeros((self.num0, self.num0)))
            ind = np.where(((k.row < self.num0) & (k.col < self.num0)))[0]
            kkk.row = np.take(k.row, ind)
            kkk.col = np.take(k.col, ind)
            kkk.data = np.take(k.data, ind)
            kkk = kkk.toarray()
            kkk = np.delete(kkk, self.excluded_dofs, axis=0)
            kkk = np.delete(kkk, self.excluded_dofs, axis=1)

        if return_kku:
            kku = coo_matrix(np.zeros((self.num0, k.shape[0])))
            ind = np.where(k.row < self.num0)[0]
            kku.row = np.take(k.row, ind)
            kku.col = np.take(k.col, ind)
            kku.data = np.take(k.data, ind)
            kku = kku.toarray()
            kku = np.delete(kku, self.excluded_dofs, axis=1)

        if return_kuk:
            kuk = coo_matrix(np.zeros((k.shape[0], self.num0)))
            ind = np.where(k.col < self.num0)[0]
            kuk.row = np.take(k.row, ind)
            kuk.col = np.take(k.col, ind)
            kuk.data = np.take(k.data, ind)
            kuk = kuk.toarray()
            kuk = np.delete(kuk, self.excluded_dofs, axis=0)

        rows = np.sort(self.excluded_dofs)[::-1]
        cols = np.sort(self.excluded_dofs)[::-1]

        kuu = k.copy()

        for r in rows:
            ind = np.where(kuu.row != r)[0]
            kuu.row[kuu.row > r] -= 1
            kuu.row = np.take(kuu.row, ind)
            kuu.col = np.take(kuu.col, ind)
            kuu.data = np.take(kuu.data, ind)
            kuu._shape = (kuu._shape[0]-1, kuu._shape[1])

        for c in cols:
            ind = np.where(kuu.col != c)[0]
            kuu.col[kuu.col > c] -= 1
            kuu.row = np.take(kuu.row, ind)
            kuu.col = np.take(kuu.col, ind)
            kuu.data = np.take(kuu.data, ind)
            kuu._shape = (kuu._shape[0], kuu._shape[1]-1)

        kuu = csr_matrix(kuu)

        out = {}
        out['kuu'] = kuu
        if return_kkk:
            out['kkk'] = kkk
        if return_kku:
            out['kku'] = kku
        if return_kuk:
            out['kuk'] = kuk

        return out


    def calc_full_c(self, cu, inc=1.):
        r"""Returns the full set of Ritz constants

        When prescribed displacements take place the matrices and the Ritz
        constants are partitioned like::

            k = | kkk    kku |
                | kuk    kuu |

        and the corresponding Ritz constants::

            c = | ck |
                | cu |

        This function adds the set of known Ritz constants (``ck``)
        to the set of unknown (``cu``) based on the prescribed displacements.

        Parameters
        ----------
        cu : np.ndarray
            The set of unknown Ritz constants
        inc : float, optional
            Load increment, necessary to calculate the full set of Ritz
            constants.

        Returns
        -------
        c : np.ndarray
            The full set of Ritz constants.

        """
        c = cu.copy()
        size = self.get_size()
        if c.shape[0] == size:
            for dof in self.excluded_dofs:
                c[dof] *= inc
            c = np.ascontiguousarray(c, dtype=DOUBLE)
            return c
        ordered = sorted(zip(self.excluded_dofs,
                             self.excluded_dofs_ck), key=lambda x:x[0])
        for dof, cai in ordered:
            c = np.insert(c, dof, inc*cai)
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        return c


    def _default_field(self, xs, ts, gridx, gridt):
        if xs is None or ts is None:
            xs = linspace(0, self.L, gridx)
            ts = linspace(-pi, pi, gridt)
            xs, ts = np.meshgrid(xs, ts, copy=True)
        xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
        ts = np.atleast_1d(np.array(ts, dtype=DOUBLE))
        xshape = xs.shape
        tshape = ts.shape
        if xshape != tshape:
            raise ValueError('Arrays xs and ts must have the same shape')
        self.Xs = xs
        self.Ts = ts
        xs = np.ascontiguousarray(xs.flatten(), dtype=DOUBLE)
        ts = np.ascontiguousarray(ts.flatten(), dtype=DOUBLE)

        return xs, ts, xshape, tshape


    def _calc_linear_matrices(self, combined_load_case=None, silent=False):
        self._rebuild()
        msg('Calculating linear matrices... ', level=2)

        fk0, fk0_cyl, fkG0, fkG0_cyl, k0edges = modelDB.get_linear_matrices(
                                                    self, combined_load_case)
        model = self.model
        alpharad = self.alpharad
        cosa = self.cosa
        r2 = self.r2
        L = self.L
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        s = self.s
        laminaprops = self.laminaprops
        plyts = self.plyts
        stack = self.stack
        P = self.P
        T = self.T
        E11 = self.E11
        nu = self.nu
        h = self.h

        Fc = self.Nxxtop[0]*(2*pi*r2*cosa)

        lam = self.lam

        if stack != [] and self.F_reuse is None:
            lam = laminate.read_stack(stack, plyts=plyts,
                                             laminaprops=laminaprops)

        if 'clpt' in model:
            if self.F_reuse is not None:
                msg('', silent=silent)
                msg('Reusing F matrix...', level=2, silent=silent)
                F = self.F_reuse
            elif lam is not None:
                F = lam.ABD
            else:
                F = self.F

        elif 'fsdt' in model:
            if self.F_reuse is not None:
                msg('', silent=silent)
                msg('Reusing F matrix...', level=2, silent=silent)
                F = self.F_reuse
            elif lam is not None:
                F = lam.ABDE
                F[6:, 6:] *= self.K
            else:
                F = self.F

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

        if self.is_cylinder:
            if 'iso_' in model:
                k0 = fk0_cyl(r2, L, E11, nu, h, m1, m2, n2)
            else:
                k0 = fk0_cyl(r2, L, F, m1, m2, n2)

            if not combined_load_case:
                kG0 = fkG0_cyl(Fc, P, T, r2, L, m1, m2, n2)
            else:
                kG0_Fc = fkG0_cyl(Fc, 0, 0, r2, L, m1, m2, n2)
                kG0_P = fkG0_cyl(0, P, 0, r2, L, m1, m2, n2)
                kG0_T = fkG0_cyl(0, 0, T, r2, L, m1, m2, n2)
        else:
            if 'iso_' in model:
                k0 = fk0(alpharad, r2, L, E11, nu, h, m1, m2, n2, s)
            else:
                k0 = fk0(alpharad, r2, L, F, m1, m2, n2, s)
            if not combined_load_case:
                kG0 = fkG0(Fc, P, T, r2, alpharad, L, m1, m2, n2, s)
            else:
                kG0_Fc = fkG0(Fc, 0, 0, r2, alpharad, L, m1, m2, n2, s)
                kG0_P = fkG0(0, P, 0, r2, alpharad, L, m1, m2, n2, s)
                kG0_T = fkG0(0, 0, T, r2, alpharad, L, m1, m2, n2, s)

        if k0edges is not None:
            k0 = csr_matrix(k0) + csr_matrix(k0edges)

        assert np.any((np.isnan(k0.data) | np.isinf(k0.data))) == False

        k0 = make_symmetric(k0)

        if not combined_load_case:
            assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
            self.kG0 = make_symmetric(kG0)
        else:
            assert np.any((np.isnan(kG0_Fc.data)
                           | np.isinf(kG0_Fc.data))) == False
            assert np.any((np.isnan(kG0_P.data)
                           | np.isinf(kG0_P.data))) == False
            assert np.any((np.isnan(kG0_T.data)
                           | np.isinf(kG0_T.data))) == False
            self.kG0_Fc = make_symmetric(kG0_Fc)
            self.kG0_P = make_symmetric(kG0_P)
            self.kG0_T = make_symmetric(kG0_T)

        k = self.exclude_dofs_matrix(k0, return_kuk=True)
        k0uk = k['kuk']
        k0uu = k['kuu']

        self.k0 = k0
        self.k0uk = k0uk
        self.k0uu = k0uu

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process

        gc.collect()

        msg('finished!', level=2)


    def calc_k0(self, silent=False):
        if self.k0uu is None:
            self._calc_linear_matrices(silent=silent)
        return self.k0uu


    def calc_kT(self, c, inc=1., silent=False):
        r"""Calculates the tangent stiffness matrix

        The following attributes will affect the numerical integration:

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
        ``nt``               ``int``, number of integration points along the
                             `\theta` coordinate
        =================    ================================================

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector of the current state.
        inc : float, optional
            Load increment, necessary to calculate the full set of Ritz
            constants using :meth:`calc_full_c`.
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        kTuu : sparse matrix
            The tangent stiffness matrix corresponding to the unknown degrees
            of freedom.

        """
        self._calc_NL_matrices(c, inc=inc, silent=silent)
        return self.kTuu


    def lb(self, c=None, tol=0, combined_load_case=None):
        r"""Performs a linear buckling analysis

        The following parameters of the ``ConeCyl`` object will affect the
        linear buckling analysis:

        =======================    =====================================
        Attribute                  Description
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

            - ``1`` : find the critical axial load for a fixed torsion load
            - ``2`` : find the critical axial load for a fixed pressure load
            - ``3`` : find the critical torsion load for a fixed axial load

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``ConeCyl`` object and the `i^{th}` eigenvector in the
        ``eigvecs[i-1, :]`` parameter.

        """
        model_dict = get_model(self.model)
        if not model_dict['linear buckling']:
            msg('________________________________________________')
            msg('')
            warn('Model {} cannot be used in linear buckling analysis!'.
                 format(self.model))
            msg('________________________________________________')

        msg('Running linear buckling analysis...')

        if self.Fc is None and self.Nxxtop is None:
            warn('using Fc = 1.', level=1)
            self.Fc = 1.
        if self.pdC is None:
            self.pdC = False

        self._calc_linear_matrices(combined_load_case=combined_load_case)

        #TODO maybe a better estimator to sigma would be to run
        #     a preliminary eigsh using a small m2 and n2
        #NOTE runs faster for self.k0 than -self.k0, so that the negative
        #     sign is applied later

        msg('Eigenvalue solver... ', level=2)

        model_dict = get_model(self.model)
        num0 = model_dict['num0']

        pos = num0

        if not combined_load_case:
            M = csr_matrix(self.k0)
            A = csr_matrix(self.kG0)
        elif combined_load_case == 1:
            M = csr_matrix(self.k0) + csr_matrix(self.kG0_T)
            A = csr_matrix(self.kG0_Fc)
        elif combined_load_case == 2:
            M = csr_matrix(self.k0) + csr_matrix(self.kG0_P)
            A = csr_matrix(self.kG0_Fc)
        elif combined_load_case == 3:
            M = csr_matrix(self.k0) + csr_matrix(self.kG0_Fc)
            A = csr_matrix(self.kG0_T)

        A = A[pos:, pos:]
        M = M[pos:, pos:]

        try:
            eigvals, eigvecs = eigsh(A=A, k=self.num_eigvalues, which='SM',
                                     M=M, tol=tol, sigma=1.,
                                     mode='cayley')
        except Exception as e:
            warn(str(e), level=3)
            size22 = M.shape[0]
            M, A, used_cols = remove_null_cols(M, A)
            msg('solver...', level=3)
            try:
                eigvals, peigvecs = eigsh(A=A, k=self.num_eigvalues,
                                          which='SM', M=M, tol=tol, sigma=1.,
                                          mode='cayley')
            except:
                eigvals, peigvecs = eigsh(A=A, k=self.num_eigvalues,
                                          which='SM', M=M, tol=tol, sigma=1.,
                                          mode='buckling')
            msg('finished!', level=3)
            eigvecs = np.zeros((size22, self.num_eigvalues), dtype=DOUBLE)
            eigvecs[used_cols, :] = peigvecs

        eigvals = -1./eigvals
        self.eigvals = eigvals
        self.eigvecs = np.vstack((np.zeros((pos, self.num_eigvalues)),
                                  eigvecs))

        msg('finished!', level=2)

        msg('first {} eigenvalues:'.format(self.num_eigvalues_print), level=1)
        for eig in eigvals[:self.num_eigvalues_print]:
            msg('{}'.format(eig), level=2)
        self.analysis.last_analysis = 'lb'


    def eigen(self, c=None, tol=0, kL=None, kG=None, combined_load_case=None):
        r"""Performs a non-linear eigenvalue analysis at a given state

        The following attributes of the ``ConeCyl`` object will affect the
        non-linear eigenvalue analysis:

        =======================    =====================================
        Attribute                  Description
        =======================    =====================================
        ``num_eigenvalues``        Number of eigenvalues to be extracted
        ``num_eigvalues_print``    Number of eigenvalues to print after
                                   the analysis is completed
        =======================    =====================================

        Additionally, the non-linear analysis parameters described in
        :meth:`static` will affect the integration of the non-linear matrices
        ``kL`` and ``kG`` if they are not given as input parameters.

        Parameters
        ----------
        combined_load_case : int or None, optional
            It tells whether the linear buckling analysis must be computed
            considering combined load cases, each value will tell
            the algorithm to rearrange the linear matrices in a different
            way. The valid values are ``1``, or ``2``, where:

            - ``1`` : find the critical axial load for a fixed torsion load
            - ``2`` : find the critical axial load for a fixed pressure load
            - ``3`` : find the critical torsion load for a fixed axial load

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``ConeCyl`` object and the `i^{th}` eigenvector in the
        ``eigvecs[i-1, :]`` parameter.

        """
        model_dict = get_model(self.model)
        if not model_dict['linear buckling']:
            msg('________________________________________________')
            msg('')
            warn('Model {} cannot be used in linear buckling analysis!'.
                 format(self.model))
            msg('________________________________________________')

        msg('Running linear buckling analysis...')

        if self.Fc is None:
            self.Fc = 1.
        if self.pdC is None:
            self.pdC = False

        self._calc_linear_matrices(combined_load_case=combined_load_case)

        #TODO maybe a better estimator to sigma would be to run
        #     a preliminary eigsh using a small m2 and n2
        #NOTE runs faster for self.k0 than -self.k0, so that the negative
        #     sign is applied later
        msg('Eigenvalue solver... ', level=2)

        model_dict = get_model(self.model)
        num0 = model_dict['num0']

        pos = num0

        if combined_load_case is None:
            M = csr_matrix(self.k0)
            A = csr_matrix(self.kG0)
        elif combined_load_case == 1:
            M = csr_matrix(self.k0) + csr_matrix(self.kG0_T)
            A = csr_matrix(self.kG0_Fc)
        elif combined_load_case == 2:
            M = csr_matrix(self.k0) + csr_matrix(self.kG0_P)
            A = csr_matrix(self.kG0_Fc)
        elif combined_load_case == 3:
            M = csr_matrix(self.k0) + csr_matrix(self.kG0_Fc)
            A = csr_matrix(self.kG0_T)
        else:
            raise ValueError('Invalid value for the "combined_load_case" parameter')

        A = A[pos:, pos:]
        M = M[pos:, pos:]

        try:
            eigvals, eigvecs = eigsh(A=A, k=self.num_eigvalues, which='SM',
                                     M=M, tol=tol, sigma=1.,
                                     mode='cayley')
        except Exception as e:
            warn(str(e), level=3)
            size22 = M.shape[0]
            M, A, used_cols = remove_null_cols(M, A)
            msg('solver...', level=3)
            try:
                eigvals, peigvecs = eigsh(A=A, k=self.num_eigvalues,
                                          which='SM', M=M, tol=tol, sigma=1.,
                                          mode='cayley')
            except:
                eigvals, peigvecs = eigsh(A=A, k=self.num_eigvalues,
                                          which='SM', M=M, tol=tol, sigma=1.,
                                          mode='buckling')
            msg('finished!', level=3)
            eigvecs = np.zeros((size22, self.num_eigvalues), dtype=DOUBLE)
            eigvecs[used_cols, :] = peigvecs

        eigvals = (-1./eigvals)
        self.eigvals = eigvals
        self.eigvecs = np.vstack((np.zeros((pos, self.num_eigvalues)),
                                  eigvecs))

        msg('finished!', level=2)

        msg('first {} eigenvalues:'.format(self.num_eigvalues_print), level=1)
        for eig in eigvals[:self.num_eigvalues_print]:
            msg('{}'.format(eig), level=2)
        self.analysis.last_analysis = 'lb'


    def _calc_NL_matrices(self, c, inc=1., with_kLL=None, with_k0L=None, silent=False):
        r"""Calculates the non-linear stiffness matrices

        Parameters
        ----------
        c : np.ndarray
            Ritz constants representing the current state to calculate the
            stiffness matrices.
        inc : float, optional
            Load increment, necessary to calculate the full set of Ritz
            constants using :meth:`calc_full_c`.
        with_kLL : bool, optional
            When ``with_kLL=False`` assumes kLL << than k0L and kG.
        with_k0L : bool, optional
            When ``with_k0L=False`` assumes k0L << than kLL and kG.
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Notes
        -----
        Nothing is returned, the calculated matrices

        """
        c = self.calc_full_c(c, inc=inc)

        if self.k0 is None:
            self._calc_linear_matrices(silent=silent)
        if with_k0L is None:
            with_k0L = self.with_k0L
        if with_kLL is None:
            with_kLL = self.with_kLL

        msg('Calculating non-linear matrices...', level=2, silent=silent)
        alpharad = self.alpharad
        r2 = self.r2
        L = self.L
        tLArad = self.tLArad
        F = self.F
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0

        model = self.model
        model_dict = get_model(model)

        nlmodule = model_dict['non-linear']
        ni_method = self.ni_method
        num_cores = self.ni_num_cores
        nx = self.nx
        nt = self.nt

        if nlmodule:
            calc_k0L = nlmodule.calc_k0L
            calc_kLL = nlmodule.calc_kLL
            if 'iso_' in model:
                calc_kG = modelDB.db[model[4:]]['non-linear'].calc_kG
            else:
                calc_kG = nlmodule.calc_kG

            kG = calc_kG(c, alpharad, r2, L, tLArad, F, m1, m2, n2, nx=nx,
                    nt=nt, num_cores=num_cores, method=ni_method, c0=c0,
                    m0=m0, n0=n0)
            kG = make_symmetric(kG)

            if 'iso_' in model:
                E11 = self.E11
                nu = self.nu
                h = self.h
                if with_k0L:
                    k0L = calc_k0L(c, alpharad, r2, L, tLArad, E11, nu, h, m1,
                            m2, n2, nx=nx, nt=nt, num_cores=num_cores,
                            method=ni_method, c0=c0, m0=m0, n0=n0)
                else:
                    k0L = kG*0
                if with_kLL:
                    kLL = calc_kLL(c, alpharad, r2, L, tLArad, E11, nu, h, m1,
                            m2, n2, nx=nx, nt=nt, num_cores=num_cores,
                            method=ni_method, c0=c0, m0=m0, n0=n0)
                    kLL = make_symmetric(kLL)

                else:
                    kLL = kG*0

            else:
                if with_k0L:
                    k0L = calc_k0L(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                            nx=nx, nt=nt, num_cores=num_cores,
                            method=ni_method, c0=c0, m0=m0, n0=n0)
                else:
                    k0L = kG*0
                if with_kLL:
                    kLL = calc_kLL(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                            nx=nx, nt=nt, num_cores=num_cores,
                            method=ni_method, c0=c0, m0=m0, n0=n0)
                    kLL = make_symmetric(kLL)

                else:
                    kLL = kG*0

        else:
            raise ValueError(
            'Non-Linear analysis not implemented for model {0}'.format(model))

        kL0 = k0L.T

        #TODO maybe slow...
        kT = coo_matrix(self.k0 + k0L + kL0 + kLL + kG)
        # kS was deprecated, now fint is integrated numerically
        #kS = coo_matrix(self.k0 + k0L/2 + kL0 + kLL/2)

        k = self.exclude_dofs_matrix(kT, return_kuk=True)
        self.kTuk = k['kuk']
        self.kTuu = k['kuu']

        #NOTE intended for non-linear eigenvalue analyses
        self.kL = csr_matrix(self.k0 + k0L + kL0 + kLL)
        self.kG = csr_matrix(kG)

        msg('finished!', level=2, silent=silent)


    def uvw(self, c, xs=None, ts=None, gridx=300, gridt=300, inc=1.):
        r"""Calculates the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phit`` of the ``ConeCyl`` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        xs : np.ndarray
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and method ``_default_field`` is used.
        ts : np.ndarray
            The ``theta`` positions where to calculate the displacement field.
            Default is ``None`` and method ``_default_field`` is used.
        gridx : int
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridt : int
            Number of points along the `theta` where to calculate the
            displacement field.
        inc : float, optional
            Load increment, necessary to calculate the full set of Ritz
            constants using :meth:`calc_full_c`.

        Returns
        -------
        out : tuple
            A tuple of ``np.ndarrays`` containing
            ``(u, v, w, phix, phit)``.

        Notes
        -----
        The returned values ``u```, ``v``, ``w``, ``phix``, ``phit`` are
        stored as parameters with the same name in the ``ConeCyl`` object.

        """
        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)
        alpharad = self.alpharad
        tLArad = self.tLArad
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        r2 = self.r2
        L = self.L

        c = self.calc_full_c(c, inc=inc)

        model_dict = get_model(self.model)
        fuvw = model_dict['commons'].fuvw
        us, vs, ws, phixs, phits = fuvw(c, m1, m2, n2, alpharad, r2, L,
                                        tLArad, xs, ts, self.out_num_cores)

        self.u = us.reshape(xshape)
        self.v = vs.reshape(xshape)
        self.w = ws.reshape(xshape)
        self.phix = phixs.reshape(xshape)
        self.phit = phits.reshape(xshape)

        return self.u, self.v, self.w, self.phix, self.phit


    def strain(self, c, xs=None, ts=None, gridx=300, gridt=300, inc=1.):
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
        inc : float, optional
            Load increment, necessary to calculate the full set of Ritz
            constants using :meth:`calc_full_c`.

        """
        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)

        L = self.L
        r2 = self.r2
        sina = self.sina
        cosa = self.cosa
        tLArad = self.tLArad
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0
        funcnum = self.funcnum
        model = self.model
        model_dict = get_model(model)
        NL_kinematics = model.split('_')[1]
        fstrain = model_dict['commons'].fstrain
        e_num = model_dict['e_num']

        if 'donnell' in NL_kinematics:
            int_NL_kinematics = 0
        elif 'sanders' in NL_kinematics:
            int_NL_kinematics = 1
        else:
            raise NotImplementedError(
                '{} is not a valid "NL_kinematics" option'.format(
                NL_kinematics))

        c = self.calc_full_c(c, inc=inc)

        es = fstrain(c, sina, cosa, tLArad, xs, ts, r2, L,
                     m1, m2, n2, c0, m0, n0, funcnum, int_NL_kinematics,
                     self.out_num_cores)

        return es.reshape((xshape + (e_num,)))


    def stress(self, c, xs=None, ts=None, gridx=300, gridt=300, inc=1.):
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
        inc : float, optional
            Load increment, necessary to calculate the full set of Ritz
            constants using :meth:`calc_full_c`.

        """
        xs, ts, xshape, tshape = self._default_field(xs, ts, gridx, gridt)

        F = self.F
        L = self.L
        r2 = self.r2
        sina = self.sina
        cosa = self.cosa
        tLArad = self.tLArad
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        c0 = self.c0
        m0 = self.m0
        n0 = self.n0
        funcnum = self.funcnum
        model = self.model
        model_dict = get_model(model)
        NL_kinematics = model.split('_')[1]
        fstress = model_dict['commons'].fstress
        e_num = model_dict['e_num']

        if 'donnell' in NL_kinematics:
            int_NL_kinematics = 0
        elif 'sanders' in NL_kinematics:
            int_NL_kinematics = 1
        else:
            raise NotImplementedError(
                    '{} is not a valid "NL_kinematics" option'.format(
                    NL_kinematics))

        c = self.calc_full_c(c, inc=inc)

        Ns = fstress(c, F, sina, cosa, tLArad, xs, ts, r2, L,
                     m1, m2, n2, c0, m0, n0, funcnum, int_NL_kinematics,
                     self.out_num_cores)
        return Ns.reshape((xshape + (e_num,)))


    def calc_fint(self, c, inc=1., m=1, return_u=True, silent=False):
        r"""Calculates the internal force vector `\{F_{int}\}`

        The following attributes will affect the numerical integration:

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
        ``nt``               ``int``, number of integration points along the
                             `\theta` coordinate
        =================    ================================================

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the internal
            forces.
        inc : float, optional
            Load increment, necessary to calculate the full set of Ritz
            constants using :meth:`calc_full_c`.
        m : integer, optional
            A multiplier to be applied to ``nx`` and ``nt``, if one
            whishes to use more integration points.
        return_u : bool, optional
            If the internal force vector corresponsing to the unknown
            set of Ritz constants should be returned.
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        fint : np.ndarray
            The internal force vector.

        """
        c = self.calc_full_c(c, inc=inc)
        if 'iso_' in self.model:
            nlmodule = modelDB.db[self.model[4:]]['non-linear']
        else:
            nlmodule = modelDB.db[self.model]['non-linear']
        ni_method = self.ni_method
        ni_num_cores = self.ni_num_cores
        nx = self.nx*m
        nt = self.nt*m
        fint = nlmodule.calc_fint_0L_L0_LL(c, self.alpharad, self.r2, self.L,
                self.tLArad, self.F, self.m1, self.m2, self.n2, nx, nt,
                ni_num_cores, ni_method, self.c0, self.m0, self.n0)
        fint += self.k0*c
        if return_u:
            fint = np.delete(fint, self.excluded_dofs)

        return fint


    def add_SPL(self, PL, pt=0.5, thetadeg=0., increment=False):
        r"""Add a Single Perturbation Load `\{{F_{PL}}_i\}`

        Adds a perturbation load to the ``ConeCyl`` object, the perturbation
        load is a particular case of the punctual load with only a normal
        component.

        Parameters
        ----------
        PL : float
            The perturbation load value.
        pt : float, optional
            The normalized position along the `x` axis in which the new SPL
            will be included.
        thetadeg : float, optional
            The angular position of the SPL in degrees.
        increment : bool, optional
            If this perturbation load should be incrementally applied in a
            non-linear analysis.

        Notes
        -----
        Each single perturbation load is added to the ``forces`` parameter
        of the ``ConeCyl`` object, which may be changed by the analyst at
        any time.

        """
        self._rebuild()
        thetarad = deg2rad(thetadeg)
        if increment:
            self.forces_inc.append([pt*self.L, thetarad, 0., 0., -PL])
        else:
            self.forces.append([pt*self.L, thetarad, 0., 0., -PL])


    def add_force(self, x, thetadeg, fx, ftheta, fz, increment=False):
        r"""Add a punctual force

        Adds a force vector `\{f_x, f_\theta, f_z\}^T` to the ``forces``
        parameter of the ``ConeCyl`` object.

        Parameters
        ----------
        x : float
            The `x` position.
        thetadeg : float
            The `\theta` position in degrees.
        fx : float
            The `x` component of the force vector.
        ftheta : float
            The `\theta` component of the force vector.
        fz : float
            The `z` component of the force vector.
        increment : bool, optional
            If this punctual force should be incrementally applied in a
            non-linear analysis.

        """
        thetarad = deg2rad(thetadeg)
        if increment:
            self.forces_inc.append([x, thetarad, fx, ftheta, fz])
        else:
            self.forces.append([x, thetarad, fx, ftheta, fz])


    def calc_fext(self, inc=1., kuk=None, silent=False):
        r"""Calculates the external force vector `\{F_{ext}\}`

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

        kuk : np.ndarray, optional
            Obsolete, created for displacement controlled analyses, but the
            implementation has not been finished, see
            :meth:`exclude_dofs_matrix`.
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        fext : np.ndarray
            The external force vector

        """
        self._rebuild()
        if self.k0 is None:
            self._calc_linear_matrices(silent=silent)

        msg('Calculating external forces...', level=2, silent=silent)


        uTM = inc*self.uTM
        Nxxtop = inc*self.Nxxtop
        thetaTrad = inc*self.thetaTrad

        sina = self.sina
        cosa = self.cosa
        r2 = self.r2
        L = self.L
        tLArad = self.tLArad
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        pdT = self.pdT
        model = self.model

        model_dict = get_model(model)

        i0 = model_dict['i0']
        j0 = model_dict['j0']
        num0 = model_dict['num0']
        num1 = model_dict['num1']
        num2 = model_dict['num2']
        dofs = model_dict['dofs']
        fg = model_dict['commons'].fg

        size = self.get_size()

        g = np.zeros((dofs, size), dtype=DOUBLE)
        fext = np.zeros(size, dtype=DOUBLE)

        fext = np.delete(fext, self.excluded_dofs)

        # constant punctual forces
        for i, force in enumerate(self.forces):
            x, theta, fx, ftheta, fz = force
            fg(g, m1, m2, n2, r2, x, theta, L, cosa, tLArad)

            gu = np.delete(g, self.excluded_dofs, axis=1)

            if dofs == 3:
                fpt = np.array([[fx, ftheta, fz]])
            elif dofs == 5:
                fpt = np.array([[fx, ftheta, fz, 0, 0]])
            fext += fpt.dot(gu).ravel()

        # incremented punctual forces
        for i, force in enumerate(self.forces_inc):
            x, theta, fx, ftheta, fz = force
            fg(g, m1, m2, n2, r2, x, theta, L, cosa, tLArad)

            gu = np.delete(g, self.excluded_dofs, axis=1)

            if dofs == 3:
                fpt = inc*np.array([[fx, ftheta, fz]])
            elif dofs == 5:
                fpt = inc*np.array([[fx, ftheta, fz, 0, 0]])
            fext += fpt.dot(gu).ravel()

        # axial load
        fext_tmp = np.zeros(size, dtype=DOUBLE)
        if not 0 in self.excluded_dofs:
            fext_tmp[0] += Nxxtop[0]*(2*pi*r2)/cosa
            if 'bc2' in model or 'bc4' in model:
                for j2 in range(j0, n2+j0):
                    for i2 in range(i0, m2+i0):
                        row = (num0 + num1*m1
                               + (i2-i0)*num2 + (j2-j0)*num2*m2)
                        rowNxx = 1+2*(j2-j0)
                        fext_tmp[row+0]+=(Nxxtop[rowNxx+0]*pi*r2)
                        fext_tmp[row+1]+=(Nxxtop[rowNxx+1]*pi*r2)
        else:
            if kuk is None:
                kuk_C = self.k0uk[:, 0].ravel()
            else:
                kuk_C = kuk[:, 0].ravel()
            fext += -uTM*kuk_C

        if not 2 in self.excluded_dofs:
            fext_tmp[2] += Nxxtop[2]*(2*pi*r2)/cosa

        # pressure
        P = self.P + inc*self.P_inc
        if P != 0:
            if 'clpt' in model:
                for i1 in range(i0, m1+i0):
                    if i1 == 0:
                        continue
                    col = num0 + (i1-i0)*num1
                    fext_tmp[col+2] += P*(L*2./i1*(r2 - (-1)**i1*(r2 + L*sina)))
            elif 'fsdt' in model:
                #TODO it might be the same as for the CLPT
                raise NotImplementedError(
                    'Pressure not implemented for static analysis for FSDT')

        fext_tmp = np.delete(fext_tmp, self.excluded_dofs)
        fext += fext_tmp

        # torsion
        if pdT:
            if kuk is None:
                kuk_T = self.k0uk[:, 1].ravel()
            else:
                kuk_T = kuk[:, 1].ravel()
            fext += -thetaTrad*kuk_T
        else:
            T = self.T + inc*self.T_inc
            if T != 0:
                fg(g, m1, m2, n2, r2, 0, 0, L, cosa, tLArad)
                gu = np.delete(g, self.excluded_dofs, axis=1)
                if dofs == 3:
                    fpt = np.array([[0, T/r2, 0]])
                elif dofs == 5:
                    fpt = np.array([[0, T/r2, 0, 0, 0]])
                fext += fpt.dot(gu).ravel()

        msg('finished!', level=2, silent=silent)

        return fext


    def static(self, NLgeom=False, silent=False):
        r"""Static analysis for cones and cylinders

        The analysis can be linear or geometrically non-linear. See
        :class:`.Analysis` for further details about the parameters
        controlling the non-linear analysis.

        Parameters
        ----------
        NLgeom : bool
            Flag to indicate whether a linear or a non-linear analysis is to
            be performed.

        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        cs : list
            A list containing the Ritz constants for each load increment of
            the static analysis. The list will have only one entry in case
            of a linear analysis.

        Notes
        -----
        The returned ``cs`` is stored in the ``cs`` parameter of the
        ``ConeCyl`` object. The actual increments used in the non-linear
        analysis are stored in the ``increments`` parameter.

        """
        self.cs = []
        self.increments = []

        if self.pdC:
            text ='Non-linear analysis with prescribed displacements'
            raise NotImplementedError(text)

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
             num_levels=400, inc=1., vecmin=None, vecmax=None):
        r"""Contour plot for a Ritz constants vector.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phit'``,
                            ``'magnitude'``
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
            - ``4``: stretched opened (`r \times \theta` vs. `H`)
            - ``5``: stretched opened (`\theta` vs. `H`)

        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
        dpi : int, optional
            Resolution of the saved file in dots per inch.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            the `name` parameter of the ``ConeCyl`` object will be used.
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
            Default is ``None`` and method ``_default_field`` is used.
        ts : np.ndarray, optional
            The ``theta`` positions where to calculate the displacement field.
            Default is ``None`` and method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridt : int, optional
            Number of points along the `theta` where to calculate the
            displacement field.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
        inc : float, optional
            Load increment, necessary to calculate the full set of Ritz
            constants using :meth:`calc_full_c`.
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

        import matplotlib
        import matplotlib.pyplot as plt

        from . plotutils import get_filename

        c = self.calc_full_c(c, inc=inc)

        msg('Computing field variables...', level=1)
        displs = ['u', 'v', 'w', 'phix', 'phit', 'magnitude', 'test']
        strains = ['exx', 'ett', 'gxt', 'kxx', 'ktt', 'kxt', 'gtz', 'gxz']
        stresses = ['Nxx', 'Ntt', 'Nxt', 'Mxx', 'Mtt', 'Mxt', 'Qt', 'Qx']
        if vec in displs or 'eq_' in vec:
            self.uvw(c, xs=xs, ts=ts, gridx=gridx, gridt=gridt, inc=inc)
            if vec == 'magnitude':
                u = self.u
                v = self.v
                w = self.w
                field = (u**2 + v**2 + w**2)**0.5
            elif 'eq_' in vec:
                u = self.u
                v = self.v
                w = self.w
                field = eval(vec[3:])
            else:
                field = getattr(self, vec)
        elif vec in strains:
            es = self.strain(c, xs=xs, ts=ts,
                             gridx=gridx, gridt=gridt, inc=inc)
            field = es[..., strains.index(vec)]
        elif vec in stresses:
            Ns = self.stress(c, xs=xs, ts=ts,
                             gridx=gridx, gridt=gridt, inc=inc)
            field = Ns[..., stresses.index(vec)]
        else:
            raise ValueError(
                    '{0} is not a valid "vec" parameter value!'.format(vec))
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
                raise ValueError('"ax" must be an Axes object')

        def r(x):
            return self.r2 + x*self.sina

        if self.is_cylinder:
            plot_type=4
        if plot_type == 1:
            r_plot = self.r2/self.sina + Xs
            r_plot_max = self.r2/self.sina + self.L
            y = r_plot_max - r_plot*cos(Ts*self.sina)
            x = r_plot*sin(Ts*self.sina)
        elif plot_type == 2:
            r_plot = self.r2/self.sina + Xs
            y = r_plot*cos(Ts*self.sina)
            x = r_plot*sin(Ts*self.sina)
        elif plot_type == 3:
            r_plot = self.r2/self.sina + Xs
            r_plot_max = self.r2/self.sina + self.L
            y = r_plot_max - r_plot*cos(Ts)
            x = r_plot*sin(Ts)
        elif plot_type == 4:
            x = r(Xs)*Ts
            y = self.L-Xs
        elif plot_type == 5:
            x = Ts
            y = Xs
        if deform_u:
            if vec in displs:
                pass
            else:
                self.uvw(c, xs=xs, ts=ts, gridx=gridx, gridt=gridt, inc=inc)
            field_u = self.u
            y -= deform_u_sf*field_u
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

        if invert_x:
            ax.invert_yaxis()

        if title!='':
            ax.set_title(str(title))

        elif add_title:
            if self.analysis.last_analysis == 'static':
                ax.set_title(r'$m_1={0}$, $m_2={1}$, $n_2={2}$'.
                             format(self.m1, self.m2, self.n2))

            elif self.analysis.last_analysis == 'lb':
                ax.set_title(
           r'$m_1={0}$, $m2={1}$, $n2={2}$, $\lambda_{{CR}}={3:1.3e}$'.format(
                self.m1, self.m2, self.n2, self.eigvals[0]))

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
                filename = get_filename(self)
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


    def plotAbaqus(self, frame, fieldOutputKey, vec, nodes, numel_cir,
                   elem_type='S4R', ignore=[],
                   ax=None, figsize=(3.3, 3.3), save=True,
                   aspect='equal', clean=True, plot_type=1,
                   outpath='', filename='', npzname='', pyname='',
                   num_levels=400):
        r"""Print a field output for a cylinder/cone model from Abaqus

        This function is intended to be used with models created by the
        `DESICOS plug-in for Abaqus <http://desicos.github.io/desicos/>`_,
        where a **mapped mesh** is used and the models comparable to the
        models of :mod:`compmech.conecyl`.

        The ``frame`` and ``nodes`` input types are described in
        `Abaqus Scripting Reference Manual
        <http://abaqus.me.chalmers.se/v6.11/books/ker/default.htm>`_ and
        they can be obtained through:

        >>> frame = session.odbs['odb_name.odb'].steps['step_name'].frames[0]
        >>> nodes = mdb.models['model_name'].parts['part_name'].nodes

        Parameters
        ----------
        frame : OdbFrame
            The frame from where the field output will be taken from.
        fieldOutputKey : str
            The field output key to be used. It must be available in
            ``frame.fieldOutputs.keys()``. This function was tested with
            ``'UT'`` and ``'U'`` only.
        vec : str
            The displacement vector to be plotted:
            ``'u'``, ``'v'`` or ``'w'``.
        nodes : MeshNodeArray
            The part nodes.
        numel_cir : int
            The number of elements around the circumference.
        elem_type : str, optional
            The element type. The elements ``'S4R', 'S4R5'`` where tested.
        ignore : list, optional
            A list with the node ids to be ignored. It must contain any nodes
            outside the mapped mesh included in ``parts['part_name'].nodes``.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it.
        figsize : tuple, optional
            The figure size given by ``(width, height)``.
        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method.
        clean : bool, optional
            Clean axes ticks, grids, spines etc.
        plot_type : int, optional
            See :meth:`plot`.
        outpath : str, optional
            Output path where the data from Abaqus and the plots are
            saved (see notes).
        filename : str, optional
            The file name for the generated image file.
        npzname : str, optional
            The file name for the generated npz file.
        pyname : str, optional
            The file name for the generated Python file.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).

        Returns
        -------
        out : tuple
            Where ``out[0]`` and ``out[1]`` contain the circumferential and
            meridional grids of coordinates and ``out[2]`` the corresponding
            field output.

        Notes
        -----
        The data is saved using ``np.savez()`` into ``outpath`` as
        ``abaqus_output.npz`` with an accompanying script for plotting
        ``abaqus_output_plot.py``, very handy when Matplotlib is not
        importable from Abaqus.

        """
        workingplt = True
        if not npzname:
            npzname = 'abaqus_output.npz'
        npzname = os.path.join(outpath, npzname)
        if not pyname:
            pyname = 'abaqus_output_plot.py'
        pyname = os.path.join(outpath, pyname)
        if not filename:
            filename = 'plot_from_abaqus.png'
        filename = os.path.join(outpath, filename)
        try:
            import matplotlib.pyplot as plt
            import matplotlib
        except:
            workingplt = False
        try:
            if not frame:
                frame = utils.get_current_frame()
            if not frame:
                raise ValueError('A frame must be selected!')
            coords = np.array([n.coordinates for n in nodes
                               if n.label not in ignore])
            #TODO include more outputs like stress etc
            field = frame.fieldOutputs[fieldOutputKey]

            uvw_rec = np.array([val.data for val in field.values
                if getattr(val.instance, 'name', None) == 'INSTANCECYLINDER'])
            u_rec = uvw_rec[:,0]
            v_rec = uvw_rec[:,1]
            w_rec = uvw_rec[:,2]

            thetas = np.arctan2(coords[:, 1], coords[:, 0])

            sina = sin(self.alpharad)
            cosa = cos(self.alpharad)

            ucyl = -w_rec
            vcyl = v_rec*cos(thetas) - u_rec*sin(thetas)
            wcyl = v_rec*sin(thetas) + u_rec*cos(thetas)
            u = wcyl*sina + ucyl*cosa
            v = vcyl
            w = wcyl*cosa - ucyl*sina

            displ_vecs = {'u':u, 'v':v, 'w':w}
            uvw = displ_vecs[vec]

            zs = coords[:, 2]

            nt = numel_cir
            if 'S8' in elem_type:
                raise NotImplementedError('{0} elements!'.format(elem_type))
                #nt *= 2

            #first sort
            asort = zs.argsort()
            zs = zs[asort].reshape(-1, nt)
            thetas = thetas[asort].reshape(-1, nt)
            uvw = uvw[asort].reshape(-1, nt)

            #second sort
            asort = thetas.argsort(axis=1)
            for i, asorti in enumerate(asort):
                zs[i,:] = zs[i,:][asorti]
                thetas[i,:] = thetas[i,:][asorti]
                uvw[i,:] = uvw[i,:][asorti]

            H = self.H
            r2 = self.r2
            r1 = self.r1
            L = H/cosa

            def fr(z):
                return r1 - z*sina/cosa

            if self.alpharad == 0.:
                plot_type=4
            if plot_type == 1:
                r_plot = fr(zs)
                if self.alpharad == 0.:
                    r_plot_max = L
                else:
                    r_plot_max = r2/sina + L
                y = r_plot_max - r_plot*cos(thetas*sina)
                x = r_plot*sin(thetas*sina)
            elif plot_type == 2:
                r_plot = fr(zs)
                y = r_plot*cos(thetas*sina)
                x = r_plot*sin(thetas*sina)
            elif plot_type == 3:
                r_plot = fr(zs)
                r_plot_max = r2/sina + L
                y = r_plot_max - r_plot*cos(thetas)
                x = r_plot*sin(thetas)
            elif plot_type == 4:
                x = fr(zs)*thetas
                y = zs
            elif plot_type == 5:
                x = thetas
                y = zs

            cir = x
            mer = y
            field = uvw

            if workingplt:
                levels = linspace(field.min(), field.max(), num_levels)
                if ax is None:
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)
                else:
                    if isinstance(ax, matplotlib.axes.Axes):
                        ax = ax
                        fig = ax.figure
                        save = False
                    else:
                        raise ValueError('"ax" must be an Axes object')
                ax.contourf(cir, mer, field, levels=levels)
                ax.grid(False)
                ax.set_aspect(aspect)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                #lim = self.r2*pi
                #ax.xaxis.set_ticks([-lim, 0, lim])
                #ax.xaxis.set_ticklabels([r'$-\pi$', '$0$', r'$+\pi$'])
                #ax.set_title(
                    #r'$PL=20 N$, $F_{{C}}=50 kN$, $w_{{PL}}=\beta$, $mm$')
                if clean:
                    ax.xaxis.set_ticks_position('none')
                    ax.yaxis.set_ticks_position('none')
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
                    ax.set_frame_on(False)
                if save:
                    msg('Plot saved at: {0}'.format(filename))
                    plt.tight_layout()
                    plt.savefig(filename, transparent=True,
                                bbox_inches='tight', pad_inches=0.05,
                                dpi=400)

            else:
                warn('Matplotlib cannot be imported from Abaqus')
            np.savez(npzname, cir=cir, mer=mer, field=field)
            with open(pyname, 'w') as f:
                f.write("import os\n")
                f.write("import numpy as np\n")
                f.write("import matplotlib.pyplot as plt\n")
                f.write("tmp = np.load('abaqus_output.npz')\n")
                f.write("cir = tmp['cir']\n")
                f.write("mer = tmp['mer']\n")
                f.write("field = tmp['field']\n")
                f.write("clean = {0}\n".format(clean))
                f.write("filename = '{0}'\n".format(filename))
                f.write("plt.figure(figsize={0})\n".format(figsize))
                f.write("ax = plt.gca()\n")
                f.write("levels = np.linspace(field.min(), field.max(), {0})\n".format(
                        num_levels))
                f.write("ax.contourf(cir, mer, field, levels=levels)\n")
                f.write("ax.grid(b=None)\n")
                f.write("ax.set_aspect('{0}')\n".format(aspect))
                f.write("ax.xaxis.set_ticks_position('bottom')\n")
                f.write("ax.yaxis.set_ticks_position('left')\n")
                f.write("ax.xaxis.set_ticks([{0}, 0, {1}])\n".format(
                        -self.r2*pi, self.r2*pi))
                f.write(r"ax.xaxis.set_ticklabels([r'$-\pi$', '$0$', r'$+\pi$'])\n")
                f.write("ax.set_title(r'Abaqus, $PL=20 N$, $F_{{C}}=50 kN$, $w_{{PL}}=\beta$, $mm$')\n")
                f.write("if clean:\n")
                f.write("    ax.xaxis.set_ticks_position('none')\n")
                f.write("    ax.yaxis.set_ticks_position('none')\n")
                f.write("    ax.xaxis.set_ticklabels([])\n")
                f.write("    ax.yaxis.set_ticklabels([])\n")
                f.write("    ax.set_frame_on(False)\n")
                f.write("if not filename:\n")
                f.write("    filename = 'abaqus_result.png'\n")
                f.write("plt.savefig(filename, transparent=True,\n")
                f.write("            bbox_inches='tight', pad_inches=0.05, dpi=400)\n")
                f.write("plt.show()\n")
            msg('Output exported to "{0}"'.format(npzname))
            msg('Please run the file "{0}" to plot the output'.format(
                  pyname))
            return cir, mer, field
        except:
            traceback.print_exc()
            error('Opened plot could not be generated! :(')


    def SPLA(self, PLs, NLgeom=True, plot=False):
        r"""Runs the Single Perturbation Load Approach (SPLA)

        A set of non-linear results will be

        Parameters
        ----------
        PLs: list
            The perturbation loads used to build the knock-down curve. It must
            be a list of float values.
        NLgeom : bool, optional
            Flag passed to the ``static()`` method that tells whether a
            geometrically non-linear analysis is to be performed.

        Returns
        -------
        curves : list
            The sequence of curves, one curve for each perturbation load given
            in the input parameter ``PLs``.
            Each curve in the list is a ``dict`` object with the keys:

            =================    ==============================================
            Key                  Description
            =================    ==============================================
            ``'wall_time_s'``    The wall time for the non-linear analysis
            ``'name'``           The name of the curve. Ex: ``'PL = 1. N'``
            ``'cs'``             A list with a vector of Ritz constants for
                                 each load increment needed
            ``'increments'``     A list with the values of increments needed
            ``'wPLs'``           A list with the normal displacement at the
                                 perturbation load application point for each
                                 load increment
            ``'uTMs'``           A list containing the axial displacement for
                                 each load increment
            ``'Fcs'``            A list containing the axial reaction force
                                 for each load increment
            =================    ==============================================

        Notes
        -----
        The curves are stores in the ``ConeCyl`` parameter
        ``outputs['SPLA_curves']``.

        """
        curves = []
        for PLi, PL in enumerate(PLs):
            curve = {}
            self.forces = []
            self.add_SPL(PL)
            t0 = time.clock()
            cs = self.static(NLgeom=NLgeom)
            if plot:
                self.plot(cs[-1])
            curve['wall_time_s'] = time.clock() - t0
            curve['name'] = 'PL = {} N'.format(PL)
            curve['cs'] = cs
            curve['increments'] = self.increments
            curve['wPLs'] = []
            curve['uTMs'] = []
            curve['Fcs'] = []
            for i, c in enumerate(self.cs):
                inc = self.increments[i]
                self.uvw(c, xs=self.L/2, ts=0)
                curve['wPLs'].append(self.w[0])
                if self.pdC:
                    ts = linspace(0, pi*2, 1000, endpoint=False)
                    xs = np.zeros_like(ts)
                    es = self.strain(c=c, xs=xs, ts=ts, inc=inc)
                    fvec = self.F.dot(es.T)
                    Fc = -fvec[0,:].mean()*(2*self.r2*pi)
                    curve['Fcs'].append(Fc/1000)
                    curve['uTMs'].append(inc*self.uTM)
                else:
                    curve['Fcs'].append(inc*self.Fc/1000)
                    curve['uTMs'].append(c[0])
            curves.append(curve)

        self.outputs['SPLA_curves'] = curves

        return curves


    def apply_shim(self, thetadeg, width, thick, ncpts=10000):
        r"""Distributes the axial load in order to simulate a shim

        The axial load distribution `{N_{xx}}_{top}` will be adjusted such
        that the resulting displacement `u` at `x=0` (top edge) will look
        similar to a case where a shim is applied.

        Parameters
        ----------
        thetadeg : float
            Position in degrees of the center of the shim.
        width : float
            Circumferential width of the shim.
        thick : float
            Thickness of the shim.
        ncpts : int, optional
            Number of control points used in the least-squares routine.

        Returns
        -------
        ts : np.ndarray
            Positions `\theta` of the control points.
        us : np.ndarray
            Displacements `u` of the control points.

        Notes
        -----
        This function changes the ``Nxxtop`` parameter of the current
        ``ConeCyl`` object. Returning ``ts`` and ``us`` is useful for post
        processing purposes only.

        Examples
        --------
        >>> ts, us = cc.apply_shim(0., 25.4, 0.1)

        """
        ts = linspace(-np.pi, np.pi, ncpts)
        us = np.zeros_like(ts)
        self.static(NLgeom=False)
        thetashim = width/self.r2
        thetarad = deg2rad(thetadeg)
        theta1 = thetarad - thetashim
        theta2 = thetarad + thetashim
        uTM = self.cs[0][0]
        us += uTM
        shim_region = (ts >= theta1) & (ts <= theta2)
        us[shim_region] += thick
        self.fit_Nxxtop(ts, us)

        return ts, us


    def fit_Nxxtop(self, ts, us, update_Nxxtop=True):
        r"""Adjusts the axial load distribution for a desired top edge
        displacement

        Parameters
        ----------
        ts : np.ndarray
            Corrdinates `\theta` of each control point.
        us : np.ndarray
            Desired displacement `u` for each control point.
        update_Nxxtop : bool, optional
            Tells whether ``self.Nxxtop`` should be updated.

        Returns
        -------
        Nxxtop : np.ndarray
            The coefficients for the `{N_{xx}}_{top}` function.

        """
        from scipy.sparse.linalg import inv as sparseinv

        assert ts.ndim == 1 and us.ndim == 1
        assert ts.shape[0] == us.shape[0]
        xs = np.zeros_like(ts)

        if not update_Nxxtop:
            Nxxtop_backup = self.Nxxtop.copy()

        k0uuinv = sparseinv(csc_matrix(self.k0uu))

        Nxxtop_new = self.Nxxtop.copy()

        def fit_Nxxtop_residual(Nxxtop_new):
            self.Nxxtop = Nxxtop_new.copy()
            fext = self.calc_fext(silent=True)
            c = k0uuinv*fext
            self.uvw(c, xs=xs, ts=ts)
            res = (self.u - us)**2
            return res
        popt, pcov = leastsq(fit_Nxxtop_residual, x0=Nxxtop_new, maxfev=10000)

        if not update_Nxxtop:
            self.Nxxtop = Nxxtop_backup
        else:
            self.Nxxtop = popt

        return popt


    def save(self):
        r"""Save the ``ConeCyl`` object using ``pickle``

        Notes
        -----
        The pickled file will have the name stored in ``ConeCyl.name``
        followed by a ``'.ConeCyl'`` extension.

        """
        name = self.name + '.ConeCyl'
        msg('Saving ConeCyl to {}'.format(name))
        self.analysis.calc_fext = None
        self.analysis.calc_k0 = None
        self.analysis.calc_fint = None
        self.analysis.calc_kT = None
        self._clear_matrices()

        with open(name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

