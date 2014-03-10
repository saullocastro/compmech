from __future__ import division
import gc
import os
import sys
from collections import Iterable
import time
import cPickle
import __main__

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, eigsh
from numpy import linspace, pi, cos, sin, tan, deg2rad

import mapy.model.properties.composite as composite
import clpt_commons
import fsdt_commons
import clpt_linear_donnell
import clpt_linear_sanders
import fsdt_linear_donnell
import non_linear

from plotutils import get_filename

def load(name):
    return cPickle.load(open(name + '.ConeCyl', 'rb'))


class ConeCyl(object):

    def __init__(self):
        self.name = ''
        self.PLvalues = []
        self.PLxs = []
        self.PLthetas = []
        self.alphadeg = 0.
        self.alpharad = 0.
        self.is_cylinder = None
        self.last_analysis = None

        # boundary conditions
        self.bc = 'ss'
        self.kuBot = 1.e15
        self.kuTop = -1.e15

        self.kvBot = 1.e15
        self.kvTop = -1.e15

        self.kwBot = 1.e15
        self.kwTop = -1.e15

        self.kphixBot = 1.e15
        self.kphixTop = -1.e15

        self.kphitBot = 1.e15
        self.kphitTop = -1.e15


        # default equations
        self.linear_kinematics = 'clpt_donnell'
        self.NL_kinematics = 'donnell_numerical'

        # approximation series
        self.m1 = 40
        self.m2 = 40
        self.n2 = 40

        # analytical integration for cones
        self.s = 80

        # numerical integration
        self.nx = 50
        self.nt = 100

        # internal pressure measured in force/area
        self.P = 0.

        # axial compression
        self.pdC = None
        self.Fc = 0.
        self.uTM = 0.

        # torsion
        self.pdT = True
        self.T = 0.
        self.thetaTdeg = 0.
        self.thetaTrad = 0.

        # load asymmetry (la)
        self.pdLA = True
        self.tLAdeg = 0.
        self.tLArad = 0.
        self.betadeg = 0.
        self.betarad = 0.
        self.LA = None

        self.num0 = 3
        self.excluded_dofs = []
        self.excluded_dofs_ck = []

        self.r1 = None
        self.r2 = None
        self.L = None
        self.h = None # total thickness, required for isotropic shells
        self.K = 5/6.
        self.sina = None
        self.cosa = None

        # material
        self.laminaprops = []
        self.stack = []
        self.plyts = []

        # constitutive law
        self.F = None
        self.E11 = None
        self.nu = None

        # eigenvalue analysis
        self.num_eigvalues = 50
        self.num_eigvalues_print = 5

        # non-linear algorithm
        self.NL_method = 'NR' # Newton-Raphson
        self.modified_NR = False # modified Newton-Raphson
        self.compute_every_n = 10 # for modified Newton-Raphson

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
        self.maxNumIter = 1000
        self.increment_PL = False

        # numerical integration
        self.ni_num_cores = 4 # showed to scale well up to 4
        self.ni_method = 'trapz2d'

        # outputs
        self.outputs = {}

        self.debug = False
        self.clear_matrices()

    def clear_matrices(self):
        self.k0 = None
        self.k0k = None
        self.k0uk = None
        self.k0uu = None
        self.kTuk = None
        self.kTuu = None
        self.kSkk = None
        self.kSku = None
        self.kSuk = None
        self.kSuu = None
        self.kG0uu = None
        self.kG0 = None
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

        gc.collect()

    def rebuild(self):
        if not self.name:
            try:
                self.name = os.path.basename(__main__.__file__).split('.py')[0]
            except AttributeError:
                print('WARNING - ConeCyl name unchanged')


        self.linear_kinematics = self.linear_kinematics.lower()

        valid_linear_kinematics = ['clpt_donnell', 'clpt_sanders',
                                   'fsdt_donnell']

        if not self.linear_kinematics in valid_linear_kinematics:
            raise ValueError('ERROR - valid linear theories are:\n' +
                     '\t' + ', '.join(valid_linear_kinematics))


        self.NL_kinematics = self.NL_kinematics.lower()

        valid_NL_kinematics = ['donnell_numerical', 'sanders_numerical']

        if not self.NL_kinematics in valid_NL_kinematics:
            raise ValueError('ERROR - valid non-linear theories are:\n' +
                     '\t' + ', '.join(valid_NL_kinematics))


        self.bc = self.bc.lower()

        if self.bc == 'ss':
            self.kphixBot = 1.e-15
            self.kphixTop = -1.e-15
        elif self.bc == 'cc':
            self.kphixBot = 1.e15
            self.kphixTop = -1.e15
        else:
            raise ValueError('{} is an invalid value for "bc"'.format(
                             self.bc))

        self.alpharad = deg2rad(self.alphadeg)
        self.sina = sin(self.alpharad)
        self.cosa = cos(self.alpharad)

        if not self.H and not self.L:
            self.H = (self.r1-self.r2)/tan(self.alpharad)
        if self.H and not self.L:
            self.L = self.H/self.cosa

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
        self.LA = -self.r2*tan(self.betarad)

        if not self.laminaprops:
            self.laminaprops = [self.laminaprop for i in self.stack]
        if not self.plyts:
            self.plyts = [self.plyt for i in self.stack]

        if self.alpharad==0:
            self.is_cylinder = True
        else:
            self.is_cylinder = False

        if self.pdC==None:
            raise ValueError('ConeCyl().pdC must be defined')
        if self.pdT==None:
            raise ValueError('ConeCyl().pdT must be defined')

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

        if not self.m1:
            self.m1 = self.m2

        self.maxInc = max(self.initialInc, self.maxInc)

        # estimating number of integration points based on convergence studies
        # the interpolation is linear but should not be proportional
        nx_nt_table = {20: 100}

        if not self.nx and self.nt:
            self.nx = int(round(self.nt*max(self.m1, self.m2)/self.n2))
        elif not self.nt and self.nx:
            self.nx = int(round(self.nt*max(self.m1, self.m2)/self.n2))
            self.nt = int(round(self.nx*self.n2/max(self.m1, self.m2)))

        if not isinstance(self.PLvalues, Iterable):
            self.PLvalues = [self.PLvalues]
        if not isinstance(self.PLxs, Iterable):
            self.PLxs = [self.PLxs]
        if not isinstance(self.PLthetas, Iterable):
            self.PLxs = [self.PLthetas]

        if self.laminaprop==None:
            h = self.h
            E11 = self.E11
            nu = self.nu
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
            self.F = np.array([[A11, A12, A16, 0, 0, 0],
                               [A12, A22, A26, 0, 0, 0],
                               [A16, A26, A66, 0, 0, 0],
                               [0, 0, 0, D11, D12, D16],
                               [0, 0, 0, D12, D22, D26],
                               [0, 0, 0, D16, D26, D66]])

    def calc_full_c(self, cu, inc=1.):
        '''Returns the full set of Ritz constants.

        When prescribed displacements take place the matrices and the Ritz
        constants are partitioned, this function takes the prescribed degrees
        of freedom (`ck`) and join them with the calculated ones (`cu`) in
        order to build a full set of Ritz constants.

        Parameters
        ----------
        cu : np.ndarray
            The set of unknown Ritz constants
        inc : float
            The increment for which `c` should be calculated.
            Default is `1.`.

        Returns
        -------
        c : np.ndarray
            The full set of Ritz constants.

        '''
        #TODO check of size

        c = cu.copy()

        ordered = sorted(zip(self.excluded_dofs,
                             self.excluded_dofs_ck), key=lambda x:x[0])
        for dof, cai in ordered:
            c = np.insert(c, dof, inc*cai)

        return c

    def exclude_dofs_matrix(self, k, return_kkk=False,
                                     return_kku=False,
                                     return_kuk=False):
        '''Makes the following partition of a given matrix:

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
            A `dict` object containing the keys for the
            corresponding sub-matrices `kkk`, `kku`, `kuk`, `kuu`. The
            sub-matrix `out['kuu']` is a `scipy.sparse.csr_matrix`, while
            the others are 2-D `np.ndarray` objects.

        '''
        if not isinstance(k, coo_matrix):
            raise ValueError('ERROR - A coo_matrix is required')

        kuu = k.copy()

        if return_kkk:
            kkk = coo_matrix(np.zeros((self.num0, self.num0)))
            check = (kuu.row < self.num0) & (kuu.col < self.num0)
            kkk.row = kuu.row[check].copy()
            kkk.col = kuu.col[check].copy()
            kkk.data = kuu.data[check].copy()
            kkk = kkk.toarray()
            kkk = np.delete(kkk, self.excluded_dofs, axis=0)
            kkk = np.delete(kkk, self.excluded_dofs, axis=1)

        if return_kku:
            kku = coo_matrix(np.zeros((self.num0, kuu.shape[0])))
            check = (kuu.row < self.num0)
            kku.row = kuu.row[check].copy()
            kku.col = kuu.col[check].copy()
            kku.data = kuu.data[check].copy()
            kku = kku.toarray()
            kku = np.delete(kku, self.excluded_dofs, axis=1)

        if return_kuk:
            kuk = coo_matrix(np.zeros((kuu.shape[0], self.num0)))
            check = (kuu.col < self.num0)
            kuk.row = kuu.row[check].copy()
            kuk.col = kuu.col[check].copy()
            kuk.data = kuu.data[check].copy()
            kuk = kuk.toarray()
            kuk = np.delete(kuk, self.excluded_dofs, axis=0)

        rows = sorted(self.excluded_dofs)[::-1]
        cols = sorted(self.excluded_dofs)[::-1]

        for r in rows:
            check = kuu.row != r
            kuu.row[kuu.row > r] -= 1
            kuu.row = kuu.row[check]
            kuu.col = kuu.col[check]
            kuu.data = kuu.data[check]
            kuu._shape = (kuu.shape[0]-1, kuu.shape[1])
            kuu = coo_matrix(kuu)

        for c in cols:
            check = kuu.col != c
            kuu.col[kuu.col > c] -= 1
            kuu.row = kuu.row[check]
            kuu.col = kuu.col[check]
            kuu.data = kuu.data[check]
            kuu._shape = (kuu.shape[0], kuu.shape[1]-1)
            kuu = coo_matrix(kuu)
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

    def default_field(self, x, t, gridx, gridt):
        if x==None or t==None:
            x = linspace(0, self.L, gridx)
            t = linspace(-pi, pi, gridt)
            x, t = np.meshgrid(x, t, copy=False)
        x = np.atleast_1d(np.array(x, dtype=float))
        t = np.atleast_1d(np.array(t, dtype=float))
        xshape = x.shape
        tshape = t.shape
        if xshape != tshape:
            raise ValueError('Arrays x and t must have the same shape')
        self.Xs = x
        self.Ts = t
        x = x.ravel()
        t = t.ravel()

        return x, t, xshape, tshape

    def uvw(self, c, x=None, t=None, gridx=100, gridt=200):
        '''Calculates the displacement field.

        For a given full set of Ritz constants `c`, the displacement field is
        calculated and stored in the parameters `u`, `v`, `w`, `phix`, `phit`
        of the `ConeCyl` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        x : np.ndarray
            The `x` positions where to calculate the displacement field.
            Default is `None` and the method `default_field` is used.
        t : np.ndarray
            The `theta` positions where to calculate the displacement field.
            Default is `None` and the method `default_field` is used.
        gridx : int
            Number of points along the `x` axis where to calculate the
            displacement field.
            Default is `100`.
        gridx : int
            Number of points along the `theta` where to calculate the
            displacement field.
            Default is `200`.

        Returns:
            For the Classical Laminated Plate Theory (CLPT)
            (u, v, w) : tuple

            For the First-order Shear Deformation Theory (FSDT)
            (u, v, w, phix, phit) : tuple

        Notes
        -----
        The returned values `u`, `v`, `w`, `phix`, `phit` are stored as
        parameters in the `ConeCyl` object.

        '''
        x, t, xshape, tshape = self.default_field(x, t, gridx, gridt)
        alpharad = self.alpharad
        tLArad = self.tLArad
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        r2 = self.r2
        L = self.L
        bc = self.bc
        uTM = 0.
        if self.uTM != None:
            uTM = self.uTM
        linear_kinematics = self.linear_kinematics

        if c.shape[0] != self.k0.shape[0]:
            print ''
            print '\t\tWARNING - completing cu with ck'
            c = self.calc_full_c(c)

        if 'fsdt_donnell' in linear_kinematics:
            u, v, w, phix, phit = fsdt_commons.fuvw(c, m1, m2, n2,
                                      alpharad, r2, L, tLArad, x, t)
            self.u = u.reshape(xshape)
            self.v = v.reshape(xshape)
            self.w = w.reshape(xshape)
            self.phix = phix.reshape(xshape)
            self.phit = phit.reshape(xshape)

            return self.u, self.v, self.w, self.phix, self.phit

        elif 'fsdt_general_donnell' in linear_kinematics:
            import fsdt_general_commons
            u, v, w, phix, phit = fsdt_general_commons.fuvw(c, m1, m2, n2,
                                      alpharad, r2, L, tLArad, x, t)
            self.u = u.reshape(xshape)
            self.v = v.reshape(xshape)
            self.w = w.reshape(xshape)
            self.phix = phix.reshape(xshape)
            self.phit = phit.reshape(xshape)

            return self.u, self.v, self.w, self.phix, self.phit

        elif 'clpt' in linear_kinematics:
            u, v, w = clpt_commons.fuvw(c, m1, m2, n2, alpharad, r2, L,
                                        tLArad, x, t)
            self.u = u.reshape(xshape)
            self.v = v.reshape(xshape)
            self.w = w.reshape(xshape)

            return self.u, self.v, self.w

    def calc_linear_matrices(self):
        print('\t\tCalculating linear matrices... '),
        self.rebuild()
        Fc = self.Fc
        if Fc==None:
            Fc = 1.
        P = self.P
        T = self.T

        alpharad = self.alpharad
        cosa = self.cosa
        r1 = self.r1
        r2 = self.r2
        L = self.L
        h = self.h
        E11 = self.E11
        nu = self.nu
        laminaprops = self.laminaprops
        stack = self.stack
        plyts = self.plyts
        lam = self.lam
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        bc = self.bc
        s = self.s
        linear_kinematics = self.linear_kinematics
        if self.stack != []:
            lam = composite.read_stack(stack, plyts, laminaprops=laminaprops)

        k0edges = None
        if 'fsdt' in linear_kinematics:
            if lam != None:
                F = lam.ABDE
                F[6:,6:] *= self.K
                self.F = F
            else:
                F = self.F

            if 'fsdt_donnell' in linear_kinematics:
                from fsdt_linear_donnell import (fk0,
                                                 fk0_cyl,
                                                 fk0edges,
                                                 fkG0,
                                                 fkG0_cyl)

                k0edges = fk0edges(m1, m2, n2, r1, r2,
                            self.kuBot, self.kuTop,
                            self.kphixBot, self.kphixTop)

            elif 'fsdt_general' in linear_kinematics:
                from fsdt_general_linear_donnell import (fk0,
                                                         fk0_cyl,
                                                         fkG0,
                                                         fkG0_cyl)

                k0edges = fsdt_general_linear_donnell.fk0edges_cyl(
                            m1, m2, n2, r2,
                            self.kuBot, self.kuTop,
                            self.kvBot, self.kvTop,
                            self.kwBot, self.kwTop,
                            self.kphixBot, self.kphixTop,
                            self.kphitBot, self.kphitTop)

        elif 'clpt' in linear_kinematics:
            if lam != None:
                F = lam.ABD
                self.F = F
            else:
                F = self.F
            if bc != 'ss':
                print('WARNING - With CLPT only bc="ss" is supported')

            if linear_kinematics=='clpt_donnell':
                from clpt_linear_donnell import (fk0,
                                                 fk0_cyl,
                                                 fkG0,
                                                 fkG0_cyl)
            elif linear_kinematics=='clpt_sanders':
                from clpt_linear_sanders import (fk0,
                                                 fk0_cyl,
                                                 fkG0,
                                                 fkG0_cyl)

        if self.is_cylinder:
            k0 = fk0_cyl(r2, L, F, m1, m2, n2)
            kG0 = fkG0_cyl(Fc, P, T, r2, L, m1, m2, n2)
        else:
            k0 = fk0(alpharad, r2, L, F, m1, m2, n2, s)
            kG0 = fkG0(Fc, P, T, r2, alpharad, L, m1, m2, n2, s)

        k = self.exclude_dofs_matrix(k0, return_kuk=True)
        k0uk = k['kuk']
        k0uu = k['kuu']
        kG0uu = self.exclude_dofs_matrix(kG0)['kuu']

        if k0edges:
            k = self.exclude_dofs_matrix(k0edges, return_kuk=True)
            k0edgesuk = k['kuk']
            k0edgesuu = k['kuu']

            k0uk = k0uk + k0edgesuk
            k0uu = k0uu + k0edgesuu

        self.k0 = k0
        self.k0uk = k0uk
        self.k0uu = k0uu
        self.kG0 = kG0
        self.kG0uu = kG0uu

        #NOTE forcing Python garbage collector to clear the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process

        gc.collect()

        print('finished!')

    def lb(self, c=None):
        '''Linear Buckling analysis.

        The following parameters of the `ConeCyl` object will affect the
        linear buckling analysis.

        =======================    ========================================
        parameter                  description
        =======================    ========================================
        ``num_eigenvalues``        Number of eigenvalues to be extracted
        ``num_eigvalues_print``    Number of eigenvalues to print after the
                                   analysis is completed
        =======================    ========================================

        Notes
        -----
        The extracted eigenvalues are stored in the `eigvals` parameter of the
        `ConeCyl` object and the ith eigenvector in the `eigvecs[i-1, :]`
        parameter.

        '''
        print('Running linear buckling analysis...')
        self.calc_linear_matrices()
        if c<>None:
            self.calc_NL_matrices(c)
            self.kG0 = self.kG
        #TODO maybe a better estimator to sigma would be to run
        #     a preliminary eigsh using a small m2 and n2
        #NOTE runs faster for self.k0 than -self.k0, so that the negative
        #     sign is applied later
        print('\t\tEigenvalue solver... '),

        #self.k0uu.data[np.abs(self.k0uu.data)<1.e-12] = 0.

        if c==None:
            eigvals, eigvecs = eigsh(self.kG0uu, k=self.num_eigvalues,
                                     which='SM', M=self.k0uu, sigma=1.)
        else:
            raise ('FIXME')
            eigvals, eigvecs = eigsh(self.kG0uu, k=self.num_eigvalues,
                                     which='SM',
                                 M=(self.k0 + self.k0L + self.kL0 + self.kLL),
                                 sigma=1.)
        eigvals = (-1./eigvals)
        self.eigvecs = eigvecs
        self.eigvals = eigvals
        pcr = eigvals[0]
        print('finished!')
        print('    first {} eigenvalues:'.format(self.num_eigvalues_print))
        for eig in eigvals[:self.num_eigvalues_print]:
            print('        {}'.format(eig))
        self.last_analysis = 'lb'

    def calc_NL_matrices(self, c, inc=1., num_cores=None):
        if not num_cores:
            num_cores=self.ni_num_cores

        c = self.calc_full_c(c, inc=inc)

        if self.k0==None:
            self.calc_linear_matrices()

        print('\t\tCalculating non-linear matrices... '),
        NL_kinematics = self.NL_kinematics
        linear_kinematics = self.linear_kinematics
        alpharad = self.alpharad
        r2 = self.r2
        L = self.L
        tLArad = self.tLArad
        F = self.F
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2

        if NL_kinematics=='donnell_numerical':
            if 'clpt' in linear_kinematics:
                from clpt_NL_donnell_numerical import (calc_k0L,
                                                       calc_kG,
                                                       calc_kLL)
                k0L = calc_k0L(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                               nx=self.nx, nt=self.nt,
                               num_cores=num_cores,
                               method=self.ni_method)
                kG = calc_kG(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                             nx=self.nx, nt=self.nt,
                             num_cores=num_cores,
                             method=self.ni_method)
                kLL = calc_kLL(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                               nx=self.nx, nt=self.nt,
                               num_cores=num_cores,
                               method=self.ni_method)
            elif 'fsdt' in linear_kinematics:
                from fsdt_NL_donnell_numerical import (calc_k0L,
                                                       calc_kG,
                                                       calc_kLL)
                k0L = calc_k0L(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                               nx=self.nx, nt=self.nt,
                               num_cores=num_cores,
                               method=self.ni_method)
                kG = calc_kG(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                             nx=self.nx, nt=self.nt,
                             num_cores=num_cores,
                             method=self.ni_method)
                kLL = calc_kLL(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                               nx=self.nx, nt=self.nt,
                               num_cores=num_cores,
                               method=self.ni_method)
            else:
                raise ValueError(
                        '{} is an nvalid linear_kinematics option'.format(
                            linear_kinematics))

        elif NL_kinematics=='sanders_numerical':
            if 'clpt' in linear_kinematics:
                from clpt_NL_sanders_numerical import (calc_k0L,
                                                       calc_kG,
                                                       calc_kLL)
                k0L = calc_k0L(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                               nx=self.nx, nt=self.nt,
                               num_cores=num_cores,
                               method=self.ni_method)
                kG = calc_kG(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                             nx=self.nx, nt=self.nt,
                             num_cores=num_cores,
                             method=self.ni_method)
                kLL = calc_kLL(c, alpharad, r2, L, tLArad, F, m1, m2, n2,
                               nx=self.nx, nt=self.nt,
                               num_cores=num_cores,
                               method=self.ni_method)
            elif 'fsdt' in linear_kinematics:
                raise NotImplementedError(
                        'Sanders not implemented for FSDT')
            else:
                raise ValueError(
                        '{} is an nvalid "linear_kinematics" option'.format(
                            linear_kinematics))

        else:
            raise ValueError(
                    '{} is an invalid "NL_kinematics" option'.format(
                    NL_kinematics))


        kL0 = k0L.T

        #TODO maybe slow...
        kT = coo_matrix(self.k0 + k0L + kL0 + kLL + kG)
        kS = coo_matrix(self.k0 + k0L/2 + kL0 + kLL/2)

        k =  self.exclude_dofs_matrix(kT, return_kuk=True)
        self.kTuk = k['kuk']
        self.kTuu = k['kuu']
        k = self.exclude_dofs_matrix(kS, return_kkk=True, return_kku=True,
                                         return_kuk=True)
        self.kSkk = k['kkk']
        self.kSku = k['kku']
        self.kSuk = k['kuk']
        self.kSuu = k['kuu']

        #NOTE intended to be used in an eigenvalue analysis using NL stresses
        self.kL = csr_matrix(self.k0 + k0L + kL0 + kLL)
        self.kG = csr_matrix(kG)

        print('finished!')

    def strain(self, c, x=None, t=None, gridx=100, gridt=200,
               inc=1.):
        x, t, xshape, tshape = self.default_field(x, t, gridx, gridt)

        alpharad = self.alpharad
        L = self.L
        r2 = self.r2
        sina = self.sina
        cosa = self.cosa
        tLArad = self.tLArad
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2

        NL_kinematics = self.NL_kinematics
        if NL_kinematics=='fsdt':
            e_num = 8
            from fsdt_commons import fstrain
        else:
            e_num = 6
            from clpt_commons import fstrain
        if 'donnell' in NL_kinematics:
            int_NL_kinematics = 0
        elif 'sanders' in NL_kinematics:
            int_NL_kinematics = 1
        else:
            raise NotImplementedError(
                    '{} is an invalid "NL_kinematics" option'.format(
                    NL_kinematics))

        c = self.calc_full_c(c, inc=inc)

        evec = fstrain(c, sina, cosa, tLArad, x, t, r2, L,
                       m1, m2, n2, int_NL_kinematics)

        return evec.reshape((x.shape + (e_num,)))

    def calc_f_ext(self, inc=None, kuk=None, increment_PL=None):
        print('\t\tCalculating external forces... '),
        if inc==None:
            Fc = self.Fc
            uTM = self.uTM
            thetaTrad = self.thetaTrad
        else:
            Fc = inc*self.Fc
            uTM = inc*self.uTM
            thetaTrad = inc*self.thetaTrad
        if increment_PL==None:
            PLvalues = self.PLvalues
        else:
            PLvalues = [PL*increment_PL for PL in self.PLvalues]
        sina = self.sina
        cosa = self.cosa
        r2 = self.r2
        L = self.L
        tLArad = self.tLArad
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2
        bc = self.bc
        pdC = self.pdC
        pdT = self.pdT
        linear_kinematics = self.linear_kinematics
        PLxs = self.PLxs
        PLthetas = self.PLthetas
        num0 = self.num0


        if 'fsdt_general' in linear_kinematics:
            dofs = 5
            num1 = 10
            num2 = 20
            import fsdt_general_commons
            fg = fsdt_general_commons.fg

        elif 'fsdt_donnell' in linear_kinematics:
            dofs = 5
            num1 = 7
            num2 = 14
            fg = fsdt_commons.fg

        elif 'clpt' in linear_kinematics:
            dofs = 3
            num1 = 3
            num2 = 6
            fg = clpt_commons.fgss

        else:
            raise ValueError(
                    '{} is an invalid "linear_kinematics" option'.format(
                    linear_kinematics))

        g = np.zeros((dofs, num0 + num1*m1 + num2*m2*n2), dtype=float)
        f_ext = np.zeros((num0 + num1*m1 + num2*m2*n2), dtype=float)
        f_ext = np.delete(f_ext, self.excluded_dofs)

        # perturbation loads
        for i, PL in enumerate(PLvalues):
            PLx = PLxs[i]
            PLtheta = PLthetas[i]
            fg(g, m1, m2, n2, r2, PLx, PLtheta, L, cosa, tLArad)
            gu = np.delete(g, self.excluded_dofs, axis=1).copy()

            if dofs==3:
                fpt = np.array([[0], [0], [PL]])
            elif dofs==5:
                fpt = np.array([[0], [0], [PL], [0], [0]])
            f_ext += -fpt.T.dot(gu).ravel()

        # axial load
        pts = float(1)
        Fci = Fc/pts
        uTMi = uTM/pts
        ts = linspace(0, 2*np.pi, pts, endpoint=False)
        for t in ts:
            fg(g, m1, m2, n2, r2, 0, t, L, cosa, tLArad)
            gu = np.delete(g, self.excluded_dofs, axis=1).copy()

            if pdC:
                if kuk==None:
                    kuk_C = self.k0uk[:, 0].ravel()
                else:
                    kuk_C = kuk[:, 0].ravel()
                f_ext += -uTMi*kuk_C
            else:
                if dofs==3:
                    fpt = np.array([[Fci/cosa], [0], [0]])
                elif dofs==5:
                    fpt = np.array([[Fci/cosa], [0], [0], [0], [0]])
                f_ext += fpt.T.dot(gu).ravel()

        # torsion
        fg(g, m1, m2, n2, r2, 0, 0, L, cosa, tLArad)
        gu = np.delete(g, self.excluded_dofs, axis=1).copy()
        if pdT:
            if kuk==None:
                kuk_T = self.k0uk[:, 1].ravel()
            else:
                kuk_T = kuk[:, 1].ravel()
            f_ext += -thetaTrad*kuk_T
        else:
            if dofs==3:
                fpt = np.array([[0], [self.T/r2], [0]])
            elif dofs==5:
                fpt = np.array([[0], [self.T/r2], [0], [0], [0]])
            f_ext += fpt.T.dot(gu).ravel()

        # pressure
        if self.P != 0.:
            if 'clpt' in linear_kinematics:
                for i1 in range(1, m1+1):
                    col = num0 + (i1-1)*num1 + 2
                    f_ext[col] += self.P*(2/i1*(r2 - (-1)**i1*(r2 + L*sina)))

            elif 'fsdt' in linear_kinematics:
                #TODO it might be the same as for the CLPT
                raise NotImplementedError('pressure not implemented for FSDT')

        print('finished!')

        return f_ext


    def static(self, NLgeom=False):
        '''Static analysis for cones and cylinders.

        The analysis can be linear or non-linear. In case of a non-linear
        analysis the following parameters of the `ConeCyl` object will
        affect the non-linear analysis:

        ====================    ======================================
        non-linear algorithm    description
        ====================    ======================================
        ``NL_method``           'NR' for the Newton-Raphson
                                'arc_length' for the Arc-Length method
        ``modified_NR``         activate the modified Newton-Raphson
        ``compute_every_n``     if `modified_NR=True`, the non-linear
                                matrices will be updated at every `n`
                                iterations
        ====================    ======================================

        ==============    ===================================================
        incrementation    description
        ==============    ===================================================
        ``initialInc``    initial load increment size
        ``minInc``        minimum increment size; if achieved the analysis is
                          terminated
        ``maxInc``        maximum increment size
        ==============    ===================================================

        ====================    ============================================
        convergence criteria    description
        ====================    ============================================
        ``absTOL``              the convergence is achieved when the maximum
                                residual force is smaller than this value
        ``maxNumIter``          maximum number of iteration; if achieved the
                                load increment is bisected
        ====================    ============================================

        =====================    =======================================
        numerical integration    description
        =====================    =======================================
        ``ni_num_cores``         number of cores used for the numerical
                                 integration
        ``ni_method``            'trapz2d' for 2-D Trapezoidal's
                                 'simps2d' for 2-D Simpsons' integration
        =====================    =======================================

        Parameters
        ----------
        NLgeom : bool
            Flag to indicate whether a linear or a non-linear analysis is to
            be performed.
            The default is `True`.

        Returns
        -------
        cs : list
            A list containing the Ritz constants for each load increment of
            the static analysis. The list will have only one entry in case
            of a linear analysis.

        Notes
        -----
        The returned `cs` is stored in the `cs` parameter in the `ConeCyl`
        object. The actual increments used in the non-linear analysis are
        stored in the `increments` parameter.

        '''
        self.cs = []
        self.increments = []
        if NLgeom:
            print('Started Non-Linear Static Analysis')
            self.calc_linear_matrices()
            if self.NL_method=='NR':
                non_linear.NR(self)
            elif self.NL_method=='NR_Broyden':
                non_linear.NR_Broyden(self)
            elif self.NL_method=='arc_length':
                non_linear.arc_length(self)
        else:
            print('Started Linear Static Analysis')
            self.calc_linear_matrices()
            f_ext = self.calc_f_ext()
            c = spsolve(self.k0uu, f_ext)
            if self.debug:
                self.uvw(c, x=self.L/2., t=0.)
                w0 = self.w.min()
                print 'DEBUG LINEAR wPL', w0
            self.cs.append(c)
            self.increments.append(1.)
            print('Finished Linear Static Analysis')
        #
        self.last_analysis = 'static'
        return self.cs

    def plot(self, c, vec='w', filename='', figsize=(3.5, 2.), save=True,
             add_title=True):
        '''Plot the contour of the

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        vec : str
            Can be `'u'`, `'v'`, `'w'`, `'phix'` or `'phit'`, which indicates
            which displacement variable should be plotted.
            The default value is `'w'` (the normal displacement).
        save : bool
            Flag telling whether the contour should be saved to an image file.
            The default value is `True`.
        filename : str
            The file name for the generated image file. If no value is given,
            the `name` parameter of the `ConeCyl` object will be used.
        figsize : tuple
            The figure size given by `(width, height)`.
            The default is `(3.5, 2.)`.
        add_title : bool
            If a title should be added to the figure. Default is `True`.

        Returns
        -------

        ax : matplotlib.axes.AxesSubplot
            The Matplotlib object that can be used to modify the current plot
            if needed.

        '''
        print('Plotting contour...'),
        ubkp, vbkp, wbkp = self.u, self.v, self.w

        import matplotlib.pyplot as plt

        c = self.calc_full_c(c)

        self.uvw(c, x=self.L/2., t=0)
        wPL = self.w[0]

        self.uvw(c)
        Xs = self.Xs
        Ts = self.Ts

        vec = getattr(self, vec)
        levels = np.linspace(vec.min(), vec.max(), 400)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # NOTE Xs must be plotted inverted because it starts at the top
        # for the semi-analytical model, but it starts at the bottom
        # for the finite-element model
        ax.contourf(self.r2*Ts, Xs[:,::-1], vec, levels=levels)

        ax.grid(False)
        ax.set_aspect('equal')
        if add_title:
            if self.last_analysis == 'static':
                ax.set_title(r'$m2={0}$, $n2={1}$, $w_{{PL}}={2:1.3f} mm$'.
                             format(self.m2, self.n2, wPL))
            elif self.last_analysis == 'lb':
                ax.set_title(r'$m2={0}$, $n2={1}$, $P_{{CR}}={2:1.3f} kN$'.
                             format(self.m2, self.n2, self.eigvals[0]/1000))

        fig.tight_layout()
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_frame_on(False)
        if save:
            if not filename:
                filename = get_filename(self)
            fig.savefig(filename, transparent=True,
                        bbox_inches='tight', pad_inches=0.05)
            plt.close()

        self.u, self.v, self.w = ubkp, vbkp, wbkp

        print('finished!')
        return ax

    def add_SPL(self, PL, pt=0.5, theta=0.):
        '''Add a Single Perturbation Load (SPL).

        Adds a perturbation load to the `ConeCyl` object.

        Parameters
        ----------
        PL : float
            The perturbation load value.
        pt : float
            The normalized position along the `x` axis in which the new SPL
            will be included.
            Default is `0.5`.
        theta : float
            The angular position of the new SPL.
            Default is `0.`.

        Notes
        -----
        The `ConeCyl` object has three parameters which must be changed when a
        SPL is included: `PLvalues`, `PLxs` and `PLthetas`. These three
        parameters are `list` objects that can be changed elsewhere if one
        needs.

        '''
        self.rebuild()
        self.PLvalues.append(PL)
        self.PLxs.append(pt*self.L)
        self.PLthetas.append(theta)

    def SPLA(self, PLs, NLgeom=True):
        '''Runs the Single Perturbation Load Approach (SPLA).

        A set of non-linear results will be

        Parameters
        ----------

        Returns
        -------
        curves : list
            The sequence of curves, one curve for each perturbation load given
            in the input parameter `PLs`.
            Each curve in the list is a `dict` object with the keys:

            ===============    ==============================================
            key                description
            ===============    ==============================================
            `'wall_time_s'`    The wall time for the non-linear analysis
            `'name'`           The name of the curve. Ex: `PL = 1. N`
            `'cs'`             A `list` with a vector of Ritz constants for
                               each load increment needed
            `'increments'`     A `list` with the values of increments needed
            `'wPLs'`           A `list` with the normal displacement at the
                               perturbation load application point for each
                               load increment
            `'uTMs'`           A `list` containing the axial displacement for
                               each load increment
            `'Fcs'`            A `list` containing the axial reaction force
                               for each load increment
            ===============    ==============================================

        Notes
        -----
        The curves are stores in the `ConeCyl` parameter
        `outputs['SPLA_curves']`.

        '''
        self.add_SPL(0.)
        curves = []
        for PLi, PL in enumerate(PLs):
            curve = {}
            self.PLvalues = [PL]
            time1 = time.clock()
            cs = self.static(NLgeom=NLgeom)
            curve['wall_time_s'] = time.clock() - time1
            curve['name'] = 'PL = {} N'.format(PL)
            curve['cs'] = cs
            curve['increments'] = self.increments
            curve['wPLs'] = []
            curve['uTMs'] = []
            curve['Fcs'] = []
            for i, c in enumerate(self.cs):
                inc = self.increments[i]
                self.uvw(c, x=self.L/2, t=0)
                curve['wPLs'].append(self.w[0])
                if self.pdC:
                    ts = np.linspace(0, np.pi*2, 1000, endpoint=False)
                    xs = np.zeros_like(ts)
                    evec = self.strain(c=c, x=xs, t=ts, inc=inc)
                    fvec = self.F.dot(evec.T)
                    Fc = -fvec[0,:].mean()*(2*self.r2*np.pi)
                    curve['Fcs'].append(Fc/1000)
                    curve['uTMs'].append(inc*self.uTM)
                else:
                    curve['Fcs'].append(inc*self.Fc/1000)
                    curve['uTMs'].append(c[0])
            curves.append(curve)

        self.outputs['SPLA_curves'] = curves

        return curves

    def save(self):
        name = self.name + '.ConeCyl'
        print('Saving ConeCyl to {}'.format(name))

        self.clear_matrices()

        with open(name, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    #execfile('tester.py')
    cc = ConeCyl()
    # data from Aluminum_r400_h800
    cc.r2 = 400.
    cc.H = 800.
    cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
    cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
    cc.plyt = 0.125
    #cc.alphadeg = 0.
    cc.r2 = 250.
    cc.H = 510.
    cc.alphadeg = 15.
    # boundary conditions
    cc.bc = 'ss'
    # shape functions
    cc.m1 = 35
    cc.m2 = 35
    cc.n2 = 35
    cc.linear_kinematics = 'clpt_sanders'
    cc.NL_linear_kinematics = 'sanders_numerical'
    #cc.linear_kinematics = 'clpt_donnell'
    #cc.NL_linear_kinematics = 'donnell_numerical'
    # testing numerical integration of k0L
    cc.Fc = 200000
    cc.Fc = 0
    cc.P = -0.1
    #cc.add_SPL(1.)
    cc.nx = 50
    cc.nt = 100
    #cc.lb()
    cc.static()
    cc.ni_num_cores=3
    cc.plot(cc.cs[0])
    if False:
        from dev_utils import regions_conditions
        regions_conditions(cc, 'k0L')

    if False:
        nts = [25, 50, 75, 100, 125, 150, 175, 200]
        from studies import mrs_ni
        cc.ni_method = 'trapz2d'
        print('TRAPZ2D')
        mrs_ni(cc, nts)
        cc.ni_method = 'simps2d'
        print('SIMPS2D')
        mrs_ni(cc, nts)
