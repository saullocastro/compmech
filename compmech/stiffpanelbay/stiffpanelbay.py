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
from scipy.linalg import eig, eigh
from numpy import linspace

import compmech.composite.laminate as laminate
from compmech.analysis import Analysis
from compmech.logger import msg, warn
from compmech.constants import DOUBLE
from compmech.sparse import (make_symmetric, make_skew_symmetric,
                             remove_null_cols)
from compmech.panel import Panel, modelDB as panmDB
from compmech.stiffener import BladeStiff2D, modelDB as stiffmDB


def load(name):
    if '.StiffPanelBay' in name:
        return cPickle.load(open(name, 'rb'))
    else:
        return cPickle.load(open(name + '.StiffPanelBay', 'rb'))


class StiffPanelBay(Panel):
    r"""Stiffened Panel Bay

    Can be used for supersonic Aeroelastic studies with the Piston Theory.

    Stiffeners are modeled either with 1D or 2D formulations.

    Main characteristics:

    - Supports both airflows along x (axis) or y (circumferential).
      Controlled by the parameter ``flow``
    - ``bladestiff1ds`` contains the :class:`.BladeStiff1D` stiffeners
    - ``bladestiff2ds`` contains the :class:`.BladeStiff2D` stiffeners

    """
    def __init__(self):
        self.name = ''

        # boundary conditions
        # "inf" is used to define the high stiffnesses (removed dofs)
        #       a high value will cause numerical instabilities
        #TODO use a marker number for self.inf and self.maxinf if the
        #     normalization of edge stiffnesses is adopted
        #     now it is already independent of self.inf and more robust
        self.forces_skin = []
        self.flow = 'x'
        self.bc = None
        self.model = 'cpanel_clt_donnell_bardell'
        self.stack = []
        self.laminaprop = None
        self.laminaprops = []
        self.plyt = None
        self.plyts = []
        self.mu = None

        # approximation series
        self.m = 11
        self.n = 12

        # panels
        self.panels = []

        # stiffeners
        self.bladestiff1ds = []
        self.bladestiff2ds = []

        # geometry
        self.a = None
        self.b = None
        self.r = None
        self.alphadeg = None

        # boundary conditions
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

        # aerodynamic properties for the Piston theory
        self.beta = None
        self.gamma = None
        self.aeromu = None
        self.rho_air = None
        self.speed_sound = None
        self.Mach = None
        self.V = None

        # eigenvalue analysis
        self.num_eigvalues = 25
        self.num_eigvalues_print = 5

        # output queries
        self.out_num_cores = cpu_count()

        # analysis
        self.analysis = Analysis(self.calc_fext, self.calc_k0, None, None)

        # outputs
        self.eigvecs = None
        self.eigvals = None

        self._clear_matrices()


    def _clear_matrices(self):
        self.k0 = None
        self.kT = None
        self.kM = None
        self.kA = None
        self.cA = None
        self.u = None
        self.v = None
        self.w = None
        self.phix = None
        self.phiy = None
        self.Xs = None
        self.Ys = None

        for panel in self.panels:
            panel.k0 = None
            panel.kM = None
            panel.kG0 = None

        for s in self.bladestiff2ds:
            s.k0 = None
            s.kM = None
            s.kG0 = None

        for s in self.bladestiff1ds:
            s.k0 = None
            s.kM = None
            s.kG0 = None

        gc.collect()


    def _rebuild(self):
        if len(self.bladestiff1ds) > 0:
            #TODO
            raise NotImplementedError('#TODO')

        if not self.name:
            try:
                self.name = os.path.basename(__main__.__file__).split('.py')[0]
            except AttributeError:
                warn('StiffPanelBay name unchanged')

        if self.a is None:
            raise ValueError('The length a must be specified')

        if self.b is None:
            raise ValueError('The width b must be specified')

        for p in self.panels:
            p._rebuild()
            if self.model is not None:
                assert self.model == p.model
            else:
                self.model = p.model

        for s in self.bladestiff1ds:
            s._rebuild()

        for s in self.bladestiff2ds:
            s._rebuild()


    def _default_field(self, xs, ys, gridx, gridy, si=None):
        if xs is None or ys is None:
            xs = linspace(0., self.a, gridx)
            if si is None:
                ys = linspace(0., self.b, gridy)
            else: # stiffener
                ys = linspace(0., self.bladestiff2ds[si].bf, gridy)
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


    def get_size(self):
        r"""Calculate the size of the stiffness matrices

        The size of the stiffness matrices can be interpreted as the number of
        rows or columns, recalling that this will be the size of the Ritz
        constants' vector `\{c\}`, the internal force vector `\{F_{int}\}` and
        the external force vector `\{F_{ext}\}`.

        It takes into account the independent degrees of freedom from each of
        the `.Stiffener2D` objects that belong to the current assembly.

        Returns
        -------
        size : int
            The size of the stiffness matrices.

        """
        num = panmDB.db[self.model]['num']
        self.size = num*self.m*self.n

        for s in self.bladestiff2ds:
            num1 = stiffmDB.db[s.model]['num1']
            self.size += num1*s.m1*s.n1

        return self.size


    def add_bladestiff1d(self):
        pass


    def add_panel(self, y1, y2, stack=None, plyt=None, plyts=None,
            laminaprop=None, laminaprops=None, model=None, mu=None, **kwargs):
        p = Panel()
        p.m = self.m
        p.n = self.n
        p.a = self.a
        p.b = self.b
        p.r = self.r
        p.y1 = y1
        p.y2 = y2
        p.d = 0.
        p.model = model if model is not None else self.model
        p.stack = stack if stack is not None else self.stack
        p.plyt = plyt if plyt is not None else self.plyt
        p.plyts = plyts if plyts is not None else self.plyts
        p.laminaprop = laminaprop if laminaprop is not None else self.laminaprop
        p.laminaprops = laminaprops if laminaprops is not None else self.laminaprops
        p.mu = mu if mu is not None else self.mu

        p.u1tx = self.u1tx
        p.u1rx = self.u1rx
        p.u2tx = self.u2tx
        p.u2rx = self.u2rx
        p.v1tx = self.v1tx
        p.v1rx = self.v1rx
        p.v2tx = self.v2tx
        p.v2rx = self.v2rx
        p.w1tx = self.w1tx
        p.w1rx = self.w1rx
        p.w2tx = self.w2tx
        p.w2rx = self.w2rx
        p.u1ty = self.u1ty
        p.u1ry = self.u1ry
        p.u2ty = self.u2ty
        p.u2ry = self.u2ry
        p.v1ty = self.v1ty
        p.v1ry = self.v1ry
        p.v2ty = self.v2ty
        p.v2ry = self.v2ry
        p.w1ty = self.w1ty
        p.w1ry = self.w1ry
        p.w2ty = self.w2ty
        p.w2ry = self.w2ry

        for k, v in kwargs.items():
            setattr(p, k, v)
        self.panels.append(p)


    def calc_k0(self, silent=False):
        self._rebuild()
        msg('Calculating k0... ', level=2, silent=silent)
        model = self.model
        a = self.a
        b = self.b
        r = self.r
        m = self.m
        n = self.n
        num = panmDB.db[self.model]['num']
        size = self.get_size()

        k0 = 0.

        # contributions from panels
        for p in self.panels:
            p.calc_k0(size=size, row0=0, col0=0, silent=True,
                          finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            k0 += p.k0

        # contributions from bladestiff1ds
        for s in self.bladestiff1ds:
            s.calc_k0(size=size, row0=0, col0=0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            k0 += s.k0

        # contributions from bladestiff2ds
        row0 = num*m*n
        col0 = num*m*n
        for i, s in enumerate(self.bladestiff2ds):
            num1 = stiffmDB.db[s.model]['num1']
            if i > 0:
                s_1 = bay.bladestiff2ds[i-1]
                row0 += num1*s_1.m1*s_1.n1
                col0 += num1*s_1.m1*s_1.n1
            s.calc_k0(size=size, row0=row0, col0=col0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            k0 += s.k0

        # performing checks for the stiffness matrices
        assert np.any(np.isnan(k0.data)) == False
        assert np.any(np.isinf(k0.data)) == False
        k0 = csr_matrix(make_symmetric(k0))

        self.k0 = k0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kG0(self, silent=False):
        self._rebuild()
        msg('Calculating kG0... ', level=2, silent=silent)
        model = self.model
        a = self.a
        b = self.b
        m = self.m
        n = self.n
        num = panmDB.db[self.model]['num']
        size = self.get_size()

        kG0 = 0.

        # contributions from panels
        for p in self.panels:
            p.calc_kG0(size=size, row0=0, col0=0, silent=True,
                       finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kG0 += p.kG0

        # contributions from bladestiff1ds
        for s in self.bladestiff1ds:
            s.calc_kG0(size=size, row0=0, col0=0, silent=True,
                       finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kG0 += s.kG0

        # contributions from bladestiff2ds
        row0 = num*m*n
        col0 = num*m*n
        for i, s in enumerate(self.bladestiff2ds):
            num1 = stiffmDB.db[s.model]['num1']
            if i > 0:
                s_1 = bay.bladestiff2ds[i-1]
                row0 += num1*s_1.m1*s_1.n1
                col0 += num1*s_1.m1*s_1.n1
            s.calc_kG0(size=size, row0=row0, col0=col0, silent=True,
                       finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kG0 += s.kG0

        assert np.any((np.isnan(kG0.data) | np.isinf(kG0.data))) == False
        kG0 = csr_matrix(make_symmetric(kG0))

        self.kG0 = kG0

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kM(self, silent=False):
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)
        model = self.model
        a = self.a
        b = self.b
        m = self.m
        n = self.n
        num = panmDB.db[self.model]['num']
        size = self.get_size()

        kM = 0.

        for p in self.panels:
            p.calc_kM(size=size, row0=0, col0=0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kM += p.kM

        # contributions from bladestiff1ds
        for s in self.bladestiff1ds:
            s.calc_kM(size=size, row0=0, col0=0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kM += s.kM

        # contributions from bladestiff2ds
        row0 = num*m*n
        col0 = num*m*n
        for i, s in enumerate(self.bladestiff2ds):
            num1 = stiffmDB.db[s.model]['num1']
            if i > 0:
                s_1 = self.bladestiff2ds[i-1]
                row0 += num1*s_1.m1*s_1.n1
                col0 += num1*s_1.m1*s_1.n1
            s.calc_kM(size=size, row0=row0, col0=col0, silent=True,
                    finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kM += s.kM

        assert np.any(np.isnan(kM.data)) == False
        assert np.any(np.isinf(kM.data)) == False
        kM = csr_matrix(make_symmetric(kM))

        self.kM = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kA(self, silent=False):
        self._rebuild()
        msg('Calculating kA... ', level=2, silent=silent)
        model = self.model
        a = self.a
        b = self.b
        r = self.r
        m = self.m
        n = self.n
        num = panmDB.db[self.model]['num']
        size = self.get_size()

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

        # contributions from panels
        #TODO summing up coo_matrix objects may be slow!
        p = self.panels[0]
        #TODO if the initialization of panel is correct, the line below is
        #     unnecessary
        p.flow = self.flow
        p.calc_kA(silent=silent, finalize=False)

        kA = p.kA

        assert np.any(np.isnan(kA.data)) == False
        assert np.any(np.isinf(kA.data)) == False
        kA = csr_matrix(make_skew_symmetric(kA))

        self.kA = kA

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_cA(self, silent=False):
        self._rebuild()
        msg('Calculating cA... ', level=2, silent=silent)
        model = self.model
        a = self.a
        b = self.b
        r = self.r
        m = self.m
        n = self.n
        num = panmDB.db[self.model]['num']
        size = self.get_size()

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

        # contributions from panels
        p = self.panels[0]
        p.calc_cA(size=size, row0=0, col0=0, silent=silent)
        cA = p.cA

        assert np.any(np.isnan(cA.data)) == False
        assert np.any(np.isinf(cA.data)) == False

        cA = csr_matrix(make_symmetric(cA))

        self.cA = cA

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def uvw_skin(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the
        :class:`.StiffPanelBay` object.

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
        :class:`.StiffPanelBay` object.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ys, xshape, tshape = self._default_field(xs, ys, gridx, gridy)
        m = self.m
        n = self.n
        a = self.a
        b = self.b
        model = self.model

        if c.shape[0] == self.get_size():
            num = panmDB.db[self.model]['num']
            c = c[:num*self.m*self.n]
        else:
            raise ValueError('c is the full vector of Ritz constants')

        fuvw = panmDB.db[model]['field'].fuvw
        us, vs, ws, phixs, phiys = fuvw(c, m, n, a, b, xs, ys,
                self.out_num_cores)

        self.u = us.reshape(xshape)
        self.v = vs.reshape(xshape)
        self.w = ws.reshape(xshape)
        self.phix = phixs.reshape(xshape)
        self.phiy = phiys.reshape(xshape)

        return self.u, self.v, self.w, self.phix, self.phiy


    def uvw_stiffener(self, c, si, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field on a stiffener

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the
        :class:`StiffPanelBay` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        si : int
            Stiffener index.
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
        :class:`StiffPanelBay` object.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ys, xshape, tshape = self._default_field(xs, ys, gridx, gridy,
                si=si)
        a = self.a
        model = self.model

        fuvw = stiffmDB.db[model]['field'].fuvw

        num = panmDB.db[self.model]['num']
        row0 = num*self.m*self.n
        m1 = self.bladestiff2ds[si].m1
        n1 = self.bladestiff2ds[si].n1
        bf = self.bladestiff2ds[si].bf
        if si > 0:
            s_1 = self.bladestiff2ds[si-1]
            num1 = stiffmDB.db[s_1.model]['num1']
            row0 += num1*s_1.m1*s_1.n1
        num1 = stiffmDB.db[self.bladestiff2ds[si].model]['num1']
        row1 = row0 + num1*m1*n1
        if c.shape[0] == self.get_size():
            c = c[row0:row1]
        else:
            raise ValueError('c is the full vector of Ritz constants')

        us, vs, ws, phixs, phiys = fuvw(c, m1, n1, a, bf, xs, ys,
                self.out_num_cores)

        self.u = us.reshape(xshape)
        self.v = vs.reshape(xshape)
        self.w = ws.reshape(xshape)
        self.phix = phixs.reshape(xshape)
        self.phiy = phiys.reshape(xshape)

        return self.u, self.v, self.w, self.phix, self.phiy


    def plot_skin(self, c, invert_y=False, plot_type=1, vec='w',
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
            the `name` parameter of the :class:`StiffPanelBay` object
            will be used.
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
            self.uvw_skin(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
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
                self.uvw_skin(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
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
                ax.set_title('$m, n={0}, {1}$'.format(self.m, self.n))

            elif self.analysis.last_analysis == 'lb':
                ax.set_title(
       r'$m, n={0}, {1}$, $\lambda_{{CR}}={4:1.3e}$'.format(
            self.m, self.n, self.eigvals[0]))

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


    def plot_stiffener(self, c, si, invert_y=False, plot_type=1, vec='w',
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
        si : int
            Stiffener index.
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
            the `name` parameter of the :class:`StiffPanelBay` object
            will be used.
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
            self.uvw_stiffener(c, si=si, xs=xs, ys=ys,
                               gridx=gridx, gridy=gridy)
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
                self.uvw_stiffener(c, si=si, xs=xs, ys=ys,
                                   gridx=gridx, gridy=gridy)
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
            m1 = self.bladestiff2ds[si].m1
            n1 = self.bladestiff2ds[si].n1
            if self.analysis.last_analysis == 'static':
                ax.set_title('$m_1, n_1={0}, {1}$'.format(m1, n1))

            elif self.analysis.last_analysis == 'lb':
                ax.set_title(
       r'$m_1, n_1={0}, {1}$, $\lambda_{{CR}}={4:1.3e}$'.format(
            m1, n1, self.eigvals[0]))

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
        """Save the :class:`StiffPanelBay` object using ``cPickle``

        Notes
        -----
        The pickled file will have the name stored in
        :property:`.StiffPanelBay.name` followed by a
        ``'.StiffPanelBay'`` extension.

        """
        name = self.name + '.StiffPanelBay'
        msg('Saving StiffPanelBay to {0}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)


    def calc_fext(self, silent=False):
        """Calculates the external force vector `\{F_{ext}\}`

        Parameters
        ----------
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        fext : np.ndarray
            The external force vector

        """
        msg('Calculating external forces...', level=2, silent=silent)
        num = panmDB.db[self.model]['num']
        fg = panmDB.db[self.model]['field'].fg

        # punctual forces on skin
        size = num*self.m*self.n
        g = np.zeros((3, size), dtype=DOUBLE)
        fext_skin = np.zeros(size, dtype=DOUBLE)
        for i, force in enumerate(self.forces_skin):
            x, y, fx, fy, fz = force
            fg(g, self.m, self.n, x, y, self.a, self.b)

            fpt = np.array([[fx, fy, fz]])
            fext_skin += fpt.dot(g).ravel()

        fext = fext_skin
        # punctual forces on stiffener
        for s in self.bladestiff2ds:
            num1 = stiffmDB.db[s.model]['num1']
            m1 = s.m1
            n1 = s.n1
            bf = s.bf
            size = num1*m1*n1
            g_stiffener = np.zeros((3, size), dtype=DOUBLE)
            fext_stiffener = np.zeros(size, dtype=DOUBLE)
            for i, force in enumerate(s.forces):
                xf, yf, fx, fy, fz = force
                fg(g_stiffener, m1, n1, xf, yf, self.a, bf)

                fpt = np.array([[fx, fy, fz]])
                fext_stiffener += fpt.dot(g_stiffener).ravel()

            fext = np.concatenate((fext, fext_stiffener))

        msg('finished!', level=2, silent=silent)

        return fext


    def static(self, silent=False):
        """Static analysis for cones and cylinders

        The analysis can be linear or geometrically non-linear. See
        :class:`.Analysis` for further details about the parameters
        controlling the non-linear analysis.

        Parameters
        ----------
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        c : np.ndarray
            The Ritz constants.

        """
        self._rebuild()
        self.calc_linear_matrices(calc_kG0=False, calc_kA=False, calc_kM=False)
        self.analysis.static(NLgeom=False, silent=silent)
        return self.analysis.cs[0]
