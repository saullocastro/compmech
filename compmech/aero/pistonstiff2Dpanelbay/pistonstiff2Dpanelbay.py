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

import modelDB


def load(name):
    if '.AeroPistonStiff2DPanelBay' in name:
        return cPickle.load(open(name, 'rb'))
    else:
        return cPickle.load(open(name + '.AeroPistonStiff2DPanelBay', 'rb'))


class Panel(object):
    def __init__(self, bay):
        self.bay = bay

        # material
        self.mu = None # laminate material density
        self.laminaprop = None
        self.plyt = None
        self.laminaprops = []
        self.stack = []
        self.plyts = []

        self.y1 = None
        self.y2 = None
        self.Nxx = None
        self.Nyy = None
        self.Nxy = None
        self.F = None
        self.k0 = None
        self.kM = None
        self.kG0 = None
        self.kG0_Nxx = None
        self.kG0_Nyy = None
        self.kG0_Nxy = None


    def rebuild(self):
        if not self.laminaprops:
            self.laminaprops = [self.laminaprop for i in self.stack]
        if not self.plyts:
            self.plyts = [self.plyt for i in self.stack]

        # defining load components from force vectors
        if self.laminaprop is None:
            raise ValueError('laminaprop must be defined')


class Stiffener2D(object):
    r"""Stiffener modeled as a 2D shell

    Blade-type of stiffener model using a 2D formulation for the flange and a
    2D formulation for the padup (base)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panel

    Both the flange and the padup are optional. The
    :class:`.AeroPistonStiff2DPanelBay` object may have any number of
    stiffeners.

    Each stiffener has a constant `y` coordinate.


    """
    def __init__(self, mu, panel, ys, bb, bf, bstack, bplyts, blaminaprops,
                 fstack, fplyts, flaminaprops):
        self.panel = panel
        self.m1 = None
        self.n1 = None
        self.mu = mu
        self.ys = ys
        self.bb = bb
        self.hb = 0.
        self.bf = bf
        self.hf = 0.
        self.forces = []

        self.Nxx = None
        self.Nxy = None

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops
        self.fstack = fstack
        self.fplyts = fplyts
        self.flaminaprops = flaminaprops
        self.blam = None
        self.flam = None

        self.k0 = None
        self.kM = None
        self.kG0 = None
        self.kG0_Nxx = None
        self.kG0_Nxy = None

        self.u1txf = 0.
        self.u1rxf = 0.
        self.u2txf = 0.
        self.u2rxf = 0.
        self.v1txf = 0.
        self.v1rxf = 0.
        self.v2txf = 0.
        self.v2rxf = 0.
        self.w1txf = 0.
        self.w1rxf = 1.
        self.w2txf = 0.
        self.w2rxf = 1.

        self.u1tyf = 1.
        self.u1ryf = 1.
        self.u2tyf = 1.
        self.u2ryf = 1.
        self.v1tyf = 1.
        self.v1ryf = 1.
        self.v2tyf = 1.
        self.v2ryf = 1.
        self.w1tyf = 1.
        self.w1ryf = 1.
        self.w2tyf = 1.
        self.w2ryf = 1.

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
            self.db = abs(-h/2.-hb/2.)
            self.blam = laminate.read_stack(self.bstack, plyts=self.bplyts,
                                            laminaprops=self.blaminaprops,
                                            offset=(-h/2.-hb/2.))
            self.hb = hb


class AeroPistonStiff2DPanelBay(object):
    r"""Stiffened Panel for Aeroelastic Studies using the Piston Theory

    Main characteristics:
        - Supports both airflows along x (axis) or y (circumferential).
          Controlled by the parameter ``flow``
        - Can have any number of :class:`.Stiffener` objects along the axial
          direction (`x`), with a fixed `y` position
        - Stiffeners modeled as 2D entities with their own approximation
          functions

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
        self.model = 'clpt_donnell_bardell'

        # approximation series
        self.m = 11
        self.n = 11

        # panels
        self.panels = []

        # stiffeners
        self.stiffeners = []

        # geometry
        self.a = None
        self.b = None
        self.r = None

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
            panel.kG0_Nxx = None
            panel.kG0_Nyy = None
            panel.kG0_Nxy = None

        for s in self.stiffeners:
            s.k0 = None
            s.kM = None
            s.kG0 = None
            s.kG0_Nxx = None
            s.kG0_Nxy = None

        gc.collect()


    def _rebuild(self):
        if not self.name:
            try:
                self.name = os.path.basename(__main__.__file__).split('.py')[0]
            except AttributeError:
                warn('AeroPistonStiff2DPanelBay name unchanged')

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

        if self.a is None:
            raise ValueError('The length a must be specified')

        if self.b is None:
            raise ValueError('The width b must be specified')

        for panel in self.panels:
            panel.rebuild()


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
        num = modelDB.db[self.model]['num']
        num1 = modelDB.db[self.model]['num1']
        self.size = num*self.m*self.n
        for s in self.stiffeners:
            self.size += num1*s.m1*s.n1

        return self.size


    def _default_field(self, xs, ys, gridx, gridy, si=None):
        if xs is None or ys is None:
            xs = linspace(0., self.a, gridx)
            if si is None:
                ys = linspace(0., self.b, gridy)
            else: # stiffener
                ys = linspace(0., self.stiffeners[si].bf, gridy)
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
        model = self.model
        a = self.a
        b = self.b
        r = self.r
        m = self.m
        n = self.n
        num = modelDB.db[self.model]['num']
        num1 = modelDB.db[self.model]['num1']
        size = self.get_size()

        if calc_kA and self.beta is None:
            if self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                self.Mach = 1.0001
            M = self.Mach
            beta = self.rho_air * self.V**2 / (M**2 - 1)**0.5
            gamma = beta*1./(2.*self.r*(M**2 - 1)**0.5)
            ainf = self.speed_sound
            aeromu = beta/(M*ainf)*(M**2 - 2)/(M**2 - 1)
        elif calc_kA and self.beta is not None:
            beta = self.beta
            gamma = self.gamma if self.gamma is not None else 0.
            aeromu = self.aeromu if self.aeromu is not None else 0.
        elif not calc_kA:
            pass
        else:
            raise NotImplementedError('check here')

        k0 = 0.
        kM = 0.
        if not combined_load_case:
            kG0 = 0.
        else:
            kG0_Nxx = 0.
            kG0_Nyy = 0.
            kG0_Nxy = 0.

        mod = modelDB.db[self.model]['linear']

        # contributions from panels
        #TODO summing up coo_matrix objects may be very slow!
        for panel in self.panels:
            y1 = panel.y1
            y2 = panel.y2
            mu = panel.mu

            if panel.stack != []:
                lam = laminate.read_stack(panel.stack, plyts=panel.plyts,
                                          laminaprops=panel.laminaprops)
            else:
                raise ValueError('Invalid panel stacking sequence!')

            F = lam.ABD

            panel.lam = lam
            panel.F = F

            # calculating panel stiffness matrices
            panel.k0 = mod.fk0y1y2(y1, y2, a, b, r, F, m, n,
                     self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                     self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                     self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                     self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                     self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                     self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                     size, 0, 0)

            if False:
                panel.k0edges = fk0edges(m, n, a, b,
                                         panel.kphixBot, panel.kphixTop,
                                         panel.kphiyLeft, panel.kphiyRight)

            if calc_kM:
                h = sum(panel.plyts)
                panel.kM = mod.fkMy1y2(y1, y2, mu, 0., h, a, b, m, n,
                                    self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                                    self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                                    self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                    self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                                    self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                                    self.w1ty, self.w1ry, self.w2ty,
                                    self.w2ry,
                                    size, 0, 0)

            if calc_kG0:
                Nxx = panel.Nxx if panel.Nxx is not None else 0.
                Nyy = panel.Nyy if panel.Nyy is not None else 0.
                Nxy = panel.Nxy if panel.Nxy is not None else 0.

                if not combined_load_case:
                    panel.kG0 = mod.fkG0y1y2(y1, y2, Nxx, Nyy, Nxy, a, b, r,
                                             m, n,
                                    self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                    self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                    size, 0, 0)
                else:
                    panel.kG0_Nxx = mod.fkG0y1y2(y1, y2, Nxx, 0, 0, a, b, r,
                                                 m, n,
                                    self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                    self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                    size, 0, 0)
                    panel.kG0_Nyy = mod.fkG0y1y2(y1, y2, 0, Nyy, 0, a, b, r,
                                                 m, n,
                                    self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                    self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                    size, 0, 0)
                    panel.kG0_Nxy = mod.fkG0y1y2(y1, y2, 0, 0, Nxy, a, b, r,
                                                 m, n,
                                    self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                    self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                    size, 0, 0)

            #TODO summing up coo_matrix objects may be very slow!
            k0 += panel.k0
            if calc_kM:
                kM += panel.kM

            if calc_kG0:
                if not combined_load_case:
                    kG0 += panel.kG0
                else:
                    kG0_Nxx += panel.kG0_Nxx
                    kG0_Nyy += panel.kG0_Nyy
                    kG0_Nxy += panel.kG0_Nxy

        # contributions from stiffeners
        row0 = num*m*n
        col0 = num*m*n
        for i, s in enumerate(self.stiffeners):
            m1 = s.m1
            n1 = s.n1
            bf = s.bf
            if s.blam is not None:
                # stiffener pad-up
                Fsb = s.blam.ABD
                y1 = s.ys - s.bb/2.
                y2 = s.ys + s.bb/2.
                k0 += mod.fk0y1y2(y1, y2, a, b, r, Fsb, m, n,
                                  self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                                  self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                                  self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                  self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                                  self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                                  self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                  size, 0, 0)
                if calc_kM:
                    kM += mod.fkMy1y2(y1, y2, s.mu, s.db, s.hb, a, b, m, n,
                                  self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                                  self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                                  self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                  self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                                  self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                                  self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                  size, 0, 0)

            kt = self.inf
            kr = self.inf
            if s.flam is not None:
                F = s.flam.ABD
                # stiffener flange
                if i > 0:
                    s_1 = self.stiffeners[i-1]
                    row0 += num1*s_1.m1*s_1*n1
                    col0 += num1*s_1.m1*s_1*n1
                k0 += mod.fk0f(a, bf, F, m1, n1,
                               s.u1txf, s.u1rxf, s.u2txf, s.u2rxf,
                               s.v1txf, s.v1rxf, s.v2txf, s.v2rxf,
                               s.w1txf, s.w1rxf, s.w2txf, s.w2rxf,
                               s.u1tyf, s.u1ryf, s.u2tyf, s.u2ryf,
                               s.v1tyf, s.v1ryf, s.v2tyf, s.v2ryf,
                               s.w1tyf, s.w1ryf, s.w2tyf, s.w2ryf,
                               size, row0, col0)
                #k0 += mod.fk0fedges(m1, n1, a, bf, kt, kr, size, row0, col0)

                if calc_kM:
                    kM += mod.fkMf(s.mu, s.hf, a, bf, 0., m1, n1,
                                   s.u1txf, s.u1rxf, s.u2txf, s.u2rxf,
                                   s.v1txf, s.v1rxf, s.v2txf, s.v2rxf,
                                   s.w1txf, s.w1rxf, s.w2txf, s.w2rxf,
                                   s.u1tyf, s.u1ryf, s.u2tyf, s.u2ryf,
                                   s.v1tyf, s.v1ryf, s.v2tyf, s.v2ryf,
                                   s.w1tyf, s.w1ryf, s.w2tyf, s.w2ryf,
                                   size, row0, col0)
                if calc_kG0:
                    Nxx = s.Nxx if s.Nxx is not None else 0.
                    Nxy = s.Nxy if s.Nxy is not None else 0.
                    kG0 += mod.fkG0f(Nxx, 0., Nxy, a, bf, m1, n1,
                                     s.w1txf, s.w1rxf, s.w2txf, s.w2rxf,
                                     s.w1tyf, s.w1ryf, s.w2tyf, s.w2ryf,
                                     size, row0, col0)

                # connectivity between skin-stiffener flange
                k0 += mod.fkCff(kt, kr, a, bf, m1, n1,
                                s.u1txf, s.u1rxf, s.u2txf, s.u2rxf,
                                s.v1txf, s.v1rxf, s.v2txf, s.v2rxf,
                                s.w1txf, s.w1rxf, s.w2txf, s.w2rxf,
                                s.u1tyf, s.u1ryf, s.u2tyf, s.u2ryf,
                                s.v1tyf, s.v1ryf, s.v2tyf, s.v2ryf,
                                s.w1tyf, s.w1ryf, s.w2tyf, s.w2ryf,
                                size, row0, col0)
                k0 += mod.fkCsf(kt, kr, s.ys, a, b, bf, m, n, m1, n1,
                                self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                                self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                                self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                                self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                                self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                s.u1txf, s.u1rxf, s.u2txf, s.u2rxf,
                                s.v1txf, s.v1rxf, s.v2txf, s.v2rxf,
                                s.w1txf, s.w1rxf, s.w2txf, s.w2rxf,
                                s.u1tyf, s.u1ryf, s.u2tyf, s.u2ryf,
                                s.v1tyf, s.v1ryf, s.v2tyf, s.v2ryf,
                                s.w1tyf, s.w1ryf, s.w2tyf, s.w2ryf,
                                size, 0, col0)
                k0 += mod.fkCss(kt, kr, s.ys, a, b, m, n,
                                self.u1tx, self.u1rx, self.u2tx, self.u2rx,
                                self.v1tx, self.v1rx, self.v2tx, self.v2rx,
                                self.w1tx, self.w1rx, self.w2tx, self.w2rx,
                                self.u1ty, self.u1ry, self.u2ty, self.u2ry,
                                self.v1ty, self.v1ry, self.v2ty, self.v2ry,
                                self.w1ty, self.w1ry, self.w2ty, self.w2ry,
                                size, 0, 0)

        # computing global matrices (in the bay domain)
        if calc_kA:
            if self.flow == 'x':
                kA = mod.fkAx(beta, gamma, a, b, m, n, size, 0, 0)
            elif self.flow == 'y':
                kA = mod.fkAy(beta, a, b, m, n, size, 0, 0)
            if fcA is None:
                cA = None
            else:
                cA = mod.fcA(aeromu, a, b, m, n, size, 0, 0)
                cA = cA*(0+1j)


        # performing checks for the stiffness matrices
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

        if False:
            print 'DEBUG, k0edges not taken into account yet'
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

        The following parameters of the ``AeroPistonStiff2DPanelBay`` object
        will affect the linear buckling analysis:

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
            :func:`scipy.sparse.linalg.eigsh` should be used.

        Notes
        -----
        The extracted eigenvalues are stored in the ``eigvals`` parameter
        of the ``AeroPistonStiff2DPanelBay`` object and the `i^{th}`
        eigenvector in the ``eigvecs[:, i-1]`` parameter.

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

        The following parameters of the ``AeroPistonStiff2DPanelBay`` object
        will affect the linear buckling analysis:

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
        of the ``AeroPistonStiff2DPanelBay`` object and the `i^{th}`
        eigenvector in the ``eigvecs[:, i-1]`` parameter.

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


    def uvw_skin(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the
        ``AeroPistonStiff2DPanelBay`` object.

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
        ``AeroPistonStiff2DPanelBay`` object.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ys, xshape, tshape = self._default_field(xs, ys, gridx, gridy)
        m = self.m
        n = self.n
        a = self.a
        b = self.b
        model = self.model

        if c.shape[0] == self.get_size():
            num = modelDB.db[self.model]['num']
            c = c[:num*self.m*self.n]
        else:
            raise ValueError('c is the full vector of Ritz constants')

        fuvw = modelDB.db[model]['commons'].fuvw
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
        ``AeroPistonStiff2DPanelBay`` object.

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
        ``AeroPistonStiff2DPanelBay`` object.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        xs, ys, xshape, tshape = self._default_field(xs, ys, gridx, gridy,
                si=si)
        a = self.a
        model = self.model

        fuvw = modelDB.db[model]['commons'].fuvw

        num = modelDB.db[self.model]['num']
        num1 = modelDB.db[self.model]['num1']
        row0 = num*self.m*self.n
        m1 = self.stiffeners[si].m1
        n1 = self.stiffeners[si].n1
        bf = self.stiffeners[si].bf
        if si > 0:
            s_1 = self.stiffeners[si-1]
            row0 += num1*s_1.m1*s_1*n1
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
            the `name` parameter of the ``AeroPistonStiff2DPanelBay`` object
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
            the `name` parameter of the ``AeroPistonStiff2DPanelBay`` object
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
            m1 = self.stiffeners[si].m1
            n1 = self.stiffeners[si].n1
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
        """Save the ``AeroPistonStiff2DPanelBay`` object using ``cPickle``

        Notes
        -----
        The pickled file will have the name stored in
        ``AeroPistonStiff2DPanelBay.name`` followed by a
        ``'.AeroPistonStiff2DPanelBay'`` extension.

        """
        name = self.name + '.AeroPistonStiff2DPanelBay'
        msg('Saving AeroPistonStiff2DPanelBay to {0}'.format(name))

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
        num = modelDB.db[self.model]['num']
        num1 = modelDB.db[self.model]['num1']
        fg = modelDB.db[self.model]['commons'].fg

        # punctual forces on skin
        size = num*self.m*self.n
        g = np.zeros((3, size), dtype=DOUBLE)
        fext_skin = np.zeros(size, dtype=DOUBLE)
        for i, force in enumerate(self.forces_skin):
            x, y, fx, fy, fz = force
            fg(g, self.m, self.n, x, y, self.a, self.b)

            fpt = np.array([[fx, fy, fz]])
            fext_skin += -fpt.dot(g).ravel()

        fext = fext_skin
        # punctual forces on stiffener
        for s in self.stiffeners:
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
                fext_stiffener += -fpt.dot(g_stiffener).ravel()

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


    def calc_k0(self):
        return self.k0
