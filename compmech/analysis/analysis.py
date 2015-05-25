import numpy as np
from numpy import dot

from compmech.sparse import solve
from compmech.logger import msg
from newton_raphson import _solver_NR
from arc_length import _solver_arc_length

class Analysis(object):
    r"""Class that embodies all data required for linear/non-linear analysis

    The parameters are described in the following tables:

    ========================  ==================================================
    Non-Linear Algorithm      Description
    ========================  ==================================================
    ``NL_method``             ``str``, ``'NR'`` for the Newton-Raphson
                              ``'arc_length'`` for the Arc-Length method
    ``line_search``           ``bool``, activate line_search (for
                              Newton-Raphson methods only)
    ``max_iter_line_search``  ``int``, maximum number of iteration attempts
                              for the line-search algorithm
    ``modified_NR``           ``bool``, activates the modified Newton-Raphson
    ``compute_every_n``       ``int``, if ``modified_NR=True``, the non-linear
                              matrices will be updated at every `n` iterations
    ``kT_initial_state``      ``bool``, tells if the tangent stiffness matrix
                              should be calculated already at the initial
                              state, which is required for example when
                              initial imperfections take place
    ========================  ==================================================

    ==============     =================================================
    Incrementation     Description
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
    Convergence Criteria    Description
    ====================    ============================================
    ``absTOL``              the convergence is achieved when the maximum
                            residual force is smaller than this value
    ``maxNumIter``          maximum number of iteration; if achieved the
                            load increment is bisected
    ``too_slow_TOL``        tolerance that tells if the convergence is too
                            slow
    ====================    ============================================

    Parameters
    ----------
    calc_fext : callable, optional
        Must return a 1-D array containing the external forces. Required for
        linear/non-linear static analysis.
    calc_k0 : callable, optional
        Must return a sparse matrix containing the linear stiffness matrix.
        Required for linear/non-linear static analysis.
    calc_fint : callable, optional
        Must return a 1-D array containing the internal forces. Required for
        non-linear analysis.
    calc_kT : callable, optional
        Must return a sparse matrix containing the tangent stiffness matrix.
        Required for non-linear analysis.

    Returns
    -------
    increments : list
        Each time increment that achieved convergence.
    cs : list
        The solution for each increment.

    """
    __slots__ = ['NL_method', 'line_search', 'max_iter_line_search',
            'modified_NR', 'compute_every_n',
            'kT_initial_state', 'initialInc', 'minInc', 'maxInc', 'absTOL',
            'relTOL', 'maxNumIter', 'too_slow_TOL', 'increments', 'cs',
            'last_analysis', 'calc_fext', 'calc_k0', 'calc_fint', 'calc_kT',
            'calc_k0_bc']


    def __init__(self, calc_fext=None, calc_k0=None, calc_fint=None,
            calc_kT=None, calc_k0_bc=None):
        # non-linear algorithm
        self.NL_method = 'NR'
        self.line_search = True
        self.max_iter_line_search = 20
        self.modified_NR = True
        self.compute_every_n = 6
        self.kT_initial_state = True
        # incrementation
        self.initialInc = 0.3
        self.minInc = 1.e-3
        self.maxInc = 1.
        # convergence criteria
        self.absTOL = 1.e-3
        self.relTOL = 1.e-3
        self.maxNumIter = 30
        self.too_slow_TOL = 0.01

        # required methods
        self.calc_fext = calc_fext
        self.calc_k0 = calc_k0
        self.calc_fint = calc_fint
        self.calc_kT = calc_kT

        # optional methods
        self.calc_k0_bc = calc_k0_bc

        # outputs to be filled
        self.increments = None
        self.cs = None

        # flag telling the last analysis
        self.last_analysis = ''


    def static(self, NLgeom=False, silent=False):
        """General solver for static analyses

        Selects the specific solver based on the ``NL_method`` parameter.

        """
        self.increments = []
        self.cs = []

        if NLgeom:
            self.maxInc = max(self.initialInc, self.maxInc)
            msg('Started Non-Linear Static Analysis', silent=silent)
            if self.NL_method is 'NR':
                _solver_NR(self)
            elif self.NL_method is 'arc_length':
                _solver_arc_length(self)
            else:
                raise ValueError('{0} is an invalid NL_method')

        else:
            msg('Started Linear Static Analysis', silent=silent)
            fext = self.calc_fext()
            k0 = self.calc_k0()

            if self.calc_k0_bc is not None:
                k0_bc = self.calc_k0_bc()
                k0max = k0.max()
                k0 /= k0max

                fextmax = fext.max()
                fext /= fextmax

                k0 = k0 + k0_bc

            c = solve(k0, fext)

            if self.calc_k0_bc is not None:
                k0 *= k0max
                fext *= fextmax

                c /= k0max
                c *= fextmax

            self.cs.append(c)
            self.increments.append(1.)
            msg('Finished Linear Static Analysis', silent=silent)

        self.last_analysis = 'static'

        return self.increments, self.cs

