#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np


cdef extern from 'bardell.h':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.h':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_fxi(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int num = 3


def fkCBFxcte11(double kt, double kr, object p1, double xcte1,
          int size, int row0, int col0):
    '''
    Penalty approach calculation to base-flange xcte panel 1 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        Panel() object
    xcte1 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Collumn position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte11 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel p1 position.
    '''

    cdef int i1, j1, k1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double u1tx1, u1rx1, u2tx1, u2rx1
    cdef double v1tx1, v1rx1, v2tx1, v2rx1
    cdef double w1tx1, w1rx1, w2tx1, w2rx1
    cdef double u1ty1, u1ry1, u2ty1, u2ry1
    cdef double v1ty1, v1ry1, v2ty1, v2ry1
    cdef double w1ty1, w1ry1, w2ty1, w2ry1

    cdef np.ndarray[cINT, ndim=1] kCBFxcte11r, kCBFxcte11c
    cdef np.ndarray[cDOUBLE, ndim=1] kCBFxcte11v

    cdef double xicte1
    cdef double f1Auf1Bu, f1Avf1Bv, f1Awf1Bw
    cdef double g1Au, g1Av, g1Aw, g1Awxi, g1Bu, g1Bv, g1Bw, g1Bwxi

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    u1tx1 = p1.u1tx ; u1rx1 = p1.u1rx ; u2tx1 = p1.u2tx ; u2rx1 = p1.u2rx
    v1tx1 = p1.v1tx ; v1rx1 = p1.v1rx ; v2tx1 = p1.v2tx ; v2rx1 = p1.v2rx
    w1tx1 = p1.w1tx ; w1rx1 = p1.w1rx ; w2tx1 = p1.w2tx ; w2rx1 = p1.w2rx
    u1ty1 = p1.u1ty ; u1ry1 = p1.u1ry ; u2ty1 = p1.u2ty ; u2ry1 = p1.u2ry
    v1ty1 = p1.v1ty ; v1ry1 = p1.v1ry ; v2ty1 = p1.v2ty ; v2ry1 = p1.v2ry
    w1ty1 = p1.w1ty ; w1ry1 = p1.w1ry ; w2ty1 = p1.w2ty ; w2ry1 = p1.w2ry

    xicte1 = 2*xcte1/a1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCBFxcte11r = np.zeros((fdim,), dtype=INT)
    kCBFxcte11c = np.zeros((fdim,), dtype=INT)
    kCBFxcte11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):
                f1Auf1Bu = integral_ff(i1, k1, u1ty1, u1ry1, u2ty1, u2ry1, u1ty1, u1ry1, u2ty1, u2ry1)
                f1Avf1Bv = integral_ff(i1, k1, v1ty1, v1ry1, v2ty1, v2ry1, v1ty1, v1ry1, v2ty1, v2ry1)
                f1Awf1Bw = integral_ff(i1, k1, w1ty1, w1ry1, w2ty1, w2ry1, w1ty1, w1ry1, w2ty1, w2ry1)

                for j1 in range(n1):
                    g1Au = calc_f(j1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                    g1Av = calc_f(j1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                    g1Aw = calc_f(j1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                    g1Awxi = calc_fxi(j1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                    for l1 in range(n1):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g1Bu = calc_f(l1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                        g1Bv = calc_f(l1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                        g1Bw = calc_f(l1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                        g1Bwxi = calc_fxi(l1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                        c += 1
                        kCBFxcte11r[c] = row+0
                        kCBFxcte11c[c] = col+0
                        kCBFxcte11v[c] += 0.5*b1*f1Auf1Bu*g1Au*g1Bu*kt
                        c += 1
                        kCBFxcte11r[c] = row+1
                        kCBFxcte11c[c] = col+1
                        kCBFxcte11v[c] += 0.5*b1*f1Avf1Bv*g1Av*g1Bv*kt
                        c += 1
                        kCBFxcte11r[c] = row+2
                        kCBFxcte11c[c] = col+2
                        kCBFxcte11v[c] += 0.5*b1*kt*(f1Awf1Bw*g1Aw*g1Bw + 4*f1Awf1Bw*g1Awxi*g1Bwxi*kr/((a1*a1)*kt))

    kCBFxcte11 = coo_matrix((kCBFxcte11v, (kCBFxcte11r, kCBFxcte11c)), shape=(size, size))

    return kCBFxcte11


def fkCBFxcte12(double kt, double kr, object p1, object p2,
          double xcte1, double xcte2,
          int size, int row0, int col0):
    '''
    Penalty approach calculation to base-flange xcte panel 1 and panel 2 coupling position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        First Panel object
    p2 : Panel
        Second Panel object
    xcte1 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    xcte2 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Collumn position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte12 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel 1 and panel 2 coupling position.
    '''
    cdef int i1, j1, k2, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, b1, b2
    cdef double u1tx1, u1rx1, u2tx1, u2rx1, u1tx2, u1rx2, u2tx2, u2rx2
    cdef double v1tx1, v1rx1, v2tx1, v2rx1, v1tx2, v1rx2, v2tx2, v2rx2
    cdef double w1tx1, w1rx1, w2tx1, w2rx1, w1tx2, w1rx2, w2tx2, w2rx2
    cdef double u1ty1, u1ry1, u2ty1, u2ry1, u1ty2, u1ry2, u2ty2, u2ry2
    cdef double v1ty1, v1ry1, v2ty1, v2ry1, v1ty2, v1ry2, v2ty2, v2ry2
    cdef double w1ty1, w1ry1, w2ty1, w2ry1, w1ty2, w1ry2, w2ty2, w2ry2

    cdef np.ndarray[cINT, ndim=1] kCBFxcte12r, kCBFxcte12c
    cdef np.ndarray[cDOUBLE, ndim=1] kCBFxcte12v

    cdef double xicte1, xicte2
    cdef double f1Auf2Bu, f1Avf2Bw, f1Awf2Bv, f1Awf2Bw
    cdef double g1Au, g1Av, g1Aw, g1Awxi, g2Bu, g2Bv, g2Bw, g2Bwxi

    a1 = p1.a
    b1 = p1.b
    b2 = p2.b
    m1 = p1.m
    n1 = p1.n
    m2 = p2.m
    n2 = p2.n
    u1tx1 = p1.u1tx ; u1rx1 = p1.u1rx ; u2tx1 = p1.u2tx ; u2rx1 = p1.u2rx
    v1tx1 = p1.v1tx ; v1rx1 = p1.v1rx ; v2tx1 = p1.v2tx ; v2rx1 = p1.v2rx
    w1tx1 = p1.w1tx ; w1rx1 = p1.w1rx ; w2tx1 = p1.w2tx ; w2rx1 = p1.w2rx
    u1ty1 = p1.u1ty ; u1ry1 = p1.u1ry ; u2ty1 = p1.u2ty ; u2ry1 = p1.u2ry
    v1ty1 = p1.v1ty ; v1ry1 = p1.v1ry ; v2ty1 = p1.v2ty ; v2ry1 = p1.v2ry
    w1ty1 = p1.w1ty ; w1ry1 = p1.w1ry ; w2ty1 = p1.w2ty ; w2ry1 = p1.w2ry

    u1tx2 = p2.u1tx ; u1rx2 = p2.u1rx ; u2tx2 = p2.u2tx ; u2rx2 = p2.u2rx
    v1tx2 = p2.v1tx ; v1rx2 = p2.v1rx ; v2tx2 = p2.v2tx ; v2rx2 = p2.v2rx
    w1tx2 = p2.w1tx ; w1rx2 = p2.w1rx ; w2tx2 = p2.w2tx ; w2rx2 = p2.w2rx
    u1ty2 = p2.u1ty ; u1ry2 = p2.u1ry ; u2ty2 = p2.u2ty ; u2ry2 = p2.u2ry
    v1ty2 = p2.v1ty ; v1ry2 = p2.v1ry ; v2ty2 = p2.v2ty ; v2ry2 = p2.v2ry
    w1ty2 = p2.w1ty ; w1ry2 = p2.w1ry ; w2ty2 = p2.w2ty ; w2ry2 = p2.w2ry

    xicte1 = 2*xcte1/a1 - 1.
    xicte2 = 2*xcte2/a2 - 1.

    fdim = 4*m1*n1*m2*n2

    kCBFxcte12r = np.zeros((fdim,), dtype=INT)
    kCBFxcte12c = np.zeros((fdim,), dtype=INT)
    kCBFxcte12v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k2 in range(m2):
                f1Auf2Bw = integral_ff(i1, k2, u1ty1, u1ry1, u2ty1, u2ry1, w1ty2, w1ry2, w2ty2, w2ry2)
                f1Avf2Bv = integral_ff(i1, k2, v1ty1, v1ry1, v2ty1, v2ry1, v1ty2, v1ry2, v2ty2, v2ry2)
                f1Awf2Bu = integral_ff(i1, k2, w1ty1, w1ry1, w2ty1, w2ry1, u1ty2, u1ry2, u2ty2, u2ry2)
                f1Awf2Bw = integral_ff(i1, k2, w1ty1, w1ry1, w2ty1, w2ry1, w1ty2, w1ry2, w2ty2, w2ry2)

                for j1 in range(n1):
                    g1Au = calc_f(j1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                    g1Av = calc_f(j1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                    g1Aw = calc_f(j1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                    g1Awxi = calc_fxi(j1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                    for l2 in range(n2):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        g2Bu = calc_f(l2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                        g2Bv = calc_f(l2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                        g2Bw = calc_f(l2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                        g2Bwxi = calc_fxi(l2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                        c += 1
                        kCBFxcte12r[c] = row+0
                        kCBFxcte12c[c] = col+0
                        kCBFxcte12v[c] += -0.5*b1*f1Auf2Bw*g1Au*g2Bw*kt
                        c += 1
                        kCBFxcte12r[c] = row+1
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -0.5*b1*f1Avf2Bv*g1Av*g2Bv*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+1
                        kCBFxcte12v[c] += 0.5*b1*f1Awf2Bu*g1Aw*g2Bu*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -2*b1*f1Awf2Bw*g1Awxi*g2Bwxi*kr/(a1*a2)

    kCBFxcte12 = coo_matrix((kCBFxcte12v, (kCBFxcte12r, kCBFxcte12c)), shape=(size, size))

    return kCBFxcte12


def fkCBFxcte22(double kt, double kr, object p1, object p2,
          double xcte2,
          int size, int row0, int col0):
    '''
    Penalty approach calculation to base-flange xcte panel 2 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        First Panel object
    p2 : Panel
        Second Panel object
    xcte2 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Collumn position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte22 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel p2 position.
    '''
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double a1, b2
    cdef double u1tx2, u1rx2, u2tx2, u2rx2
    cdef double v1tx2, v1rx2, v2tx2, v2rx2
    cdef double w1tx2, w1rx2, w2tx2, w2rx2
    cdef double u1ty2, u1ry2, u2ty2, u2ry2
    cdef double v1ty2, v1ry2, v2ty2, v2ry2
    cdef double w1ty2, w1ry2, w2ty2, w2ry2

    cdef np.ndarray[cINT, ndim=1] kCBFxcte22r, kCBFxcte22c
    cdef np.ndarray[cDOUBLE, ndim=1] kCBFxcte22v

    cdef double xicte2
    cdef double f2Auf2Bu, f2Avf2Bv, f2Awf2Bw
    cdef double g2Au, g2Bu, g2Av, g2Bv, g2Aw, g2Bw, g2Awxi, g2Bwxi
    a1 = p1.a
    b2 = p2.b
    m2 = p2.m
    n2 = p2.n
    u1tx2 = p2.u1tx ; u1rx2 = p2.u1rx ; u2tx2 = p2.u2tx ; u2rx2 = p2.u2rx
    v1tx2 = p2.v1tx ; v1rx2 = p2.v1rx ; v2tx2 = p2.v2tx ; v2rx2 = p2.v2rx
    w1tx2 = p2.w1tx ; w1rx2 = p2.w1rx ; w2tx2 = p2.w2tx ; w2rx2 = p2.w2rx
    u1ty2 = p2.u1ty ; u1ry2 = p2.u1ry ; u2ty2 = p2.u2ty ; u2ry2 = p2.u2ry
    v1ty2 = p2.v1ty ; v1ry2 = p2.v1ry ; v2ty2 = p2.v2ty ; v2ry2 = p2.v2ry
    w1ty2 = p2.w1ty ; w1ry2 = p2.w1ry ; w2ty2 = p2.w2ty ; w2ry2 = p2.w2ry

    xicte2 = 2*xcte2/a2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCBFxcte22r = np.zeros((fdim,), dtype=INT)
    kCBFxcte22c = np.zeros((fdim,), dtype=INT)
    kCBFxcte22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i2 in range(m2):
            for k2 in range(m2):
                f2Auf2Bu = integral_ff(i2, k2, u1ty2, u1ry2, u2ty2, u2ry2, u1ty2, u1ry2, u2ty2, u2ry2)
                f2Avf2Bv = integral_ff(i2, k2, v1ty2, v1ry2, v2ty2, v2ry2, v1ty2, v1ry2, v2ty2, v2ry2)
                f2Awf2Bw = integral_ff(i2, k2, w1ty2, w1ry2, w2ty2, w2ry2, w1ty2, w1ry2, w2ty2, w2ry2)

                for j2 in range(n2):
                    g2Au = calc_f(j2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                    g2Av = calc_f(j2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                    g2Aw = calc_f(j2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                    g2Awxi = calc_fxi(j2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                    for l2 in range(n2):
                        row = row0 + num*(j2*m2 + i2)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g2Bu = calc_f(l2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                        g2Bv = calc_f(l2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                        g2Bw = calc_f(l2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                        g2Bwxi = calc_fxi(l2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                        c += 1
                        kCBFxcte22r[c] = row+0
                        kCBFxcte22c[c] = col+0
                        kCBFxcte22v[c] += 0.5*b1*f2Auf2Bu*g2Au*g2Bu*kt
                        c += 1
                        kCBFxcte22r[c] = row+1
                        kCBFxcte22c[c] = col+1
                        kCBFxcte22v[c] += 0.5*b1*f2Avf2Bv*g2Av*g2Bv*kt
                        c += 1
                        kCBFxcte22r[c] = row+2
                        kCBFxcte22c[c] = col+2
                        kCBFxcte22v[c] += 0.5*b1*kt*(f2Awf2Bw*g2Aw*g2Bw + 4*f2Awf2Bw*g2Awxi*g2Bwxi*kr/((a2*a2)*kt))

    kCBFxcte22 = coo_matrix((kCBFxcte22v, (kCBFxcte22r, kCBFxcte22c)), shape=(size, size))

    return kCBFxcte22
