#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np


cdef extern from 'bardell.h':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.h':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_fxi(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil

DOUBLE = np.float64
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

    cdef long [:] kCBFxcte11r, kCBFxcte11c
    cdef double [:] kCBFxcte11v

    cdef double xicte1
    cdef double g1Auf1Bu, g1Avf1Bv, g1Awf1Bw
    cdef double f1Au, f1Av, f1Aw, f1Awxi, f1Bu, f1Bv, f1Bw, f1Bwxi

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
        for j1 in range(n1):
            for l1 in range(n1):
                g1Auf1Bu = integral_ff(j1, l1, u1ty1, u1ry1, u2ty1, u2ry1, u1ty1, u1ry1, u2ty1, u2ry1)
                g1Avf1Bv = integral_ff(j1, l1, v1ty1, v1ry1, v2ty1, v2ry1, v1ty1, v1ry1, v2ty1, v2ry1)
                g1Awf1Bw = integral_ff(j1, l1, w1ty1, w1ry1, w2ty1, w2ry1, w1ty1, w1ry1, w2ty1, w2ry1)

                for i1 in range(m1):
                    f1Au = calc_f(i1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                    f1Av = calc_f(i1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                    f1Aw = calc_f(i1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                    f1Awxi = calc_fxi(i1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                    for k1 in range(m1):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f1Bu = calc_f(k1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                        f1Bv = calc_f(k1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                        f1Bw = calc_f(k1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                        f1Bwxi = calc_fxi(k1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                        c += 1
                        kCBFxcte11r[c] = row+0
                        kCBFxcte11c[c] = col+0
                        kCBFxcte11v[c] += 0.5*b1*f1Au*f1Bu*g1Auf1Bu*kt
                        c += 1
                        kCBFxcte11r[c] = row+1
                        kCBFxcte11c[c] = col+1
                        kCBFxcte11v[c] += 0.5*b1*f1Av*f1Bv*g1Avf1Bv*kt
                        c += 1
                        kCBFxcte11r[c] = row+2
                        kCBFxcte11c[c] = col+2
                        kCBFxcte11v[c] += 0.5*b1*kt*(f1Aw*f1Bw*g1Awf1Bw + 4*f1Awxi*f1Bwxi*g1Awf1Bw*kr/((a1*a1)*kt))

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
    cdef double a1, a2, b1, b2
    cdef double u1tx1, u1rx1, u2tx1, u2rx1, u1tx2, u1rx2, u2tx2, u2rx2
    cdef double v1tx1, v1rx1, v2tx1, v2rx1, v1tx2, v1rx2, v2tx2, v2rx2
    cdef double w1tx1, w1rx1, w2tx1, w2rx1, w1tx2, w1rx2, w2tx2, w2rx2
    cdef double u1ty1, u1ry1, u2ty1, u2ry1, u1ty2, u1ry2, u2ty2, u2ry2
    cdef double v1ty1, v1ry1, v2ty1, v2ry1, v1ty2, v1ry2, v2ty2, v2ry2
    cdef double w1ty1, w1ry1, w2ty1, w2ry1, w1ty2, w1ry2, w2ty2, w2ry2

    cdef long [:] kCBFxcte12r, kCBFxcte12c
    cdef double [:] kCBFxcte12v

    cdef double xicte1, xicte2
    cdef double g1Auf2Bw, g1Avf2Bv, g1Awf2Bu, g1Awf2Bw
    cdef double f1Au, f1Av, f1Aw, f1Awxi, f2Bu, f2Bv, f2Bw, f2Bwxi

    a1 = p1.a
    a2 = p2.a
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
        for j1 in range(n1):
            for l2 in range(n2):
                g1Auf2Bw = integral_ff(j1, l2, u1ty1, u1ry1, u2ty1, u2ry1, w1ty2, w1ry2, w2ty2, w2ry2)
                g1Avf2Bv = integral_ff(j1, l2, v1ty1, v1ry1, v2ty1, v2ry1, v1ty2, v1ry2, v2ty2, v2ry2)
                g1Awf2Bu = integral_ff(j1, l2, w1ty1, w1ry1, w2ty1, w2ry1, u1ty2, u1ry2, u2ty2, u2ry2)
                g1Awf2Bw = integral_ff(j1, l2, w1ty1, w1ry1, w2ty1, w2ry1, w1ty2, w1ry2, w2ty2, w2ry2)

                for i1 in range(m1):
                    f1Au = calc_f(i1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                    f1Av = calc_f(i1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                    f1Aw = calc_f(i1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                    f1Awxi = calc_fxi(i1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                    for k2 in range(m2):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        f2Bu = calc_f(k2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                        f2Bv = calc_f(k2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                        f2Bw = calc_f(k2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                        f2Bwxi = calc_fxi(k2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                        c += 1
                        kCBFxcte12r[c] = row+0
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -0.5*b1*f1Au*f2Bw*g1Auf2Bw*kt
                        c += 1
                        kCBFxcte12r[c] = row+1
                        kCBFxcte12c[c] = col+1
                        kCBFxcte12v[c] += -0.5*b1*f1Av*f2Bv*g1Avf2Bv*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+0
                        kCBFxcte12v[c] += 0.5*b1*f1Aw*f2Bu*g1Awf2Bu*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -2*b1*f1Awxi*f2Bwxi*g1Awf2Bw*kr/(a1*a2)

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
    cdef double b1, a2
    cdef double u1tx2, u1rx2, u2tx2, u2rx2
    cdef double v1tx2, v1rx2, v2tx2, v2rx2
    cdef double w1tx2, w1rx2, w2tx2, w2rx2
    cdef double u1ty2, u1ry2, u2ty2, u2ry2
    cdef double v1ty2, v1ry2, v2ty2, v2ry2
    cdef double w1ty2, w1ry2, w2ty2, w2ry2

    cdef long [:] kCBFxcte22r, kCBFxcte22c
    cdef double [:] kCBFxcte22v

    cdef double xicte2
    cdef double g2Auf2Bu, g2Avf2Bv, g2Awf2Bw
    cdef double f2Au, f2Bu, f2Av, f2Bv, f2Aw, f2Bw, f2Awxi, f2Bwxi
    b1 = p1.b
    a2 = p2.a
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
        for j2 in range(n2):
            for l2 in range(n2):
                g2Auf2Bu = integral_ff(j2, l2, u1ty2, u1ry2, u2ty2, u2ry2, u1ty2, u1ry2, u2ty2, u2ry2)
                g2Avf2Bv = integral_ff(j2, l2, v1ty2, v1ry2, v2ty2, v2ry2, v1ty2, v1ry2, v2ty2, v2ry2)
                g2Awf2Bw = integral_ff(j2, l2, w1ty2, w1ry2, w2ty2, w2ry2, w1ty2, w1ry2, w2ty2, w2ry2)

                for i2 in range(m2):
                    f2Au = calc_f(i2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                    f2Av = calc_f(i2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                    f2Aw = calc_f(i2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                    f2Awxi = calc_fxi(i2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                    for k2 in range(m2):
                        row = row0 + num*(j2*m2 + i2)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f2Bu = calc_f(k2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                        f2Bv = calc_f(k2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                        f2Bw = calc_f(k2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                        f2Bwxi = calc_fxi(k2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                        c += 1
                        kCBFxcte22r[c] = row+0
                        kCBFxcte22c[c] = col+0
                        kCBFxcte22v[c] += 0.5*b1*f2Au*f2Bu*g2Auf2Bu*kt
                        c += 1
                        kCBFxcte22r[c] = row+1
                        kCBFxcte22c[c] = col+1
                        kCBFxcte22v[c] += 0.5*b1*f2Av*f2Bv*g2Avf2Bv*kt
                        c += 1
                        kCBFxcte22r[c] = row+2
                        kCBFxcte22c[c] = col+2
                        kCBFxcte22v[c] += 0.5*b1*kt*(f2Aw*f2Bw*g2Awf2Bw + 4*f2Awxi*f2Bwxi*g2Awf2Bw*kr/((a2*a2)*kt))

    kCBFxcte22 = coo_matrix((kCBFxcte22v, (kCBFxcte22r, kCBFxcte22c)), shape=(size, size))

    return kCBFxcte22
