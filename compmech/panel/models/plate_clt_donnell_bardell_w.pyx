#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np

from compmech import INT


cdef extern from 'bardell.h':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxifxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxixifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_12.h':
    double integral_ff_12(double eta1, double eta2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxi_12(double eta1, double eta2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxixi_12(double eta1, double eta2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxifxi_12(double eta1, double eta2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxifxixi_12(double eta1, double eta2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxixifxixi_12(double eta1, double eta2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

DOUBLE = np.float64

cdef int num = 1


def fk0(object panel, int size, int row0, int col0):
    cdef double a, b
    cdef double [:, ::1] F
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry
    cdef int i, j, k, l, c, row, col
    cdef double D11, D12, D16, D22, D26, D66

    cdef long [:] k0r, k0c
    cdef double [:] k0v

    cdef double fAwxifBvxi, fAwxixifBwxixi, fAwfBwxixi, fAwxixifBw,
    cdef double fAwxifBwxixi, fAwxixifBwxi, fAwfBw, fAwfBwxi, fAwxifBw,
    cdef double fAwxifBwxi
    cdef double gAwetaetagBwetaeta, gAwgBwetaeta, gAwetaetagBw,
    cdef double gAwetagBwetaeta, gAwetaetagBweta, gAwgBw, gAwgBweta, gAwetagBw,
    cdef double gAwetagBweta

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    F = panel.lam.ABD
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*m*n*n

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        D11 = F[3,3]
        D12 = F[3,4]
        D16 = F[3,5]
        D22 = F[4,4]
        D26 = F[4,5]
        D66 = F[5,5]

        # k0
        c = -1
        for i in range(m):
            for k in range(m):

                fAwxixifBwxixi = integral_fxixifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwfBwxixi = integral_ffxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwxixifBw = integral_ffxixi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwxifBwxixi = integral_fxifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwxixifBwxi = integral_fxifxixi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwfBwxi = integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwxifBw = integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j in range(n):
                    for l in range(n):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAwetaetagBwetaeta = integral_fxixifxixi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwgBwetaeta = integral_ffxixi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwetaetagBw = integral_ffxixi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwetagBwetaeta = integral_fxifxixi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwetaetagBweta = integral_fxifxixi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwgBw = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwgBweta = integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwetagBw = integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwetagBweta = integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*D12*(fAwfBwxixi*gAwetaetagBw + fAwxixifBw*gAwgBwetaeta)/(a*b) + 8*D16*(fAwxifBwxixi*gAwetagBw + fAwxixifBwxi*gAwgBweta)/(a*a) + 4*D22*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*D26*(fAwfBwxi*gAwetaetagBweta + fAwxifBw*gAwetagBwetaeta)/(b*b) + 16*D66*fAwxifBwxi*gAwetagBweta/(a*b)

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0y1y2(double y1, double y2, object panel, int size, int row0, int col0):
    cdef double a, b
    cdef double [:, ::1] F
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, k, l, row, col, c
    cdef double eta1, eta2
    cdef double D11, D12, D16, D22, D26, D66

    cdef long [:] k0y1y2r, k0y1y2c
    cdef double [:] k0y1y2v

    cdef double fAwxifBvxi, fAwxixifBwxixi, fAwfBwxixi, fAwxixifBw,
    cdef double fAwxifBwxixi, fAwxixifBwxi, fAwfBw, fAwfBwxi, fAwxifBw,
    cdef double fAwxifBwxi
    cdef double gAwetaetagBwetaeta, gAwgBwetaeta, gAwetaetagBw,
    cdef double gAwetagBwetaeta, gAwetaetagBweta, gAwgBw, gAwgBweta, gAwetagBw,
    cdef double gAwetagBweta

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    F = panel.lam.ABD
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*m*n*n

    k0y1y2r = np.zeros((fdim,), dtype=INT)
    k0y1y2c = np.zeros((fdim,), dtype=INT)
    k0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        D11 = F[3,3]
        D12 = F[3,4]
        D16 = F[3,5]
        D22 = F[4,4]
        D26 = F[4,5]
        D66 = F[5,5]

        eta1 = 2*y1/b - 1.
        eta2 = 2*y2/b - 1.

        # k0y1y2
        c = -1
        for j in range(n):
            for l in range(n):

                gAwetaetagBwetaeta = integral_fxixifxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwgBwetaeta = integral_ffxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetaetagBw = integral_ffxixi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetagBwetaeta = integral_fxifxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetaetagBweta = integral_fxifxixi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwgBw = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwgBweta = integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetagBw = integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetagBweta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                for i in range(m):
                    for k in range(m):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fAwxixifBwxixi = integral_fxixifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwfBwxixi = integral_ffxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxixifBw = integral_ffxixi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxifBwxixi = integral_fxifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxixifBwxi = integral_fxifxixi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwfBwxi = integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxifBw = integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*D12*(fAwfBwxixi*gAwetaetagBw + fAwxixifBw*gAwgBwetaeta)/(a*b) + 8*D16*(fAwxifBwxixi*gAwetagBw + fAwxixifBwxi*gAwgBweta)/(a*a) + 4*D22*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*D26*(fAwfBwxi*gAwetaetagBweta + fAwxifBw*gAwetagBwetaeta)/(b*b) + 16*D66*fAwxifBwxi*gAwetagBweta/(a*b)

    k0y1y2 = coo_matrix((k0y1y2v, (k0y1y2r, k0y1y2c)), shape=(size, size))

    return k0y1y2


def fkG0(double Nxx, double Nyy, double Nxy, object panel,
         int size, int row0, int col0):
    cdef double a, b
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry
    
    cdef int i, k, j, l, c, row, col

    cdef long [:] kG0r, kG0c
    cdef double [:] kG0v

    cdef double fAwxifBwxi, fAwfBwxi, fAwxifBw, fAwfBw
    cdef double gAwetagBweta, gAwgBweta, gAwetagBw, gAwgBw

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*m*n*n

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kG0
        c = -1
        for i in range(m):
            for k in range(m):

                fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwfBwxi = integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwxifBw = integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j in range(n):
                    for l in range(n):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAwetagBw = integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwgBw = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwgBweta = integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwetagBweta = integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        kG0r[c] = row+0
                        kG0c[c] = col+0
                        kG0v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkG0y1y2(double y1, double y2, double Nxx, double Nyy, double Nxy,
             object panel, int size, int row0, int col0):
    cdef double a, b
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef double eta1, eta2

    cdef long [:] kG0y1y2r, kG0y1y2c
    cdef double [:] kG0y1y2v

    cdef double fAwxifBwxi, fAwfBwxi, fAwxifBw, fAwfBw
    cdef double gAwetagBweta, gAwgBweta, gAwetagBw, gAwgBw

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*n*m*n

    kG0y1y2r = np.zeros((fdim,), dtype=INT)
    kG0y1y2c = np.zeros((fdim,), dtype=INT)
    kG0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        eta1 = 2*y1/b - 1.
        eta2 = 2*y2/b - 1.

        # kG0y1y2
        c = -1
        for j in range(n):
            for l in range(n):

                gAwetagBw = integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwgBw = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwgBweta = integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetagBweta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                for i in range(m):
                    for k in range(m):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwfBwxi = integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxifBw = integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                        c += 1
                        kG0y1y2r[c] = row+0
                        kG0y1y2c[c] = col+0
                        kG0y1y2v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

    kG0y1y2 = coo_matrix((kG0y1y2v, (kG0y1y2r, kG0y1y2c)), shape=(size, size))

    return kG0y1y2


def fkM(double d, object panel, int size, int row0, int col0):
    cdef double a, b, mu, h
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col

    cdef long [:] kMr, kMc
    cdef double [:] kMv

    cdef double fAwfBw, fAwxifBwxi
    cdef double gAwgBw, gAwetagBweta

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    mu = panel.mu
    h = sum(panel.plyts)
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*n*m*n

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kM
        c = -1
        for i in range(m):
            for k in range(m):

                fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j in range(n):
                    for l in range(n):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAwgBw = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                        gAwetagBweta = integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+0
                        kMv[c] += 0.25*a*b*h*mu*(fAwfBw*gAwgBw + 4*fAwfBw*gAwetagBweta*((d*d) + 0.0833333333333333*(h*h))/(b*b) + 4*fAwxifBwxi*gAwgBw*((d*d) + 0.0833333333333333*(h*h))/(a*a))

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkMy1y2(double y1, double y2, double d, object panel,
            int size, int row0, int col0):
    cdef double a, b, mu, h
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef double eta1, eta2

    cdef long [:] kMy1y2r, kMy1y2c
    cdef double [:] kMy1y2v

    cdef double fAwfBw, fAwxifBwxi
    cdef double gAwgBw, gAwetagBweta

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    mu = panel.mu
    h = sum(panel.plyts)
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*n*m*n

    kMy1y2r = np.zeros((fdim,), dtype=INT)
    kMy1y2c = np.zeros((fdim,), dtype=INT)
    kMy1y2v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        eta1 = 2*y1/b - 1.
        eta2 = 2*y2/b - 1.

        # kMy1y2
        c = -1
        for j in range(n):
            for l in range(n):

                gAwgBw = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetagBweta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                for i in range(m):
                    for k in range(m):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += 0.25*a*b*h*mu*(fAwfBw*gAwgBw + 4*fAwfBw*gAwetagBweta*((d*d) + 0.0833333333333333*(h*h))/(b*b) + 4*fAwxifBwxi*gAwgBw*((d*d) + 0.0833333333333333*(h*h))/(a*a))

    kMy1y2 = coo_matrix((kMy1y2v, (kMy1y2r, kMy1y2c)), shape=(size, size))

    return kMy1y2


def fkAx(double beta, double gamma, object panel,
         int size, int row0, int col0):
    cdef double a, b
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef long [:] kAxr, kAxc
    cdef double [:] kAxv

    cdef double fAwxifBw, gAwgBw

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*n*m*n

    kAxr = np.zeros((fdim,), dtype=INT)
    kAxc = np.zeros((fdim,), dtype=INT)
    kAxv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kAx
        c = -1
        for i in range(m):
            for k in range(m):

                fAwxifBw = integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j in range(n):
                    for l in range(n):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAwgBw = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        kAxr[c] = row+0
                        kAxc[c] = col+0
                        kAxv[c] += -0.5*b*beta*fAwxifBw*gAwgBw

    kAx = coo_matrix((kAxv, (kAxr, kAxc)), shape=(size, size))

    return kAx


def fkAy(double beta, object panel, int size, int row0, int col0):
    cdef double a, b
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef long [:] kAyr, kAyc
    cdef double [:] kAyv

    cdef double fAwfBw, gAwetagBw

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*n*m*n

    kAyr = np.zeros((fdim,), dtype=INT)
    kAyc = np.zeros((fdim,), dtype=INT)
    kAyv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kAy
        c = -1
        for i in range(m):
            for k in range(m):

                fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j in range(n):
                    for l in range(n):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAwetagBw = integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        kAyr[c] = row+0
                        kAyc[c] = col+0
                        kAyv[c] += -0.5*a*beta*fAwfBw*gAwetagBw

    kAy = coo_matrix((kAyv, (kAyr, kAyc)), shape=(size, size))

    return kAy


def fcA(double aeromu, object panel, int size, int row0, int col0):
    cdef double a, b
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef long [:] cAr, cAc
    cdef double [:] cAv

    cdef double fAwfBw, gAwgBw

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*n*m*n

    cAr = np.zeros((fdim,), dtype=INT)
    cAc = np.zeros((fdim,), dtype=INT)
    cAv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # cA
        c = -1
        for i in range(m):
            for k in range(m):

                fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j in range(n):
                    for l in range(n):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAwgBw = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        cAr[c] = row+0
                        cAc[c] = col+0
                        cAv[c] += -0.25*a*aeromu*b*fAwfBw*gAwgBw

    cA = coo_matrix((cAv, (cAr, cAc)), shape=(size, size))

    return cA
