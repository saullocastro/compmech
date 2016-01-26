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


cdef extern from 'bardell_functions.h':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_fxi(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil
    double calc_fxixi(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil

cdef extern from 'legendre_gauss_quadrature.h':
    void leggauss_quad(int n, double *points, double* weights) nogil


ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int num = 3


def fkG_num(np.ndarray[cDOUBLE, ndim=1] cs, np.ndarray[cDOUBLE, ndim=2] F,
             double a, double b, double r, double alpharad, int m, int n,
             double u1tx, double u1rx, double u2tx, double u2rx,
             double u1ty, double u1ry, double u2ty, double u2ry,
             double v1tx, double v1rx, double v2tx, double v2rx,
             double v1ty, double v1ry, double v2ty, double v2ry,
             double w1tx, double w1rx, double w2tx, double w2rx,
             double w1ty, double w1ry, double w2ty, double w2ry,
             int size, int row0, int col0, int nx, int ny):
    cdef int i, k, j, l, c, row, col, ptx, pty
    cdef double xi, eta, x, y, alpha

    cdef np.ndarray[cINT, ndim=1] kGr, kGc
    cdef np.ndarray[cDOUBLE, ndim=1] kGv

    cdef double fAu, fAv, fAw, fAuxi, fAvxi, fAwxi, fAwxixi
    cdef double gAu, gAv, gAw, gAueta, gAveta, gAweta, gAwetaeta
    cdef double gBw, gBweta, fBw, fBwxi

    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double Nxx, Nyy, Nxy

    cdef double *css

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weightsxi, weightseta

    A11 = F[0,0]
    A12 = F[0,1]
    A16 = F[0,2]
    A22 = F[1,1]
    A26 = F[1,2]
    A66 = F[2,2]

    B11 = F[0,3]
    B12 = F[0,4]
    B16 = F[0,5]
    B22 = F[1,4]
    B26 = F[1,5]
    B66 = F[2,5]

    fdim = 1*m*m*n*n

    xis = np.zeros(nx, dtype=DOUBLE)
    weightsxi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weightseta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weightsxi[0])
    leggauss_quad(ny, &etas[0], &weightseta[0])

    kGr = np.zeros((fdim,), dtype=INT)
    kGc = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                alpha = weightsxi[ptx]*weightseta[pty]

                # Nxx, Nyy and Nxy

                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.
                for i in range(m):
                    fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                    fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                    fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                    fAuxi = calc_fxi(i, xi, u1tx, u1rx, u2tx, u2rx)
                    fAvxi = calc_fxi(i, xi, v1tx, v1rx, v2tx, v2rx)
                    fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)
                    fAwxixi = calc_fxixi(i, xi, w1tx, w1rx, w2tx, w2rx)
                    for j in range(n):
                        gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                        gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                        gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                        gAueta = calc_fxi(j, eta, u1ty, u1ry, u2ty, u2ry)
                        gAveta = calc_fxi(j, eta, v1ty, v1ry, v2ty, v2ry)
                        gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)
                        gAwetaeta = calc_fxixi(j, eta, w1ty, w1ry, w2ty, w2ry)

                        row = row0 + num*(j*m + i)

                        exx += cs[row+0]*(2/a)*fAuxi*gAu
                        eyy += cs[row+1]*fAv*(2/b)*gAveta
                        gxy += cs[row+0]*fAu*(2/b)*gAueta + cs[row+1]*(2/a)*fAvxi*gAv
                        kxx += -cs[row+2]*(2/a)*(2/a)*fAwxixi*gAw
                        kyy += -cs[row+2]*(2/b)*fAw*gAwetaeta
                        kxy += -2*cs[row+2]*(2/a)*fAwxi*(2/b)*gAweta

                Nxx = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

                # computing kG

                c = -1
                for i in range(m):
                    fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                    fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)

                    for k in range(m):
                        fBw = calc_f(k, xi, w1tx, w1rx, w2tx, w2rx)
                        fBwxi = calc_fxi(k, xi, w1tx, w1rx, w2tx, w2rx)

                        for j in range(n):
                            gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                            gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)

                            for l in range(n):
                                gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)
                                gBweta = calc_fxi(l, eta, w1ty, w1ry, w2ty, w2ry)

                                row = row0 + num*(j*m + i)
                                col = col0 + num*(l*m + k)

                                if row > col:
                                    continue

                                c += 1

                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2

                                kGv[c] = kGv[c] + alpha*(Nxx*b*fAwxi*fBwxi*gAw*gBw/a + Nxy*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta) + Nyy*a*fAw*fBw*gAweta*gBweta/b)

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


