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


def fkL_num(np.ndarray[cDOUBLE, ndim=1] cs, object Finput, object panel,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double a, b, ra, rb
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, k, l, c, row, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef np.ndarray[cINT, ndim=1] kLr, kLc
    cdef np.ndarray[cDOUBLE, ndim=1] kLv

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double fBu, fBuxi, fBv, fBvxi, fBw, fBwxi, fBwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta
    cdef double gBu, gBueta, gBv, gBveta, gBw, gBweta, gBwetaeta
    cdef double xi, eta, weight
    cdef double wxi, weta

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weightsxi, weightseta

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef np.ndarray[cDOUBLE, ndim=4] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.asarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = np.ascontiguousarray(Finput)
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        Finput = np.ascontiguousarray(Finput)
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    bbot = panel.b
    ra = panel.ra
    rb = panel.rb
    alpharad = panel.alpharad
    m = panel.m
    n = panel.n
    u1tx = panel.u1tx; u1rx = panel.u1rx; u2tx = panel.u2tx; u2rx = panel.u2rx
    v1tx = panel.v1tx; v1rx = panel.v1rx; v2tx = panel.v2tx; v2rx = panel.v2rx
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    u1ty = panel.u1ty; u1ry = panel.u1ry; u2ty = panel.u2ty; u2ry = panel.u2ry
    v1ty = panel.v1ty; v1ry = panel.v1ry; v2ty = panel.v2ty; v2ry = panel.v2ry
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 9*m*m*n*n

    xis = np.zeros(nx, dtype=DOUBLE)
    weightsxi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weightseta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weightsxi[0])
    leggauss_quad(ny, &etas[0], &weightseta[0])

    kLr = np.zeros((fdim,), dtype=INT)
    kLc = np.zeros((fdim,), dtype=INT)
    kLv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        sina = sin(alpharad)
        cosa = cos(alpharad)

        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                weight = weightsxi[ptx]*weightseta[pty]

                x = a*(xi + 1)/2

                r = rbot - sina*(x)

                b = r*bbot/rbot

                wxi = 0
                weta = 0
                if NLgeom == 1:
                    for j in range(n):
                        #TODO put these in a lookup vector
                        gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                        gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)
                        for i in range(m):
                            #TODO put these in a lookup vector
                            fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                            fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)

                            col = col0 + num*(j*m + i)

                            wxi += cs[col+2]*fAwxi*gAw
                            weta += cs[col+2]*fAw*gAweta

                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            #TODO could assume symmetry
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                A11 = F[0*6 + 0]
                A12 = F[0*6 + 1]
                A16 = F[0*6 + 2]
                A22 = F[1*6 + 1]
                A26 = F[1*6 + 2]
                A66 = F[2*6 + 2]

                B11 = F[0*6 + 3]
                B12 = F[0*6 + 4]
                B16 = F[0*6 + 5]
                B22 = F[1*6 + 4]
                B26 = F[1*6 + 5]
                B66 = F[2*6 + 5]

                D11 = F[3*6 + 3]
                D12 = F[3*6 + 4]
                D16 = F[3*6 + 5]
                D22 = F[4*6 + 4]
                D26 = F[4*6 + 5]
                D66 = F[5*6 + 5]

                # kL
                c = -1
                for i in range(m):
                    fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                    fAuxi = calc_fxi(i, xi, u1tx, u1rx, u2tx, u2rx)
                    fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                    fAvxi = calc_fxi(i, xi, v1tx, v1rx, v2tx, v2rx)
                    fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                    fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)
                    fAwxixi = calc_fxixi(i, xi, w1tx, w1rx, w2tx, w2rx)

                    for k in range(m):
                        fBu = calc_f(k, xi, u1tx, u1rx, u2tx, u2rx)
                        fBuxi = calc_fxi(k, xi, u1tx, u1rx, u2tx, u2rx)
                        fBv = calc_f(k, xi, v1tx, v1rx, v2tx, v2rx)
                        fBvxi = calc_fxi(k, xi, v1tx, v1rx, v2tx, v2rx)
                        fBw = calc_f(k, xi, w1tx, w1rx, w2tx, w2rx)
                        fBwxi = calc_fxi(k, xi, w1tx, w1rx, w2tx, w2rx)
                        fBwxixi = calc_fxixi(k, xi, w1tx, w1rx, w2tx, w2rx)

                        for j in range(n):
                            gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                            gAueta = calc_fxi(j, eta, u1ty, u1ry, u2ty, u2ry)
                            gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                            gAveta = calc_fxi(j, eta, v1ty, v1ry, v2ty, v2ry)
                            gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                            gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)
                            gAwetaeta = calc_fxixi(j, eta, w1ty, w1ry, w2ty, w2ry)

                            for l in range(n):

                                row = row0 + num*(j*m + i)
                                col = col0 + num*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = calc_f(l, eta, u1ty, u1ry, u2ty, u2ry)
                                gBueta = calc_fxi(l, eta, u1ty, u1ry, u2ty, u2ry)
                                gBv = calc_f(l, eta, v1ty, v1ry, v2ty, v2ry)
                                gBveta = calc_fxi(l, eta, v1ty, v1ry, v2ty, v2ry)
                                gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)
                                gBweta = calc_fxi(l, eta, w1ty, w1ry, w2ty, w2ry)
                                gBwetaeta = calc_fxixi(l, eta, w1ty, w1ry, w2ty, w2ry)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+0
                                    kLc[c] = col+0
                                kLv[c] += A11*b*fAuxifBuxi*gAugBu/a + A12*(0.5*b*fAufBuxi*gAugBu*sina/r + 0.5*b*fAuxifBu*gAugBu*sina/r) + A16*(fAufBuxi*gAuetagBu + fAuxifBu*gAugBueta) + 0.25*A22*a*b*fAufBu*gAugBu*(sina*sina)/(r*r) + A26*(0.5*a*fAufBu*gAugBueta*sina/r + 0.5*a*fAufBu*gAuetagBu*sina/r) + A66*a*fAufBu*gAuetagBueta/b
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+0
                                    kLc[c] = col+1
                                kLv[c] += A12*fAuxifBv*gAugBveta + A16*(-0.5*b*fAuxifBv*gAugBv*sina/r + b*fAuxifBvxi*gAugBv/a) + 0.5*A22*a*fAufBv*gAugBveta*sina/r + A26*(-0.25*a*b*fAufBv*gAugBv*(sina*sina)/(r*r) + a*fAufBv*gAuetagBveta/b + 0.5*b*fAufBvxi*gAugBv*sina/r) + A66*(-0.5*a*fAufBv*gAuetagBv*sina/r + fAufBvxi*gAuetagBv)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+0
                                    kLc[c] = col+2
                                kLv[c] += 0.5*A12*b*cosa*fAuxifBw*gAugBw/r + 0.25*A22*a*b*cosa*fAufBw*gAugBw*sina/(r*r) + 0.5*A26*a*cosa*fAufBw*gAuetagBw/r - 2*B11*b*fAuxifBwxixi*gAugBw/(a*a) + B12*(-2*fAuxifBw*gAugBwetaeta/b - b*fAufBwxixi*gAugBw*sina/(a*r) - b*fAuxifBwxi*gAugBw*sina/(a*r)) + B16*(fAuxifBw*gAugBweta*sina/r - 2*fAufBwxixi*gAuetagBw/a - 4*fAuxifBwxi*gAugBweta/a) + B22*(-a*fAufBw*gAugBwetaeta*sina/(b*r) - 0.5*b*fAufBwxi*gAugBw*(sina*sina)/(r*r)) + B26*(0.5*a*fAufBw*gAugBweta*(sina*sina)/(r*r) - 2*a*fAufBw*gAuetagBwetaeta/(b*b) - 2*fAufBwxi*gAugBweta*sina/r - fAufBwxi*gAuetagBw*sina/r) + B66*(a*fAufBw*gAuetagBweta*sina/(b*r) - 4*fAufBwxi*gAuetagBweta/b)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+1
                                    kLc[c] = col+0
                                kLv[c] += A12*fAvfBuxi*gAvetagBu + A16*(-0.5*b*fAvfBuxi*gAvgBu*sina/r + b*fAvxifBuxi*gAvgBu/a) + 0.5*A22*a*fAvfBu*gAvetagBu*sina/r + A26*(-0.25*a*b*fAvfBu*gAvgBu*(sina*sina)/(r*r) + a*fAvfBu*gAvetagBueta/b + 0.5*b*fAvxifBu*gAvgBu*sina/r) + A66*(-0.5*a*fAvfBu*gAvgBueta*sina/r + fAvxifBu*gAvgBueta)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+1
                                    kLc[c] = col+1
                                kLv[c] += A22*a*fAvfBv*gAvetagBveta/b + A26*(-0.5*a*fAvfBv*gAvgBveta*sina/r - 0.5*a*fAvfBv*gAvetagBv*sina/r + fAvfBvxi*gAvetagBv + fAvxifBv*gAvgBveta) + A66*(0.25*a*b*fAvfBv*gAvgBv*(sina*sina)/(r*r) - 0.5*b*fAvfBvxi*gAvgBv*sina/r - 0.5*b*fAvxifBv*gAvgBv*sina/r + b*fAvxifBvxi*gAvgBv/a)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+1
                                    kLc[c] = col+2
                                kLv[c] += 0.5*A22*a*cosa*fAvfBw*gAvetagBw/r + A26*(-0.25*a*b*cosa*fAvfBw*gAvgBw*sina/(r*r) + 0.5*b*cosa*fAvxifBw*gAvgBw/r) - 2*B12*fAvfBwxixi*gAvetagBw/a + B16*(b*fAvfBwxixi*gAvgBw*sina/(a*r) - 2*b*fAvxifBwxixi*gAvgBw/(a*a)) + B22*(-2*a*fAvfBw*gAvetagBwetaeta/(b*b) - fAvfBwxi*gAvetagBw*sina/r) + B26*(a*fAvfBw*gAvgBwetaeta*sina/(b*r) + a*fAvfBw*gAvetagBweta*sina/(b*r) + 0.5*b*fAvfBwxi*gAvgBw*(sina*sina)/(r*r) - 4*fAvfBwxi*gAvetagBweta/b - 2*fAvxifBw*gAvgBwetaeta/b - b*fAvxifBwxi*gAvgBw*sina/(a*r)) + B66*(-0.5*a*fAvfBw*gAvgBweta*(sina*sina)/(r*r) + 2*fAvfBwxi*gAvgBweta*sina/r + fAvxifBw*gAvgBweta*sina/r - 4*fAvxifBwxi*gAvgBweta/a)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+2
                                    kLc[c] = col+0
                                kLv[c] += 0.5*A12*b*cosa*fAwfBuxi*gAwgBu/r + 0.25*A22*a*b*cosa*fAwfBu*gAwgBu*sina/(r*r) + 0.5*A26*a*cosa*fAwfBu*gAwgBueta/r - 2*B11*b*fAwxixifBuxi*gAwgBu/(a*a) + B12*(-2*fAwfBuxi*gAwetaetagBu/b - b*fAwxifBuxi*gAwgBu*sina/(a*r) - b*fAwxixifBu*gAwgBu*sina/(a*r)) + B16*(fAwfBuxi*gAwetagBu*sina/r - 4*fAwxifBuxi*gAwetagBu/a - 2*fAwxixifBu*gAwgBueta/a) + B22*(-a*fAwfBu*gAwetaetagBu*sina/(b*r) - 0.5*b*fAwxifBu*gAwgBu*(sina*sina)/(r*r)) + B26*(0.5*a*fAwfBu*gAwetagBu*(sina*sina)/(r*r) - 2*a*fAwfBu*gAwetaetagBueta/(b*b) - fAwxifBu*gAwgBueta*sina/r - 2*fAwxifBu*gAwetagBu*sina/r) + B66*(a*fAwfBu*gAwetagBueta*sina/(b*r) - 4*fAwxifBu*gAwetagBueta/b)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+2
                                    kLc[c] = col+1
                                kLv[c] += 0.5*A22*a*cosa*fAwfBv*gAwgBveta/r + A26*(-0.25*a*b*cosa*fAwfBv*gAwgBv*sina/(r*r) + 0.5*b*cosa*fAwfBvxi*gAwgBv/r) - 2*B12*fAwxixifBv*gAwgBveta/a + B16*(b*fAwxixifBv*gAwgBv*sina/(a*r) - 2*b*fAwxixifBvxi*gAwgBv/(a*a)) + B22*(-2*a*fAwfBv*gAwetaetagBveta/(b*b) - fAwxifBv*gAwgBveta*sina/r) + B26*(a*fAwfBv*gAwetagBveta*sina/(b*r) + a*fAwfBv*gAwetaetagBv*sina/(b*r) + 0.5*b*fAwxifBv*gAwgBv*(sina*sina)/(r*r) - 2*fAwfBvxi*gAwetaetagBv/b - 4*fAwxifBv*gAwetagBveta/b - b*fAwxifBvxi*gAwgBv*sina/(a*r)) + B66*(-0.5*a*fAwfBv*gAwetagBv*(sina*sina)/(r*r) + fAwfBvxi*gAwetagBv*sina/r + 2*fAwxifBv*gAwetagBv*sina/r - 4*fAwxifBvxi*gAwetagBv/a)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+2
                                    kLc[c] = col+2
                                kLv[c] += 0.25*A22*a*b*(cosa*cosa)*fAwfBw*gAwgBw/(r*r) + B12*(-b*cosa*fAwfBwxixi*gAwgBw/(a*r) - b*cosa*fAwxixifBw*gAwgBw/(a*r)) + B22*(-a*cosa*fAwfBw*gAwgBwetaeta/(b*r) - a*cosa*fAwfBw*gAwetaetagBw/(b*r) - 0.5*b*cosa*fAwfBwxi*gAwgBw*sina/(r*r) - 0.5*b*cosa*fAwxifBw*gAwgBw*sina/(r*r)) + B26*(0.5*a*cosa*fAwfBw*gAwgBweta*sina/(r*r) + 0.5*a*cosa*fAwfBw*gAwetagBw*sina/(r*r) - 2*cosa*fAwfBwxi*gAwgBweta/r - 2*cosa*fAwxifBw*gAwetagBw/r) + 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + D12*(4*fAwfBwxixi*gAwetaetagBw/(a*b) + 4*fAwxixifBw*gAwgBwetaeta/(a*b) + 2*b*fAwxifBwxixi*gAwgBw*sina/((a*a)*r) + 2*b*fAwxixifBwxi*gAwgBw*sina/((a*a)*r)) + D16*(-2*fAwfBwxixi*gAwetagBw*sina/(a*r) - 2*fAwxixifBw*gAwgBweta*sina/(a*r) + 8*fAwxifBwxixi*gAwetagBw/(a*a) + 8*fAwxixifBwxi*gAwgBweta/(a*a)) + D22*(4*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 2*fAwfBwxi*gAwetaetagBw*sina/(b*r) + 2*fAwxifBw*gAwgBwetaeta*sina/(b*r) + b*fAwxifBwxi*gAwgBw*(sina*sina)/(a*(r*r))) + D26*(-2*a*fAwfBw*gAwetagBwetaeta*sina/((b*b)*r) - 2*a*fAwfBw*gAwetaetagBweta*sina/((b*b)*r) - fAwfBwxi*gAwetagBw*(sina*sina)/(r*r) - fAwxifBw*gAwgBweta*(sina*sina)/(r*r) + 8*fAwfBwxi*gAwetaetagBweta/(b*b) + 8*fAwxifBw*gAwetagBwetaeta/(b*b) + 4*fAwxifBwxi*gAwgBweta*sina/(a*r) + 4*fAwxifBwxi*gAwetagBw*sina/(a*r)) + D66*(a*fAwfBw*gAwetagBweta*(sina*sina)/(b*(r*r)) - 4*fAwfBwxi*gAwetagBweta*sina/(b*r) - 4*fAwxifBw*gAwetagBweta*sina/(b*r) + 16*fAwxifBwxi*gAwetagBweta/(a*b))




    kL = coo_matrix((kLv, (kLr, kLc)), shape=(size, size))

    return kL


def fkG_num(np.ndarray[cDOUBLE, ndim=1] cs, object Finput, object panel,
            int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double a, b, r
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col, ptx, pty
    cdef double xi, eta, x, y, weight

    cdef np.ndarray[cINT, ndim=1] kGr, kGc
    cdef np.ndarray[cDOUBLE, ndim=1] kGv

    cdef double fAu, fAv, fAw, fAuxi, fAvxi, fAwxi, fAwxixi
    cdef double gAu, gAv, gAw, gAueta, gAveta, gAweta, gAwetaeta
    cdef double gBw, gBweta, fBw, fBwxi

    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double wxi, weta, Nxx, Nyy, Nxy

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weightsxi, weightseta

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef np.ndarray[cDOUBLE, ndim=4] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.asarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = np.ascontiguousarray(Finput)
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        Finput = np.ascontiguousarray(Finput)
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    r = panel.r
    m = panel.m
    n = panel.n
    u1tx = panel.u1tx; u1rx = panel.u1rx; u2tx = panel.u2tx; u2rx = panel.u2rx
    v1tx = panel.v1tx; v1rx = panel.v1rx; v2tx = panel.v2tx; v2rx = panel.v2rx
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    u1ty = panel.u1ty; u1ry = panel.u1ry; u2ty = panel.u2ty; u2ry = panel.u2ry
    v1ty = panel.v1ty; v1ry = panel.v1ry; v2ty = panel.v2ty; v2ry = panel.v2ry
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

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
                weight = weightsxi[ptx]*weightseta[pty]

                # Reading laminate constitutive data
                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            #TODO could assume symmetry
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                A11 = F[0*6 + 0]
                A12 = F[0*6 + 1]
                A16 = F[0*6 + 2]
                A22 = F[1*6 + 1]
                A26 = F[1*6 + 2]
                A66 = F[2*6 + 2]

                B11 = F[0*6 + 3]
                B12 = F[0*6 + 4]
                B16 = F[0*6 + 5]
                B22 = F[1*6 + 4]
                B26 = F[1*6 + 5]
                B66 = F[2*6 + 5]

                wxi = 0
                weta = 0
                if NLgeom == 1:
                    for j in range(n):
                        #TODO put these in a lookup vector
                        gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                        gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)
                        for i in range(m):
                            fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                            fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)

                            col = col0 + num*(j*m + i)

                            wxi += cs[col+2]*fAwxi*gAw
                            weta += cs[col+2]*fAw*gAweta

                # Calculating strain components
                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.
                for j in range(n):
                    #TODO put these in a lookup vector
                    gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAueta = calc_fxi(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAveta = calc_fxi(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAwetaeta = calc_fxixi(j, eta, w1ty, w1ry, w2ty, w2ry)

                    for i in range(m):
                        fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAuxi = calc_fxi(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAvxi = calc_fxi(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxixi = calc_fxixi(i, xi, w1tx, w1rx, w2tx, w2rx)

                        col = col0 + num*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAuxi*gAu
                        eyy += cs[col+1]*fAv*(2/b)*gAveta + 1/r*cs[col+2]*fAw*gAw
                        gxy += cs[col+0]*fAu*(2/b)*gAueta + cs[col+1]*(2/a)*fAvxi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta
                        kxy += -2*cs[col+2]*(2/a)*fAwxi*(2/b)*gAweta

                exx += 0.5*(2/a)*(2/a)*wxi*wxi
                eyy += 0.5*(2/b)*(2/b)*weta*weta
                gxy += (2/a*2/b)*wxi*weta

                # Calculating membrane stress components
                Nxx = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

                # computing kG

                c = -1
                for j in range(n):
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)

                    for l in range(n):
                        gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)
                        gBweta = calc_fxi(l, eta, w1ty, w1ry, w2ty, w2ry)

                        for i in range(m):
                            fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                            fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)

                            for k in range(m):
                                fBw = calc_f(k, xi, w1tx, w1rx, w2tx, w2rx)
                                fBwxi = calc_fxi(k, xi, w1tx, w1rx, w2tx, w2rx)

                                row = row0 + num*(j*m + i)
                                col = col0 + num*(l*m + k)

                                if row > col:
                                    continue

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] = kGv[c] + weight*(Nxx*b*fAwxi*fBwxi*gAw*gBw/a + Nxy*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta) + Nyy*a*fAw*fBw*gAweta*gBweta/b)

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


def calc_fint(np.ndarray[cDOUBLE, ndim=1] cs, object Finput, object panel,
        int size, int col0, int nx, int ny):
    cdef double a, b, r
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, c, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy
    cdef double exx, eyy, gxy, kxx, kyy, kxy

    cdef double xi, eta, weight
    cdef double wxi, weta

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double fBu, fBuxi, fBv, fBvxi, fBw, fBwxi, fBwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta
    cdef double gBu, gBueta, gBv, gBveta, gBw, gBweta, gBwetaeta

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weightsxi, weightseta, fint

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef np.ndarray[cDOUBLE, ndim=4] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.asarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = np.ascontiguousarray(Finput)
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        Finput = np.ascontiguousarray(Finput)
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    b = panel.b
    r = panel.r
    m = panel.m
    n = panel.n
    u1tx = panel.u1tx; u1rx = panel.u1rx; u2tx = panel.u2tx; u2rx = panel.u2rx
    v1tx = panel.v1tx; v1rx = panel.v1rx; v2tx = panel.v2tx; v2rx = panel.v2rx
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    u1ty = panel.u1ty; u1ry = panel.u1ry; u2ty = panel.u2ty; u2ry = panel.u2ry
    v1ty = panel.v1ty; v1ry = panel.v1ry; v2ty = panel.v2ty; v2ry = panel.v2ry
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    xis = np.zeros(nx, dtype=DOUBLE)
    weightsxi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weightseta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weightsxi[0])
    leggauss_quad(ny, &etas[0], &weightseta[0])

    fint = np.zeros(size, dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                weight = weightsxi[ptx]*weightseta[pty]

                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            #TODO could assume symmetry
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                A11 = F[0*6 + 0]
                A12 = F[0*6 + 1]
                A16 = F[0*6 + 2]
                A22 = F[1*6 + 1]
                A26 = F[1*6 + 2]
                A66 = F[2*6 + 2]

                B11 = F[0*6 + 3]
                B12 = F[0*6 + 4]
                B16 = F[0*6 + 5]
                B22 = F[1*6 + 4]
                B26 = F[1*6 + 5]
                B66 = F[2*6 + 5]

                D11 = F[3*6 + 3]
                D12 = F[3*6 + 4]
                D16 = F[3*6 + 5]
                D22 = F[4*6 + 4]
                D26 = F[4*6 + 5]
                D66 = F[5*6 + 5]

                wxi = 0
                weta = 0
                for j in range(n):
                    #TODO save in buffer
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)
                    for i in range(m):
                        #TODO save in buffer
                        fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)

                        col = col0 + num*(j*m + i)

                        wxi += cs[col+2]*fAwxi*gAw
                        weta += cs[col+2]*fAw*gAweta

                # current strain state
                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.

                for j in range(n):
                    #TODO save in buffer
                    gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAueta = calc_fxi(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAveta = calc_fxi(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAwetaeta = calc_fxixi(j, eta, w1ty, w1ry, w2ty, w2ry)

                    for i in range(m):
                        #TODO save in buffer
                        fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAuxi = calc_fxi(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAvxi = calc_fxi(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxixi = calc_fxixi(i, xi, w1tx, w1rx, w2tx, w2rx)

                        col = col0 + num*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAuxi*gAu
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + 1./r*cs[col+2]*fAw*gAw
                        gxy += cs[col+0]*(2/b)*fAu*gAueta + cs[col+1]*(2/a)*fAvxi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta
                        kxy += -2*cs[col+2]*(2/a*2/b)*fAwxi*gAweta

                exx += 0.5*(2/a)*(2/a)*wxi*wxi
                eyy += 0.5*(2/b)*(2/b)*weta*weta
                gxy += (2/a)*(2/b)*wxi*weta

                # current stress state
                Nxx = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy
                Mxx = B11*exx + B12*eyy + B16*gxy + D11*kxx + D12*kyy + D16*kxy
                Myy = B12*exx + B22*eyy + B26*gxy + D12*kxx + D22*kyy + D26*kxy
                Mxy = B16*exx + B26*eyy + B66*gxy + D16*kxx + D26*kyy + D66*kxy

                for j in range(n):
                    gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAueta = calc_fxi(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAveta = calc_fxi(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAwetaeta = calc_fxixi(j, eta, w1ty, w1ry, w2ty, w2ry)
                    for i in range(m):
                        fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAuxi = calc_fxi(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAvxi = calc_fxi(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxi = calc_fxi(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxixi = calc_fxixi(i, xi, w1tx, w1rx, w2tx, w2rx)

                        col = col0 + num*(j*m + i)

                        fint[col+0] += weight*( 0.25*a*b * ((2/a)*fAuxi*gAu*Nxx + (2/b)*fAu*gAueta*Nxy) )
                        fint[col+1] += weight*( 0.25*a*b * ((2/b)*fAv*gAveta*Nyy + (2/a)*fAvxi*gAv*Nxy) )
                        fint[col+2] += weight*( 0.25*a*b * ((2/a)*fAwxi*gAw*(2/a)*wxi*Nxx + 1./r*fAw*gAw*Nyy + (2/b)*fAw*gAweta*(2/b)*weta*Nyy + (2/a*2/b)*(fAwxi*gAw*weta + wxi*fAw*gAweta)*Nxy - (2/a*2/a)*fAwxixi*gAw*Mxx - (2/b*2/b)*fAw*gAwetaeta*Myy -2*(2/a*2/b)*fAwxi*gAweta*Mxy) )

    return fint
