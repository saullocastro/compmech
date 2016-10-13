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

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

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
    cdef double a, b, r, rbot, bbot, alpharad, sina, cosa
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
    cdef double xi, eta, x, weight
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
    rbot = panel.r
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
                                kLv[c] += A11*b*fAuxi*fBuxi*gAu*gBu/a + 0.5*A12*b*gAu*gBu*sina*(fAu*fBuxi + fAuxi*fBu)/r + A16*(fAu*fBuxi*gAueta*gBu + fAuxi*fBu*gAu*gBueta) + 0.25*A22*a*b*fAu*fBu*gAu*gBu*(sina*sina)/(r*r) + 0.5*A26*a*fAu*fBu*sina*(gAu*gBueta + gAueta*gBu)/r + A66*a*fAu*fBu*gAueta*gBueta/b
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+0
                                    kLc[c] = col+1
                                kLv[c] += A12*fAuxi*fBv*gAu*gBveta + 0.5*A16*b*fAuxi*gAu*(-fBv*gBv*sina/r + 2*fBvxi*gBv/a) + 0.5*A22*a*fAu*fBv*gAu*gBveta*sina/r + A26*(-0.25*a*b*fAu*fBv*gAu*gBv*(sina*sina)/(r*r) + a*fAu*fBv*gAueta*gBveta/b + 0.5*b*fAu*fBvxi*gAu*gBv*sina/r) + A66*fAu*(-0.5*a*fBv*gAueta*gBv*sina/r + fBvxi*gAueta*gBv)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+0
                                    kLc[c] = col+2
                                kLv[c] += 2*A11*b*fAuxi*fBwxi*gAu*gBw*wxi/(a*a) + A12*(2*fAuxi*fBw*gAu*gBweta*weta/b + 0.5*b*gAu*gBw*(a*cosa*fAuxi*fBw + 2*fAu*fBwxi*sina*wxi)/(a*r)) + 2*A16*(fAu*fBwxi*gAueta*gBw*wxi + fAuxi*gAu*(fBw*gBweta*wxi + fBwxi*gBw*weta))/a + 0.25*A22*a*fAu*fBw*gAu*sina*((b*b)*cosa*gBw + 4*gBweta*r*weta)/(b*(r*r)) + 0.5*A26*fAu*(a*fBw*gAueta*((b*b)*cosa*gBw + 4*gBweta*r*weta) + 2*(b*b)*gAu*sina*(fBw*gBweta*wxi + fBwxi*gBw*weta))/((b*b)*r) + 2*A66*fAu*gAueta*(fBw*gBweta*wxi + fBwxi*gBw*weta)/b - 2*B11*b*fAuxi*fBwxixi*gAu*gBw/(a*a) - B12*gAu*(2*a*fAuxi*fBw*gBwetaeta*r + (b*b)*gBw*sina*(fAu*fBwxixi + fAuxi*fBwxi))/(a*b*r) + B16*(-2*fAu*fBwxixi*gAueta*gBw*r + fAuxi*gAu*gBweta*(a*fBw*sina - 4*fBwxi*r))/(a*r) - 0.5*B22*fAu*gAu*sina*(2*a*fBw*gBwetaeta*r + (b*b)*fBwxi*gBw*sina)/(b*(r*r)) - 0.5*B26*fAu*(a*fBw*(-(b*b)*gAu*gBweta*(sina*sina) + 4*gAueta*gBwetaeta*(r*r)) + 2*(b*b)*fBwxi*r*sina*(2*gAu*gBweta + gAueta*gBw))/((b*b)*(r*r)) + B66*fAu*gAueta*gBweta*(a*fBw*sina - 4*fBwxi*r)/(b*r)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+1
                                    kLc[c] = col+0
                                kLv[c] += A12*fAv*fBuxi*gAveta*gBu + 0.5*A16*b*fBuxi*gBu*(-fAv*gAv*sina/r + 2*fAvxi*gAv/a) + 0.5*A22*a*fAv*fBu*gAveta*gBu*sina/r + A26*(-0.25*a*b*fAv*fBu*gAv*gBu*(sina*sina)/(r*r) + a*fAv*fBu*gAveta*gBueta/b + 0.5*b*fAvxi*fBu*gAv*gBu*sina/r) + A66*(-0.5*a*fAv*fBu*gAv*gBueta*sina/r + fAvxi*fBu*gAv*gBueta)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+1
                                    kLc[c] = col+1
                                kLv[c] += A22*a*fAv*fBv*gAveta*gBveta/b + 0.5*A26*(fAv*(-a*fBv*sina*(gAv*gBveta + gAveta*gBv) + 2*fBvxi*gAveta*gBv*r) + 2*fAvxi*fBv*gAv*gBveta*r)/r + 0.25*A66*b*gAv*gBv*(a*fAv*sina - 2*fAvxi*r)*(a*fBv*sina - 2*fBvxi*r)/(a*(r*r))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+1
                                    kLc[c] = col+2
                                kLv[c] += 2*A12*fAv*fBwxi*gAveta*gBw*wxi/a + A16*b*fBwxi*gAv*gBw*wxi*(-a*fAv*sina + 2*fAvxi*r)/((a*a)*r) + 0.5*A22*a*fAv*gAveta*(cosa*fBw*gBw/r + 4*fBw*gBweta*weta/(b*b)) + 0.25*A26*a*b*((cosa*fBw*gBw/r + 4*fBw*gBweta*weta/(b*b))*(-fAv*gAv*sina/r + 2*fAvxi*gAv/a) + 8*fAv*gAveta*(fBw*gBweta*wxi + fBwxi*gBw*weta)/(a*(b*b))) + A66*(fBw*gBweta*wxi + fBwxi*gBw*weta)*(-fAv*gAv*sina/r + 2*fAvxi*gAv/a) - 2*B12*fAv*fBwxixi*gAveta*gBw/a + B16*b*fBwxixi*gAv*gBw*(a*fAv*sina - 2*fAvxi*r)/((a*a)*r) + B22*fAv*gAveta*(-2*a*fBw*gBwetaeta/(b*b) - fBwxi*gBw*sina/r) + 0.5*B26*(2*(a*a)*fAv*fBw*r*sina*(gAv*gBwetaeta + gAveta*gBweta) + a*((b*b)*fAv*fBwxi*gAv*gBw*(sina*sina) - 8*fAv*fBwxi*gAveta*gBweta*(r*r) - 4*fAvxi*fBw*gAv*gBwetaeta*(r*r)) - 2*(b*b)*fAvxi*fBwxi*gAv*gBw*r*sina)/(a*b*(r*r)) - 0.5*B66*gAv*gBweta*(a*fAv*sina - 2*fAvxi*r)*(a*fBw*sina - 4*fBwxi*r)/(a*(r*r))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+2
                                    kLc[c] = col+0
                                kLv[c] += 2*A11*b*fAwxi*fBuxi*gAw*gBu*wxi/(a*a) + A12*(2*fAw*fBuxi*gAweta*gBu*weta/b + 0.5*b*gAw*gBu*(a*cosa*fAw*fBuxi + 2*fAwxi*fBu*sina*wxi)/(a*r)) + 2*A16*(fAw*fBuxi*gAweta*gBu*wxi + fAwxi*gAw*(fBu*gBueta*wxi + fBuxi*gBu*weta))/a + 0.25*A22*a*fAw*fBu*gBu*sina*((b*b)*cosa*gAw + 4*gAweta*r*weta)/(b*(r*r)) + 0.5*A26*fBu*(a*fAw*gBueta*((b*b)*cosa*gAw + 4*gAweta*r*weta) + 2*(b*b)*gBu*sina*(fAw*gAweta*wxi + fAwxi*gAw*weta))/((b*b)*r) + 2*A66*fBu*gBueta*(fAw*gAweta*wxi + fAwxi*gAw*weta)/b - 2*B11*b*fAwxixi*fBuxi*gAw*gBu/(a*a) - B12*gBu*(2*a*fAw*fBuxi*gAwetaeta*r + (b*b)*gAw*sina*(fAwxi*fBuxi + fAwxixi*fBu))/(a*b*r) + B16*(a*fAw*fBuxi*gAweta*gBu*sina - 4*fAwxi*fBuxi*gAweta*gBu*r - 2*fAwxixi*fBu*gAw*gBueta*r)/(a*r) - 0.5*B22*fBu*gBu*sina*(2*a*fAw*gAwetaeta*r + (b*b)*fAwxi*gAw*sina)/(b*(r*r)) - 0.5*B26*fBu*(a*fAw*(-(b*b)*gAweta*gBu*(sina*sina) + 4*gAwetaeta*gBueta*(r*r)) + 2*(b*b)*fAwxi*r*sina*(gAw*gBueta + 2*gAweta*gBu))/((b*b)*(r*r)) + B66*fBu*gAweta*gBueta*(a*fAw*sina - 4*fAwxi*r)/(b*r)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+2
                                    kLc[c] = col+1
                                kLv[c] += 2*A12*fAwxi*fBv*gAw*gBveta*wxi/a + A16*b*fAwxi*gAw*gBv*wxi*(-a*fBv*sina + 2*fBvxi*r)/((a*a)*r) + 0.5*A22*a*fBv*gBveta*(cosa*fAw*gAw/r + 4*fAw*gAweta*weta/(b*b)) + 0.25*A26*a*b*((cosa*fAw*gAw/r + 4*fAw*gAweta*weta/(b*b))*(-fBv*gBv*sina/r + 2*fBvxi*gBv/a) + 8*fBv*gBveta*(fAw*gAweta*wxi + fAwxi*gAw*weta)/(a*(b*b))) + A66*(fAw*gAweta*wxi + fAwxi*gAw*weta)*(-fBv*gBv*sina/r + 2*fBvxi*gBv/a) - 2*B12*fAwxixi*fBv*gAw*gBveta/a + B16*b*fAwxixi*gAw*gBv*(a*fBv*sina - 2*fBvxi*r)/((a*a)*r) + B22*fBv*gBveta*(-2*a*fAw*gAwetaeta/(b*b) - fAwxi*gAw*sina/r) + 0.5*B26*(2*(a*a)*fAw*fBv*r*sina*(gAweta*gBveta + gAwetaeta*gBv) + a*(-4*fAw*fBvxi*gAwetaeta*gBv*(r*r) + fAwxi*fBv*((b*b)*gAw*gBv*(sina*sina) - 8*gAweta*gBveta*(r*r))) - 2*(b*b)*fAwxi*fBvxi*gAw*gBv*r*sina)/(a*b*(r*r)) - 0.5*B66*gAweta*gBv*(a*fAw*sina - 4*fAwxi*r)*(a*fBv*sina - 2*fBvxi*r)/(a*(r*r))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kLr[c] = row+2
                                    kLc[c] = col+2
                                kLv[c] += 4*A11*b*fAwxi*fBwxi*gAw*gBw*(wxi*wxi)/(a*a*a) + A12*wxi*((b*b)*cosa*gAw*gBw*(fAw*fBwxi + fAwxi*fBw) + r*weta*(4*fAw*fBwxi*gAweta*gBw + 4*fAwxi*fBw*gAw*gBweta))/(a*b*r) + 4*A16*wxi*(fAw*fBwxi*gAweta*gBw*wxi + fAwxi*gAw*(fBw*gBweta*wxi + 2*fBwxi*gBw*weta))/(a*a) + 0.25*A22*a*fAw*fBw*((b*b)*cosa*gAw + 4*gAweta*r*weta)*((b*b)*cosa*gBw + 4*gBweta*r*weta)/((b*b*b)*(r*r)) + A26*((b*b)*cosa*(fAw*(fBw*gAw*gBweta*wxi + fBw*gAweta*gBw*wxi + fBwxi*gAw*gBw*weta) + fAwxi*fBw*gAw*gBw*weta) + 4*r*weta*(fAw*gAweta*(2*fBw*gBweta*wxi + fBwxi*gBw*weta) + fAwxi*fBw*gAw*gBweta*weta))/((b*b)*r) + 4*A66*(fAw*gAweta*wxi + fAwxi*gAw*weta)*(fBw*gBweta*wxi + fBwxi*gBw*weta)/(a*b) - 4*B11*b*gAw*gBw*wxi*(fAwxi*fBwxixi + fAwxixi*fBwxi)/(a*a*a) - B12*(a*((b*b)*cosa*gAw*gBw*(fAw*fBwxixi + fAwxixi*fBw) + 4*r*(fAw*fBwxi*gAwetaeta*gBw*wxi + fAw*fBwxixi*gAweta*gBw*weta + fAwxi*fBw*gAw*gBwetaeta*wxi + fAwxixi*fBw*gAw*gBweta*weta)) + 4*(b*b)*fAwxi*fBwxi*gAw*gBw*sina*wxi)/((a*a)*b*r) + 2*B16*(fAw*gAweta*gBw*wxi*(a*fBwxi*sina - 2*fBwxixi*r) + fAwxi*(-2*fBwxixi*gAw*gBw*r*weta + wxi*(a*fBw*gAw*gBweta*sina - 4*fBwxi*r*(gAw*gBweta + gAweta*gBw))) - 2*fAwxixi*gAw*r*(fBw*gBweta*wxi + fBwxi*gBw*weta))/((a*a)*r) - 0.5*B22*(2*a*fAw*fBw*r*((b*b)*cosa*(gAw*gBwetaeta + gAwetaeta*gBw) + r*weta*(4*gAweta*gBwetaeta + 4*gAwetaeta*gBweta)) + (b*b)*sina*((b*b)*cosa*gAw*gBw*(fAw*fBwxi + fAwxi*fBw) + r*weta*(4*fAw*fBwxi*gAweta*gBw + 4*fAwxi*fBw*gAw*gBweta)))/((b*b*b)*(r*r)) + 0.5*B26*((a*a)*fAw*fBw*sina*((b*b)*cosa*(gAw*gBweta + gAweta*gBw) + 8*gAweta*gBweta*r*weta) - 4*a*r*((b*b)*cosa*(fAw*fBwxi*gAw*gBweta + fAwxi*fBw*gAweta*gBw) + 2*r*(fAw*(fBw*gAweta*gBwetaeta*wxi + fBw*gAwetaeta*gBweta*wxi + 2*fBwxi*gAweta*gBweta*weta + fBwxi*gAwetaeta*gBw*weta) + fAwxi*fBw*weta*(gAw*gBwetaeta + 2*gAweta*gBweta))) - 4*(b*b)*r*sina*(fAw*fBwxi*gAweta*gBw*wxi + fAwxi*gAw*(fBw*gBweta*wxi + 2*fBwxi*gBw*weta)))/(a*(b*b)*(r*r)) + 2*B66*(fAw*gAweta*(a*sina*(2*fBw*gBweta*wxi + fBwxi*gBw*weta) - 4*fBwxi*gBweta*r*wxi) + fAwxi*(fBw*gBweta*(a*gAw*sina*weta - 4*gAweta*r*wxi) - 4*fBwxi*r*weta*(gAw*gBweta + gAweta*gBw)))/(a*b*r) + 4*D11*b*fAwxixi*fBwxixi*gAw*gBw/(a*a*a) + 2*D12*(2*a*r*(fAw*fBwxixi*gAwetaeta*gBw + fAwxixi*fBw*gAw*gBwetaeta) + (b*b)*gAw*gBw*sina*(fAwxi*fBwxixi + fAwxixi*fBwxi))/((a*a)*b*r) - 2*D16*(a*fAw*fBwxixi*gAweta*gBw*sina - 4*fAwxi*fBwxixi*gAweta*gBw*r + fAwxixi*gAw*gBweta*(a*fBw*sina - 4*fBwxi*r))/((a*a)*r) + D22*(2*a*fAw*gAwetaeta*r + (b*b)*fAwxi*gAw*sina)*(2*a*fBw*gBwetaeta*r + (b*b)*fBwxi*gBw*sina)/(a*(b*b*b)*(r*r)) + D26*(-2*(a*a)*fAw*fBw*r*sina*(gAweta*gBwetaeta + gAwetaeta*gBweta) + a*(fAw*fBwxi*(-(b*b)*gAweta*gBw*(sina*sina) + 8*gAwetaeta*gBweta*(r*r)) + fAwxi*fBw*(-(b*b)*gAw*gBweta*(sina*sina) + 8*gAweta*gBwetaeta*(r*r))) + 4*(b*b)*fAwxi*fBwxi*r*sina*(gAw*gBweta + gAweta*gBw))/(a*(b*b)*(r*r)) + D66*gAweta*gBweta*(a*fAw*sina - 4*fAwxi*r)*(a*fBw*sina - 4*fBwxi*r)/(a*b*(r*r))

    kL = coo_matrix((kLv, (kLr, kLc)), shape=(size, size))

    return kL


def fkG_num(np.ndarray[cDOUBLE, ndim=1] cs, object Finput, object panel,
            int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double a, b, r, rbot, bbot, alpharad, sina, cosa
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col, ptx, pty
    cdef double xi, eta, x, weight

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
    bbot = panel.b
    rbot = panel.r
    alpharad = panel.alpharad
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
                        eyy += cs[col+1]*fAv*(2/b)*gAveta + 1/r*(cosa*cs[col+2]*fAw*gAw + sina*cs[col+0]*fAu*gAu)
                        gxy += cs[col+0]*fAu*(2/b)*gAueta - 1/r*sina*cs[col+1]*fAv*gAv + cs[col+1]*(2/a)*fAvxi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta - 1/r*sina*cs[col+2]*(2/a)*fAwxi*gAw
                        kxy += -2*cs[col+2]*(2/a)*fAwxi*(2/b)*gAweta + 1/r*sina*cs[col+2]*(2/b)*fAw*gAweta

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
    cdef double a, b, r, rbot, bbot, alpharad, sina, cosa
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

    cdef double xi, eta, x, weight
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
    bbot = panel.b
    rbot = panel.r
    alpharad = panel.alpharad
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
                        eyy += cs[col+1]*fAv*(2/b)*gAveta + 1/r*(cosa*cs[col+2]*fAw*gAw + sina*cs[col+0]*fAu*gAu)
                        gxy += cs[col+0]*fAu*(2/b)*gAueta - 1/r*sina*cs[col+1]*fAv*gAv + cs[col+1]*(2/a)*fAvxi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta - 1/r*sina*cs[col+2]*(2/a)*fAwxi*gAw
                        kxy += -2*cs[col+2]*(2/a)*fAwxi*(2/b)*gAweta + 1/r*sina*cs[col+2]*(2/b)*fAw*gAweta

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

                        fint[col+0] += weight*( 0.25*a*b*(2*Nxx*fAuxi*gAu/a + 2*Nxy*fAu*gAueta/b + Nyy*fAu*gAu*sina/r) )
                        fint[col+1] += weight*( 0.25*a*b*(Nxy*(-fAv*gAv*sina/r + 2*fAvxi*gAv/a) + 2*Nyy*fAv*gAveta/b) )
                        fint[col+2] += weight*( 0.25*a*b*(-4*Mxx*fAwxixi*gAw/(a*a) + Mxy*(2*fAw*gAweta*sina/(b*r) - 8*fAwxi*gAweta/(a*b)) + Myy*(-4*fAw*gAwetaeta/(b*b) - 2*fAwxi*gAw*sina/(a*r)) + 4*Nxx*fAwxi*gAw*wxi/(a*a) + 4*Nxy*(fAw*gAweta*wxi + fAwxi*gAw*weta)/(a*b) + Nyy*(cosa*fAw*gAw/r + 4*fAw*gAweta*weta/(b*b))) )

    return fint
