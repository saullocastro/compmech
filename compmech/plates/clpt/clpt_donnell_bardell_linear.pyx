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
cimport cython

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef double pi = 3.141592653589793
cdef int num1 = 3
cdef int nmax = 30


def fk0(double a, double b, np.ndarray[cDOUBLE, ndim=2] F,
        double u1tx, double u1rx, double u2tx, double u2rx,
        double v1tx, double v1rx, double v2tx, double v2rx,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double u1ty, double u1ry, double u2ty, double u2ry,
        double v1ty, double v1ry, double v2ty, double v2ry,
        double w1ty, double w1ry, double w2ty, double w2ry,
        int m1, int n1):
    cdef int i1, j1, k1, l1, c, row, col
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    cdef double fAufBu, fAufBuxi, fAuxifBu, fAuxifBuxi, fAufBv, fAufBvxi,
    cdef double fAuxifBv, fAuxifBvxi, fAuxifBwxixi, fAuxifBw, fAufBwxixi,
    cdef double fAuxifBwxi, fAufBw, fAufBwxi, fAvfBuxi, fAvxifBuxi, fAvfBu,
    cdef double fAvxifBu, fAvfBv, fAvfBvxi, fAvxifBv, fAvxifBvxi, fAvfBwxixi,
    cdef double fAvxifBwxixi, fAvfBw, fAvfBwxi, fAvxifBw, fAvxifBwxi,
    cdef double fAwxixifBuxi, fAwfBuxi, fAwxifBuxi, fAwxixifBu, fAwfBu,
    cdef double fAwxifBu, fAwxixifBv, fAwxixifBvxi, fAwfBv, fAwfBvxi, fAwxifBv,
    cdef double fAwxifBvxi, fAwxixifBwxixi, fAwfBwxixi, fAwxixifBw,
    cdef double fAwxifBwxixi, fAwxixifBwxi, fAwfBw, fAwfBwxi, fAwxifBw,
    cdef double fAwxifBwxi
    cdef double gAugBu, gAugBueta, gAuetagBu, gAuetagBueta, gAugBv, gAugBveta,
    cdef double gAuetagBv, gAuetagBveta, gAuetagBwetaeta, gAuetagBw,
    cdef double gAugBwetaeta, gAuetagBweta, gAugBw, gAugBweta, gAvgBueta,
    cdef double gAvetagBueta, gAvgBu, gAvetagBu, gAvgBv, gAvgBveta, gAvetagBv,
    cdef double gAvetagBveta, gAvgBwetaeta, gAvetagBwetaeta, gAvgBw, gAvgBweta,
    cdef double gAvetagBw, gAvetagBweta, gAwetaetagBueta, gAwgBueta,
    cdef double gAwetagBueta, gAwetaetagBu, gAwgBu, gAwetagBu, gAwetaetagBv,
    cdef double gAwetaetagBveta, gAwgBv, gAwgBveta, gAwetagBv, gAwetagBveta,
    cdef double gAwetaetagBwetaeta, gAwgBwetaeta, gAwetaetagBw,
    cdef double gAwetagBwetaeta, gAwetaetagBweta, gAwgBw, gAwgBweta, gAwetagBw,
    cdef double gAwetagBweta

    fdim = 9*m1*m1*n1*n1

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

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

    D11 = F[3,3]
    D12 = F[3,4]
    D16 = F[3,5]
    D22 = F[4,4]
    D26 = F[4,5]
    D66 = F[5,5]

    # k0
    c = -1
    for i1 in range(0, m1):
        for k1 in range(0, m1):

            fAufBu = calc_ff(i1, k1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAufBuxi = calc_ffxi(i1, k1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAuxifBu = calc_ffxi(k1, i1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAuxifBuxi = calc_fxifxi(i1, k1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAufBv = calc_ff(i1, k1, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
            fAufBvxi = calc_ffxi(i1, k1, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
            fAuxifBv = calc_ffxi(k1, i1, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
            fAuxifBvxi = calc_fxifxi(i1, k1, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
            fAuxifBwxixi = calc_fxifxixi(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAuxifBw = calc_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAufBwxixi = calc_ffxixi(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAuxifBwxi = calc_fxifxi(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAufBw = calc_ff(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAufBwxi = calc_ffxi(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAvfBuxi = calc_ffxi(i1, k1, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
            fAvxifBuxi = calc_fxifxi(i1, k1, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
            fAvfBu = calc_ff(i1, k1, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
            fAvxifBu = calc_ffxi(k1, i1, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
            fAvfBv = calc_ff(i1, k1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvfBvxi = calc_ffxi(i1, k1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvxifBv = calc_ffxi(k1, i1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvxifBvxi = calc_fxifxi(i1, k1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvfBwxixi = calc_ffxixi(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAvxifBwxixi = calc_fxifxixi(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAvfBw = calc_ff(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAvfBwxi = calc_ffxi(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAvxifBw = calc_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAvxifBwxi = calc_fxifxi(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBuxi = calc_fxifxixi(k1, i1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBuxi = calc_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAwxifBuxi = calc_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAwxixifBu = calc_ffxixi(k1, i1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBu = calc_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAwxifBu = calc_ffxi(k1, i1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBv = calc_ffxixi(k1, i1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBvxi = calc_fxifxixi(k1, i1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBv = calc_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAwfBvxi = calc_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAwxifBv = calc_ffxi(k1, i1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBvxi = calc_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAwxixifBwxixi = calc_fxixifxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxixi = calc_ffxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBw = calc_ffxixi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBwxixi = calc_fxifxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBwxi = calc_fxifxixi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBw = calc_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxi = calc_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBw = calc_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBwxi = calc_fxifxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)


            for j1 in range(0, n1):
                for l1 in range(0, n1):

                    row = num1*(j1*m1 + i1)
                    col = num1*(l1*m1 + k1)

                    if row > col:
                        continue

                    gAugBu = calc_ff(j1, l1, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                    gAugBueta = calc_ffxi(j1, l1, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                    gAuetagBu = calc_ffxi(l1, j1, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                    gAuetagBueta = calc_fxifxi(j1, l1, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                    gAugBv = calc_ff(j1, l1, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                    gAugBveta = calc_ffxi(j1, l1, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                    gAuetagBv = calc_ffxi(l1, j1, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                    gAuetagBveta = calc_fxifxi(j1, l1, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                    gAuetagBwetaeta = calc_fxifxixi(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAuetagBw = calc_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                    gAugBwetaeta = calc_ffxixi(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAuetagBweta = calc_fxifxi(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAugBw = calc_ff(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAugBweta = calc_ffxi(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvgBueta = calc_ffxi(j1, l1, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                    gAvetagBueta = calc_fxifxi(j1, l1, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                    gAvgBu = calc_ff(j1, l1, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                    gAvetagBu = calc_ffxi(l1, j1, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvgBv = calc_ff(j1, l1, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvgBveta = calc_ffxi(j1, l1, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvetagBv = calc_ffxi(l1, j1, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvetagBveta = calc_fxifxi(j1, l1, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvgBwetaeta = calc_ffxixi(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvetagBwetaeta = calc_fxifxixi(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvgBw = calc_ff(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvgBweta = calc_ffxi(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvetagBw = calc_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvetagBweta = calc_fxifxi(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBueta = calc_fxifxixi(l1, j1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBueta = calc_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                    gAwetagBueta = calc_fxifxi(j1, l1, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                    gAwetaetagBu = calc_ffxixi(l1, j1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBu = calc_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                    gAwetagBu = calc_ffxi(l1, j1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBv = calc_ffxixi(l1, j1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBveta = calc_fxifxixi(l1, j1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBv = calc_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                    gAwgBveta = calc_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                    gAwetagBv = calc_ffxi(l1, j1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBveta = calc_fxifxi(j1, l1, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                    gAwetaetagBwetaeta = calc_fxixifxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBwetaeta = calc_ffxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBw = calc_ffxixi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBwetaeta = calc_fxifxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBweta = calc_fxifxixi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBw = calc_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBweta = calc_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBw = calc_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBweta = calc_fxifxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += A11*b*fAuxifBuxi*gAugBu/a + A16*(fAufBuxi*gAuetagBu + fAuxifBu*gAugBueta) + A66*a*fAufBu*gAuetagBueta/b
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+1
                    k0v[c] += A12*fAuxifBv*gAugBveta + A16*b*fAuxifBvxi*gAugBv/a + A26*a*fAufBv*gAuetagBveta/b + A66*fAufBvxi*gAuetagBv
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+2
                    k0v[c] += -2*B11*b*fAuxifBwxixi*gAugBw/(a*a) - 2*B12*fAuxifBw*gAugBwetaeta/b - 2*B16*(fAufBwxixi*gAuetagBw + 2*fAuxifBwxi*gAugBweta)/a - 2*B26*a*fAufBw*gAuetagBwetaeta/(b*b) - 4*B66*fAufBwxi*gAuetagBweta/b
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += A12*fAvfBuxi*gAvetagBu + A16*b*fAvxifBuxi*gAvgBu/a + A26*a*fAvfBu*gAvetagBueta/b + A66*fAvxifBu*gAvgBueta
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += A22*a*fAvfBv*gAvetagBveta/b + A26*(fAvfBvxi*gAvetagBv + fAvxifBv*gAvgBveta) + A66*b*fAvxifBvxi*gAvgBv/a
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+2
                    k0v[c] += -2*B12*fAvfBwxixi*gAvetagBw/a - 2*B16*b*fAvxifBwxixi*gAvgBw/(a*a) - 2*B22*a*fAvfBw*gAvetagBwetaeta/(b*b) - 2*B26*(2*fAvfBwxi*gAvetagBweta + fAvxifBw*gAvgBwetaeta)/b - 4*B66*fAvxifBwxi*gAvgBweta/a
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+0
                    k0v[c] += -2*B11*b*fAwxixifBuxi*gAwgBu/(a*a) - 2*B12*fAwfBuxi*gAwetaetagBu/b - 2*B16*(2*fAwxifBuxi*gAwetagBu + fAwxixifBu*gAwgBueta)/a - 2*B26*a*fAwfBu*gAwetaetagBueta/(b*b) - 4*B66*fAwxifBu*gAwetagBueta/b
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+1
                    k0v[c] += -2*B12*fAwxixifBv*gAwgBveta/a - 2*B16*b*fAwxixifBvxi*gAwgBv/(a*a) - 2*B22*a*fAwfBv*gAwetaetagBveta/(b*b) - 2*B26*(fAwfBvxi*gAwetaetagBv + 2*fAwxifBv*gAwetagBveta)/b - 4*B66*fAwxifBvxi*gAwetagBv/a
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*D12*(fAwfBwxixi*gAwetaetagBw + fAwxixifBw*gAwgBwetaeta)/(a*b) + 8*D16*(fAwxifBwxixi*gAwetagBw + fAwxixifBwxi*gAwgBweta)/(a*a) + 4*D22*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*D26*(fAwfBwxi*gAwetaetagBweta + fAwxifBw*gAwetagBwetaeta)/(b*b) + 16*D66*fAwxifBwxi*gAwetagBweta/(a*b)

    size = num1*m1*n1

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fkG0(double Nxx, double Nyy, double Nxy, double a, double b,
         double w1tx, double w1rx, double w2tx, double w2rx,
         double w1ty, double w1ry, double w2ty, double w2ry,
         int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col

    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    cdef double fAwxifBwxi, fAwfBwxi, fAwxifBw, fAwfBw
    cdef double gAwetagBweta, gAwgBweta, gAwetagBw, gAwgBw

    fdim = 1*m1*m1*n1*n1

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    # kG0

    c = -1
    for i1 in range(0, m1):
        for k1 in range(0, m1):

            fAwxifBwxi = calc_fxifxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxi = calc_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBw = calc_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBw = calc_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

            for j1 in range(0, n1):
                for l1 in range(0, n1):

                    row = num1*(j1*m1 + i1)
                    col = num1*(l1*m1 + k1)

                    if row > col:
                        continue

                    gAwetagBw = calc_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBw = calc_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBweta = calc_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBweta = calc_fxifxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                    c += 1
                    kG0r[c] = row+2
                    kG0c[c] = col+2
                    kG0v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

    size = num1*m1*n1

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


cdef double calc_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                    double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return 0.742857142857143*x1t*y1t
        if j == 1:
            return 0.104761904761905*x1t*y1r
        if j == 2:
            return 0.257142857142857*x1t*y2t
        if j == 3:
            return -0.0619047619047619*x1t*y2r
        if j == 4:
            return 0.0666666666666667*x1t
        if j == 5:
            return -0.0126984126984127*x1t
        if j == 7:
            return 0.000288600288600289*x1t
    if i == 1:
        if j == 0:
            return 0.104761904761905*x1r*y1t
        if j == 1:
            return 0.019047619047619*x1r*y1r
        if j == 2:
            return 0.0619047619047619*x1r*y2t
        if j == 3:
            return -0.0142857142857143*x1r*y2r
        if j == 4:
            return 0.0142857142857143*x1r
        if j == 5:
            return -0.00158730158730159*x1r
        if j == 6:
            return -0.000529100529100529*x1r
        if j == 7:
            return 0.000144300144300144*x1r
    if i == 2:
        if j == 0:
            return 0.257142857142857*x2t*y1t
        if j == 1:
            return 0.0619047619047619*x2t*y1r
        if j == 2:
            return 0.742857142857143*x2t*y2t
        if j == 3:
            return -0.104761904761905*x2t*y2r
        if j == 4:
            return 0.0666666666666667*x2t
        if j == 5:
            return 0.0126984126984127*x2t
        if j == 7:
            return -0.000288600288600289*x2t
    if i == 3:
        if j == 0:
            return -0.0619047619047619*x2r*y1t
        if j == 1:
            return -0.0142857142857143*x2r*y1r
        if j == 2:
            return -0.104761904761905*x2r*y2t
        if j == 3:
            return 0.019047619047619*x2r*y2r
        if j == 4:
            return -0.0142857142857143*x2r
        if j == 5:
            return -0.00158730158730159*x2r
        if j == 6:
            return 0.000529100529100529*x2r
        if j == 7:
            return 0.000144300144300144*x2r
    if i == 4:
        if j == 0:
            return 0.0666666666666667*y1t
        if j == 1:
            return 0.0142857142857143*y1r
        if j == 2:
            return 0.0666666666666667*y2t
        if j == 3:
            return -0.0142857142857143*y2r
        if j == 4:
            return 0.0126984126984127
        if j == 6:
            return -0.000769600769600770
        if j == 8:
            return 4.44000444000444e-5
    if i == 5:
        if j == 0:
            return -0.0126984126984127*y1t
        if j == 1:
            return -0.00158730158730159*y1r
        if j == 2:
            return 0.0126984126984127*y2t
        if j == 3:
            return -0.00158730158730159*y2r
        if j == 5:
            return 0.00115440115440115
        if j == 7:
            return -0.000177600177600178
        if j == 9:
            return 1.48000148000148e-5
    if i == 6:
        if j == 1:
            return -0.000529100529100529*y1r
        if j == 3:
            return 0.000529100529100529*y2r
        if j == 4:
            return -0.000769600769600770
        if j == 6:
            return 0.000266400266400266
        if j == 8:
            return -5.92000592000592e-5
        if j == 10:
            return 6.09412374118256e-6
    if i == 7:
        if j == 0:
            return 0.000288600288600289*y1t
        if j == 1:
            return 0.000144300144300144*y1r
        if j == 2:
            return -0.000288600288600289*y2t
        if j == 3:
            return 0.000144300144300144*y2r
        if j == 5:
            return -0.000177600177600178
        if j == 7:
            return 8.88000888000888e-5
        if j == 9:
            return -2.43764949647303e-5
        if j == 11:
            return 2.88669019319174e-6
    if i == 8:
        if j == 4:
            return 4.44000444000444e-5
        if j == 6:
            return -5.92000592000592e-5
        if j == 8:
            return 3.65647424470954e-5
        if j == 10:
            return -1.15467607727670e-5
        if j == 12:
            return 1.51207581548139e-6
    if i == 9:
        if j == 5:
            return 1.48000148000148e-5
        if j == 7:
            return -2.43764949647303e-5
        if j == 9:
            return 1.73201411591504e-5
        if j == 11:
            return -6.04830326192555e-6
        if j == 13:
            return 8.54651547880785e-7
    if i == 10:
        if j == 6:
            return 6.09412374118256e-6
        if j == 8:
            return -1.15467607727670e-5
        if j == 10:
            return 9.07245489288833e-6
        if j == 12:
            return -3.41860619152314e-6
        if j == 14:
            return 5.12790928728471e-7
    if i == 11:
        if j == 7:
            return 2.88669019319174e-6
        if j == 9:
            return -6.04830326192555e-6
        if j == 11:
            return 5.12790928728471e-6
        if j == 13:
            return -2.05116371491388e-6
        if j == 15:
            return 3.22868362532741e-7
    if i == 12:
        if j == 8:
            return 1.51207581548139e-6
        if j == 10:
            return -3.41860619152314e-6
        if j == 12:
            return 3.07674557237082e-6
        if j == 14:
            return -1.29147345013096e-6
        if j == 16:
            return 2.11534444418003e-7
    if i == 13:
        if j == 9:
            return 8.54651547880785e-7
        if j == 11:
            return -2.05116371491388e-6
        if j == 13:
            return 1.93721017519645e-6
        if j == 15:
            return -8.46137777672011e-7
        if j == 17:
            return 1.43297526863808e-7
    if i == 14:
        if j == 10:
            return 5.12790928728471e-7
        if j == 12:
            return -1.29147345013096e-6
        if j == 14:
            return 1.26920666650802e-6
        if j == 16:
            return -5.73190107455233e-7
        if j == 18:
            return 9.98740338747754e-8
    if i == 15:
        if j == 11:
            return 3.22868362532741e-7
        if j == 13:
            return -8.46137777672011e-7
        if j == 15:
            return 8.59785161182849e-7
        if j == 17:
            return -3.99496135499102e-7
        if j == 19:
            return 7.13385956248396e-8
    if i == 16:
        if j == 12:
            return 2.11534444418003e-7
        if j == 14:
            return -5.73190107455233e-7
        if j == 16:
            return 5.99244203248653e-7
        if j == 18:
            return -2.85354382499358e-7
        if j == 20:
            return 5.20578941046127e-8
    if i == 17:
        if j == 13:
            return 1.43297526863808e-7
        if j == 15:
            return -3.99496135499102e-7
        if j == 17:
            return 4.28031573749038e-7
        if j == 19:
            return -2.08231576418451e-7
        if j == 21:
            return 3.87097161290710e-8
    if i == 18:
        if j == 14:
            return 9.98740338747754e-8
        if j == 16:
            return -2.85354382499358e-7
        if j == 18:
            return 3.12347364627676e-7
        if j == 20:
            return -1.54838864516284e-7
        if j == 22:
            return 2.92683219512488e-8
    if i == 19:
        if j == 15:
            return 7.13385956248396e-8
        if j == 17:
            return -2.08231576418451e-7
        if j == 19:
            return 2.32258296774426e-7
        if j == 21:
            return -1.17073287804995e-7
        if j == 23:
            return 2.24617354509584e-8
    if i == 20:
        if j == 16:
            return 5.20578941046127e-8
        if j == 18:
            return -1.54838864516284e-7
        if j == 20:
            return 1.75609931707493e-7
        if j == 22:
            return -8.98469418038335e-8
        if j == 24:
            return 1.74702386840787e-8
    if i == 21:
        if j == 17:
            return 3.87097161290710e-8
        if j == 19:
            return -1.17073287804995e-7
        if j == 21:
            return 1.34770412705750e-7
        if j == 23:
            return -6.98809547363149e-8
        if j == 25:
            return 1.37531666236364e-8
    if i == 22:
        if j == 18:
            return 2.92683219512488e-8
        if j == 20:
            return -8.98469418038335e-8
        if j == 22:
            return 1.04821432104472e-7
        if j == 24:
            return -5.50126664945458e-8
        if j == 26:
            return 1.09463979249351e-8
    if i == 23:
        if j == 19:
            return 2.24617354509584e-8
        if j == 21:
            return -6.98809547363149e-8
        if j == 23:
            return 8.25189997418187e-8
        if j == 25:
            return -4.37855916997405e-8
        if j == 27:
            return 8.80004539063412e-9
    if i == 24:
        if j == 20:
            return 1.74702386840787e-8
        if j == 22:
            return -5.50126664945458e-8
        if j == 24:
            return 6.56783875496108e-8
        if j == 26:
            return -3.52001815625365e-8
        if j == 28:
            return 7.13965946787297e-9
    if i == 25:
        if j == 21:
            return 1.37531666236364e-8
        if j == 23:
            return -4.37855916997405e-8
        if j == 25:
            return 5.28002723438047e-8
        if j == 27:
            return -2.85586378714919e-8
        if j == 29:
            return 5.84153956462334e-9
    if i == 26:
        if j == 22:
            return 1.09463979249351e-8
        if j == 24:
            return -3.52001815625365e-8
        if j == 26:
            return 4.28379568072378e-8
        if j == 28:
            return -2.33661582584934e-8
    if i == 27:
        if j == 23:
            return 8.80004539063412e-9
        if j == 25:
            return -2.85586378714919e-8
        if j == 27:
            return 3.50492373877400e-8
        if j == 29:
            return -1.92668322482314e-8
    if i == 28:
        if j == 24:
            return 7.13965946787297e-9
        if j == 26:
            return -2.33661582584934e-8
        if j == 28:
            return 2.89002483723470e-8
    if i == 29:
        if j == 25:
            return 5.84153956462334e-9
        if j == 27:
            return -1.92668322482314e-8
        if j == 29:
            return 2.40019011905933e-8


cdef double calc_ffxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return -0.5*x1t*y1t
        if j == 1:
            return 0.1*x1t*y1r
        if j == 2:
            return 0.5*x1t*y2t
        if j == 3:
            return -0.1*x1t*y2r
        if j == 4:
            return 0.0857142857142857*x1t
        if j == 6:
            return -0.00317460317460317*x1t
    if i == 1:
        if j == 0:
            return -0.1*x1r*y1t
        if j == 2:
            return 0.1*x1r*y2t
        if j == 3:
            return -0.0166666666666667*x1r*y2r
        if j == 4:
            return 0.00952380952380952*x1r
        if j == 5:
            return 0.00476190476190476*x1r
        if j == 6:
            return -0.00158730158730159*x1r
    if i == 2:
        if j == 0:
            return -0.5*x2t*y1t
        if j == 1:
            return -0.1*x2t*y1r
        if j == 2:
            return 0.5*x2t*y2t
        if j == 3:
            return 0.1*x2t*y2r
        if j == 4:
            return -0.0857142857142857*x2t
        if j == 6:
            return 0.00317460317460317*x2t
    if i == 3:
        if j == 0:
            return 0.1*x2r*y1t
        if j == 1:
            return 0.0166666666666667*x2r*y1r
        if j == 2:
            return -0.1*x2r*y2t
        if j == 4:
            return 0.00952380952380952*x2r
        if j == 5:
            return -0.00476190476190476*x2r
        if j == 6:
            return -0.00158730158730159*x2r
    if i == 4:
        if j == 0:
            return -0.0857142857142857*y1t
        if j == 1:
            return -0.00952380952380952*y1r
        if j == 2:
            return 0.0857142857142857*y2t
        if j == 3:
            return -0.00952380952380952*y2r
        if j == 5:
            return 0.00634920634920635
        if j == 7:
            return -0.000577200577200577
    if i == 5:
        if j == 1:
            return -0.00476190476190476*y1r
        if j == 3:
            return 0.00476190476190476*y2r
        if j == 4:
            return -0.00634920634920635
        if j == 6:
            return 0.00173160173160173
        if j == 8:
            return -0.000222000222000222
    if i == 6:
        if j == 0:
            return 0.00317460317460317*y1t
        if j == 1:
            return 0.00158730158730159*y1r
        if j == 2:
            return -0.00317460317460317*y2t
        if j == 3:
            return 0.00158730158730159*y2r
        if j == 5:
            return -0.00173160173160173
        if j == 7:
            return 0.000666000666000666
        if j == 9:
            return -0.000103600103600104
    if i == 7:
        if j == 4:
            return 0.000577200577200577
        if j == 6:
            return -0.000666000666000666
        if j == 8:
            return 0.000310800310800311
        if j == 10:
            return -5.48471136706431e-5
    if i == 8:
        if j == 5:
            return 0.000222000222000222
        if j == 7:
            return -0.000310800310800311
        if j == 9:
            return 0.000164541341011929
        if j == 11:
            return -3.17535921251092e-5
    if i == 9:
        if j == 6:
            return 0.000103600103600104
        if j == 8:
            return -0.000164541341011929
        if j == 10:
            return 9.52607763753275e-5
        if j == 12:
            return -1.96569856012580e-5
    if i == 10:
        if j == 7:
            return 5.48471136706431e-5
        if j == 9:
            return -9.52607763753275e-5
        if j == 11:
            return 5.89709568037741e-5
        if j == 13:
            return -1.28197732182118e-5
    if i == 11:
        if j == 8:
            return 3.17535921251092e-5
        if j == 10:
            return -5.89709568037741e-5
        if j == 12:
            return 3.84593196546353e-5
        if j == 14:
            return -8.71744578838400e-6
    if i == 12:
        if j == 9:
            return 1.96569856012580e-5
        if j == 11:
            return -3.84593196546353e-5
        if j == 13:
            return 2.61523373651520e-5
        if j == 15:
            return -6.13449888812208e-6
    if i == 13:
        if j == 10:
            return 1.28197732182118e-5
        if j == 12:
            return -2.61523373651520e-5
        if j == 14:
            return 1.84034966643662e-5
        if j == 16:
            return -4.44222333277806e-6
    if i == 14:
        if j == 11:
            return 8.71744578838400e-6
        if j == 13:
            return -1.84034966643662e-5
        if j == 15:
            return 1.33266699983342e-5
        if j == 17:
            return -3.29584311786759e-6
    if i == 15:
        if j == 12:
            return 6.13449888812208e-6
        if j == 14:
            return -1.33266699983342e-5
        if j == 16:
            return 9.88752935360277e-6
        if j == 18:
            return -2.49685084686939e-6
    if i == 16:
        if j == 13:
            return 4.44222333277806e-6
        if j == 15:
            return -9.88752935360277e-6
        if j == 17:
            return 7.49055254060816e-6
        if j == 19:
            return -1.92614208187067e-6
    if i == 17:
        if j == 14:
            return 3.29584311786759e-6
        if j == 16:
            return -7.49055254060816e-6
        if j == 18:
            return 5.77842624561201e-6
        if j == 20:
            return -1.50967892903377e-6
    if i == 18:
        if j == 15:
            return 2.49685084686939e-6
        if j == 17:
            return -5.77842624561201e-6
        if j == 19:
            return 4.52903678710130e-6
        if j == 21:
            return -1.20000120000120e-6
    if i == 19:
        if j == 16:
            return 1.92614208187067e-6
        if j == 18:
            return -4.52903678710130e-6
        if j == 20:
            return 3.60000360000360e-6
        if j == 22:
            return -9.65854624391210e-7
    if i == 20:
        if j == 17:
            return 1.50967892903377e-6
        if j == 19:
            return -3.60000360000360e-6
        if j == 21:
            return 2.89756387317363e-6
        if j == 23:
            return -7.86160740783543e-7
    if i == 21:
        if j == 18:
            return 1.20000120000120e-6
        if j == 20:
            return -2.89756387317363e-6
        if j == 22:
            return 2.35848222235063e-6
        if j == 24:
            return -6.46398831310913e-7
    if i == 22:
        if j == 19:
            return 9.65854624391210e-7
        if j == 21:
            return -2.35848222235063e-6
        if j == 23:
            return 1.93919649393274e-6
        if j == 25:
            return -5.36373498321821e-7
    if i == 23:
        if j == 20:
            return 7.86160740783543e-7
        if j == 22:
            return -1.93919649393274e-6
        if j == 24:
            return 1.60912049496546e-6
        if j == 26:
            return -4.48802314922340e-7
    if i == 24:
        if j == 21:
            return 6.46398831310913e-7
        if j == 23:
            return -1.60912049496546e-6
        if j == 25:
            return 1.34640694476702e-6
        if j == 27:
            return -3.78401951797267e-7
    if i == 25:
        if j == 22:
            return 5.36373498321821e-7
        if j == 24:
            return -1.34640694476702e-6
        if j == 26:
            return 1.13520585539180e-6
        if j == 28:
            return -3.21284676054284e-7
    if i == 26:
        if j == 23:
            return 4.48802314922340e-7
        if j == 25:
            return -1.13520585539180e-6
        if j == 27:
            return 9.63854028162851e-7
        if j == 29:
            return -2.74552359537297e-7
    if i == 27:
        if j == 24:
            return 3.78401951797267e-7
        if j == 26:
            return -9.63854028162851e-7
        if j == 28:
            return 8.23657078611891e-7
    if i == 28:
        if j == 25:
            return 3.21284676054284e-7
        if j == 27:
            return -8.23657078611891e-7
        if j == 29:
            return 7.08056085122503e-7
    if i == 29:
        if j == 26:
            return 2.74552359537297e-7
        if j == 28:
            return -7.08056085122503e-7


cdef double calc_ffxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return -0.6*x1t*y1t
        if j == 1:
            return -0.55*x1t*y1r
        if j == 2:
            return 0.6*x1t*y2t
        if j == 3:
            return -0.05*x1t*y2r
        if j == 5:
            return 0.0285714285714286*x1t
    if i == 1:
        if j == 0:
            return -0.05*x1r*y1t
        if j == 1:
            return -0.0666666666666667*x1r*y1r
        if j == 2:
            return 0.05*x1r*y2t
        if j == 3:
            return 0.0166666666666667*x1r*y2r
        if j == 4:
            return -0.0333333333333333*x1r
        if j == 5:
            return 0.0142857142857143*x1r
    if i == 2:
        if j == 0:
            return 0.6*x2t*y1t
        if j == 1:
            return 0.05*x2t*y1r
        if j == 2:
            return -0.6*x2t*y2t
        if j == 3:
            return 0.55*x2t*y2r
        if j == 5:
            return -0.0285714285714286*x2t
    if i == 3:
        if j == 0:
            return -0.05*x2r*y1t
        if j == 1:
            return 0.0166666666666667*x2r*y1r
        if j == 2:
            return 0.05*x2r*y2t
        if j == 3:
            return -0.0666666666666667*x2r*y2r
        if j == 4:
            return 0.0333333333333333*x2r
        if j == 5:
            return 0.0142857142857143*x2r
    if i == 4:
        if j == 1:
            return -0.0333333333333333*y1r
        if j == 3:
            return 0.0333333333333333*y2r
        if j == 4:
            return -0.0380952380952381
        if j == 6:
            return 0.00634920634920635
    if i == 5:
        if j == 0:
            return 0.0285714285714286*y1t
        if j == 1:
            return 0.0142857142857143*y1r
        if j == 2:
            return -0.0285714285714286*y2t
        if j == 3:
            return 0.0142857142857143*y2r
        if j == 5:
            return -0.0126984126984127
        if j == 7:
            return 0.00288600288600289
    if i == 6:
        if j == 4:
            return 0.00634920634920635
        if j == 6:
            return -0.00577200577200577
        if j == 8:
            return 0.00155400155400155
    if i == 7:
        if j == 5:
            return 0.00288600288600289
        if j == 7:
            return -0.00310800310800311
        if j == 9:
            return 0.000932400932400932
    if i == 8:
        if j == 6:
            return 0.00155400155400155
        if j == 8:
            return -0.00186480186480186
        if j == 10:
            return 0.000603318250377074
    if i == 9:
        if j == 7:
            return 0.000932400932400932
        if j == 9:
            return -0.00120663650075415
        if j == 11:
            return 0.000412796697626419
    if i == 10:
        if j == 8:
            return 0.000603318250377074
        if j == 10:
            return -0.000825593395252838
        if j == 12:
            return 0.000294854784018871
    if i == 11:
        if j == 9:
            return 0.000412796697626419
        if j == 11:
            return -0.000589709568037741
        if j == 13:
            return 0.000217936144709600
    if i == 12:
        if j == 10:
            return 0.000294854784018871
        if j == 12:
            return -0.000435872289419200
        if j == 14:
            return 0.000165631469979296
    if i == 13:
        if j == 11:
            return 0.000217936144709600
        if j == 13:
            return -0.000331262939958592
        if j == 15:
            return 0.000128824476650564
    if i == 14:
        if j == 12:
            return 0.000165631469979296
        if j == 14:
            return -0.000257648953301127
        if j == 16:
            return 0.000102171136653895
    if i == 15:
        if j == 13:
            return 0.000128824476650564
        if j == 15:
            return -0.000204342273307791
        if j == 17:
            return 8.23960779466897e-5
    if i == 16:
        if j == 14:
            return 0.000102171136653895
        if j == 16:
            return -0.000164792155893379
        if j == 18:
            return 6.74149728654734e-5
    if i == 17:
        if j == 15:
            return 8.23960779466897e-5
        if j == 17:
            return -0.000134829945730947
        if j == 19:
            return 5.58581203742494e-5
    if i == 18:
        if j == 16:
            return 6.74149728654734e-5
        if j == 18:
            return -0.000111716240748499
        if j == 20:
            return 4.68000468000468e-5
    if i == 19:
        if j == 17:
            return 5.58581203742494e-5
        if j == 19:
            return -9.36000936000936e-5
        if j == 21:
            return 3.96000396000396e-5
    if i == 20:
        if j == 18:
            return 4.68000468000468e-5
        if j == 20:
            return -7.92000792000792e-5
        if j == 22:
            return 3.38049118536923e-5
    if i == 21:
        if j == 19:
            return 3.96000396000396e-5
        if j == 21:
            return -6.76098237073847e-5
        if j == 23:
            return 2.90879474089911e-5
    if i == 22:
        if j == 20:
            return 3.38049118536923e-5
        if j == 22:
            return -5.81758948179822e-5
        if j == 24:
            return 2.52095544211256e-5
    if i == 23:
        if j == 21:
            return 2.90879474089911e-5
        if j == 23:
            return -5.04191088422512e-5
        if j == 25:
            return 2.19913134311947e-5
    if i == 24:
        if j == 22:
            return 2.52095544211256e-5
        if j == 24:
            return -4.39826268623894e-5
        if j == 26:
            return 1.92984995416606e-5
    if i == 25:
        if j == 23:
            return 2.19913134311947e-5
        if j == 25:
            return -3.85969990833213e-5
        if j == 27:
            return 1.70280878308770e-5
    if i == 26:
        if j == 24:
            return 1.92984995416606e-5
        if j == 26:
            return -3.40561756617541e-5
        if j == 28:
            return 1.51003797745513e-5
    if i == 27:
        if j == 25:
            return 1.70280878308770e-5
        if j == 27:
            return -3.02007595491027e-5
        if j == 29:
            return 1.34530656173275e-5
    if i == 28:
        if j == 26:
            return 1.51003797745513e-5
        if j == 28:
            return -2.69061312346551e-5
    if i == 29:
        if j == 27:
            return 1.34530656173275e-5
        if j == 29:
            return -2.40739068941651e-5


cdef double calc_fxifxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return 0.6*x1t*y1t
        if j == 1:
            return 0.05*x1t*y1r
        if j == 2:
            return -0.6*x1t*y2t
        if j == 3:
            return 0.05*x1t*y2r
        if j == 5:
            return -0.0285714285714286*x1t
    if i == 1:
        if j == 0:
            return 0.05*x1r*y1t
        if j == 1:
            return 0.0666666666666667*x1r*y1r
        if j == 2:
            return -0.05*x1r*y2t
        if j == 3:
            return -0.0166666666666667*x1r*y2r
        if j == 4:
            return 0.0333333333333333*x1r
        if j == 5:
            return -0.0142857142857143*x1r
    if i == 2:
        if j == 0:
            return -0.6*x2t*y1t
        if j == 1:
            return -0.05*x2t*y1r
        if j == 2:
            return 0.6*x2t*y2t
        if j == 3:
            return -0.05*x2t*y2r
        if j == 5:
            return 0.0285714285714286*x2t
    if i == 3:
        if j == 0:
            return 0.05*x2r*y1t
        if j == 1:
            return -0.0166666666666667*x2r*y1r
        if j == 2:
            return -0.05*x2r*y2t
        if j == 3:
            return 0.0666666666666667*x2r*y2r
        if j == 4:
            return -0.0333333333333333*x2r
        if j == 5:
            return -0.0142857142857143*x2r
    if i == 4:
        if j == 1:
            return 0.0333333333333333*y1r
        if j == 3:
            return -0.0333333333333333*y2r
        if j == 4:
            return 0.0380952380952381
        if j == 6:
            return -0.00634920634920635
    if i == 5:
        if j == 0:
            return -0.0285714285714286*y1t
        if j == 1:
            return -0.0142857142857143*y1r
        if j == 2:
            return 0.0285714285714286*y2t
        if j == 3:
            return -0.0142857142857143*y2r
        if j == 5:
            return 0.0126984126984127
        if j == 7:
            return -0.00288600288600289
    if i == 6:
        if j == 4:
            return -0.00634920634920635
        if j == 6:
            return 0.00577200577200577
        if j == 8:
            return -0.00155400155400155
    if i == 7:
        if j == 5:
            return -0.00288600288600289
        if j == 7:
            return 0.00310800310800311
        if j == 9:
            return -0.000932400932400932
    if i == 8:
        if j == 6:
            return -0.00155400155400155
        if j == 8:
            return 0.00186480186480186
        if j == 10:
            return -0.000603318250377074
    if i == 9:
        if j == 7:
            return -0.000932400932400932
        if j == 9:
            return 0.00120663650075415
        if j == 11:
            return -0.000412796697626419
    if i == 10:
        if j == 8:
            return -0.000603318250377074
        if j == 10:
            return 0.000825593395252838
        if j == 12:
            return -0.000294854784018871
    if i == 11:
        if j == 9:
            return -0.000412796697626419
        if j == 11:
            return 0.000589709568037741
        if j == 13:
            return -0.000217936144709600
    if i == 12:
        if j == 10:
            return -0.000294854784018871
        if j == 12:
            return 0.000435872289419200
        if j == 14:
            return -0.000165631469979296
    if i == 13:
        if j == 11:
            return -0.000217936144709600
        if j == 13:
            return 0.000331262939958592
        if j == 15:
            return -0.000128824476650564
    if i == 14:
        if j == 12:
            return -0.000165631469979296
        if j == 14:
            return 0.000257648953301127
        if j == 16:
            return -0.000102171136653895
    if i == 15:
        if j == 13:
            return -0.000128824476650564
        if j == 15:
            return 0.000204342273307791
        if j == 17:
            return -8.23960779466897e-5
    if i == 16:
        if j == 14:
            return -0.000102171136653895
        if j == 16:
            return 0.000164792155893379
        if j == 18:
            return -6.74149728654734e-5
    if i == 17:
        if j == 15:
            return -8.23960779466897e-5
        if j == 17:
            return 0.000134829945730947
        if j == 19:
            return -5.58581203742494e-5
    if i == 18:
        if j == 16:
            return -6.74149728654734e-5
        if j == 18:
            return 0.000111716240748499
        if j == 20:
            return -4.68000468000468e-5
    if i == 19:
        if j == 17:
            return -5.58581203742494e-5
        if j == 19:
            return 9.36000936000936e-5
        if j == 21:
            return -3.96000396000396e-5
    if i == 20:
        if j == 18:
            return -4.68000468000468e-5
        if j == 20:
            return 7.92000792000792e-5
        if j == 22:
            return -3.38049118536923e-5
    if i == 21:
        if j == 19:
            return -3.96000396000396e-5
        if j == 21:
            return 6.76098237073847e-5
        if j == 23:
            return -2.90879474089911e-5
    if i == 22:
        if j == 20:
            return -3.38049118536923e-5
        if j == 22:
            return 5.81758948179822e-5
        if j == 24:
            return -2.52095544211256e-5
    if i == 23:
        if j == 21:
            return -2.90879474089911e-5
        if j == 23:
            return 5.04191088422512e-5
        if j == 25:
            return -2.19913134311947e-5
    if i == 24:
        if j == 22:
            return -2.52095544211256e-5
        if j == 24:
            return 4.39826268623894e-5
        if j == 26:
            return -1.92984995416606e-5
    if i == 25:
        if j == 23:
            return -2.19913134311947e-5
        if j == 25:
            return 3.85969990833213e-5
        if j == 27:
            return -1.70280878308770e-5
    if i == 26:
        if j == 24:
            return -1.92984995416606e-5
        if j == 26:
            return 3.40561756617541e-5
        if j == 28:
            return -1.51003797745513e-5
    if i == 27:
        if j == 25:
            return -1.70280878308770e-5
        if j == 27:
            return 3.02007595491027e-5
        if j == 29:
            return -1.34530656173275e-5
    if i == 28:
        if j == 26:
            return -1.51003797745513e-5
        if j == 28:
            return 2.69061312346551e-5
    if i == 29:
        if j == 27:
            return -1.34530656173275e-5
        if j == 29:
            return 2.40739068941651e-5


cdef double calc_fxifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 1:
            return 0.25*x1t*y1r
        if j == 3:
            return -0.25*x1t*y2r
        if j == 4:
            return 0.2*x1t
    if i == 1:
        if j == 0:
            return -0.25*x1r*y1t
        if j == 1:
            return -0.125*x1r*y1r
        if j == 2:
            return 0.25*x1r*y2t
        if j == 3:
            return -0.125*x1r*y2r
        if j == 4:
            return 0.1*x1r
    if i == 2:
        if j == 1:
            return -0.25*x2t*y1r
        if j == 3:
            return 0.25*x2t*y2r
        if j == 4:
            return -0.2*x2t
    if i == 3:
        if j == 0:
            return 0.25*x2r*y1t
        if j == 1:
            return 0.125*x2r*y1r
        if j == 2:
            return -0.25*x2r*y2t
        if j == 3:
            return 0.125*x2r*y2r
        if j == 4:
            return 0.1*x2r
    if i == 4:
        if j == 0:
            return -0.2*y1t
        if j == 1:
            return -0.1*y1r
        if j == 2:
            return 0.2*y2t
        if j == 3:
            return -0.1*y2r
        if j == 5:
            return 0.0571428571428571
    if i == 5:
        if j == 4:
            return -0.0571428571428571
        if j == 6:
            return 0.0317460317460317
    if i == 6:
        if j == 5:
            return -0.0317460317460317
        if j == 7:
            return 0.0202020202020202
    if i == 7:
        if j == 6:
            return -0.0202020202020202
        if j == 8:
            return 0.0139860139860140
    if i == 8:
        if j == 7:
            return -0.0139860139860140
        if j == 9:
            return 0.0102564102564103
    if i == 9:
        if j == 8:
            return -0.0102564102564103
        if j == 10:
            return 0.00784313725490196
    if i == 10:
        if j == 9:
            return -0.00784313725490196
        if j == 11:
            return 0.00619195046439629
    if i == 11:
        if j == 10:
            return -0.00619195046439629
        if j == 12:
            return 0.00501253132832080
    if i == 12:
        if j == 11:
            return -0.00501253132832080
        if j == 13:
            return 0.00414078674948240
    if i == 13:
        if j == 12:
            return -0.00414078674948240
        if j == 14:
            return 0.00347826086956522
    if i == 14:
        if j == 13:
            return -0.00347826086956522
        if j == 15:
            return 0.00296296296296296
    if i == 15:
        if j == 14:
            return -0.00296296296296296
        if j == 16:
            return 0.00255427841634738
    if i == 16:
        if j == 15:
            return -0.00255427841634738
        if j == 17:
            return 0.00222469410456062
    if i == 17:
        if j == 16:
            return -0.00222469410456062
        if j == 18:
            return 0.00195503421309873
    if i == 18:
        if j == 17:
            return -0.00195503421309873
        if j == 19:
            return 0.00173160173160173
    if i == 19:
        if j == 18:
            return -0.00173160173160173
        if j == 20:
            return 0.00154440154440154
    if i == 20:
        if j == 19:
            return -0.00154440154440154
        if j == 21:
            return 0.00138600138600139
    if i == 21:
        if j == 20:
            return -0.00138600138600139
        if j == 22:
            return 0.00125078173858662
    if i == 22:
        if j == 21:
            return -0.00125078173858662
        if j == 23:
            return 0.00113442994895065
    if i == 23:
        if j == 22:
            return -0.00113442994895065
        if j == 24:
            return 0.00103359173126615
    if i == 24:
        if j == 23:
            return -0.00103359173126615
        if j == 25:
            return 0.000945626477541371
    if i == 25:
        if j == 24:
            return -0.000945626477541371
        if j == 26:
            return 0.000868432479374729
    if i == 26:
        if j == 25:
            return -0.000868432479374729
        if j == 27:
            return 0.000800320128051221
    if i == 27:
        if j == 26:
            return -0.000800320128051221
        if j == 28:
            return 0.000739918608953015
    if i == 28:
        if j == 27:
            return -0.000739918608953015
        if j == 29:
            return 0.000686106346483705
    if i == 29:
        if j == 28:
            return -0.000686106346483705


cdef double calc_fxixifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return 1.5*x1t*y1t
        if j == 1:
            return 0.75*x1t*y1r
        if j == 2:
            return -1.5*x1t*y2t
        if j == 3:
            return 0.75*x1t*y2r
    if i == 1:
        if j == 0:
            return 0.75*x1r*y1t
        if j == 1:
            return 0.5*x1r*y1r
        if j == 2:
            return -0.75*x1r*y2t
        if j == 3:
            return 0.25*x1r*y2r
    if i == 2:
        if j == 0:
            return -1.5*x2t*y1t
        if j == 1:
            return -0.75*x2t*y1r
        if j == 2:
            return 1.5*x2t*y2t
        if j == 3:
            return -0.75*x2t*y2r
    if i == 3:
        if j == 0:
            return 0.75*x2r*y1t
        if j == 1:
            return 0.25*x2r*y1r
        if j == 2:
            return -0.75*x2r*y2t
        if j == 3:
            return 0.5*x2r*y2r
    if i == 4:
        if j == 4:
            return 0.400000000000000
    if i == 5:
        if j == 5:
            return 0.285714285714286
    if i == 6:
        if j == 6:
            return 0.222222222222222
    if i == 7:
        if j == 7:
            return 0.181818181818182
    if i == 8:
        if j == 8:
            return 0.153846153846154
    if i == 9:
        if j == 9:
            return 0.133333333333333
    if i == 10:
        if j == 10:
            return 0.117647058823529
    if i == 11:
        if j == 11:
            return 0.105263157894737
    if i == 12:
        if j == 12:
            return 0.0952380952380952
    if i == 13:
        if j == 13:
            return 0.0869565217391304
    if i == 14:
        if j == 14:
            return 0.0800000000000000
    if i == 15:
        if j == 15:
            return 0.0740740740740741
    if i == 16:
        if j == 16:
            return 0.0689655172413793
    if i == 17:
        if j == 17:
            return 0.0645161290322581
    if i == 18:
        if j == 18:
            return 0.0606060606060606
    if i == 19:
        if j == 19:
            return 0.0571428571428571
    if i == 20:
        if j == 20:
            return 0.0540540540540541
    if i == 21:
        if j == 21:
            return 0.0512820512820513
    if i == 22:
        if j == 22:
            return 0.0487804878048781
    if i == 23:
        if j == 23:
            return 0.0465116279069767
    if i == 24:
        if j == 24:
            return 0.0444444444444444
    if i == 25:
        if j == 25:
            return 0.0425531914893617
    if i == 26:
        if j == 26:
            return 0.0408163265306122
    if i == 27:
        if j == 27:
            return 0.0392156862745098
    if i == 28:
        if j == 28:
            return 0.0377358490566038
    if i == 29:
        if j == 29:
            return 0.0363636363636364
