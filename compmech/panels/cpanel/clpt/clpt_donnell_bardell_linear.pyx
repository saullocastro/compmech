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
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_ffxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_ffxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxifxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxixifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef double pi = 3.141592653589793
cdef int num1 = 3


def fk0(double a, double b, double r, np.ndarray[cDOUBLE, ndim=2] F,
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

            fAufBu = integral_ff(i1, k1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAufBuxi = integral_ffxi(i1, k1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAuxifBu = integral_ffxi(k1, i1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAuxifBuxi = integral_fxifxi(i1, k1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAufBv = integral_ff(i1, k1, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
            fAufBvxi = integral_ffxi(i1, k1, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
            fAuxifBv = integral_ffxi(k1, i1, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
            fAuxifBvxi = integral_fxifxi(i1, k1, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
            fAuxifBwxixi = integral_fxifxixi(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAuxifBw = integral_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAufBwxixi = integral_ffxixi(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAuxifBwxi = integral_fxifxi(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAufBw = integral_ff(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAufBwxi = integral_ffxi(i1, k1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAvfBuxi = integral_ffxi(i1, k1, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
            fAvxifBuxi = integral_fxifxi(i1, k1, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
            fAvfBu = integral_ff(i1, k1, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
            fAvxifBu = integral_ffxi(k1, i1, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
            fAvfBv = integral_ff(i1, k1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvfBvxi = integral_ffxi(i1, k1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvxifBv = integral_ffxi(k1, i1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvxifBvxi = integral_fxifxi(i1, k1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvfBwxixi = integral_ffxixi(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAvxifBwxixi = integral_fxifxixi(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAvfBw = integral_ff(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAvfBwxi = integral_ffxi(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAvxifBw = integral_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAvxifBwxi = integral_fxifxi(i1, k1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBuxi = integral_fxifxixi(k1, i1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBuxi = integral_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAwxifBuxi = integral_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAwxixifBu = integral_ffxixi(k1, i1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBu = integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAwxifBu = integral_ffxi(k1, i1, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBv = integral_ffxixi(k1, i1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBvxi = integral_fxifxixi(k1, i1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBv = integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAwfBvxi = integral_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAwxifBv = integral_ffxi(k1, i1, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBvxi = integral_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAwxixifBwxixi = integral_fxixifxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxixi = integral_ffxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBw = integral_ffxixi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBwxixi = integral_fxifxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBwxi = integral_fxifxixi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBw = integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxi = integral_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBw = integral_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBwxi = integral_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)


            for j1 in range(0, n1):
                for l1 in range(0, n1):

                    row = num1*(j1*m1 + i1)
                    col = num1*(l1*m1 + k1)

                    if row > col:
                        continue

                    gAugBu = integral_ff(j1, l1, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                    gAugBueta = integral_ffxi(j1, l1, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                    gAuetagBu = integral_ffxi(l1, j1, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                    gAuetagBueta = integral_fxifxi(j1, l1, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                    gAugBv = integral_ff(j1, l1, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                    gAugBveta = integral_ffxi(j1, l1, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                    gAuetagBv = integral_ffxi(l1, j1, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                    gAuetagBveta = integral_fxifxi(j1, l1, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                    gAuetagBwetaeta = integral_fxifxixi(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAuetagBw = integral_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                    gAugBwetaeta = integral_ffxixi(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAuetagBweta = integral_fxifxi(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAugBw = integral_ff(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAugBweta = integral_ffxi(j1, l1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvgBueta = integral_ffxi(j1, l1, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                    gAvetagBueta = integral_fxifxi(j1, l1, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                    gAvgBu = integral_ff(j1, l1, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                    gAvetagBu = integral_ffxi(l1, j1, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvgBv = integral_ff(j1, l1, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvgBveta = integral_ffxi(j1, l1, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvetagBv = integral_ffxi(l1, j1, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvetagBveta = integral_fxifxi(j1, l1, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvgBwetaeta = integral_ffxixi(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvetagBwetaeta = integral_fxifxixi(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvgBw = integral_ff(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvgBweta = integral_ffxi(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAvetagBw = integral_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                    gAvetagBweta = integral_fxifxi(j1, l1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBueta = integral_fxifxixi(l1, j1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBueta = integral_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                    gAwetagBueta = integral_fxifxi(j1, l1, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                    gAwetaetagBu = integral_ffxixi(l1, j1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBu = integral_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                    gAwetagBu = integral_ffxi(l1, j1, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBv = integral_ffxixi(l1, j1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBveta = integral_fxifxixi(l1, j1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBv = integral_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                    gAwgBveta = integral_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                    gAwetagBv = integral_ffxi(l1, j1, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBveta = integral_fxifxi(j1, l1, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                    gAwetaetagBwetaeta = integral_fxixifxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBwetaeta = integral_ffxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBw = integral_ffxixi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBwetaeta = integral_fxifxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBweta = integral_fxifxixi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBw = integral_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBweta = integral_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBw = integral_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBweta = integral_fxifxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

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
                    k0v[c] += 0.5*A12*b*fAuxifBw*gAugBw/r + 0.5*A26*a*fAufBw*gAuetagBw/r - 2*B11*b*fAuxifBwxixi*gAugBw/(a*a) - 2*B12*fAuxifBw*gAugBwetaeta/b - 2*B16*(fAufBwxixi*gAuetagBw + 2*fAuxifBwxi*gAugBweta)/a - 2*B26*a*fAufBw*gAuetagBwetaeta/(b*b) - 4*B66*fAufBwxi*gAuetagBweta/b
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
                    k0v[c] += 0.5*A22*a*fAvfBw*gAvetagBw/r + 0.5*A26*b*fAvxifBw*gAvgBw/r - 2*B12*fAvfBwxixi*gAvetagBw/a - 2*B16*b*fAvxifBwxixi*gAvgBw/(a*a) - 2*B22*a*fAvfBw*gAvetagBwetaeta/(b*b) - 2*B26*(2*fAvfBwxi*gAvetagBweta + fAvxifBw*gAvgBwetaeta)/b - 4*B66*fAvxifBwxi*gAvgBweta/a
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+0
                    k0v[c] += 0.5*A12*b*fAwfBuxi*gAwgBu/r + 0.5*A26*a*fAwfBu*gAwgBueta/r - 2*B11*b*fAwxixifBuxi*gAwgBu/(a*a) - 2*B12*fAwfBuxi*gAwetaetagBu/b - 2*B16*(2*fAwxifBuxi*gAwetagBu + fAwxixifBu*gAwgBueta)/a - 2*B26*a*fAwfBu*gAwetaetagBueta/(b*b) - 4*B66*fAwxifBu*gAwetagBueta/b
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+1
                    k0v[c] += 0.5*A22*a*fAwfBv*gAwgBveta/r + 0.5*A26*b*fAwfBvxi*gAwgBv/r - 2*B12*fAwxixifBv*gAwgBveta/a - 2*B16*b*fAwxixifBvxi*gAwgBv/(a*a) - 2*B22*a*fAwfBv*gAwetaetagBveta/(b*b) - 2*B26*(fAwfBvxi*gAwetaetagBv + 2*fAwxifBv*gAwetagBveta)/b - 4*B66*fAwxifBvxi*gAwetagBv/a
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += 0.25*A22*a*b*fAwfBw*gAwgBw/(r*r) - B12*b*gAwgBw*(fAwfBwxixi + fAwxixifBw)/(a*r) - B22*a*fAwfBw*(gAwgBwetaeta + gAwetaetagBw)/(b*r) - 2*B26*(fAwfBwxi*gAwgBweta + fAwxifBw*gAwetagBw)/r + 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*D12*(fAwfBwxixi*gAwetaetagBw + fAwxixifBw*gAwgBwetaeta)/(a*b) + 8*D16*(fAwxifBwxixi*gAwetagBw + fAwxixifBwxi*gAwgBweta)/(a*a) + 4*D22*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*D26*(fAwfBwxi*gAwetaetagBweta + fAwxifBw*gAwetagBwetaeta)/(b*b) + 16*D66*fAwxifBwxi*gAwetagBweta/(a*b)

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

            fAwxifBwxi = integral_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxi = integral_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBw = integral_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBw = integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

            for j1 in range(0, n1):
                for l1 in range(0, n1):

                    row = num1*(j1*m1 + i1)
                    col = num1*(l1*m1 + k1)

                    if row > col:
                        continue

                    gAwgBw = integral_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBweta = integral_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBw = integral_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBweta = integral_fxifxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                    c += 1
                    kG0r[c] = row+2
                    kG0c[c] = col+2
                    kG0v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

    size = num1*m1*n1

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
