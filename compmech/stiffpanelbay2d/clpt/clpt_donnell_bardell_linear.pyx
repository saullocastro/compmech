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

cdef extern from 'bardell_12.h':
    double integral_ff_12(double xi1, double xi2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_ffxi_12(double xi1, double xi2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_ffxixi_12(double xi1, double xi2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxifxi_12(double xi1, double xi2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxifxixi_12(double xi1, double xi2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxixifxixi_12(double xi1, double xi2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)

cdef extern from 'bardell_functions.h':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r)
    double calc_fxi(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r)

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int num = 3
cdef int num1 = 3
cdef double pi = 3.141592653589793


def fk0y1y2(double y1, double y2, double a, double b, double r,
            np.ndarray[cDOUBLE, ndim=2] F, int m, int n,
            double u1tx, double u1rx, double u2tx, double u2rx,
            double v1tx, double v1rx, double v2tx, double v2rx,
            double w1tx, double w1rx, double w2tx, double w2rx,
            double u1ty, double u1ry, double u2ty, double u2ry,
            double v1ty, double v1ry, double v2ty, double v2ry,
            double w1ty, double w1ry, double w2ty, double w2ry,
            int size, int row0, int col0):
    cdef int i, j, k, l, row, col, c
    cdef double xi1, xi2
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef np.ndarray[cINT, ndim=1] k0y1y2r, k0y1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] k0y1y2v

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

    fdim = 9*m*n*m*n

    k0y1y2r = np.zeros((fdim,), dtype=INT)
    k0y1y2c = np.zeros((fdim,), dtype=INT)
    k0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

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

    xi1 = 2*y1/b - 1.
    xi2 = 2*y2/b - 1.

    # k0y1y2
    c = -1
    for j in range(n):
        for l in range(n):

            gAugBu = integral_ff_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
            gAugBueta = integral_ffxi_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
            gAuetagBu = integral_ffxi_12(xi1, xi2, l, j, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
            gAuetagBueta = integral_fxifxi_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
            gAugBv = integral_ff_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
            gAugBveta = integral_ffxi_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
            gAuetagBv = integral_ffxi_12(xi1, xi2, l, j, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
            gAuetagBveta = integral_fxifxi_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
            gAuetagBwetaeta = integral_fxifxixi_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAuetagBw = integral_ffxi_12(xi1, xi2, l, j, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
            gAugBwetaeta = integral_ffxixi_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAuetagBweta = integral_fxifxi_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAugBw = integral_ff_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAugBweta = integral_ffxi_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAvgBueta = integral_ffxi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
            gAvetagBueta = integral_fxifxi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
            gAvgBu = integral_ff_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
            gAvetagBu = integral_ffxi_12(xi1, xi2, l, j, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
            gAvgBv = integral_ff_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
            gAvgBveta = integral_ffxi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
            gAvetagBv = integral_ffxi_12(xi1, xi2, l, j, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
            gAvetagBveta = integral_fxifxi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
            gAvgBwetaeta = integral_ffxixi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAvetagBwetaeta = integral_fxifxixi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAvgBw = integral_ff_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAvgBweta = integral_ffxi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAvetagBw = integral_ffxi_12(xi1, xi2, l, j, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
            gAvetagBweta = integral_fxifxi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetaetagBueta = integral_fxifxixi_12(xi1, xi2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBueta = integral_ffxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
            gAwetagBueta = integral_fxifxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
            gAwetaetagBu = integral_ffxixi_12(xi1, xi2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBu = integral_ff_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
            gAwetagBu = integral_ffxi_12(xi1, xi2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetaetagBv = integral_ffxixi_12(xi1, xi2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetaetagBveta = integral_fxifxixi_12(xi1, xi2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBv = integral_ff_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
            gAwgBveta = integral_ffxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
            gAwetagBv = integral_ffxi_12(xi1, xi2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetagBveta = integral_fxifxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
            gAwetaetagBwetaeta = integral_fxixifxixi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBwetaeta = integral_ffxixi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetaetagBw = integral_ffxixi_12(xi1, xi2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetagBwetaeta = integral_fxifxixi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetaetagBweta = integral_fxifxixi_12(xi1, xi2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBw = integral_ff_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBweta = integral_ffxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetagBw = integral_ffxi_12(xi1, xi2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetagBweta = integral_fxifxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

            for i in range(m):
                for k in range(m):
                    row = row0 + num*(j*m + i)
                    col = col0 + num*(l*m + k)

                    #NOTE symmetry
                    if row > col:
                        continue

                    fAufBu = integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBuxi = integral_ffxi(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBu = integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBuxi = integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBv = integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAufBvxi = integral_ffxi(i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAuxifBv = integral_ffxi(k, i, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBvxi = integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAuxifBwxixi = integral_fxifxixi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAuxifBw = integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBwxixi = integral_ffxixi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAuxifBwxi = integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAufBw = integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAufBwxi = integral_ffxi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBuxi = integral_ffxi(i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvxifBuxi = integral_fxifxi(i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvfBu = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvxifBu = integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBv = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBvxi = integral_ffxi(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBv = integral_ffxi(k, i, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBvxi = integral_fxifxi(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBwxixi = integral_ffxixi(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvxifBwxixi = integral_fxifxixi(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBw = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBwxi = integral_ffxi(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvxifBw = integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBwxi = integral_fxifxi(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBuxi = integral_fxifxixi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBuxi = integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxifBuxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxixifBu = integral_ffxixi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBu = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxifBu = integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBv = integral_ffxixi(k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBvxi = integral_fxifxixi(k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBv = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwfBvxi = integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwxifBv = integral_ffxi(k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBvxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
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
                    k0y1y2v[c] += A11*b*fAuxifBuxi*gAugBu/a + A16*(fAufBuxi*gAuetagBu + fAuxifBu*gAugBueta) + A66*a*fAufBu*gAuetagBueta/b
                    c += 1
                    k0y1y2r[c] = row+0
                    k0y1y2c[c] = col+1
                    k0y1y2v[c] += A12*fAuxifBv*gAugBveta + A16*b*fAuxifBvxi*gAugBv/a + A26*a*fAufBv*gAuetagBveta/b + A66*fAufBvxi*gAuetagBv
                    c += 1
                    k0y1y2r[c] = row+0
                    k0y1y2c[c] = col+2
                    k0y1y2v[c] += 0.5*A12*b*fAuxifBw*gAugBw/r + 0.5*A26*a*fAufBw*gAuetagBw/r - 2*B11*b*fAuxifBwxixi*gAugBw/(a*a) - 2*B12*fAuxifBw*gAugBwetaeta/b - 2*B16*(fAufBwxixi*gAuetagBw + 2*fAuxifBwxi*gAugBweta)/a - 2*B26*a*fAufBw*gAuetagBwetaeta/(b*b) - 4*B66*fAufBwxi*gAuetagBweta/b
                    c += 1
                    k0y1y2r[c] = row+1
                    k0y1y2c[c] = col+0
                    k0y1y2v[c] += A12*fAvfBuxi*gAvetagBu + A16*b*fAvxifBuxi*gAvgBu/a + A26*a*fAvfBu*gAvetagBueta/b + A66*fAvxifBu*gAvgBueta
                    c += 1
                    k0y1y2r[c] = row+1
                    k0y1y2c[c] = col+1
                    k0y1y2v[c] += A22*a*fAvfBv*gAvetagBveta/b + A26*(fAvfBvxi*gAvetagBv + fAvxifBv*gAvgBveta) + A66*b*fAvxifBvxi*gAvgBv/a
                    c += 1
                    k0y1y2r[c] = row+1
                    k0y1y2c[c] = col+2
                    k0y1y2v[c] += 0.5*A22*a*fAvfBw*gAvetagBw/r + 0.5*A26*b*fAvxifBw*gAvgBw/r - 2*B12*fAvfBwxixi*gAvetagBw/a - 2*B16*b*fAvxifBwxixi*gAvgBw/(a*a) - 2*B22*a*fAvfBw*gAvetagBwetaeta/(b*b) - 2*B26*(2*fAvfBwxi*gAvetagBweta + fAvxifBw*gAvgBwetaeta)/b - 4*B66*fAvxifBwxi*gAvgBweta/a
                    c += 1
                    k0y1y2r[c] = row+2
                    k0y1y2c[c] = col+0
                    k0y1y2v[c] += 0.5*A12*b*fAwfBuxi*gAwgBu/r + 0.5*A26*a*fAwfBu*gAwgBueta/r - 2*B11*b*fAwxixifBuxi*gAwgBu/(a*a) - 2*B12*fAwfBuxi*gAwetaetagBu/b - 2*B16*(2*fAwxifBuxi*gAwetagBu + fAwxixifBu*gAwgBueta)/a - 2*B26*a*fAwfBu*gAwetaetagBueta/(b*b) - 4*B66*fAwxifBu*gAwetagBueta/b
                    c += 1
                    k0y1y2r[c] = row+2
                    k0y1y2c[c] = col+1
                    k0y1y2v[c] += 0.5*A22*a*fAwfBv*gAwgBveta/r + 0.5*A26*b*fAwfBvxi*gAwgBv/r - 2*B12*fAwxixifBv*gAwgBveta/a - 2*B16*b*fAwxixifBvxi*gAwgBv/(a*a) - 2*B22*a*fAwfBv*gAwetaetagBveta/(b*b) - 2*B26*(fAwfBvxi*gAwetaetagBv + 2*fAwxifBv*gAwetagBveta)/b - 4*B66*fAwxifBvxi*gAwetagBv/a
                    c += 1
                    k0y1y2r[c] = row+2
                    k0y1y2c[c] = col+2
                    k0y1y2v[c] += 0.25*A22*a*b*fAwfBw*gAwgBw/(r*r) - B12*b*gAwgBw*(fAwfBwxixi + fAwxixifBw)/(a*r) - B22*a*fAwfBw*(gAwgBwetaeta + gAwetaetagBw)/(b*r) - 2*B26*(fAwfBwxi*gAwgBweta + fAwxifBw*gAwetagBw)/r + 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*D12*(fAwfBwxixi*gAwetaetagBw + fAwxixifBw*gAwgBwetaeta)/(a*b) + 8*D16*(fAwxifBwxixi*gAwetagBw + fAwxixifBwxi*gAwgBweta)/(a*a) + 4*D22*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*D26*(fAwfBwxi*gAwetaetagBweta + fAwxifBw*gAwetagBwetaeta)/(b*b) + 16*D66*fAwxifBwxi*gAwetagBweta/(a*b)


    k0 = coo_matrix((k0y1y2v, (k0y1y2r, k0y1y2c)), shape=(size, size))

    return k0


def fkG0y1y2(double y1, double y2, double Nxx, double Nyy, double Nxy,
             double a, double b, double r, int m, int n,
             double w1tx, double w1rx, double w2tx, double w2rx,
             double w1ty, double w1ry, double w2ty, double w2ry,
             int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double xi1, xi2

    cdef np.ndarray[cINT, ndim=1] kG0y1y2r, kG0y1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0y1y2v

    cdef double fAwxifBwxi, fAwfBwxi, fAwxifBw, fAwfBw
    cdef double gAwetagBweta, gAwgBweta, gAwetagBw, gAwgBw

    fdim = 1*m*n*m*n

    kG0y1y2r = np.zeros((fdim,), dtype=INT)
    kG0y1y2c = np.zeros((fdim,), dtype=INT)
    kG0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

    xi1 = 2*y1/b - 1.
    xi2 = 2*y2/b - 1.

    # kG0y1y2
    c = -1
    for j in range(n):
        for l in range(n):

            gAwetagBw = integral_ffxi_12(xi1, xi2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBw = integral_ff_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBweta = integral_ffxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetagBweta = integral_fxifxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

            for i in range(m):
                for k in range(m):
                    col = col0 + num*(l*m + k)
                    row = row0 + num*(j*m + i)

                    #NOTE symmetry
                    if row > col:
                        continue

                    fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBwxi = integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBw = integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                    c += 1
                    kG0y1y2r[c] = row+2
                    kG0y1y2c[c] = col+2
                    kG0y1y2v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

    kG0 = coo_matrix((kG0y1y2v, (kG0y1y2r, kG0y1y2c)), shape=(size, size))

    return kG0


def fkMy1y2(double y1, double y2, double mu, double d, double h,
            double a, double b, int m, int n,
            double u1tx, double u1rx, double u2tx, double u2rx,
            double v1tx, double v1rx, double v2tx, double v2rx,
            double w1tx, double w1rx, double w2tx, double w2rx,
            double u1ty, double u1ry, double u2ty, double u2ry,
            double v1ty, double v1ry, double v2ty, double v2ry,
            double w1ty, double w1ry, double w2ty, double w2ry,
            int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double xi1, xi2

    cdef np.ndarray[cINT, ndim=1] kMy1y2r, kMy1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] kMy1y2v

    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwxifBu, fAwfBv, fAwfBw, fAwxifBwxi
    cdef double gAugBu, gAugBw, gAvgBv, gAvgBweta, gAwgBu, gAwetagBv, gAwgBw, gAwetagBweta

    fdim = 7*m*n*m*n

    kMy1y2r = np.zeros((fdim,), dtype=INT)
    kMy1y2c = np.zeros((fdim,), dtype=INT)
    kMy1y2v = np.zeros((fdim,), dtype=DOUBLE)

    xi1 = 2*y1/b - 1.
    xi2 = 2*y2/b - 1.

    # kMy1y2
    c = -1
    for j in range(n):
        for l in range(n):

            gAugBu = integral_ff_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
            gAugBw = integral_ff_12(xi1, xi2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
            gAvgBv = integral_ff_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
            gAvgBweta = integral_ffxi_12(xi1, xi2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBu = integral_ff_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
            gAwetagBv = integral_ffxi_12(xi1, xi2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
            gAwgBw = integral_ff_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
            gAwetagBweta = integral_fxifxi_12(xi1, xi2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

            for i in range(m):
                for k in range(m):

                    row = row0 + num*(j*m + i)
                    col = col0 + num*(l*m + k)

                    #NOTE symmetry
                    if row > col:
                        continue

                    fAufBu = integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBwxi = integral_ffxi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBv = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBw = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBu = integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBv = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                    c += 1
                    kMy1y2r[c] = row+0
                    kMy1y2c[c] = col+0
                    kMy1y2v[c] += 0.25*a*b*fAufBu*gAugBu*h*mu
                    c += 1
                    kMy1y2r[c] = row+0
                    kMy1y2c[c] = col+2
                    kMy1y2v[c] += 0.5*b*d*fAufBwxi*gAugBw*h*mu
                    c += 1
                    kMy1y2r[c] = row+1
                    kMy1y2c[c] = col+1
                    kMy1y2v[c] += 0.25*a*b*fAvfBv*gAvgBv*h*mu
                    c += 1
                    kMy1y2r[c] = row+1
                    kMy1y2c[c] = col+2
                    kMy1y2v[c] += 0.5*a*d*fAvfBw*gAvgBweta*h*mu
                    c += 1
                    kMy1y2r[c] = row+2
                    kMy1y2c[c] = col+0
                    kMy1y2v[c] += 0.5*b*d*fAwxifBu*gAwgBu*h*mu
                    c += 1
                    kMy1y2r[c] = row+2
                    kMy1y2c[c] = col+1
                    kMy1y2v[c] += 0.5*a*d*fAwfBv*gAwetagBv*h*mu
                    c += 1
                    kMy1y2r[c] = row+2
                    kMy1y2c[c] = col+2
                    kMy1y2v[c] += 0.25*a*b*h*mu*(fAwfBw*gAwgBw + 4*fAwfBw*gAwetagBweta*((d*d) + 0.0833333333333333*(h*h))/(b*b) + 4*fAwxifBwxi*gAwgBw*((d*d) + 0.0833333333333333*(h*h))/(a*a))

    kM = coo_matrix((kMy1y2v, (kMy1y2r, kMy1y2c)), shape=(size, size))

    return kM


def fkAx(double beta, double gamma, double a, double b, int m, int n,
         double w1tx, double w1rx, double w2tx, double w2rx,
         double w1ty, double w1ry, double w2ty, double w2ry,
         int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef np.ndarray[cINT, ndim=1] kAxr, kAxc
    cdef np.ndarray[cDOUBLE, ndim=1] kAxv

    cdef double fAwfBw, fAwxifBw, gAwgBw

    fdim = 1*m*n*m*n

    kAxr = np.zeros((fdim,), dtype=INT)
    kAxc = np.zeros((fdim,), dtype=INT)
    kAxv = np.zeros((fdim,), dtype=DOUBLE)

    # kAx
    c = -1
    for i in range(m):
        for k in range(m):

            fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
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
                    kAxr[c] = row+2
                    kAxc[c] = col+2
                    kAxv[c] += -0.25*a*b*(fAwfBw*gAwgBw*gamma + 2*beta*fAwxifBw*gAwgBw/a)

    kAx = coo_matrix((kAxv, (kAxr, kAxc)), shape=(size, size))

    return kAx


def fkAy(double beta, double a, double b, int m, int n,
         double w1tx, double w1rx, double w2tx, double w2rx,
         double w1ty, double w1ry, double w2ty, double w2ry,
         int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef np.ndarray[cINT, ndim=1] kAyr, kAyc
    cdef np.ndarray[cDOUBLE, ndim=1] kAyv

    cdef double fAwfBw, gAwetagBw

    fdim = 1*m*n*m*n

    kAyr = np.zeros((fdim,), dtype=INT)
    kAyc = np.zeros((fdim,), dtype=INT)
    kAyv = np.zeros((fdim,), dtype=DOUBLE)

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
                    kAyr[c] = row+2
                    kAyc[c] = col+2
                    kAyv[c] += -0.5*a*beta*fAwfBw*gAwetagBw

    kAy = coo_matrix((kAyv, (kAyr, kAyc)), shape=(size, size))

    return kAy


def fcA(double aeromu, double a, double b, int m, int n,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double w1ty, double w1ry, double w2ty, double w2ry,
        int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef np.ndarray[cINT, ndim=1] cAr, cAc
    cdef np.ndarray[cDOUBLE, ndim=1] cAv

    cdef double fAwfBw, gAwgBw

    fdim = 1*m*n*m*n

    cAr = np.zeros((fdim,), dtype=INT)
    cAc = np.zeros((fdim,), dtype=INT)
    cAv = np.zeros((fdim,), dtype=DOUBLE)

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
                    cAr[c] = row+2
                    cAc[c] = col+2
                    cAv[c] += -0.25*a*aeromu*b*fAwfBw*gAwgBw

    cA = coo_matrix((cAv, (cAr, cAc)), shape=(size, size))

    return cA


def fk0f(double a, double bf, np.ndarray[cDOUBLE, ndim=2] F, int m1, int n1,
         double u1txf, double u1rxf, double u2txf, double u2rxf,
         double v1txf, double v1rxf, double v2txf, double v2rxf,
         double w1txf, double w1rxf, double w2txf, double w2rxf,
         double u1tyf, double u1ryf, double u2tyf, double u2ryf,
         double v1tyf, double v1ryf, double v2tyf, double v2ryf,
         double w1tyf, double w1ryf, double w2tyf, double w2ryf,
         int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef np.ndarray[cINT, ndim=1] k0fr, k0fc
    cdef np.ndarray[cDOUBLE, ndim=1] k0fv

    cdef double rAurBu, rAurBuxi, rAuxirBu, rAuxirBuxi, rAurBv, rAurBvxi,
    cdef double rAuxirBv, rAuxirBvxi, rAuxirBwxixi, rAuxirBw, rAurBwxixi,
    cdef double rAuxirBwxi, rAurBw, rAurBwxi, rAvrBuxi, rAvxirBuxi, rAvrBu,
    cdef double rAvxirBu, rAvrBv, rAvrBvxi, rAvxirBv, rAvxirBvxi, rAvrBwxixi,
    cdef double rAvxirBwxixi, rAvrBw, rAvrBwxi, rAvxirBw, rAvxirBwxi,
    cdef double rAwxixirBuxi, rAwrBuxi, rAwxirBuxi, rAwxixirBu, rAwrBu,
    cdef double rAwxirBu, rAwxixirBv, rAwxixirBvxi, rAwrBv, rAwrBvxi, rAwxirBv,
    cdef double rAwxirBvxi, rAwxixirBwxixi, rAwrBwxixi, rAwxixirBw,
    cdef double rAwxirBwxixi, rAwxixirBwxi, rAwrBw, rAwrBwxi, rAwxirBw,
    cdef double rAwxirBwxi
    cdef double sAusBu, sAusBueta, sAuetasBu, sAuetasBueta, sAusBv, sAusBveta,
    cdef double sAuetasBv, sAuetasBveta, sAuetasBwetaeta, sAuetasBw,
    cdef double sAusBwetaeta, sAuetasBweta, sAusBw, sAusBweta, sAvsBueta,
    cdef double sAvetasBueta, sAvsBu, sAvetasBu, sAvsBv, sAvsBveta, sAvetasBv,
    cdef double sAvetasBveta, sAvsBwetaeta, sAvetasBwetaeta, sAvsBw, sAvsBweta,
    cdef double sAvetasBw, sAvetasBweta, sAwetaetasBueta, sAwsBueta,
    cdef double sAwetasBueta, sAwetaetasBu, sAwsBu, sAwetasBu, sAwetaetasBv,
    cdef double sAwetaetasBveta, sAwsBv, sAwsBveta, sAwetasBv, sAwetasBveta,
    cdef double sAwetaetasBwetaeta, sAwsBwetaeta, sAwetaetasBw,
    cdef double sAwetasBwetaeta, sAwetaetasBweta, sAwsBw, sAwsBweta, sAwetasBw,
    cdef double sAwetasBweta

    fdim = 9*m1*n1*m1*n1

    k0fr = np.zeros((fdim,), dtype=INT)
    k0fc = np.zeros((fdim,), dtype=INT)
    k0fv = np.zeros((fdim,), dtype=DOUBLE)

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

    # k0f
    c = -1
    for i1 in range(m1):
        for k1 in range(m1):

            rAurBu = integral_ff(i1, k1, u1txf, u1rxf, u2txf, u2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAurBuxi = integral_ffxi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAuxirBu = integral_ffxi(k1, i1, u1txf, u1rxf, u2txf, u2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAuxirBuxi = integral_fxifxi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAurBv = integral_ff(i1, k1, u1txf, u1rxf, u2txf, u2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAurBvxi = integral_ffxi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAuxirBv = integral_ffxi(k1, i1, v1txf, v1rxf, v2txf, v2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAuxirBvxi = integral_fxifxi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAuxirBwxixi = integral_fxifxixi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAuxirBw = integral_ffxi(k1, i1, w1txf, w1rxf, w2txf, w2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAurBwxixi = integral_ffxixi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAuxirBwxi = integral_fxifxi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAurBw = integral_ff(i1, k1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAurBwxi = integral_ffxi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAvrBuxi = integral_ffxi(i1, k1, v1txf, v1rxf, v2txf, v2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAvxirBuxi = integral_fxifxi(i1, k1, v1txf, v1rxf, v2txf, v2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAvrBu = integral_ff(i1, k1, v1txf, v1rxf, v2txf, v2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAvxirBu = integral_ffxi(k1, i1, u1txf, u1rxf, u2txf, u2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAvrBv = integral_ff(i1, k1, v1txf, v1rxf, v2txf, v2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAvrBvxi = integral_ffxi(i1, k1, v1txf, v1rxf, v2txf, v2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAvxirBv = integral_ffxi(k1, i1, v1txf, v1rxf, v2txf, v2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAvxirBvxi = integral_fxifxi(i1, k1, v1txf, v1rxf, v2txf, v2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAvrBwxixi = integral_ffxixi(i1, k1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAvxirBwxixi = integral_fxifxixi(i1, k1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAvrBw = integral_ff(i1, k1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAvrBwxi = integral_ffxi(i1, k1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAvxirBw = integral_ffxi(k1, i1, w1txf, w1rxf, w2txf, w2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAvxirBwxi = integral_fxifxi(i1, k1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxixirBuxi = integral_fxifxixi(k1, i1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBuxi = integral_ffxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAwxirBuxi = integral_fxifxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAwxixirBu = integral_ffxixi(k1, i1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBu = integral_ff(i1, k1, w1txf, w1rxf, w2txf, w2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAwxirBu = integral_ffxi(k1, i1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxixirBv = integral_ffxixi(k1, i1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxixirBvxi = integral_fxifxixi(k1, i1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBv = integral_ff(i1, k1, w1txf, w1rxf, w2txf, w2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAwrBvxi = integral_ffxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAwxirBv = integral_ffxi(k1, i1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxirBvxi = integral_fxifxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAwxixirBwxixi = integral_fxixifxixi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBwxixi = integral_ffxixi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxixirBw = integral_ffxixi(k1, i1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxirBwxixi = integral_fxifxixi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxixirBwxi = integral_fxifxixi(k1, i1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBw = integral_ff(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBwxi = integral_ffxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxirBw = integral_ffxi(k1, i1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxirBwxi = integral_fxifxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)


            for j1 in range(n1):
                for l1 in range(n1):

                    row = row0 + num1*(j1*m1 + i1)
                    col = col0 + num1*(l1*m1 + k1)

                    #NOTE symmetry
                    if row > col:
                        continue

                    sAusBu = integral_ff(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAusBueta = integral_ffxi(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAuetasBu = integral_ffxi(l1, j1, u1tyf, u1ryf, u2tyf, u2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAuetasBueta = integral_fxifxi(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAusBv = integral_ff(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAusBveta = integral_ffxi(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAuetasBv = integral_ffxi(l1, j1, v1tyf, v1ryf, v2tyf, v2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAuetasBveta = integral_fxifxi(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAuetasBwetaeta = integral_fxifxixi(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAuetasBw = integral_ffxi(l1, j1, w1tyf, w1ryf, w2tyf, w2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAusBwetaeta = integral_ffxixi(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAuetasBweta = integral_fxifxi(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAusBw = integral_ff(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAusBweta = integral_ffxi(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAvsBueta = integral_ffxi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAvetasBueta = integral_fxifxi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAvsBu = integral_ff(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAvetasBu = integral_ffxi(l1, j1, u1tyf, u1ryf, u2tyf, u2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAvsBv = integral_ff(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAvsBveta = integral_ffxi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAvetasBv = integral_ffxi(l1, j1, v1tyf, v1ryf, v2tyf, v2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAvetasBveta = integral_fxifxi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAvsBwetaeta = integral_ffxixi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAvetasBwetaeta = integral_fxifxixi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAvsBw = integral_ff(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAvsBweta = integral_ffxi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAvetasBw = integral_ffxi(l1, j1, w1tyf, w1ryf, w2tyf, w2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAvetasBweta = integral_fxifxi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetaetasBueta = integral_fxifxixi(l1, j1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBueta = integral_ffxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAwetasBueta = integral_fxifxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAwetaetasBu = integral_ffxixi(l1, j1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBu = integral_ff(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAwetasBu = integral_ffxi(l1, j1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetaetasBv = integral_ffxixi(l1, j1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetaetasBveta = integral_fxifxixi(l1, j1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBv = integral_ff(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAwsBveta = integral_ffxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAwetasBv = integral_ffxi(l1, j1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetasBveta = integral_fxifxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAwetaetasBwetaeta = integral_fxixifxixi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBwetaeta = integral_ffxixi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetaetasBw = integral_ffxixi(l1, j1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetasBwetaeta = integral_fxifxixi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetaetasBweta = integral_fxifxixi(l1, j1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBw = integral_ff(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBweta = integral_ffxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetasBw = integral_ffxi(l1, j1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetasBweta = integral_fxifxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)

                    c += 1
                    k0fr[c] = row+0
                    k0fc[c] = col+0
                    k0fv[c] += A11*bf*rAuxirBuxi*sAusBu/a + A16*(rAurBuxi*sAuetasBu + rAuxirBu*sAusBueta) + A66*a*rAurBu*sAuetasBueta/bf
                    c += 1
                    k0fr[c] = row+0
                    k0fc[c] = col+1
                    k0fv[c] += A12*rAuxirBv*sAusBveta + A16*bf*rAuxirBvxi*sAusBv/a + A26*a*rAurBv*sAuetasBveta/bf + A66*rAurBvxi*sAuetasBv
                    c += 1
                    k0fr[c] = row+0
                    k0fc[c] = col+2
                    k0fv[c] += -2*B11*bf*rAuxirBwxixi*sAusBw/(a*a) - 2*B12*rAuxirBw*sAusBwetaeta/bf - 2*B16*(rAurBwxixi*sAuetasBw + 2*rAuxirBwxi*sAusBweta)/a - 2*B26*a*rAurBw*sAuetasBwetaeta/(bf*bf) - 4*B66*rAurBwxi*sAuetasBweta/bf
                    c += 1
                    k0fr[c] = row+1
                    k0fc[c] = col+0
                    k0fv[c] += A12*rAvrBuxi*sAvetasBu + A16*bf*rAvxirBuxi*sAvsBu/a + A26*a*rAvrBu*sAvetasBueta/bf + A66*rAvxirBu*sAvsBueta
                    c += 1
                    k0fr[c] = row+1
                    k0fc[c] = col+1
                    k0fv[c] += A22*a*rAvrBv*sAvetasBveta/bf + A26*(rAvrBvxi*sAvetasBv + rAvxirBv*sAvsBveta) + A66*bf*rAvxirBvxi*sAvsBv/a
                    c += 1
                    k0fr[c] = row+1
                    k0fc[c] = col+2
                    k0fv[c] += -2*B12*rAvrBwxixi*sAvetasBw/a - 2*B16*bf*rAvxirBwxixi*sAvsBw/(a*a) - 2*B22*a*rAvrBw*sAvetasBwetaeta/(bf*bf) - 2*B26*(2*rAvrBwxi*sAvetasBweta + rAvxirBw*sAvsBwetaeta)/bf - 4*B66*rAvxirBwxi*sAvsBweta/a
                    c += 1
                    k0fr[c] = row+2
                    k0fc[c] = col+0
                    k0fv[c] += -2*B11*bf*rAwxixirBuxi*sAwsBu/(a*a) - 2*B12*rAwrBuxi*sAwetaetasBu/bf - 2*B16*(2*rAwxirBuxi*sAwetasBu + rAwxixirBu*sAwsBueta)/a - 2*B26*a*rAwrBu*sAwetaetasBueta/(bf*bf) - 4*B66*rAwxirBu*sAwetasBueta/bf
                    c += 1
                    k0fr[c] = row+2
                    k0fc[c] = col+1
                    k0fv[c] += -2*B12*rAwxixirBv*sAwsBveta/a - 2*B16*bf*rAwxixirBvxi*sAwsBv/(a*a) - 2*B22*a*rAwrBv*sAwetaetasBveta/(bf*bf) - 2*B26*(rAwrBvxi*sAwetaetasBv + 2*rAwxirBv*sAwetasBveta)/bf - 4*B66*rAwxirBvxi*sAwetasBv/a
                    c += 1
                    k0fr[c] = row+2
                    k0fc[c] = col+2
                    k0fv[c] += 4*D11*bf*rAwxixirBwxixi*sAwsBw/(a*a*a) + 4*D12*(rAwrBwxixi*sAwetaetasBw + rAwxixirBw*sAwsBwetaeta)/(a*bf) + 8*D16*(rAwxirBwxixi*sAwetasBw + rAwxixirBwxi*sAwsBweta)/(a*a) + 4*D22*a*rAwrBw*sAwetaetasBwetaeta/(bf*bf*bf) + 8*D26*(rAwrBwxi*sAwetaetasBweta + rAwxirBw*sAwetasBwetaeta)/(bf*bf) + 16*D66*rAwxirBwxi*sAwetasBweta/(a*bf)


    k0f = coo_matrix((k0fv, (k0fr, k0fc)), shape=(size, size))

    return k0f


def fkMf(double mu, double hf, double a, double bf, double d,
         int m1, int n1,
         double u1txf, double u1rxf, double u2txf, double u2rxf,
         double v1txf, double v1rxf, double v2txf, double v2rxf,
         double w1txf, double w1rxf, double w2txf, double w2rxf,
         double u1tyf, double u1ryf, double u2tyf, double u2ryf,
         double v1tyf, double v1ryf, double v2tyf, double v2ryf,
         double w1tyf, double w1ryf, double w2tyf, double w2ryf,
         int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col

    cdef np.ndarray[cINT, ndim=1] kMfr, kMfc
    cdef np.ndarray[cDOUBLE, ndim=1] kMfv

    cdef double rAurBu, rAurBwxi, rAvrBv, rAvrBw, rAwxirBu, rAwrBv, rAwrBw, rAwxirBwxi
    cdef double sAusBu, sAusBw, sAvsBv, sAvsBweta, sAwsBu, sAwetasBv, sAwsBw, sAwetasBweta

    fdim = 7*m1*n1*m1*n1

    kMfr = np.zeros((fdim,), dtype=INT)
    kMfc = np.zeros((fdim,), dtype=INT)
    kMfv = np.zeros((fdim,), dtype=DOUBLE)

    # kMf
    c = -1
    for i1 in range(m1):
        for k1 in range(m1):

            rAurBu = integral_ff(i1, k1, u1txf, u1rxf, u2txf, u2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAurBwxi = integral_ffxi(i1, k1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAvrBv = integral_ff(i1, k1, v1txf, v1rxf, v2txf, v2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAvrBw = integral_ff(i1, k1, v1txf, v1rxf, v2txf, v2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxirBu = integral_ffxi(k1, i1, u1txf, u1rxf, u2txf, u2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBv = integral_ff(i1, k1, w1txf, w1rxf, w2txf, w2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAwrBw = integral_ff(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxirBwxi = integral_fxifxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)

            for j1 in range(n1):
                for l1 in range(n1):

                    row = row0 + num1*(j1*m1 + i1)
                    col = col0 + num1*(l1*m1 + k1)

                    #NOTE symmetry
                    if row > col:
                        continue

                    sAusBu = integral_ff(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAusBw = integral_ff(j1, l1, u1tyf, u1ryf, u2tyf, u2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAvsBv = integral_ff(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, v1tyf, v1ryf, v2tyf, v2ryf)
                    sAvsBweta = integral_ffxi(j1, l1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBu = integral_ff(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, u1tyf, u1ryf, u2tyf, u2ryf)
                    sAwetasBv = integral_ffxi(l1, j1, v1tyf, v1ryf, v2tyf, v2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBw = integral_ff(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetasBweta = integral_fxifxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)

                    c += 1
                    kMfr[c] = row+0
                    kMfc[c] = col+0
                    kMfv[c] += 0.25*a*bf*hf*mu*rAurBu*sAusBu
                    c += 1
                    kMfr[c] = row+0
                    kMfc[c] = col+2
                    kMfv[c] += 0.5*bf*d*hf*mu*rAurBwxi*sAusBw
                    c += 1
                    kMfr[c] = row+1
                    kMfc[c] = col+1
                    kMfv[c] += 0.25*a*bf*hf*mu*rAvrBv*sAvsBv
                    c += 1
                    kMfr[c] = row+1
                    kMfc[c] = col+2
                    kMfv[c] += 0.5*a*d*hf*mu*rAvrBw*sAvsBweta
                    c += 1
                    kMfr[c] = row+2
                    kMfc[c] = col+0
                    kMfv[c] += 0.5*bf*d*hf*mu*rAwxirBu*sAwsBu
                    c += 1
                    kMfr[c] = row+2
                    kMfc[c] = col+1
                    kMfv[c] += 0.5*a*d*hf*mu*rAwrBv*sAwetasBv
                    c += 1
                    kMfr[c] = row+2
                    kMfc[c] = col+2
                    kMfv[c] += 0.0833333333333333*a*bf*hf*mu*(3*rAwrBw*sAwsBw + rAwrBw*sAwetasBweta*(12*(d*d) + (hf*hf))/(bf*bf) + rAwxirBwxi*sAwsBw*(12*(d*d) + (hf*hf))/(a*a))

    kMf = coo_matrix((kMfv, (kMfr, kMfc)), shape=(size, size))

    return kMf


def fkCff(double kt, double kr, double a, double bf, int m1, int n1,
          double u1txf, double u1rxf, double u2txf, double u2rxf,
          double v1txf, double v1rxf, double v2txf, double v2rxf,
          double w1txf, double w1rxf, double w2txf, double w2rxf,
          double u1tyf, double u1ryf, double u2tyf, double u2ryf,
          double v1tyf, double v1ryf, double v2tyf, double v2ryf,
          double w1tyf, double w1ryf, double w2tyf, double w2ryf,
          int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col

    cdef np.ndarray[cINT, ndim=1] kCffr, kCffc
    cdef np.ndarray[cDOUBLE, ndim=1] kCffv

    cdef double rAurBu, rAvrBv, rAwrBw
    cdef double sAu, sBu, sAv, sBv, sAw, sBw, sAweta, sBweta

    fdim = 5*m1*n1*m1*n1

    kCffr = np.zeros((fdim,), dtype=INT)
    kCffc = np.zeros((fdim,), dtype=INT)
    kCffv = np.zeros((fdim,), dtype=DOUBLE)

    # kCff
    c = -1
    for i1 in range(m1):
        for k1 in range(m1):

            rAurBu = integral_ff(i1, k1, u1txf, u1rxf, u2txf, u2rxf, u1txf, u1rxf, u2txf, u2rxf)
            rAvrBv = integral_ff(i1, k1, v1txf, v1rxf, v2txf, v2rxf, v1txf, v1rxf, v2txf, v2rxf)
            rAwrBw = integral_ff(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)

            for j1 in range(n1):
                sAu = calc_f(j1, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                sAv = calc_f(j1, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                sAw = calc_f(j1, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                sAweta = calc_fxi(j1, -1., w1tyf, w1ryf, w2tyf, w2ryf)

                for l1 in range(n1):

                    row = row0 + num1*(j1*m1 + i1)
                    col = col0 + num1*(l1*m1 + k1)

                    #NOTE symmetry
                    if row > col:
                        continue

                    sBu = calc_f(l1, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                    sBv = calc_f(l1, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                    sBw = calc_f(l1, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                    sBweta = calc_fxi(l1, -1., w1tyf, w1ryf, w2tyf, w2ryf)

                    c += 1
                    kCffr[c] = row+0
                    kCffc[c] = col+0
                    kCffv[c] += 0.5*a*kt*rAurBu*sAu*sBu
                    c += 1
                    kCffr[c] = row+1
                    kCffc[c] = col+1
                    kCffv[c] += 0.5*a*kt*rAvrBv*sAv*sBv
                    c += 1
                    kCffr[c] = row+2
                    kCffc[c] = col+2
                    kCffv[c] += 0.5*a*(kt*rAwrBw*sAw*sBw + 4*kr*rAwrBw*sAweta*sBweta/(bf*bf))

    kCff = coo_matrix((kCffv, (kCffr, kCffc)), shape=(size, size))

    return kCff


def fkCsf(double kt, double kr, double ys, double a, double b, double bf,
          int m, int n, int m1, int n1,
          double u1tx, double u1rx, double u2tx, double u2rx,
          double v1tx, double v1rx, double v2tx, double v2rx,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double u1ty, double u1ry, double u2ty, double u2ry,
          double v1ty, double v1ry, double v2ty, double v2ry,
          double w1ty, double w1ry, double w2ty, double w2ry,
          double u1txf, double u1rxf, double u2txf, double u2rxf,
          double v1txf, double v1rxf, double v2txf, double v2rxf,
          double w1txf, double w1rxf, double w2txf, double w2rxf,
          double u1tyf, double u1ryf, double u2tyf, double u2ryf,
          double v1tyf, double v1ryf, double v2tyf, double v2ryf,
          double w1tyf, double w1ryf, double w2tyf, double w2ryf,
          int size, int row0, int col0):
    cdef int i, j, k1, l1, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kCsfr, kCsfc
    cdef np.ndarray[cDOUBLE, ndim=1] kCsfv

    cdef double fAurBu, fAvrBw, fAwrBv, fAwrBw
    cdef double gAu, gAv, gAw, gAweta, sBu, sBv, sBw, sBweta

    eta = 2*ys/b - 1.

    fdim = 4*m*n*m1*n1

    kCsfr = np.zeros((fdim,), dtype=INT)
    kCsfc = np.zeros((fdim,), dtype=INT)
    kCsfv = np.zeros((fdim,), dtype=DOUBLE)

    # kCsf
    c = -1
    for i in range(m):
        for k1 in range(m1):

            fAurBu = integral_ff(i, k1, u1tx, u1rx, u2tx, u2rx, u1txf, u1rxf, u2txf, u2rxf)
            fAvrBw = integral_ff(i, k1, v1tx, v1rx, v2tx, v2rx, w1txf, w1rxf, w2txf, w2rxf)
            fAwrBv = integral_ff(i, k1, w1tx, w1rx, w2tx, w2rx, v1txf, v1rxf, v2txf, v2rxf)
            fAwrBw = integral_ff(i, k1, w1tx, w1rx, w2tx, w2rx, w1txf, w1rxf, w2txf, w2rxf)

            for j in range(n):
                gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)

                for l1 in range(n1):

                    row = row0 + num1*(j*m + i)
                    col = col0 + num1*(l1*m1 + k1)

                    #NOTE symmetry
                    if row > col:
                        continue

                    sBu = calc_f(l1, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                    sBv = calc_f(l1, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                    sBw = calc_f(l1, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                    sBweta = calc_fxi(l1, -1., w1tyf, w1ryf, w2tyf, w2ryf)

                    c += 1
                    kCsfr[c] = row+0
                    kCsfc[c] = col+0
                    kCsfv[c] += -0.5*a*fAurBu*gAu*kt*sBu
                    c += 1
                    kCsfr[c] = row+1
                    kCsfc[c] = col+2
                    kCsfv[c] += -0.5*a*fAvrBw*gAv*kt*sBw
                    c += 1
                    kCsfr[c] = row+2
                    kCsfc[c] = col+1
                    kCsfv[c] += 0.5*a*fAwrBv*gAw*kt*sBv
                    c += 1
                    kCsfr[c] = row+2
                    kCsfc[c] = col+2
                    kCsfv[c] += -2*a*fAwrBw*gAweta*kr*sBweta/(b*bf)

    kCsf = coo_matrix((kCsfv, (kCsfr, kCsfc)), shape=(size, size))

    return kCsf


def fkCss(double kt, double kr, double ys, double a, double b, int m, int n,
          double u1tx, double u1rx, double u2tx, double u2rx,
          double v1tx, double v1rx, double v2tx, double v2rx,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double u1ty, double u1ry, double u2ty, double u2ry,
          double v1ty, double v1ry, double v2ty, double v2ry,
          double w1ty, double w1ry, double w2ty, double w2ry,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kCssr, kCssc
    cdef np.ndarray[cDOUBLE, ndim=1] kCssv

    cdef double fAufBu, fAvfBv, fAwfBw
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAweta, gBweta

    eta = 2*ys/b - 1.

    fdim = 3*m*n*m*n

    kCssr = np.zeros((fdim,), dtype=INT)
    kCssc = np.zeros((fdim,), dtype=INT)
    kCssv = np.zeros((fdim,), dtype=DOUBLE)

    # kCss
    c = -1
    for i in range(m):
        for k in range(m):

            fAufBu = integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAvfBv = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

            for j in range(n):
                gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)

                for l in range(n):
                    row = row0 + num*(j*m + i)
                    col = col0 + num*(l*m + k)

                    #NOTE symmetry
                    if row > col:
                        continue

                    gBu = calc_f(l, eta, u1ty, u1ry, u2ty, u2ry)
                    gBv = calc_f(l, eta, v1ty, v1ry, v2ty, v2ry)
                    gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)
                    gBweta = calc_fxi(l, eta, w1ty, w1ry, w2ty, w2ry)

                    c += 1
                    kCssr[c] = row+0
                    kCssc[c] = col+0
                    kCssv[c] += 0.5*a*fAufBu*gAu*gBu*kt
                    c += 1
                    kCssr[c] = row+1
                    kCssc[c] = col+1
                    kCssv[c] += 0.5*a*fAvfBv*gAv*gBv*kt
                    c += 1
                    kCssr[c] = row+2
                    kCssc[c] = col+2
                    kCssv[c] += 0.5*a*(fAwfBw*gAw*gBw*kt + 4*fAwfBw*gAweta*gBweta*kr/(b*b))

    kCss = coo_matrix((kCssv, (kCssr, kCssc)), shape=(size, size))

    return kCss


def fkG0f(double Nxx, double Nyy, double Nxy, double a, double bf,
          int m1, int n1,
          double w1txf, double w1rxf, double w2txf, double w2rxf,
          double w1tyf, double w1ryf, double w2tyf, double w2ryf,
          int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col

    cdef np.ndarray[cINT, ndim=1] kG0fr, kG0fc
    cdef np.ndarray[cDOUBLE, ndim=1] kG0fv

    cdef double rAwxirBwxi, rAwrBwxi, rAwxirBw, rAwrBw
    cdef double sAwsBw, sAwetasBw, sAwsBweta, sAwetasBweta

    fdim = 1*m1*n1*m1*n1

    kG0fr = np.zeros((fdim,), dtype=INT)
    kG0fc = np.zeros((fdim,), dtype=INT)
    kG0fv = np.zeros((fdim,), dtype=DOUBLE)

    # kG0f
    c = -1
    for i1 in range(m1):
        for k1 in range(m1):
            rAwxirBwxi = integral_fxifxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBwxi = integral_ffxi(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwxirBw = integral_ffxi(k1, i1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)
            rAwrBw = integral_ff(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)

            for j1 in range(n1):
                for l1 in range(n1):
                    row = row0 + num1*(j1*m1 + i1)
                    col = col0 + num1*(l1*m1 + k1)

                    #NOTE symmetry
                    if row > col:
                        continue

                    sAwsBw = integral_ff(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwsBweta = integral_ffxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetasBw = integral_ffxi(l1, j1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)
                    sAwetasBweta = integral_fxifxi(j1, l1, w1tyf, w1ryf, w2tyf, w2ryf, w1tyf, w1ryf, w2tyf, w2ryf)

                    c += 1
                    kG0fr[c] = row+2
                    kG0fc[c] = col+2
                    kG0fv[c] += Nxx*bf*rAwxirBwxi*sAwsBw/a + Nxy*(rAwrBwxi*sAwetasBw + rAwxirBw*sAwsBweta) + Nyy*a*rAwrBw*sAwetasBweta/bf

    kG0f = coo_matrix((kG0fv, (kG0fr, kG0fc)), shape=(size, size))

    return kG0f

