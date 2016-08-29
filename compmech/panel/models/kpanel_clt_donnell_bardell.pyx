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

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int num = 3
cdef int s = 41


def fk0(object panel, int size, int row0, int col0):
    cdef double a, bbot, rbot, alpharad
    cdef np.ndarray[cDOUBLE, ndim=2] F
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, k, l, c, row, col
    cdef int section
    cdef double x1, x2, xi1, xi2, r, b, sina, cosa
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

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    bbot = panel.b
    rbot = panel.r
    alpharad = panel.alpharad
    F = panel.lam.ABD
    m = panel.m
    n = panel.n
    u1tx = panel.u1tx; u1rx = panel.u1rx; u2tx = panel.u2tx; u2rx = panel.u2rx
    v1tx = panel.v1tx; v1rx = panel.v1rx; v2tx = panel.v2tx; v2rx = panel.v2rx
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    u1ty = panel.u1ty; u1ry = panel.u1ry; u2ty = panel.u2ty; u2ry = panel.u2ry
    v1ty = panel.v1ty; v1ry = panel.v1ry; v2ty = panel.v2ty; v2ry = panel.v2ry
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 9*m*m*n*n

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    # k0
    with nogil:
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

        sina = sin(alpharad)
        cosa = cos(alpharad)

        #TODO this can be put at the innermost loop
        for section in range(s):
            x1 = a*float(section)/s
            x2 = a*float(section+1)/s

            xi1 = 2*x1/a - 1.
            xi2 = 2*x2/a - 1.

            r = rbot - sina*((x1+x2)/2.)
            b = r*bbot/rbot

            c = -1
            for i in range(m):
                for k in range(m):

                    fAufBu = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBuxi = integral_ffxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBu = integral_ffxi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBuxi = integral_fxifxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBv = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAufBvxi = integral_ffxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAuxifBv = integral_ffxi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBvxi = integral_fxifxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAuxifBwxixi = integral_fxifxixi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAuxifBw = integral_ffxi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBwxixi = integral_ffxixi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAuxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAufBw = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAufBwxi = integral_ffxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBuxi = integral_ffxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvxifBuxi = integral_fxifxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvfBu = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvxifBu = integral_ffxi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBv = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBvxi = integral_ffxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBv = integral_ffxi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBvxi = integral_fxifxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBwxixi = integral_ffxixi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvxifBwxixi = integral_fxifxixi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBw = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBwxi = integral_ffxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvxifBw = integral_ffxi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBuxi = integral_fxifxixi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBuxi = integral_ffxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxifBuxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxixifBu = integral_ffxixi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBu = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxifBu = integral_ffxi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBv = integral_ffxixi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBvxi = integral_fxifxixi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBv = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwfBvxi = integral_ffxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwxifBv = integral_ffxi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBvxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwxixifBwxixi = integral_fxixifxixi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBwxixi = integral_ffxixi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBw = integral_ffxixi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBwxixi = integral_fxifxixi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBwxi = integral_fxifxixi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBw = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBwxi = integral_ffxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBw = integral_ffxi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                    for j in range(n):
                        for l in range(n):

                            row = row0 + num*(j*m + i)
                            col = col0 + num*(l*m + k)

                            #NOTE symmetry
                            if row > col:
                                continue

                            gAugBu = integral_ff(j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAugBueta = integral_ffxi(j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAuetagBu = integral_ffxi(l, j, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAuetagBueta = integral_fxifxi(j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAugBv = integral_ff(j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                            gAugBveta = integral_ffxi(j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                            gAuetagBv = integral_ffxi(l, j, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                            gAuetagBveta = integral_fxifxi(j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                            gAuetagBwetaeta = integral_fxifxixi(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAuetagBw = integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAugBwetaeta = integral_ffxixi(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAuetagBweta = integral_fxifxi(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAugBw = integral_ff(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAugBweta = integral_ffxi(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvgBueta = integral_ffxi(j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                            gAvetagBueta = integral_fxifxi(j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                            gAvgBu = integral_ff(j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                            gAvetagBu = integral_ffxi(l, j, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvgBv = integral_ff(j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvgBveta = integral_ffxi(j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvetagBv = integral_ffxi(l, j, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvetagBveta = integral_fxifxi(j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvgBwetaeta = integral_ffxixi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvetagBwetaeta = integral_fxifxixi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvgBw = integral_ff(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvgBweta = integral_ffxi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvetagBw = integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvetagBweta = integral_fxifxi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetaetagBueta = integral_fxifxixi(l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBueta = integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAwetagBueta = integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAwetaetagBu = integral_ffxixi(l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBu = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAwetagBu = integral_ffxi(l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetaetagBv = integral_ffxixi(l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetaetagBveta = integral_fxifxixi(l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBv = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                            gAwgBveta = integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                            gAwetagBv = integral_ffxi(l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBveta = integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
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
                            k0v[c] += A11*b*fAuxifBuxi*gAugBu/a + A12*(0.5*b*fAufBuxi*gAugBu*sina/r + 0.5*b*fAuxifBu*gAugBu*sina/r) + A16*(fAufBuxi*gAuetagBu + fAuxifBu*gAugBueta) + 0.25*A22*a*b*fAufBu*gAugBu*(sina*sina)/(r*r) + A26*(0.5*a*fAufBu*gAugBueta*sina/r + 0.5*a*fAufBu*gAuetagBu*sina/r) + A66*a*fAufBu*gAuetagBueta/b
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += A12*fAuxifBv*gAugBveta + A16*(-0.5*b*fAuxifBv*gAugBv*sina/r + b*fAuxifBvxi*gAugBv/a) + 0.5*A22*a*fAufBv*gAugBveta*sina/r + A26*(-0.25*a*b*fAufBv*gAugBv*(sina*sina)/(r*r) + a*fAufBv*gAuetagBveta/b + 0.5*b*fAufBvxi*gAugBv*sina/r) + A66*(-0.5*a*fAufBv*gAuetagBv*sina/r + fAufBvxi*gAuetagBv)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += 0.5*A12*b*cosa*fAuxifBw*gAugBw/r + 0.25*A22*a*b*cosa*fAufBw*gAugBw*sina/(r*r) + 0.5*A26*a*cosa*fAufBw*gAuetagBw/r - 2*B11*b*fAuxifBwxixi*gAugBw/(a*a) + B12*(-2*fAuxifBw*gAugBwetaeta/b - b*fAufBwxixi*gAugBw*sina/(a*r) - b*fAuxifBwxi*gAugBw*sina/(a*r)) + B16*(fAuxifBw*gAugBweta*sina/r - 2*fAufBwxixi*gAuetagBw/a - 4*fAuxifBwxi*gAugBweta/a) + B22*(-a*fAufBw*gAugBwetaeta*sina/(b*r) - 0.5*b*fAufBwxi*gAugBw*(sina*sina)/(r*r)) + B26*(0.5*a*fAufBw*gAugBweta*(sina*sina)/(r*r) - 2*a*fAufBw*gAuetagBwetaeta/(b*b) - 2*fAufBwxi*gAugBweta*sina/r - fAufBwxi*gAuetagBw*sina/r) + B66*(a*fAufBw*gAuetagBweta*sina/(b*r) - 4*fAufBwxi*gAuetagBweta/b)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += A12*fAvfBuxi*gAvetagBu + A16*(-0.5*b*fAvfBuxi*gAvgBu*sina/r + b*fAvxifBuxi*gAvgBu/a) + 0.5*A22*a*fAvfBu*gAvetagBu*sina/r + A26*(-0.25*a*b*fAvfBu*gAvgBu*(sina*sina)/(r*r) + a*fAvfBu*gAvetagBueta/b + 0.5*b*fAvxifBu*gAvgBu*sina/r) + A66*(-0.5*a*fAvfBu*gAvgBueta*sina/r + fAvxifBu*gAvgBueta)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += A22*a*fAvfBv*gAvetagBveta/b + A26*(-0.5*a*fAvfBv*gAvgBveta*sina/r - 0.5*a*fAvfBv*gAvetagBv*sina/r + fAvfBvxi*gAvetagBv + fAvxifBv*gAvgBveta) + A66*(0.25*a*b*fAvfBv*gAvgBv*(sina*sina)/(r*r) - 0.5*b*fAvfBvxi*gAvgBv*sina/r - 0.5*b*fAvxifBv*gAvgBv*sina/r + b*fAvxifBvxi*gAvgBv/a)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += 0.5*A22*a*cosa*fAvfBw*gAvetagBw/r + A26*(-0.25*a*b*cosa*fAvfBw*gAvgBw*sina/(r*r) + 0.5*b*cosa*fAvxifBw*gAvgBw/r) - 2*B12*fAvfBwxixi*gAvetagBw/a + B16*(b*fAvfBwxixi*gAvgBw*sina/(a*r) - 2*b*fAvxifBwxixi*gAvgBw/(a*a)) + B22*(-2*a*fAvfBw*gAvetagBwetaeta/(b*b) - fAvfBwxi*gAvetagBw*sina/r) + B26*(a*fAvfBw*gAvgBwetaeta*sina/(b*r) + a*fAvfBw*gAvetagBweta*sina/(b*r) + 0.5*b*fAvfBwxi*gAvgBw*(sina*sina)/(r*r) - 4*fAvfBwxi*gAvetagBweta/b - 2*fAvxifBw*gAvgBwetaeta/b - b*fAvxifBwxi*gAvgBw*sina/(a*r)) + B66*(-0.5*a*fAvfBw*gAvgBweta*(sina*sina)/(r*r) + 2*fAvfBwxi*gAvgBweta*sina/r + fAvxifBw*gAvgBweta*sina/r - 4*fAvxifBwxi*gAvgBweta/a)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += 0.5*A12*b*cosa*fAwfBuxi*gAwgBu/r + 0.25*A22*a*b*cosa*fAwfBu*gAwgBu*sina/(r*r) + 0.5*A26*a*cosa*fAwfBu*gAwgBueta/r - 2*B11*b*fAwxixifBuxi*gAwgBu/(a*a) + B12*(-2*fAwfBuxi*gAwetaetagBu/b - b*fAwxifBuxi*gAwgBu*sina/(a*r) - b*fAwxixifBu*gAwgBu*sina/(a*r)) + B16*(fAwfBuxi*gAwetagBu*sina/r - 4*fAwxifBuxi*gAwetagBu/a - 2*fAwxixifBu*gAwgBueta/a) + B22*(-a*fAwfBu*gAwetaetagBu*sina/(b*r) - 0.5*b*fAwxifBu*gAwgBu*(sina*sina)/(r*r)) + B26*(0.5*a*fAwfBu*gAwetagBu*(sina*sina)/(r*r) - 2*a*fAwfBu*gAwetaetagBueta/(b*b) - fAwxifBu*gAwgBueta*sina/r - 2*fAwxifBu*gAwetagBu*sina/r) + B66*(a*fAwfBu*gAwetagBueta*sina/(b*r) - 4*fAwxifBu*gAwetagBueta/b)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += 0.5*A22*a*cosa*fAwfBv*gAwgBveta/r + A26*(-0.25*a*b*cosa*fAwfBv*gAwgBv*sina/(r*r) + 0.5*b*cosa*fAwfBvxi*gAwgBv/r) - 2*B12*fAwxixifBv*gAwgBveta/a + B16*(b*fAwxixifBv*gAwgBv*sina/(a*r) - 2*b*fAwxixifBvxi*gAwgBv/(a*a)) + B22*(-2*a*fAwfBv*gAwetaetagBveta/(b*b) - fAwxifBv*gAwgBveta*sina/r) + B26*(a*fAwfBv*gAwetagBveta*sina/(b*r) + a*fAwfBv*gAwetaetagBv*sina/(b*r) + 0.5*b*fAwxifBv*gAwgBv*(sina*sina)/(r*r) - 2*fAwfBvxi*gAwetaetagBv/b - 4*fAwxifBv*gAwetagBveta/b - b*fAwxifBvxi*gAwgBv*sina/(a*r)) + B66*(-0.5*a*fAwfBv*gAwetagBv*(sina*sina)/(r*r) + fAwfBvxi*gAwetagBv*sina/r + 2*fAwxifBv*gAwetagBv*sina/r - 4*fAwxifBvxi*gAwetagBv/a)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.25*A22*a*b*(cosa*cosa)*fAwfBw*gAwgBw/(r*r) + B12*(-b*cosa*fAwfBwxixi*gAwgBw/(a*r) - b*cosa*fAwxixifBw*gAwgBw/(a*r)) + B22*(-a*cosa*fAwfBw*gAwgBwetaeta/(b*r) - a*cosa*fAwfBw*gAwetaetagBw/(b*r) - 0.5*b*cosa*fAwfBwxi*gAwgBw*sina/(r*r) - 0.5*b*cosa*fAwxifBw*gAwgBw*sina/(r*r)) + B26*(0.5*a*cosa*fAwfBw*gAwgBweta*sina/(r*r) + 0.5*a*cosa*fAwfBw*gAwetagBw*sina/(r*r) - 2*cosa*fAwfBwxi*gAwgBweta/r - 2*cosa*fAwxifBw*gAwetagBw/r) + 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + D12*(4*fAwfBwxixi*gAwetaetagBw/(a*b) + 4*fAwxixifBw*gAwgBwetaeta/(a*b) + 2*b*fAwxifBwxixi*gAwgBw*sina/((a*a)*r) + 2*b*fAwxixifBwxi*gAwgBw*sina/((a*a)*r)) + D16*(-2*fAwfBwxixi*gAwetagBw*sina/(a*r) - 2*fAwxixifBw*gAwgBweta*sina/(a*r) + 8*fAwxifBwxixi*gAwetagBw/(a*a) + 8*fAwxixifBwxi*gAwgBweta/(a*a)) + D22*(4*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 2*fAwfBwxi*gAwetaetagBw*sina/(b*r) + 2*fAwxifBw*gAwgBwetaeta*sina/(b*r) + b*fAwxifBwxi*gAwgBw*(sina*sina)/(a*(r*r))) + D26*(-2*a*fAwfBw*gAwetagBwetaeta*sina/((b*b)*r) - 2*a*fAwfBw*gAwetaetagBweta*sina/((b*b)*r) - fAwfBwxi*gAwetagBw*(sina*sina)/(r*r) - fAwxifBw*gAwgBweta*(sina*sina)/(r*r) + 8*fAwfBwxi*gAwetaetagBweta/(b*b) + 8*fAwxifBw*gAwetagBwetaeta/(b*b) + 4*fAwxifBwxi*gAwgBweta*sina/(a*r) + 4*fAwxifBwxi*gAwetagBw*sina/(a*r)) + D66*(a*fAwfBw*gAwetagBweta*(sina*sina)/(b*(r*r)) - 4*fAwfBwxi*gAwetagBweta*sina/(b*r) - 4*fAwxifBw*gAwetagBweta*sina/(b*r) + 16*fAwxifBwxi*gAwetagBweta/(a*b))

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0y1y2(double y1, double y2, object panel, int size, int row0, int col0):
    cdef double a, bbot, rbot, alpharad
    cdef np.ndarray[cDOUBLE, ndim=2] F
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, k, l, row, col, c
    cdef int section
    cdef double x1, x2, xi1, xi2, r, b, sina, cosa
    cdef double eta1, eta2
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

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    bbot = panel.b
    rbot = panel.r
    alpharad = panel.alpharad
    F = panel.lam.ABD
    m = panel.m
    n = panel.n
    u1tx = panel.u1tx; u1rx = panel.u1rx; u2tx = panel.u2tx; u2rx = panel.u2rx
    v1tx = panel.v1tx; v1rx = panel.v1rx; v2tx = panel.v2tx; v2rx = panel.v2rx
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    u1ty = panel.u1ty; u1ry = panel.u1ry; u2ty = panel.u2ty; u2ry = panel.u2ry
    v1ty = panel.v1ty; v1ry = panel.v1ry; v2ty = panel.v2ty; v2ry = panel.v2ry
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 9*m*n*m*n

    k0y1y2r = np.zeros((fdim,), dtype=INT)
    k0y1y2c = np.zeros((fdim,), dtype=INT)
    k0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

    # k0y1y2
    with nogil:
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

        sina = sin(alpharad)
        cosa = cos(alpharad)

        eta1 = 2*y1/bbot - 1.
        eta2 = 2*y2/bbot - 1.
        #TODO this can be put at the innermost loop
        for section in range(s):
            x1 = a*float(section)/s
            x2 = a*float(section+1)/s

            xi1 = 2*x1/a - 1.
            xi2 = 2*x2/a - 1.

            r = rbot - sina*((x1+x2)/2.)
            b = r*bbot/rbot

            c = -1
            for i in range(m):
                for k in range(m):

                    fAufBu = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBuxi = integral_ffxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBu = integral_ffxi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBuxi = integral_fxifxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBv = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAufBvxi = integral_ffxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAuxifBv = integral_ffxi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAuxifBvxi = integral_fxifxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAuxifBwxixi = integral_fxifxixi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAuxifBw = integral_ffxi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBwxixi = integral_ffxixi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAuxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAufBw = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAufBwxi = integral_ffxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBuxi = integral_ffxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvxifBuxi = integral_fxifxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvfBu = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx)
                    fAvxifBu = integral_ffxi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBv = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBvxi = integral_ffxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBv = integral_ffxi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBvxi = integral_fxifxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBwxixi = integral_ffxixi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvxifBwxixi = integral_fxifxixi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBw = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBwxi = integral_ffxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvxifBw = integral_ffxi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBuxi = integral_fxifxixi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBuxi = integral_ffxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxifBuxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxixifBu = integral_ffxixi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBu = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
                    fAwxifBu = integral_ffxi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBv = integral_ffxixi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBvxi = integral_fxifxixi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBv = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwfBvxi = integral_ffxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwxifBv = integral_ffxi_12(xi1, xi2, k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBvxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwxixifBwxixi = integral_fxixifxixi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBwxixi = integral_ffxixi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBw = integral_ffxixi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBwxixi = integral_fxifxixi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxixifBwxi = integral_fxifxixi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBw = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBwxi = integral_ffxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBw = integral_ffxi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                    for j in range(n):
                        for l in range(n):

                            row = row0 + num*(j*m + i)
                            col = col0 + num*(l*m + k)

                            #NOTE symmetry
                            if row > col:
                                continue

                            gAugBu = integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAugBueta = integral_ffxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAuetagBu = integral_ffxi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAuetagBueta = integral_fxifxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAugBv = integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                            gAugBveta = integral_ffxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                            gAuetagBv = integral_ffxi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                            gAuetagBveta = integral_fxifxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                            gAuetagBwetaeta = integral_fxifxixi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAuetagBw = integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAugBwetaeta = integral_ffxixi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAuetagBweta = integral_fxifxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAugBw = integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAugBweta = integral_ffxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvgBueta = integral_ffxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                            gAvetagBueta = integral_fxifxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                            gAvgBu = integral_ff_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry)
                            gAvetagBu = integral_ffxi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvgBv = integral_ff_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvgBveta = integral_ffxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvetagBv = integral_ffxi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvetagBveta = integral_fxifxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvgBwetaeta = integral_ffxixi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvetagBwetaeta = integral_fxifxixi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvgBw = integral_ff_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvgBweta = integral_ffxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvetagBw = integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvetagBweta = integral_fxifxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetaetagBueta = integral_fxifxixi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBueta = integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAwetagBueta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAwetaetagBu = integral_ffxixi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBu = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAwetagBu = integral_ffxi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetaetagBv = integral_ffxixi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetaetagBveta = integral_fxifxixi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBv = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                            gAwgBveta = integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                            gAwetagBv = integral_ffxi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBveta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry)
                            gAwetaetagBwetaeta = integral_fxixifxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBwetaeta = integral_ffxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetaetagBw = integral_ffxixi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBwetaeta = integral_fxifxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetaetagBweta = integral_fxifxixi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBw = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBweta = integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBw = integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBweta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                            c += 1
                            k0y1y2r[c] = row+0
                            k0y1y2c[c] = col+0
                            k0y1y2v[c] += A11*b*fAuxifBuxi*gAugBu/a + A12*(0.5*b*fAufBuxi*gAugBu*sina/r + 0.5*b*fAuxifBu*gAugBu*sina/r) + A16*(fAufBuxi*gAuetagBu + fAuxifBu*gAugBueta) + 0.25*A22*a*b*fAufBu*gAugBu*(sina*sina)/(r*r) + A26*(0.5*a*fAufBu*gAugBueta*sina/r + 0.5*a*fAufBu*gAuetagBu*sina/r) + A66*a*fAufBu*gAuetagBueta/b
                            c += 1
                            k0y1y2r[c] = row+0
                            k0y1y2c[c] = col+1
                            k0y1y2v[c] += A12*fAuxifBv*gAugBveta + A16*(-0.5*b*fAuxifBv*gAugBv*sina/r + b*fAuxifBvxi*gAugBv/a) + 0.5*A22*a*fAufBv*gAugBveta*sina/r + A26*(-0.25*a*b*fAufBv*gAugBv*(sina*sina)/(r*r) + a*fAufBv*gAuetagBveta/b + 0.5*b*fAufBvxi*gAugBv*sina/r) + A66*(-0.5*a*fAufBv*gAuetagBv*sina/r + fAufBvxi*gAuetagBv)
                            c += 1
                            k0y1y2r[c] = row+0
                            k0y1y2c[c] = col+2
                            k0y1y2v[c] += 0.5*A12*b*cosa*fAuxifBw*gAugBw/r + 0.25*A22*a*b*cosa*fAufBw*gAugBw*sina/(r*r) + 0.5*A26*a*cosa*fAufBw*gAuetagBw/r - 2*B11*b*fAuxifBwxixi*gAugBw/(a*a) + B12*(-2*fAuxifBw*gAugBwetaeta/b - b*fAufBwxixi*gAugBw*sina/(a*r) - b*fAuxifBwxi*gAugBw*sina/(a*r)) + B16*(fAuxifBw*gAugBweta*sina/r - 2*fAufBwxixi*gAuetagBw/a - 4*fAuxifBwxi*gAugBweta/a) + B22*(-a*fAufBw*gAugBwetaeta*sina/(b*r) - 0.5*b*fAufBwxi*gAugBw*(sina*sina)/(r*r)) + B26*(0.5*a*fAufBw*gAugBweta*(sina*sina)/(r*r) - 2*a*fAufBw*gAuetagBwetaeta/(b*b) - 2*fAufBwxi*gAugBweta*sina/r - fAufBwxi*gAuetagBw*sina/r) + B66*(a*fAufBw*gAuetagBweta*sina/(b*r) - 4*fAufBwxi*gAuetagBweta/b)
                            c += 1
                            k0y1y2r[c] = row+1
                            k0y1y2c[c] = col+0
                            k0y1y2v[c] += A12*fAvfBuxi*gAvetagBu + A16*(-0.5*b*fAvfBuxi*gAvgBu*sina/r + b*fAvxifBuxi*gAvgBu/a) + 0.5*A22*a*fAvfBu*gAvetagBu*sina/r + A26*(-0.25*a*b*fAvfBu*gAvgBu*(sina*sina)/(r*r) + a*fAvfBu*gAvetagBueta/b + 0.5*b*fAvxifBu*gAvgBu*sina/r) + A66*(-0.5*a*fAvfBu*gAvgBueta*sina/r + fAvxifBu*gAvgBueta)
                            c += 1
                            k0y1y2r[c] = row+1
                            k0y1y2c[c] = col+1
                            k0y1y2v[c] += A22*a*fAvfBv*gAvetagBveta/b + A26*(-0.5*a*fAvfBv*gAvgBveta*sina/r - 0.5*a*fAvfBv*gAvetagBv*sina/r + fAvfBvxi*gAvetagBv + fAvxifBv*gAvgBveta) + A66*(0.25*a*b*fAvfBv*gAvgBv*(sina*sina)/(r*r) - 0.5*b*fAvfBvxi*gAvgBv*sina/r - 0.5*b*fAvxifBv*gAvgBv*sina/r + b*fAvxifBvxi*gAvgBv/a)
                            c += 1
                            k0y1y2r[c] = row+1
                            k0y1y2c[c] = col+2
                            k0y1y2v[c] += 0.5*A22*a*cosa*fAvfBw*gAvetagBw/r + A26*(-0.25*a*b*cosa*fAvfBw*gAvgBw*sina/(r*r) + 0.5*b*cosa*fAvxifBw*gAvgBw/r) - 2*B12*fAvfBwxixi*gAvetagBw/a + B16*(b*fAvfBwxixi*gAvgBw*sina/(a*r) - 2*b*fAvxifBwxixi*gAvgBw/(a*a)) + B22*(-2*a*fAvfBw*gAvetagBwetaeta/(b*b) - fAvfBwxi*gAvetagBw*sina/r) + B26*(a*fAvfBw*gAvgBwetaeta*sina/(b*r) + a*fAvfBw*gAvetagBweta*sina/(b*r) + 0.5*b*fAvfBwxi*gAvgBw*(sina*sina)/(r*r) - 4*fAvfBwxi*gAvetagBweta/b - 2*fAvxifBw*gAvgBwetaeta/b - b*fAvxifBwxi*gAvgBw*sina/(a*r)) + B66*(-0.5*a*fAvfBw*gAvgBweta*(sina*sina)/(r*r) + 2*fAvfBwxi*gAvgBweta*sina/r + fAvxifBw*gAvgBweta*sina/r - 4*fAvxifBwxi*gAvgBweta/a)
                            c += 1
                            k0y1y2r[c] = row+2
                            k0y1y2c[c] = col+0
                            k0y1y2v[c] += 0.5*A12*b*cosa*fAwfBuxi*gAwgBu/r + 0.25*A22*a*b*cosa*fAwfBu*gAwgBu*sina/(r*r) + 0.5*A26*a*cosa*fAwfBu*gAwgBueta/r - 2*B11*b*fAwxixifBuxi*gAwgBu/(a*a) + B12*(-2*fAwfBuxi*gAwetaetagBu/b - b*fAwxifBuxi*gAwgBu*sina/(a*r) - b*fAwxixifBu*gAwgBu*sina/(a*r)) + B16*(fAwfBuxi*gAwetagBu*sina/r - 4*fAwxifBuxi*gAwetagBu/a - 2*fAwxixifBu*gAwgBueta/a) + B22*(-a*fAwfBu*gAwetaetagBu*sina/(b*r) - 0.5*b*fAwxifBu*gAwgBu*(sina*sina)/(r*r)) + B26*(0.5*a*fAwfBu*gAwetagBu*(sina*sina)/(r*r) - 2*a*fAwfBu*gAwetaetagBueta/(b*b) - fAwxifBu*gAwgBueta*sina/r - 2*fAwxifBu*gAwetagBu*sina/r) + B66*(a*fAwfBu*gAwetagBueta*sina/(b*r) - 4*fAwxifBu*gAwetagBueta/b)
                            c += 1
                            k0y1y2r[c] = row+2
                            k0y1y2c[c] = col+1
                            k0y1y2v[c] += 0.5*A22*a*cosa*fAwfBv*gAwgBveta/r + A26*(-0.25*a*b*cosa*fAwfBv*gAwgBv*sina/(r*r) + 0.5*b*cosa*fAwfBvxi*gAwgBv/r) - 2*B12*fAwxixifBv*gAwgBveta/a + B16*(b*fAwxixifBv*gAwgBv*sina/(a*r) - 2*b*fAwxixifBvxi*gAwgBv/(a*a)) + B22*(-2*a*fAwfBv*gAwetaetagBveta/(b*b) - fAwxifBv*gAwgBveta*sina/r) + B26*(a*fAwfBv*gAwetagBveta*sina/(b*r) + a*fAwfBv*gAwetaetagBv*sina/(b*r) + 0.5*b*fAwxifBv*gAwgBv*(sina*sina)/(r*r) - 2*fAwfBvxi*gAwetaetagBv/b - 4*fAwxifBv*gAwetagBveta/b - b*fAwxifBvxi*gAwgBv*sina/(a*r)) + B66*(-0.5*a*fAwfBv*gAwetagBv*(sina*sina)/(r*r) + fAwfBvxi*gAwetagBv*sina/r + 2*fAwxifBv*gAwetagBv*sina/r - 4*fAwxifBvxi*gAwetagBv/a)
                            c += 1
                            k0y1y2r[c] = row+2
                            k0y1y2c[c] = col+2
                            k0y1y2v[c] += 0.25*A22*a*b*(cosa*cosa)*fAwfBw*gAwgBw/(r*r) + B12*(-b*cosa*fAwfBwxixi*gAwgBw/(a*r) - b*cosa*fAwxixifBw*gAwgBw/(a*r)) + B22*(-a*cosa*fAwfBw*gAwgBwetaeta/(b*r) - a*cosa*fAwfBw*gAwetaetagBw/(b*r) - 0.5*b*cosa*fAwfBwxi*gAwgBw*sina/(r*r) - 0.5*b*cosa*fAwxifBw*gAwgBw*sina/(r*r)) + B26*(0.5*a*cosa*fAwfBw*gAwgBweta*sina/(r*r) + 0.5*a*cosa*fAwfBw*gAwetagBw*sina/(r*r) - 2*cosa*fAwfBwxi*gAwgBweta/r - 2*cosa*fAwxifBw*gAwetagBw/r) + 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + D12*(4*fAwfBwxixi*gAwetaetagBw/(a*b) + 4*fAwxixifBw*gAwgBwetaeta/(a*b) + 2*b*fAwxifBwxixi*gAwgBw*sina/((a*a)*r) + 2*b*fAwxixifBwxi*gAwgBw*sina/((a*a)*r)) + D16*(-2*fAwfBwxixi*gAwetagBw*sina/(a*r) - 2*fAwxixifBw*gAwgBweta*sina/(a*r) + 8*fAwxifBwxixi*gAwetagBw/(a*a) + 8*fAwxixifBwxi*gAwgBweta/(a*a)) + D22*(4*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 2*fAwfBwxi*gAwetaetagBw*sina/(b*r) + 2*fAwxifBw*gAwgBwetaeta*sina/(b*r) + b*fAwxifBwxi*gAwgBw*(sina*sina)/(a*(r*r))) + D26*(-2*a*fAwfBw*gAwetagBwetaeta*sina/((b*b)*r) - 2*a*fAwfBw*gAwetaetagBweta*sina/((b*b)*r) - fAwfBwxi*gAwetagBw*(sina*sina)/(r*r) - fAwxifBw*gAwgBweta*(sina*sina)/(r*r) + 8*fAwfBwxi*gAwetaetagBweta/(b*b) + 8*fAwxifBw*gAwetagBwetaeta/(b*b) + 4*fAwxifBwxi*gAwgBweta*sina/(a*r) + 4*fAwxifBwxi*gAwetagBw*sina/(a*r)) + D66*(a*fAwfBw*gAwetagBweta*(sina*sina)/(b*(r*r)) - 4*fAwfBwxi*gAwetagBweta*sina/(b*r) - 4*fAwxifBw*gAwetagBweta*sina/(b*r) + 16*fAwxifBwxi*gAwetagBweta/(a*b))


    k0y1y2 = coo_matrix((k0y1y2v, (k0y1y2r, k0y1y2c)), shape=(size, size))

    return k0y1y2


def fkG0(double Nxx, double Nyy, double Nxy, object panel,
         int size, int row0, int col0):
    cdef double a, bbot, rbot, alpharad
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef int section
    cdef double r, b, x1, x2, xi1, xi2, sina

    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    cdef double fAwxifBwxi, fAwfBwxi, fAwxifBw, fAwfBw
    cdef double gAwetagBweta, gAwgBweta, gAwetagBw, gAwgBw

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    bbot = panel.b
    rbot = panel.r
    alpharad = panel.alpharad
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*m*n*n

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)

    # kG0
    with nogil:
        #TODO this can be put at the innermost loop
        for section in range(s):
            x1 = a*float(section)/s
            x2 = a*float(section+1)/s

            xi1 = 2*x1/a - 1.
            xi2 = 2*x2/a - 1.

            r = rbot - sina*((x1+x2)/2.)
            b = r*bbot/rbot

            c = -1
            for i in range(m):
                for k in range(m):

                    fAwxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBwxi = integral_ffxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBw = integral_ffxi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBw = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                    for j in range(n):
                        for l in range(n):

                            row = row0 + num*(j*m + i)
                            col = col0 + num*(l*m + k)

                            #NOTE symmetry
                            if row > col:
                                continue

                            gAwgBw = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBweta = integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBw = integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBweta = integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkG0y1y2(double y1, double y2, double Nxx, double Nyy, double Nxy,
             object panel, int size, int row0, int col0):
    cdef double a, bbot, rbot, alpharad
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef double eta1, eta2
    cdef int section
    cdef double r, b, x1, x2, xi1, xi2, sina

    cdef np.ndarray[cINT, ndim=1] kG0y1y2r, kG0y1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0y1y2v

    cdef double fAwxifBwxi, fAwfBwxi, fAwxifBw, fAwfBw
    cdef double gAwetagBweta, gAwgBweta, gAwetagBw, gAwgBw

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    bbot = panel.b
    rbot = panel.r
    alpharad = panel.alpharad
    m = panel.m
    n = panel.n
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 1*m*n*m*n

    kG0y1y2r = np.zeros((fdim,), dtype=INT)
    kG0y1y2c = np.zeros((fdim,), dtype=INT)
    kG0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

    # kG0y1y2
    with nogil:
        sina = sin(alpharad)

        eta1 = 2*y1/bbot - 1.
        eta2 = 2*y2/bbot - 1.

        #TODO this can be put at the innermost loop
        for section in range(s):
            x1 = a*float(section)/s
            x2 = a*float(section+1)/s

            xi1 = 2*x1/a - 1.
            xi2 = 2*x2/a - 1.

            r = rbot - sina*((x1+x2)/2.)
            b = r*bbot/rbot

            c = -1
            for i in range(m):
                for k in range(m):

                    fAwxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBwxi = integral_ffxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBw = integral_ffxi_12(xi1, xi2, k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBw = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                    for j in range(n):
                        for l in range(n):

                            row = row0 + num*(j*m + i)
                            col = col0 + num*(l*m + k)

                            #NOTE symmetry
                            if row > col:
                                continue

                            gAwetagBw = integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBw = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBweta = integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBweta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                            c += 1
                            kG0y1y2r[c] = row+2
                            kG0y1y2c[c] = col+2
                            kG0y1y2v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

    kG0y1y2 = coo_matrix((kG0y1y2v, (kG0y1y2r, kG0y1y2c)), shape=(size, size))

    return kG0y1y2


def fkM(double d, object panel, int size, int row0, int col0):
    cdef double a, bbot, rbot, alpharad, mu, h
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef int section
    cdef double r, b, x1, x2, xi1, xi2, sina

    cdef np.ndarray[cINT, ndim=1] kMr, kMc
    cdef np.ndarray[cDOUBLE, ndim=1] kMv

    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwxifBu, fAwfBv, fAwfBw, fAwxifBwxi
    cdef double gAugBu, gAugBw, gAvgBv, gAvgBweta, gAwgBu, gAwetagBv, gAwgBw, gAwetagBweta

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    bbot = panel.b
    rbot = panel.r
    alpharad = panel.alpharad
    mu = panel.mu
    h = sum(panel.plyts)
    m = panel.m
    n = panel.n
    u1tx = panel.u1tx; u1rx = panel.u1rx; u2tx = panel.u2tx; u2rx = panel.u2rx
    v1tx = panel.v1tx; v1rx = panel.v1rx; v2tx = panel.v2tx; v2rx = panel.v2rx
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    u1ty = panel.u1ty; u1ry = panel.u1ry; u2ty = panel.u2ty; u2ry = panel.u2ry
    v1ty = panel.v1ty; v1ry = panel.v1ry; v2ty = panel.v2ty; v2ry = panel.v2ry
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 7*m*n*m*n

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)

    # kM
    with nogil:
        #TODO this can be put at the innermost loop
        for section in range(s):
            x1 = a*float(section)/s
            x2 = a*float(section+1)/s

            xi1 = 2*x1/a - 1.
            xi2 = 2*x2/a - 1.

            r = rbot - sina*((x1+x2)/2.)
            b = r*bbot/rbot

            c = -1
            for i in range(m):
                for k in range(m):

                    fAufBu = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBwxi = integral_ffxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBv = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBw = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBu = integral_ffxi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBv = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwfBw = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                    for j in range(n):
                        for l in range(n):

                            row = row0 + num*(j*m + i)
                            col = col0 + num*(l*m + k)

                            #NOTE symmetry
                            if row > col:
                                continue

                            gAugBu = integral_ff(j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAugBw = integral_ff(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvgBv = integral_ff(j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvgBweta = integral_ffxi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBu = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAwetagBv = integral_ffxi(l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBw = integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBweta = integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                            c += 1
                            kMr[c] = row+0
                            kMc[c] = col+0
                            kMv[c] += 0.25*a*b*fAufBu*gAugBu*h*mu
                            c += 1
                            kMr[c] = row+0
                            kMc[c] = col+2
                            kMv[c] += 0.5*b*d*fAufBwxi*gAugBw*h*mu
                            c += 1
                            kMr[c] = row+1
                            kMc[c] = col+1
                            kMv[c] += 0.25*a*b*fAvfBv*gAvgBv*h*mu
                            c += 1
                            kMr[c] = row+1
                            kMc[c] = col+2
                            kMv[c] += 0.5*a*d*fAvfBw*gAvgBweta*h*mu
                            c += 1
                            kMr[c] = row+2
                            kMc[c] = col+0
                            kMv[c] += 0.5*b*d*fAwxifBu*gAwgBu*h*mu
                            c += 1
                            kMr[c] = row+2
                            kMc[c] = col+1
                            kMv[c] += 0.5*a*d*fAwfBv*gAwetagBv*h*mu
                            c += 1
                            kMr[c] = row+2
                            kMc[c] = col+2
                            kMv[c] += 0.25*a*b*h*mu*(fAwfBw*gAwgBw + 4*fAwfBw*gAwetagBweta*((d*d) + 0.0833333333333333*(h*h))/(b*b) + 4*fAwxifBwxi*gAwgBw*((d*d) + 0.0833333333333333*(h*h))/(a*a))

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkMy1y2(double y1, double y2, double d, object panel,
            int size, int row0, int col0):
    cdef double a, bbot, rbot, alpharad, mu, h
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col
    cdef double eta1, eta2
    cdef int section
    cdef double r, b, x1, x2, xi1, xi2, sina

    cdef np.ndarray[cINT, ndim=1] kMy1y2r, kMy1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] kMy1y2v

    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwxifBu, fAwfBv, fAwfBw, fAwxifBwxi
    cdef double gAugBu, gAugBw, gAvgBv, gAvgBweta, gAwgBu, gAwetagBv, gAwgBw, gAwetagBweta

    if not 'Panel' in panel.__class__.__name__:
        raise ValueError('a Panel object must be given as input')
    a = panel.a
    bbot = panel.b
    rbot = panel.r
    alpharad = panel.alpharad
    mu = panel.mu
    h = sum(panel.plyts)
    m = panel.m
    n = panel.n
    u1tx = panel.u1tx; u1rx = panel.u1rx; u2tx = panel.u2tx; u2rx = panel.u2rx
    v1tx = panel.v1tx; v1rx = panel.v1rx; v2tx = panel.v2tx; v2rx = panel.v2rx
    w1tx = panel.w1tx; w1rx = panel.w1rx; w2tx = panel.w2tx; w2rx = panel.w2rx
    u1ty = panel.u1ty; u1ry = panel.u1ry; u2ty = panel.u2ty; u2ry = panel.u2ry
    v1ty = panel.v1ty; v1ry = panel.v1ry; v2ty = panel.v2ty; v2ry = panel.v2ry
    w1ty = panel.w1ty; w1ry = panel.w1ry; w2ty = panel.w2ty; w2ry = panel.w2ry

    fdim = 7*m*n*m*n

    kMy1y2r = np.zeros((fdim,), dtype=INT)
    kMy1y2c = np.zeros((fdim,), dtype=INT)
    kMy1y2v = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)

    eta1 = 2*y1/bbot - 1.
    eta2 = 2*y2/bbot - 1.

    # kMy1y2
    with nogil:
        #TODO this can be put at the innermost loop
        for section in range(s):
            x1 = a*float(section)/s
            x2 = a*float(section+1)/s

            xi1 = 2*x1/a - 1.
            xi2 = 2*x2/a - 1.

            r = rbot - sina*((x1+x2)/2.)
            b = r*bbot/rbot

            c = -1
            for i in range(m):
                for k in range(m):

                    fAufBu = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                    fAufBwxi = integral_ffxi_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAvfBv = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                    fAvfBw = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBu = integral_ffxi_12(xi1, xi2, k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwfBv = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                    fAwfBw = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                    fAwxifBwxi = integral_fxifxi_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                    for j in range(n):
                        for l in range(n):

                            row = row0 + num*(j*m + i)
                            col = col0 + num*(l*m + k)

                            #NOTE symmetry
                            if row > col:
                                continue

                            gAugBu = integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                            gAugBw = integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                            gAvgBv = integral_ff_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                            gAvgBweta = integral_ffxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBu = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                            gAwetagBv = integral_ffxi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwgBw = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                            gAwetagBweta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

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

    kMy1y2 = coo_matrix((kMy1y2v, (kMy1y2r, kMy1y2c)), shape=(size, size))

    return kMy1y2
