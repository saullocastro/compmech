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
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxi(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxifxi(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_12.h':
    double integral_ff_12(double eta1, double eta2, int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxi_12(double eta1, double eta2, int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxifxi_12(double eta1, double eta2, int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_c0c1.h':
    double integral_ff_c0c1(double c0, double c1, int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxi_c0c1(double c0, double c1, int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

DOUBLE = np.float64

cdef int num = 3
cdef int num1 = 3
cdef int num2 = 3


def fkCppy1y2(double y1, double y2,
          double kt, double a, double b, double dpb, int m, int n,
          double u1tx, double u1rx, double u2tx, double u2rx,
          double v1tx, double v1rx, double v2tx, double v2rx,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double u1ty, double u1ry, double u2ty, double u2ry,
          double v1ty, double v1ry, double v2ty, double v2ry,
          double w1ty, double w1ry, double w2ty, double w2ry,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta1, eta2

    cdef long [:] kCppr, kCppc
    cdef double [:] kCppv

    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwfBv, fAwfBw, fAwxifBu, fAwxifBwxi
    cdef double gAugBu, gAugBw, gAvgBv, gAvgBweta, gAwgBu, gAwgBw, gAwetagBv, gAwetagBweta

    eta1 = 2*y1/b - 1.
    eta2 = 2*y2/b - 1.

    fdim = 7*m*n*m*n

    kCppr = np.zeros((fdim,), dtype=INT)
    kCppc = np.zeros((fdim,), dtype=INT)
    kCppv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCpp
        c = -1
        for j in range(n):
            for l in range(n):
                gAugBu = integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                gAugBw = integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry)
                gAvgBv = integral_ff_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                gAvgBweta = integral_ffxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                gAwgBu = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry)
                gAwgBw = integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetagBv = integral_ffxi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry)
                gAwetagBweta = integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

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
                        fAwfBv = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
                        fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxifBu = integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
                        fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                        c += 1
                        kCppr[c] = row+0
                        kCppc[c] = col+0
                        kCppv[c] += 0.25*a*b*fAufBu*gAugBu*kt
                        c += 1
                        kCppr[c] = row+0
                        kCppc[c] = col+2
                        kCppv[c] += 0.5*b*dpb*fAufBwxi*gAugBw*kt
                        c += 1
                        kCppr[c] = row+1
                        kCppc[c] = col+1
                        kCppv[c] += 0.25*a*b*fAvfBv*gAvgBv*kt
                        c += 1
                        kCppr[c] = row+1
                        kCppc[c] = col+2
                        kCppv[c] += 0.5*a*dpb*fAvfBw*gAvgBweta*kt
                        c += 1
                        kCppr[c] = row+2
                        kCppc[c] = col+0
                        kCppv[c] += 0.5*b*dpb*fAwxifBu*gAwgBu*kt
                        c += 1
                        kCppr[c] = row+2
                        kCppc[c] = col+1
                        kCppv[c] += 0.5*a*dpb*fAwfBv*gAwetagBv*kt
                        c += 1
                        kCppr[c] = row+2
                        kCppc[c] = col+2
                        kCppv[c] += 0.25*a*b*kt*(fAwfBw*gAwgBw + 4*(dpb*dpb)*fAwfBw*gAwetagBweta/(b*b) + 4*(dpb*dpb)*fAwxifBwxi*gAwgBw/(a*a))

    kCpp = coo_matrix((kCppv, (kCppr, kCppc)), shape=(size, size))

    return kCpp


def fkCpby1y2(double y1, double y2,
          double kt, double a, double b, double dpb,
          int m, int n, int m1, int n1,
          double u1tx, double u1rx, double u2tx, double u2rx,
          double v1tx, double v1rx, double v2tx, double v2rx,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double u1ty, double u1ry, double u2ty, double u2ry,
          double v1ty, double v1ry, double v2ty, double v2ry,
          double w1ty, double w1ry, double w2ty, double w2ry,
          double u1txb, double u1rxb, double u2txb, double u2rxb,
          double v1txb, double v1rxb, double v2txb, double v2rxb,
          double w1txb, double w1rxb, double w2txb, double w2rxb,
          double u1tyb, double u1ryb, double u2tyb, double u2ryb,
          double v1tyb, double v1ryb, double v2tyb, double v2ryb,
          double w1tyb, double w1ryb, double w2tyb, double w2ryb,
          int size, int row0, int col0):
    cdef int i, j, k1, l1, c, row, col
    cdef double eta1, eta2, c0, c1

    cdef long [:] kCpbr, kCpbc
    cdef double [:] kCpbv

    cdef double fAupBu, fAvpBv, fAwpBw, fAwxipBu, fAwpBv
    cdef double gAuqBu, gAvqBv, gAwqBw, gAwetaqBv, gAwqBu

    eta1 = 2*y1/b - 1.
    eta2 = 2*y2/b - 1.

    fdim = 5*m*n*m1*n1

    # integrating along eta' from -1 to +1
    # and eta is related to eta' doing eta = c0 + c1*eta'
    c0 = 0.5*(eta1 + eta2)
    c1 = 0.5*(eta2 - eta1)

    kCpbr = np.zeros((fdim,), dtype=INT)
    kCpbc = np.zeros((fdim,), dtype=INT)
    kCpbv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCpb
        c = -1
        for j in range(n):
            for l1 in range(n1):
                gAuqBu = integral_ff_c0c1(c0, c1, l1, j, u1tyb, u1ryb, u2tyb, u2ryb, u1ty, u1ry, u2ty, u2ry)
                gAvqBv = integral_ff_c0c1(c0, c1, l1, j, v1tyb, v1ryb, v2tyb, v2ryb, v1ty, v1ry, v2ty, v2ry)
                gAwqBw = integral_ff_c0c1(c0, c1, l1, j, w1tyb, w1ryb, w2tyb, w2ryb, w1ty, w1ry, w2ty, w2ry)
                gAwetaqBv = integral_ffxi_c0c1(c0, c1, l1, j, v1tyb, v1ryb, v2tyb, v2ryb, w1ty, w1ry, w2ty, w2ry)
                gAwqBu = integral_ff_c0c1(c0, c1, l1, j, u1tyb, u1ryb, u2tyb, u2ryb, w1ty, w1ry, w2ty, w2ry)

                for i in range(m):
                    for k1 in range(m1):
                        row = row0 + num*(j*m + i)
                        col = col0 + num1*(l1*m1 + k1)

                        fAupBu = integral_ff(i, k1, u1tx, u1rx, u2tx, u2rx, u1txb, u1rxb, u2txb, u2rxb)
                        fAvpBv = integral_ff(i, k1, v1tx, v1rx, v2tx, v2rx, v1txb, v1rxb, v2txb, v2rxb)
                        fAwpBw = integral_ff(i, k1, w1tx, w1rx, w2tx, w2rx, w1txb, w1rxb, w2txb, w2rxb)
                        fAwxipBu = integral_ffxi(k1, i, u1txb, u1rxb, u2txb, u2rxb, w1tx, w1rx, w2tx, w2rx)
                        fAwpBv = integral_ff(i, k1, w1tx, w1rx, w2tx, w2rx, v1txb, v1rxb, v2txb, v2rxb)

                        c += 1
                        kCpbr[c] = row+0
                        kCpbc[c] = col+0
                        kCpbv[c] += -0.25*a*b*c1*fAupBu*gAuqBu*kt
                        c += 1
                        kCpbr[c] = row+1
                        kCpbc[c] = col+1
                        kCpbv[c] += -0.25*a*b*c1*fAvpBv*gAvqBv*kt
                        c += 1
                        kCpbr[c] = row+2
                        kCpbc[c] = col+0
                        kCpbv[c] += -0.5*b*c1*dpb*fAwxipBu*gAwqBu*kt
                        c += 1
                        kCpbr[c] = row+2
                        kCpbc[c] = col+1
                        kCpbv[c] += -0.5*a*c1*dpb*fAwpBv*gAwetaqBv*kt
                        c += 1
                        kCpbr[c] = row+2
                        kCpbc[c] = col+2
                        kCpbv[c] += -0.25*a*b*c1*fAwpBw*gAwqBw*kt

    kCpb = coo_matrix((kCpbv, (kCpbr, kCpbc)), shape=(size, size))

    return kCpb


def fkCbbpby1y2(double y1, double y2,
        double kt, double a, double b, int m1, int n1,
        double u1txb, double u1rxb, double u2txb, double u2rxb,
        double v1txb, double v1rxb, double v2txb, double v2rxb,
        double w1txb, double w1rxb, double w2txb, double w2rxb,
        double u1tyb, double u1ryb, double u2tyb, double u2ryb,
        double v1tyb, double v1ryb, double v2tyb, double v2ryb,
        double w1tyb, double w1ryb, double w2tyb, double w2ryb,
        int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col
    cdef double eta1, eta2, c1

    cdef long [:] kCbbpbr, kCbbpbc
    cdef double [:] kCbbpbv

    cdef double pAupBu, pAvpBv, pAwpBw
    cdef double qAuqBu, qAvqBv, qAwqBw

    eta1 = 2*y1/b - 1.
    eta2 = 2*y2/b - 1.
    c1 = 0.5*(eta2 - eta1)

    fdim = 3*m1*n1*m1*n1

    kCbbpbr = np.zeros((fdim,), dtype=INT)
    kCbbpbc = np.zeros((fdim,), dtype=INT)
    kCbbpbv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCbbpb
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):
                pAupBu = integral_ff(i1, k1, u1txb, u1rxb, u2txb, u2rxb, u1txb, u1rxb, u2txb, u2rxb)
                pAvpBv = integral_ff(i1, k1, v1txb, v1rxb, v2txb, v2rxb, v1txb, v1rxb, v2txb, v2rxb)
                pAwpBw = integral_ff(i1, k1, w1txb, w1rxb, w2txb, w2rxb, w1txb, w1rxb, w2txb, w2rxb)

                for j1 in range(n1):
                    for l1 in range(n1):
                        row = row0 + num1*(j1*m1 + i1)
                        col = col0 + num1*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        qAuqBu = integral_ff(j1, l1, u1tyb, u1ryb, u2tyb, u2ryb, u1tyb, u1ryb, u2tyb, u2ryb)
                        qAvqBv = integral_ff(j1, l1, v1tyb, v1ryb, v2tyb, v2ryb, v1tyb, v1ryb, v2tyb, v2ryb)
                        qAwqBw = integral_ff(j1, l1, w1tyb, w1ryb, w2tyb, w2ryb, w1tyb, w1ryb, w2tyb, w2ryb)

                        c += 1
                        kCbbpbr[c] = row+0
                        kCbbpbc[c] = col+0
                        kCbbpbv[c] += 0.25*a*b*c1*kt*pAupBu*qAuqBu
                        c += 1
                        kCbbpbr[c] = row+1
                        kCbbpbc[c] = col+1
                        kCbbpbv[c] += 0.25*a*b*c1*kt*pAvpBv*qAvqBv
                        c += 1
                        kCbbpbr[c] = row+2
                        kCbbpbc[c] = col+2
                        kCbbpbv[c] += 0.25*a*b*c1*kt*pAwpBw*qAwqBw

    kCbbpb = coo_matrix((kCbbpbv, (kCbbpbr, kCbbpbc)), shape=(size, size))

    return kCbbpb
