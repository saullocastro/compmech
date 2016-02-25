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
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                         double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.h':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_fxi(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil

cdef extern from 'bardell_12.h':
    double integral_ff_12(double eta1, double eta2, int i, int j,
                          double x1t, double x1r, double x2t, double x2r,
                          double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxi_12(double eta1, double eta2, int i, int j,
                       double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_c0c1.h':
    double integral_ff_c0c1(double c0, double c1, int i, int j,
                            double x1t, double x1r, double x2t, double x2r,
                            double y1t, double y1r, double y2t, double y2r) nogil

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int num = 3
cdef int num1 = 3
cdef double pi = 3.141592653589793


def fkCppx1x2y1y2(double x1, double x2, double y1, double y2,
          double kt, double a, double b, int m, int n,
          double u1tx, double u1rx, double u2tx, double u2rx,
          double v1tx, double v1rx, double v2tx, double v2rx,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double u1ty, double u1ry, double u2ty, double u2ry,
          double v1ty, double v1ry, double v2ty, double v2ry,
          double w1ty, double w1ry, double w2ty, double w2ry,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double xi1, xi2, eta1, eta2

    cdef np.ndarray[cINT, ndim=1] kCppr, kCppc
    cdef np.ndarray[cDOUBLE, ndim=1] kCppv

    cdef double fAufBu, fAvfBv, fAwfBw
    cdef double gAugBu, gAvgBv, gAwgBw

    xi1 = 2*x1/a - 1.
    xi2 = 2*x2/a - 1.
    eta1 = 2*y1/b - 1.
    eta2 = 2*y2/b - 1.

    fdim = 3*m*n*m*n

    kCppr = np.zeros((fdim,), dtype=INT)
    kCppc = np.zeros((fdim,), dtype=INT)
    kCppv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCpp
        c = -1
        for i in range(m):
            for k in range(m):

                fAufBu = integral_ff_12(xi1, xi2, i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                fAvfBv = integral_ff_12(xi1, xi2, i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                fAwfBw = integral_ff_12(xi1, xi2, i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j in range(n):
                    for l in range(n):
                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAugBu = integral_ff_12(y1, y2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry)
                        gAvgBv = integral_ff_12(y1, y2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry)
                        gAwgBw = integral_ff_12(y1, y2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        kCppr[c] = row+0
                        kCppc[c] = col+0
                        kCppv[c] += 0.25*a*b*fAufBu*gAugBu*kt
                        c += 1
                        kCppr[c] = row+1
                        kCppc[c] = col+1
                        kCppv[c] += 0.25*a*b*fAvfBv*gAvgBv*kt
                        c += 1
                        kCppr[c] = row+2
                        kCppc[c] = col+2
                        kCppv[c] += 0.25*a*b*fAwfBw*gAwgBw*kt

    kCpp = coo_matrix((kCppv, (kCppr, kCppc)), shape=(size, size))

    return kCpp


def fkCpbx1x2y1y2(double x1, double x2, double y1, double y2,
          double kt, double a, double b, double bb,
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
    cdef double xi1, xi2, eta1, eta2, c0, c1

    cdef np.ndarray[cINT, ndim=1] kCpbr, kCpbc
    cdef np.ndarray[cDOUBLE, ndim=1] kCpbv

    cdef double fAupBu, fAvpBv, fAwpBw
    cdef double gAuqBu, gAvqBv, gAwqBw

    xi1 = 2*x1/a - 1.
    xi2 = 2*x2/a - 1.
    eta1 = 2*y1/b - 1.
    eta2 = 2*y2/b - 1.

    fdim = 3*m*n*m1*n1

    # integrating along eta' from -1 to +1
    # and eta is related to eta' doing eta' = c0 + c1*eta
    c0 = 0.5*(eta1 + eta2)
    c1 = 0.5*(eta2 - eta1)

    kCpbr = np.zeros((fdim,), dtype=INT)
    kCpbc = np.zeros((fdim,), dtype=INT)
    kCpbv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCpb
        c = -1
        for i in range(m):
            for k1 in range(m1):

                fAupBu = integral_ff_12(xi1, xi2, i, k1, u1tx, u1rx, u2tx, u2rx, u1txb, u1rxb, u2txb, u2rxb)
                fAvpBv = integral_ff_12(xi1, xi2, i, k1, v1tx, v1rx, v2tx, v2rx, v1txb, v1rxb, v2txb, v2rxb)
                fAwpBw = integral_ff_12(xi1, xi2, i, k1, w1tx, w1rx, w2tx, w2rx, w1txb, w1rxb, w2txb, w2rxb)

                for j in range(n):
                    for l1 in range(n1):

                        row = row0 + num1*(j*m + i)
                        col = col0 + num1*(l1*m1 + k1)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        gAuqBu = integral_ff_c0c1(c0, c1, l1, j, u1tyb, u1ryb, u2tyb, u2ryb, u1ty, u1ry, u2ty, u2ry)
                        gAvqBv = integral_ff_c0c1(c0, c1, l1, j, v1tyb, v1ryb, v2tyb, v2ryb, v1ty, v1ry, v2ty, v2ry)
                        gAwqBw = integral_ff_c0c1(c0, c1, l1, j, w1tyb, w1ryb, w2tyb, w2ryb, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        kCpbr[c] = row+0
                        kCpbc[c] = col+0
                        kCpbv[c] += -0.25*a*b*fAupBu*gAuqBu*kt
                        c += 1
                        kCpbr[c] = row+1
                        kCpbc[c] = col+1
                        kCpbv[c] += -0.25*a*b*fAvpBv*gAvqBv*kt
                        c += 1
                        kCpbr[c] = row+2
                        kCpbc[c] = col+2
                        kCpbv[c] += -0.25*a*b*fAwpBw*gAwqBw*kt

    kCpb = coo_matrix((kCpbv, (kCpbr, kCpbc)), shape=(size, size))

    return kCpb


def fkCbbpbx1x2(double x1, double x2, double kt, double kr,
        double ys, double a, double b, int m1, int n1,
        double u1txb, double u1rxb, double u2txb, double u2rxb,
        double v1txb, double v1rxb, double v2txb, double v2rxb,
        double w1txb, double w1rxb, double w2txb, double w2rxb,
        double u1tyb, double u1ryb, double u2tyb, double u2ryb,
        double v1tyb, double v1ryb, double v2tyb, double v2ryb,
        double w1tyb, double w1ryb, double w2tyb, double w2ryb,
        int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col
    cdef double xi1, xi2

    cdef np.ndarray[cINT, ndim=1] kCbbpbr, kCbbpbc
    cdef np.ndarray[cDOUBLE, ndim=1] kCbbpbv

    cdef double pAupBu, pAvpBv, pAwpBw
    cdef double qAuqBu, qAvqBv, qAwqBw, qAwetaqBweta

    xi1 = 2*x1/a - 1.
    xi2 = 2*x2/a - 1.

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
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        qAuqBu = integral_ff_12(xi1, xi2, j1, l1, u1tyb, u1ryb, u2tyb, u2ryb, u1tyb, u1ryb, u2tyb, u2ryb)
                        qAvqBv = integral_ff_12(xi1, xi2, j1, l1, v1tyb, v1ryb, v2tyb, v2ryb, v1tyb, v1ryb, v2tyb, v2ryb)
                        qAwqBw = integral_ff_12(xi1, xi2, j1, l1, w1tyb, w1ryb, w2tyb, w2ryb, w1tyb, w1ryb, w2tyb, w2ryb)

                        c += 1
                        kCbbpbr[c] = row+0
                        kCbbpbc[c] = col+0
                        kCbbpbv[c] += 0.25*a*b*kt*pAupBu*qAuqBu
                        c += 1
                        kCbbpbr[c] = row+1
                        kCbbpbc[c] = col+1
                        kCbbpbv[c] += 0.25*a*b*kt*pAvpBv*qAvqBv
                        c += 1
                        kCbbpbr[c] = row+2
                        kCbbpbc[c] = col+2
                        kCbbpbv[c] += 0.25*a*b*kt*pAwpBw*qAwqBw

    kCbbpb = coo_matrix((kCbbpbv, (kCbbpbr, kCbbpbc)), shape=(size, size))

    return kCbbpb


def fkCbbbf(double kt, double kr, double a, double bb,
          int m1, int n1,
          double u1txb, double u1rxb, double u2txb, double u2rxb,
          double v1txb, double v1rxb, double v2txb, double v2rxb,
          double w1txb, double w1rxb, double w2txb, double w2rxb,
          double u1tyb, double u1ryb, double u2tyb, double u2ryb,
          double v1tyb, double v1ryb, double v2tyb, double v2ryb,
          double w1tyb, double w1ryb, double w2tyb, double w2ryb,
          int size, int row0, int col0):
    cdef int i1, j1, k1, l1, c, row, col

    cdef np.ndarray[cINT, ndim=1] kCbbbfr, kCbbbfc
    cdef np.ndarray[cDOUBLE, ndim=1] kCbbbfv

    cdef double pAupBu, pAvpBv, pAwpBw
    cdef double qAu, qAv, qAw, qAweta, qBu, qBv, qBw, qBweta

    cdef double eta = 0. # connection at the middle of the stiffener's base

    fdim = 3*m1*n1*m1*n1

    kCbbbfr = np.zeros((fdim,), dtype=INT)
    kCbbbfc = np.zeros((fdim,), dtype=INT)
    kCbbbfv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCbbbf
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):

                pAupBu = integral_ff(i1, k1, u1txb, u1rxb, u2txb, u2rxb, u1txb, u1rxb, u2txb, u2rxb)
                pAvpBv = integral_ff(i1, k1, v1txb, v1rxb, v2txb, v2rxb, v1txb, v1rxb, v2txb, v2rxb)
                pAwpBw = integral_ff(i1, k1, w1txb, w1rxb, w2txb, w2rxb, w1txb, w1rxb, w2txb, w2rxb)

                for j1 in range(n1):
                    qAu = calc_f(j1, eta, u1tyb, u1ryb, u2tyb, u2ryb)
                    qAv = calc_f(j1, eta, v1tyb, v1ryb, v2tyb, v2ryb)
                    qAw = calc_f(j1, eta, w1tyb, w1ryb, w2tyb, w2ryb)
                    qAweta = calc_fxi(j1, eta, w1tyb, w1ryb, w2tyb, w2ryb)

                    for l1 in range(n1):

                        row = row0 + num1*(j1*m1 + i1)
                        col = col0 + num1*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        qBu = calc_f(l1, eta, u1tyb, u1ryb, u2tyb, u2ryb)
                        qBv = calc_f(l1, eta, v1tyb, v1ryb, v2tyb, v2ryb)
                        qBw = calc_f(l1, eta, w1tyb, w1ryb, w2tyb, w2ryb)
                        qBweta = calc_fxi(l1, eta, w1tyb, w1ryb, w2tyb, w2ryb)

                        c += 1
                        kCbbbfr[c] = row+0
                        kCbbbfc[c] = col+0
                        kCbbbfv[c] += 0.5*a*kt*pAupBu*qAu*qBu
                        c += 1
                        kCbbbfr[c] = row+1
                        kCbbbfc[c] = col+1
                        kCbbbfv[c] += 0.5*a*kt*pAvpBv*qAv*qBv
                        c += 1
                        kCbbbfr[c] = row+2
                        kCbbbfc[c] = col+2
                        kCbbbfv[c] += 0.5*a*kt*(pAwpBw*qAw*qBw + 4*kr*pAwpBw*qAweta*qBweta/((bb*bb)*kt))

    kCbbbf = coo_matrix((kCbbbfv, (kCbbbfr, kCbbbfc)), shape=(size, size))

    return kCbbbf


def fkCbf(double kt, double kr, double a, double bb, double bf,
          int m1, int n1, int m2, int n2,
          double u1txb, double u1rxb, double u2txb, double u2rxb,
          double v1txb, double v1rxb, double v2txb, double v2rxb,
          double w1txb, double w1rxb, double w2txb, double w2rxb,
          double u1tyb, double u1ryb, double u2tyb, double u2ryb,
          double v1tyb, double v1ryb, double v2tyb, double v2ryb,
          double w1tyb, double w1ryb, double w2tyb, double w2ryb,
          double u1txf, double u1rxf, double u2txf, double u2rxf,
          double v1txf, double v1rxf, double v2txf, double v2rxf,
          double w1txf, double w1rxf, double w2txf, double w2rxf,
          double u1tyf, double u1ryf, double u2tyf, double u2ryf,
          double v1tyf, double v1ryf, double v2tyf, double v2ryf,
          double w1tyf, double w1ryf, double w2tyf, double w2ryf,
          int size, int row0, int col0):
    cdef int i1, j1, k2, l2, c, row, col

    cdef np.ndarray[cINT, ndim=1] kCbfr, kCbfc
    cdef np.ndarray[cDOUBLE, ndim=1] kCbfv

    cdef double pAurBu, pAvrBw, pAwrBv, pAwrBw
    cdef double qAu, qAv, qAw, qAweta, sBu, sBv, sBw, sBweta

    cdef double eta = 0. # connection at the middle of the stiffener's base

    fdim = 4*m1*n1*m2*n2

    kCbfr = np.zeros((fdim,), dtype=INT)
    kCbfc = np.zeros((fdim,), dtype=INT)
    kCbfv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCbf
        c = -1
        for i1 in range(m1):
            for k2 in range(m2):

                pAurBu = integral_ff(i1, k2, u1txb, u1rxb, u2txb, u2rxb, u1txf, u1rxf, u2txf, u2rxf)
                pAvrBw = integral_ff(i1, k2, v1txb, v1rxb, v2txb, v2rxb, w1txf, w1rxf, w2txf, w2rxf)
                pAwrBv = integral_ff(i1, k2, w1txb, w1rxb, w2txb, w2rxb, v1txf, v1rxf, v2txf, v2rxf)
                pAwrBw = integral_ff(i1, k2, w1txb, w1rxb, w2txb, w2rxb, w1txf, w1rxf, w2txf, w2rxf)

                for j1 in range(n1):
                    qAu = calc_f(j1, eta, u1tyb, u1ryb, u2tyb, u2ryb)
                    qAv = calc_f(j1, eta, v1tyb, v1ryb, v2tyb, v2ryb)
                    qAw = calc_f(j1, eta, w1tyb, w1ryb, w2tyb, w2ryb)
                    qAweta = calc_fxi(j1, eta, w1tyb, w1ryb, w2tyb, w2ryb)

                    for l2 in range(n2):

                        row = row0 + num1*(j1*m1 + i1)
                        col = col0 + num1*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        # connection at eta = -1 for stiffener's flange
                        sBu = calc_f(l2, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                        sBv = calc_f(l2, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                        sBw = calc_f(l2, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                        sBweta = calc_fxi(l2, -1., w1tyf, w1ryf, w2tyf, w2ryf)

                        c += 1
                        kCbfr[c] = row+0
                        kCbfc[c] = col+0
                        kCbfv[c] += -0.5*a*kt*pAurBu*qAu*sBu
                        c += 1
                        kCbfr[c] = row+1
                        kCbfc[c] = col+2
                        kCbfv[c] += -0.5*a*kt*pAvrBw*qAv*sBw
                        c += 1
                        kCbfr[c] = row+2
                        kCbfc[c] = col+1
                        kCbfv[c] += 0.5*a*kt*pAwrBv*qAw*sBv
                        c += 1
                        kCbfr[c] = row+2
                        kCbfc[c] = col+2
                        kCbfv[c] += -2*a*kr*pAwrBw*qAweta*sBweta/(bb*bf)

    kCbf = coo_matrix((kCbfv, (kCbfr, kCbfc)), shape=(size, size))

    return kCbf


def fkCff(double kt, double kr, double a, double bf, int m2, int n2,
          double u1txf, double u1rxf, double u2txf, double u2rxf,
          double v1txf, double v1rxf, double v2txf, double v2rxf,
          double w1txf, double w1rxf, double w2txf, double w2rxf,
          double u1tyf, double u1ryf, double u2tyf, double u2ryf,
          double v1tyf, double v1ryf, double v2tyf, double v2ryf,
          double w1tyf, double w1ryf, double w2tyf, double w2ryf,
          int size, int row0, int col0):
    cdef int i2, k2, j2, l2, c, row, col

    cdef np.ndarray[cINT, ndim=1] kCffr, kCffc
    cdef np.ndarray[cDOUBLE, ndim=1] kCffv

    cdef double rAurBu, rAvrBv, rAwrBw
    cdef double sAu, sBu, sAv, sBv, sAw, sBw, sAweta, sBweta

    fdim = 5*m2*n2*m2*n2

    kCffr = np.zeros((fdim,), dtype=INT)
    kCffc = np.zeros((fdim,), dtype=INT)
    kCffv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:

        # kCff
        c = -1
        for i2 in range(m2):
            for k2 in range(m2):

                rAurBu = integral_ff(i2, k2, u1txf, u1rxf, u2txf, u2rxf, u1txf, u1rxf, u2txf, u2rxf)
                rAvrBv = integral_ff(i2, k2, v1txf, v1rxf, v2txf, v2rxf, v1txf, v1rxf, v2txf, v2rxf)
                rAwrBw = integral_ff(i2, k2, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)

                for j2 in range(n2):
                    sAu = calc_f(j2, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                    sAv = calc_f(j2, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                    sAw = calc_f(j2, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                    sAweta = calc_fxi(j2, -1., w1tyf, w1ryf, w2tyf, w2ryf)

                    for l2 in range(n2):

                        row = row0 + num1*(j2*m2 + i2)
                        col = col0 + num1*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        sBu = calc_f(l2, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                        sBv = calc_f(l2, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                        sBw = calc_f(l2, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                        sBweta = calc_fxi(l2, -1., w1tyf, w1ryf, w2tyf, w2ryf)

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
                        kCffv[c] += 0.5*a*kt*(rAwrBw*sAw*sBw + 4*kr*rAwrBw*sAweta*sBweta/((bf*bf)*kt))

    kCff = coo_matrix((kCffv, (kCffr, kCffc)), shape=(size, size))

    return kCff


