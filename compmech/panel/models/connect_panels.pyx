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
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffxi(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fxifxi(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.h':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_fxi(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int num = 3
cdef double pi = 3.141592653589793


def fkCppycte(double kt, double kr, double a, double b,
          int m1, int n1, double eta_conn_p,
          double u1tx, double u1rx, double u2tx, double u2rx,
          double v1tx, double v1rx, double v2tx, double v2rx,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double u1ty, double u1ry, double u2ty, double u2ry,
          double v1ty, double v1ry, double v2ty, double v2ry,
          double w1ty, double w1ry, double w2ty, double w2ry,
          int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col

    cdef np.ndarray[cINT, ndim=1] kCppr, kCppc
    cdef np.ndarray[cDOUBLE, ndim=1] kCppv

    cdef double f1Auf1Bu, f1Avf1Bv, f1Awf1Bw, f1Awxif1Bwxi
    cdef double g1Au, g1Bu, g1Av, g1Bv, g1Aw, g1Bw, g1Aweta, g1Bweta

    fdim = 3*m1*n1*m1*n1

    kCppr = np.zeros((fdim,), dtype=INT)
    kCppc = np.zeros((fdim,), dtype=INT)
    kCppv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):

                f1Auf1Bu = integral_ff(i1, k1, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                f1Avf1Bv = integral_ff(i1, k1, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                f1Awf1Bw = integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
                f1Awxif1Bwxi = integral_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j1 in range(n1):
                    g1Au = calc_f(j1, eta_conn_p, u1ty, u1ry, u2ty, u2ry)
                    g1Av = calc_f(j1, eta_conn_p, v1ty, v1ry, v2ty, v2ry)
                    g1Aw = calc_f(j1, eta_conn_p, w1ty, w1ry, w2ty, w2ry)
                    g1Aweta = calc_fxi(j1, eta_conn_p, w1ty, w1ry, w2ty, w2ry)

                    for l1 in range(n1):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g1Bu = calc_f(l1, eta_conn_p, u1ty, u1ry, u2ty, u2ry)
                        g1Bv = calc_f(l1, eta_conn_p, v1ty, v1ry, v2ty, v2ry)
                        g1Bw = calc_f(l1, eta_conn_p, w1ty, w1ry, w2ty, w2ry)
                        g1Bweta = calc_fxi(l1, eta_conn_p, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        kCppr[c] = row+0
                        kCppc[c] = col+0
                        kCppv[c] += 0.5*a*f1Auf1Bu*g1Au*g1Bu*kt
                        c += 1
                        kCppr[c] = row+1
                        kCppc[c] = col+1
                        kCppv[c] += 0.5*a*f1Avf1Bv*g1Av*g1Bv*kt
                        c += 1
                        kCppr[c] = row+2
                        kCppc[c] = col+2
                        kCppv[c] += 0.5*a*kt*(f1Awf1Bw*g1Aw*g1Bw + f1Awf1Bw*g1Aweta*g1Bweta*kr/kt + f1Awxif1Bwxi*g1Aw*g1Bw*kr/kt)

    kCpp = coo_matrix((kCppv, (kCppr, kCppc)), shape=(size, size))

    return kCpp


def fkCp1p2ycte(double kt, double kr, double a, double b,
          int m1, int n1, int m2, int n2,
          double eta_conn_p1, double eta_conn_p2,
          double u1tx1, double u1rx1, double u2tx1, double u2rx1,
          double v1tx1, double v1rx1, double v2tx1, double v2rx1,
          double w1tx1, double w1rx1, double w2tx1, double w2rx1,
          double u1ty1, double u1ry1, double u2ty1, double u2ry1,
          double v1ty1, double v1ry1, double v2ty1, double v2ry1,
          double w1ty1, double w1ry1, double w2ty1, double w2ry1,
          double u1tx2, double u1rx2, double u2tx2, double u2rx2,
          double v1tx2, double v1rx2, double v2tx2, double v2rx2,
          double w1tx2, double w1rx2, double w2tx2, double w2rx2,
          double u1ty2, double u1ry2, double u2ty2, double u2ry2,
          double v1ty2, double v1ry2, double v2ty2, double v2ry2,
          double w1ty2, double w1ry2, double w2ty2, double w2ry2,
          int size, int row0, int col0):
    cdef int i1, k2, j1, l2, c, row, col

    cdef np.ndarray[cINT, ndim=1] kCp1p2r, kCp1p2c
    cdef np.ndarray[cDOUBLE, ndim=1] kCp1p2v

    cdef double f1Auf2Bu, f1Avf2Bv, f1Awf2Bw, f1Awxif2Bwxi
    cdef double g1Au, g2Bu, g1Av, g2Bv, g1Aw, g2Bw, g1Aweta, g2Bweta

    fdim = 3*m1*n1*m2*n2

    kCp1p2r = np.zeros((fdim,), dtype=INT)
    kCp1p2c = np.zeros((fdim,), dtype=INT)
    kCp1p2v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k2 in range(m2):

                f1Auf2Bu = integral_ff(i1, k2, u1tx1, u1rx1, u2tx1, u2rx1, u1tx2, u1rx2, u2tx2, u2rx2)
                f1Avf2Bv = integral_ff(i1, k2, v1tx1, v1rx1, v2tx1, v2rx1, v1tx2, v1rx2, v2tx2, v2rx2)
                f1Awf2Bw = integral_ff(i1, k2, w1tx1, w1rx1, w2tx1, w2rx1, w1tx2, w1rx2, w2tx2, w2rx2)
                f1Awxif2Bwxi = integral_fxifxi(i1, k2, w1tx1, w1rx1, w2tx1, w2rx1, w1tx2, w1rx2, w2tx2, w2rx2)

                for j1 in range(n1):

                    g1Au = calc_f(j1, eta_conn_p1, u1ty1, u1ry1, u2ty1, u2ry1)
                    g1Av = calc_f(j1, eta_conn_p1, v1ty1, v1ry1, v2ty1, v2ry1)
                    g1Aw = calc_f(j1, eta_conn_p1, w1ty1, w1ry1, w2ty1, w2ry1)
                    g1Aweta = calc_fxi(j1, eta_conn_p1, w1ty1, w1ry1, w2ty1, w2ry1)

                    for l2 in range(n2):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry
                        #if row > col:
                            #continue

                        g2Bu = calc_f(l2, eta_conn_p2, u1ty2, u1ry2, u2ty2, u2ry2)
                        g2Bv = calc_f(l2, eta_conn_p2, v1ty2, v1ry2, v2ty2, v2ry2)
                        g2Bw = calc_f(l2, eta_conn_p2, w1ty2, w1ry2, w2ty2, w2ry2)
                        g2Bweta = calc_fxi(l2, eta_conn_p2, w1ty2, w1ry2, w2ty2, w2ry2)

                        c += 1
                        kCp1p2r[c] = row+0
                        kCp1p2c[c] = col+0
                        kCp1p2v[c] += 0.5*a*f1Auf2Bu*g1Au*g2Bu*kt
                        c += 1
                        kCp1p2r[c] = row+1
                        kCp1p2c[c] = col+1
                        kCp1p2v[c] += 0.5*a*f1Avf2Bv*g1Av*g2Bv*kt
                        c += 1
                        kCp1p2r[c] = row+2
                        kCp1p2c[c] = col+2
                        kCp1p2v[c] += 0.5*a*kt*(f1Awf2Bw*g1Aw*g2Bw + f1Awf2Bw*g1Aweta*g2Bweta*kr/kt + f1Awxif2Bwxi*g1Aw*g2Bw*kr/kt)

    kCp1p2 = coo_matrix((kCp1p2v, (kCp1p2r, kCp1p2c)), shape=(size, size))

    return kCp1p2
