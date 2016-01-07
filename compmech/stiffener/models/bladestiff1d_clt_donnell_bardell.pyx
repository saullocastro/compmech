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


def fk0f(double ys, double a, double b, double bf, double df, double E1, double F1,
         double S1, double Jxx, int m, int n,
         double u1tx, double u1rx, double u2tx, double u2rx,
         double w1tx, double w1rx, double w2tx, double w2rx,
         double u1ty, double u1ry, double u2ty, double u2ry,
         double w1ty, double w1ry, double w2ty, double w2ry,
         int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef double fAuxifBuxi, fAuxifBwxixi, fAuxifBwxi, fAwxixifBuxi
    cdef double fAwxifBuxi, fAwxifBwxi, fAwxifBwxixi, fAwxixifBwxi
    cdef double fAwxixifBwxixi
    cdef double gAu, gBu, gAw, gBw, gAweta, gBweta

    cdef np.ndarray[cINT, ndim=1] k0fr, k0fc
    cdef np.ndarray[cDOUBLE, ndim=1] k0fv

    eta = 2*ys/b - 1.

    fdim = 4*m*n*m*n

    k0fr = np.zeros((fdim,), dtype=INT)
    k0fc = np.zeros((fdim,), dtype=INT)
    k0fv = np.zeros((fdim,), dtype=DOUBLE)

    # k0f
    c = -1
    for i in range(m):
        for k in range(m):
            fAuxifBuxi = integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAuxifBwxixi = integral_fxifxixi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAuxifBwxi = integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBuxi = integral_fxifxixi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBuxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx)
            fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBwxixi = integral_fxifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBwxi = integral_fxifxixi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBwxixi = integral_fxixifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

            for j in range(n):
                gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                gAweta = calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry)

                for l in range(n):

                    row = row0 + num*(j*m + i)
                    col = col0 + num*(l*m + k)

                    #NOTE symmetry
                    if row > col:
                        continue

                    gBu = calc_f(l, eta, u1ty, u1ry, u2ty, u2ry)
                    gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)
                    gBweta = calc_fxi(l, eta, w1ty, w1ry, w2ty, w2ry)

                    c += 1
                    k0fr[c] = row+0
                    k0fc[c] = col+0
                    k0fv[c] += E1*(bf*bf)*fAuxifBuxi*gAu*gBu/a
                    c += 1
                    k0fr[c] = row+0
                    k0fc[c] = col+2
                    k0fv[c] += 2*(bf*bf)*(E1*b*df*fAuxifBwxixi*gAu*gBw - S1*a*fAuxifBwxi*gAu*gBweta)/((a*a)*b)
                    c += 1
                    k0fr[c] = row+2
                    k0fc[c] = col+0
                    k0fv[c] += 2*(bf*bf)*(E1*b*df*fAwxixifBuxi*gAw*gBu - S1*a*fAwxifBuxi*gAweta*gBu)/((a*a)*b)
                    c += 1
                    k0fr[c] = row+2
                    k0fc[c] = col+2
                    k0fv[c] += 4*(bf*bf)*(Jxx*(a*a)*fAwxifBwxi*gAweta*gBweta - S1*a*b*df*(fAwxifBwxixi*gAweta*gBw + fAwxixifBwxi*gAw*gBweta) + (b*b)*fAwxixifBwxixi*gAw*gBw*(E1*(df*df) + F1))/((a*a*a)*(b*b))

    k0f = coo_matrix((k0fv, (k0fr, k0fc)), shape=(size, size))

    return k0f


def fkG0f(double ys, double Fx, double a, double b, double bf, int m, int n,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double w1ty, double w1ry, double w2ty, double w2ry,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kG0fr, kG0fc
    cdef np.ndarray[cDOUBLE, ndim=1] kG0fv

    cdef double fAwxifBwxi, gAw, gBw

    eta = 2*ys/b - 1.

    fdim = 1*m*n*m*n

    kG0fr = np.zeros((fdim,), dtype=INT)
    kG0fc = np.zeros((fdim,), dtype=INT)
    kG0fv = np.zeros((fdim,), dtype=DOUBLE)

    # kG0f
    c = -1
    for i in range(m):
        for k in range(m):
            fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

            for j in range(n):
                gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)

                for l in range(n):
                    row = row0 + num*(j*m + i)
                    col = col0 + num*(l*m + k)

                    #NOTE symmetry
                    if row > col:
                        continue

                    gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)

                    c += 1
                    kG0fr[c] = row+2
                    kG0fc[c] = col+2
                    kG0fv[c] += Fx*bf*fAwxifBwxi*gAw*gBw/a

    kG0f = coo_matrix((kG0fv, (kG0fr, kG0fc)), shape=(size, size))

    return kG0f


def fkMf(double ys, double mu, double h, double hb, double hf, double a,
         double b, double bf, double df,
         int m, int n,
         double u1tx, double u1rx, double u2tx, double u2rx,
         double v1tx, double v1rx, double v2tx, double v2rx,
         double w1tx, double w1rx, double w2tx, double w2rx,
         double u1ty, double u1ry, double u2ty, double u2ry,
         double v1ty, double v1ry, double v2ty, double v2ry,
         double w1ty, double w1ry, double w2ty, double w2ry,
         int size, int row0, int col0):
    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwxifBu, fAwfBv, fAwfBw
    cdef double fAwxifBwxi
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAweta, gBweta

    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kMfr, kMfc
    cdef np.ndarray[cDOUBLE, ndim=1] kMfv

    eta = 2*ys/b - 1.

    fdim = 7*m*n*m*n

    kMfr = np.zeros((fdim,), dtype=INT)
    kMfc = np.zeros((fdim,), dtype=INT)
    kMfv = np.zeros((fdim,), dtype=DOUBLE)

    # kMf
    c = -1
    for i in range(m):
        for k in range(m):

            fAufBu = integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
            fAufBwxi = integral_ffxi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAvfBv = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
            fAvfBw = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBu = integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBv = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx)
            fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBwxi = integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

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
                    kMfr[c] = row+0
                    kMfc[c] = col+0
                    kMfv[c] += 0.25*a*(bf*bf)*fAufBu*gAu*gBu*hf*mu
                    c += 1
                    kMfr[c] = row+0
                    kMfc[c] = col+2
                    kMfv[c] += (bf*bf)*df*fAufBwxi*gAu*gBw*hf*mu
                    c += 1
                    kMfr[c] = row+1
                    kMfc[c] = col+1
                    kMfv[c] += 0.25*a*(bf*bf)*fAvfBv*gAv*gBv*hf*mu
                    c += 1
                    kMfr[c] = row+1
                    kMfc[c] = col+2
                    kMfv[c] += a*(bf*bf)*df*fAvfBw*gAv*gBweta*hf*mu/b
                    c += 1
                    kMfr[c] = row+2
                    kMfc[c] = col+0
                    kMfv[c] += (bf*bf)*df*fAwxifBu*gAw*gBu*hf*mu
                    c += 1
                    kMfr[c] = row+2
                    kMfc[c] = col+1
                    kMfv[c] += a*(bf*bf)*df*fAwfBv*gAweta*gBv*hf*mu/b
                    c += 1
                    kMfr[c] = row+2
                    kMfc[c] = col+2
                    kMfv[c] += 0.0833333333333333*(bf*bf)*hf*mu*((a*a)*fAwfBw*(3*(b*b)*gAw*gBw + gAweta*gBweta*(4*(bf*bf) + 6*bf*(h + 2*hb) + 3*(h + 2*hb)**2)) + (b*b)*fAwxifBwxi*gAw*gBw*(4*(bf*bf) + 6*bf*(h + 2*hb) + 3*(h + 2*hb)**2))/(a*(b*b))

    kMf = coo_matrix((kMfv, (kMfr, kMfc)), shape=(size, size))

    return kMf

