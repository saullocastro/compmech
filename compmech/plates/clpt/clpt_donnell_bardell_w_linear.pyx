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

include '../../func/bardell/bardell.pyx'

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef double pi = 3.141592653589793
cdef int num1 = 1


def fk0(double a, double b, np.ndarray[cDOUBLE, ndim=2] F,
        double u1tx, double u1rx, double u2tx, double u2rx,
        double v1tx, double v1rx, double v2tx, double v2rx,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double u1ty, double u1ry, double u2ty, double u2ry,
        double v1ty, double v1ry, double v2ty, double v2ry,
        double w1ty, double w1ry, double w2ty, double w2ry,
        int m1, int n1):
    cdef int i1, j1, k1, l1, c, row, col
    cdef double D11, D12, D16, D22, D26, D66

    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    cdef double fAwxifBvxi, fAwxixifBwxixi, fAwfBwxixi, fAwxixifBw,
    cdef double fAwxifBwxixi, fAwxixifBwxi, fAwfBw, fAwfBwxi, fAwxifBw,
    cdef double fAwxifBwxi
    cdef double gAwetaetagBwetaeta, gAwgBwetaeta, gAwetaetagBw,
    cdef double gAwetagBwetaeta, gAwetaetagBweta, gAwgBw, gAwgBweta, gAwetagBw,
    cdef double gAwetagBweta

    fdim = 1*m1*m1*n1*n1

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

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

            fAwxixifBwxixi = integral_fxixifxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxixi = integral_ffxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBw = integral_ffxixi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBwxixi = integral_fxifxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxixifBwxi = integral_fxifxixi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBw = integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxi = integral_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBw = integral_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBwxi = integral_fxifxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

            for j1 in range(0, n1):
                for l1 in range(0, n1):

                    row = num1*(j1*m1 + i1)
                    col = num1*(l1*m1 + k1)

                    if row > col:
                        continue

                    gAwetaetagBwetaeta = integral_fxixifxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBwetaeta = integral_ffxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBw = integral_ffxixi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBwetaeta = integral_fxifxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetaetagBweta = integral_fxifxixi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBw = integral_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBweta = integral_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBw = integral_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBweta = integral_fxifxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
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

            fAwxifBwxi = integral_fxifxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBwxi = integral_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwxifBw = integral_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)
            fAwfBw = integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

            for j1 in range(0, n1):
                for l1 in range(0, n1):

                    row = num1*(j1*m1 + i1)
                    col = num1*(l1*m1 + k1)

                    if row > col:
                        continue

                    gAwetagBw = integral_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBw = integral_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwgBweta = integral_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)
                    gAwetagBweta = integral_fxifxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry)

                    c += 1
                    kG0r[c] = row+0
                    kG0c[c] = col+0
                    kG0v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

    size = num1*m1*n1

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0

