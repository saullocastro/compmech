#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange


DOUBLE = np.float64
INT = np.int64
ctypedef np.double_t cDOUBLE
ctypedef np.int64_t cINT


cdef extern from "math.h":
    double cos(double theta) nogil
    double sin(double theta) nogil


cdef int num0 = 0
cdef int num1 = 3
cdef int e_num = 6
cdef double pi=3.141592653589793


def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m1, int n1, double a, double b,
        np.ndarray[cDOUBLE, ndim=1] xs, np.ndarray[cDOUBLE, ndim=1] ys,
        int num_cores=4):
    cdef int size_core, i
    cdef np.ndarray[cDOUBLE, ndim=2] us, vs, ws, phixs, phiys
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ys_core = np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ys_core = ys.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phiys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m1, n1, a, b, &xs_core[i,0],
              &ys_core[i,0], size_core, &us[i,0], &vs[i,0], &ws[i,0])

        cfwx(&c[0], m1, n1, &xs_core[i,0], &ys_core[i,0],
             size_core, a, b, &phixs[i,0])

        cfwy(&c[0], m1, n1, &xs_core[i,0], &ys_core[i,0],
             size_core, a, b, &phiys[i,0])

    phixs *= -1.
    phiys *= -1.
    return (us.ravel()[:size], vs.ravel()[:size], ws.ravel()[:size],
            phixs.ravel()[:size], phiys.ravel()[:size])


cdef void cfuvw(double *c, int m1, int n1, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws) nogil:
    cdef int i1, j1, col, i
    cdef double sini1bx, sinj1by
    cdef double x, y, u, v, w, bx, by

    for i in range(size):
        x = xs[i]
        y = ys[i]

        bx = x/a
        by = y/b

        u = 0
        v = 0
        w = 0

        for j1 in range(1, n1+1):
            sinj1by = sin(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num0 + num1*((j1-1)*m1 + (i1-1))
                sini1bx = sin(i1*pi*bx)
                u += c[col+0]*sini1bx*sinj1by
                v += c[col+1]*sini1bx*sinj1by
                w += c[col+2]*sini1bx*sinj1by

        us[i] = u
        vs[i] = v
        ws[i] = w


cdef void cfwx(double *c, int m1, int n1, double *xs, double *ys, int size,
        double a, double b, double *outwx) nogil:
    cdef double dsini1bx, sinj1by, wx, x, y, bx, by
    cdef int i1, j1, col, i

    for i in range(size):
        x = xs[i]
        y = ys[i]
        bx = x/a
        by = y/b

        wx = 0.

        for j1 in range(1, n1+1):
            sinj1by = sin(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num0 + num1*((j1-1)*m1 + (i1-1))
                dsini1bx = i1*pi/a*cos(i1*pi*bx)
                wx += c[col+2]*dsini1bx*sinj1by

        outwx[i] = wx


cdef void cfwy(double *c, int m1, int n1, double *xs, double *ys, int size,
        double a, double b, double *outwt) nogil:
    cdef double sini1bx, dsinj1by, wy, x, y, bx, by
    cdef int i1, j1, col, i

    for i in range(size):
        x = xs[i]
        y = ys[i]
        bx = x/a
        by = y/b

        wy = 0.

        for j1 in range(1, n1+1):
            dsinj1by = j1*pi/b*cos(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num0 + num1*((j1-1)*m1 + (i1-1))
                sini1bx = sin(i1*pi*bx)
                wy += c[col+2]*sini1bx*dsinj1by

        outwt[i] = wy


def fg(double[:,::1] g, int m1, int n1,
       double x, double y, double a, double b):
    cfg(g, m1, n1, x, y, a, b)


cdef void cfg(double[:,::1] g, int m1, int n1,
              double x, double y, double a, double b) nogil:
    cdef int i1, j1, col, i
    cdef double sini1bx, sinj1by
    cdef double bx, by

    bx = x/a
    by = y/b

    for j1 in range(1, n1+1):
        sinj1by = sin(j1*pi*by)
        for i1 in range(1, m1+1):
            col = num0 + num1*((j1-1)*m1 + (i1-1))
            sini1bx = sin(i1*pi*bx)
            g[0, col+0] = sini1bx*sinj1by
            g[1, col+1] = sini1bx*sinj1by
            g[2, col+2] = sini1bx*sinj1by
