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


cdef int num = 3
cdef int num1 = 4
cdef int e_num = 6
cdef double pi=3.141592653589793


def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m, int n, double a, double b,
        np.ndarray[cDOUBLE, ndim=1] xs, np.ndarray[cDOUBLE, ndim=1] ys,
        int num_cores=4, skin=True):
    cdef int size_core, ii
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

    if skin:
        for ii in prange(num_cores, nogil=True, chunksize=1,
                         num_threads=num_cores, schedule='static'):
            cfuvw_skin(&c[0], m, n, a, b, &xs_core[ii,0],
                  &ys_core[ii,0], size_core, &us[ii,0], &vs[ii,0], &ws[ii,0])

            cfwx_skin(&c[0], m, n, &xs_core[ii,0], &ys_core[ii,0],
                 size_core, a, b, &phixs[ii,0])

            cfwy_skin(&c[0], m, n, &xs_core[ii,0], &ys_core[ii,0],
                 size_core, a, b, &phiys[ii,0])
    else:
        for ii in prange(num_cores, nogil=True, chunksize=1,
                         num_threads=num_cores, schedule='static'):
            cfuvw_stiffener(&c[0], m, n, a, b, &xs_core[ii,0],
                  &ys_core[ii,0], size_core, &us[ii,0], &vs[ii,0], &ws[ii,0])

            cfwx_stiffener(&c[0], m, n, &xs_core[ii,0], &ys_core[ii,0],
                 size_core, a, b, &phixs[ii,0])

            cfwy_stiffener(&c[0], m, n, &xs_core[ii,0], &ys_core[ii,0],
                 size_core, a, b, &phiys[ii,0])

    phixs *= -1.
    phiys *= -1.
    return (us.ravel()[:size], vs.ravel()[:size], ws.ravel()[:size],
            phixs.ravel()[:size], phiys.ravel()[:size])


cdef void cfuvw_skin(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws) nogil:
    cdef int i, j, col, ii
    cdef double sinibx, sinjby
    cdef double x, y, u, v, w, bx, by

    for ii in range(size):
        x = xs[ii]
        y = ys[ii]

        bx = x/a
        by = y/b

        u = 0
        v = 0
        w = 0

        for j in range(1, n+1):
            sinjby = sin(j*pi*by)
            for i in range(1, m+1):
                col = num*((j-1)*m + (i-1))
                sinibx = sin(i*pi*bx)
                u += c[col+0]*sinibx*sinjby
                v += c[col+1]*sinibx*sinjby
                w += c[col+2]*sinibx*sinjby

        us[ii] = u
        vs[ii] = v
        ws[ii] = w


cdef void cfwx_skin(double *c, int m, int n, double *xs, double *ys, int size,
        double a, double b, double *outwx) nogil:
    cdef double dsinibx, sinjby, wx, x, y, bx, by
    cdef int i, j, col, ii

    for ii in range(size):
        x = xs[ii]
        y = ys[ii]
        bx = x/a
        by = y/b

        wx = 0.

        for j in range(1, n+1):
            sinjby = sin(j*pi*by)
            for i in range(1, m+1):
                col = num*((j-1)*m + (i-1))
                dsinibx = i*pi/a*cos(i*pi*bx)
                wx += c[col+2]*dsinibx*sinjby

        outwx[ii] = wx


cdef void cfwy_skin(double *c, int m, int n, double *xs, double *ys, int size,
        double a, double b, double *outwt) nogil:
    cdef double sinibx, dsinjby, wy, x, y, bx, by
    cdef int i, j, col, ii

    for ii in range(size):
        x = xs[ii]
        y = ys[ii]
        bx = x/a
        by = y/b

        wy = 0.

        for j in range(1, n+1):
            dsinjby = j*pi/b*cos(j*pi*by)
            for i in range(1, m+1):
                col = num*((j-1)*m + (i-1))
                sinibx = sin(i*pi*bx)
                wy += c[col+2]*sinibx*dsinjby

        outwt[ii] = wy


cdef void cfuvw_stiffener(double *c, int m1, int n1, double a, double bf,
                          double *xs, double *ys, int size, double *us,
                          double *vs, double *ws) nogil:
    cdef int i1, j1, col, ii
    cdef double sini1bx, sinj1by, cosi1bx, cosj1by
    cdef double xf, yf, u, v, w, bx, by

    for ii in range(size):
        xf = xs[ii]
        yf = ys[ii]

        bx = xf/a
        by = yf/bf

        u = 0
        v = 0
        w = 0

        for j1 in range(1, n1+1):
            sinj1by = sin(j1*pi*by)
            cosj1by = cos(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num1*((j1-1)*m1 + (i1-1))
                sini1bx = sin(i1*pi*bx)
                cosi1bx = cos(i1*pi*bx)
                u += c[col+0]*sini1bx*sinj1by
                v += c[col+1]*sini1bx*sinj1by
                w += c[col+2]*sini1bx*sinj1by
                w += c[col+3]*cosi1bx*cosj1by

        us[ii] = u
        vs[ii] = v
        ws[ii] = w


cdef void cfwx_stiffener(double *c, int m1, int n1, double *xs, double *ys,
                         int size, double a, double bf, double *outwx) nogil:
    cdef double dsini1bx, sinj1by, dcosi1bx, cosj1by, wx, xf, yf, bx, by
    cdef int i1, j1, col, ii

    for ii in range(size):
        xf = xs[ii]
        yf = ys[ii]
        bx = xf/a
        by = yf/bf

        wx = 0.

        for j1 in range(1, n1+1):
            sinj1by = sin(j1*pi*by)
            cosj1by = cos(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num1*((j1-1)*m1 + (i1-1))
                dsini1bx = i1*pi/a*cos(i1*pi*bx)
                dcosi1bx = -i1*pi/a*sin(i1*pi*bx)
                wx += c[col+2]*dsini1bx*sinj1by
                wx += c[col+3]*dcosi1bx*cosj1by

        outwx[ii] = wx


cdef void cfwy_stiffener(double *c, int m1, int n1, double *xs, double *ys,
                         int size, double a, double bf, double *outwt) nogil:
    cdef double sini1bx, dsinj1by, cosi1bx, dcosj1by, wy, xf, yf, bx, by
    cdef int i1, j1, col, ii

    for ii in range(size):
        xf = xs[ii]
        yf = ys[ii]
        bx = xf/a
        by = yf/bf

        wy = 0.

        for j1 in range(1, n1+1):
            dsinj1by = j1*pi/bf*cos(j1*pi*by)
            dcosj1by = -j1*pi/bf*sin(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num*((j1-1)*m1 + (i1-1))
                sini1bx = sin(i1*pi*bx)
                cosi1bx = cos(i1*pi*bx)
                wy += c[col+2]*sini1bx*dsinj1by
                wy += c[col+3]*cosi1bx*dcosj1by

        outwt[ii] = wy
