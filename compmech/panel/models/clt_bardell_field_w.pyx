#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange

cdef extern from 'bardell_functions.h':
    double calc_vec_f(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_vec_fxi(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil

DOUBLE = np.float64

cdef int nmax = 30
cdef int num = 1


def fuvw(double [:] c, object p, double [:] xs, double [:] ys, int num_cores=4):
    cdef double a, b
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry
    a = p.a
    b = p.b
    m = p.m
    n = p.n
    w1tx = p.w1tx ; w1rx = p.w1rx ; w2tx = p.w2tx ; w2rx = p.w2rx
    w1ty = p.w1ty ; w1ry = p.w1ry ; w2ty = p.w2ty ; w2ry = p.w2ry
    cdef int size_core, pti, j
    cdef double [:, ::1] us, vs, ws, phixs, phiys
    cdef double [:, ::1] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size == num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores) != 0:
        xs_core = np.ascontiguousarray(np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
    else:                              
        xs_core = np.ascontiguousarray(np.reshape(xs, (num_cores, -1)), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.reshape(ys, (num_cores, -1)), dtype=DOUBLE)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phiys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for pti in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfw(&c[0], m, n, a, b, &xs_core[pti,0],
              &ys_core[pti,0], size_core, &us[pti,0], &vs[pti,0], &ws[pti,0],
              w1tx, w1rx, w2tx, w2rx,
              w1ty, w1ry, w2ty, w2ry)

        cfwx(&c[0], m, n, a, b, &xs_core[pti,0], &ys_core[pti,0],
             size_core, &phixs[pti,0],
             w1tx, w1rx, w2tx, w2rx,
             w1ty, w1ry, w2ty, w2ry)

        cfwy(&c[0], m, n, a, b, &xs_core[pti,0], &ys_core[pti,0],
             size_core, &phiys[pti,0],
             w1tx, w1rx, w2tx, w2rx,
             w1ty, w1ry, w2ty, w2ry)

    for pti in range(num_cores):
        for j in range(size_core):
            phixs[pti, j] *= -1.
            phiys[pti, j] *= -1.
    return (np.ravel(us)[:size], np.ravel(vs)[:size], np.ravel(ws)[:size],
            np.ravel(phixs)[:size], np.ravel(phiys)[:size])


cdef void cfw(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double w1ty, double w1ry, double w2ty, double w2ry) nogil:
    cdef int i, j, col, pti
    cdef double x, y, u, v, w, xi, eta
    cdef double *fw
    cdef double *gw

    fw = <double *>malloc(nmax * sizeof(double *))
    gw = <double *>malloc(nmax * sizeof(double *))

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)

        w = 0

        for j in range(n):
            for i in range(m):
                col = num*(j*m + i)
                w += c[col+0]*fw[i]*gw[j]

        ws[pti] = w

    free(fw)
    free(gw)


cdef void cfwx(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *wxs,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double w1ty, double w1ry, double w2ty, double w2ry) nogil: 
    cdef int i, j, col, pti
    cdef double x, y, wx, xi, eta
    cdef double *fwxi
    cdef double *gw

    fwxi = <double *>malloc(nmax * sizeof(double *))
    gw = <double *>malloc(nmax * sizeof(double *))

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        calc_vec_fxi(fwxi, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)

        wx = 0

        for j in range(n):
            for i in range(m):
                col = num*(j*m + i)
                wx += (2/a)*c[col+0]*fwxi[i]*gw[j]

        wxs[pti] = wx

    free(fwxi)
    free(gw)


cdef void cfwy(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *wys,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double w1ty, double w1ry, double w2ty, double w2ry) nogil: 
    cdef int i, j, col, pti
    cdef double x, y, wy, xi, eta
    cdef double *fw
    cdef double *gweta

    fw = <double *>malloc(nmax * sizeof(double *))
    gweta = <double *>malloc(nmax * sizeof(double *))

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_fxi(gweta, eta, w1ty, w1ry, w2ty, w2ry)

        wy = 0

        for j in range(n):
            for i in range(m):
                col = num*(j*m + i)
                wy += (2/b)*c[col+0]*fw[i]*gweta[j]

        wys[pti] = wy

    free(fw)
    free(gweta)


def fg(double[:,::1] g, double x, double y, object p):
    a = p.a
    b = p.b
    m = p.m
    n = p.n
    w1tx = p.w1tx ; w1rx = p.w1rx ; w2tx = p.w2tx ; w2rx = p.w2rx
    w1ty = p.w1ty ; w1ry = p.w1ry ; w2ty = p.w2ty ; w2ry = p.w2ry
    cfg(g, m, n, x, y, a, b,
        w1tx, w1rx, w2tx, w2rx,
        w1ty, w1ry, w2ty, w2ry)


cdef void cfg(double[:,::1] g, int m, int n,
              double x, double y, double a, double b,
              double w1tx, double w1rx, double w2tx, double w2rx,
              double w1ty, double w1ry, double w2ty, double w2ry) nogil:
    cdef int i, j, col
    cdef double xi, eta
    cdef double *fw
    cdef double *gw

    fw = <double *>malloc(nmax * sizeof(double *))
    gw = <double *>malloc(nmax * sizeof(double *))

    xi = 2*x/a - 1.
    eta = 2*y/b - 1.

    calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
    calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)

    for j in range(n):
        for i in range(m):
            col = num*(j*m + i)
            g[0, col+0] = fw[i]*gw[j]

    free(fw)
    free(gw)
