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

cdef extern from 'bardell_functions.h':
    double calc_vec_f(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_vec_fxi(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil

DOUBLE = np.float64
ctypedef np.double_t cDOUBLE

cdef int nmax = 30
cdef int num = 3


def fuvw(np.ndarray[cDOUBLE, ndim=1] c, object p,
        np.ndarray[cDOUBLE, ndim=1] xs, np.ndarray[cDOUBLE, ndim=1] ys,
        int num_cores=4):
    cdef double a, b
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry
    a = p.a
    b = p.b
    m = p.m
    n = p.n
    u1tx = p.u1tx ; u1rx = p.u1rx ; u2tx = p.u2tx ; u2rx = p.u2rx
    v1tx = p.v1tx ; v1rx = p.v1rx ; v2tx = p.v2tx ; v2rx = p.v2rx
    w1tx = p.w1tx ; w1rx = p.w1rx ; w2tx = p.w2tx ; w2rx = p.w2rx
    u1ty = p.u1ty ; u1ry = p.u1ry ; u2ty = p.u2ty ; u2ry = p.u2ry
    v1ty = p.v1ty ; v1ry = p.v1ry ; v2ty = p.v2ty ; v2ry = p.v2ry
    w1ty = p.w1ty ; w1ry = p.w1ry ; w2ty = p.w2ty ; w2ry = p.w2ry

    cdef int size_core, pti
    cdef np.ndarray[cDOUBLE, ndim=2] us, vs, ws, phixs, phiys
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size == num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores) != 0:
        xs_core = np.ascontiguousarray(np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
    else:                              
        xs_core = np.ascontiguousarray(xs.reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(ys.reshape(num_cores, -1), dtype=DOUBLE)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phiys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for pti in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m, n, a, b, &xs_core[pti,0],
              &ys_core[pti,0], size_core, &us[pti,0], &vs[pti,0], &ws[pti,0],
              u1tx, u1rx, u2tx, u2rx,
              v1tx, v1rx, v2tx, v2rx,
              w1tx, w1rx, w2tx, w2rx,
              u1ty, u1ry, u2ty, u2ry,
              v1ty, v1ry, v2ty, v2ry,
              w1ty, w1ry, w2ty, w2ry)

        cfwx(&c[0], m, n, a, b, &xs_core[pti,0], &ys_core[pti,0],
             size_core, &phixs[pti,0],
             w1tx, w1rx, w2tx, w2rx,
             w1ty, w1ry, w2ty, w2ry)

        cfwy(&c[0], m, n, a, b, &xs_core[pti,0], &ys_core[pti,0],
             size_core, &phiys[pti,0],
             w1tx, w1rx, w2tx, w2rx,
             w1ty, w1ry, w2ty, w2ry)

    phixs *= -1.
    phiys *= -1.
    return (us.ravel()[:size], vs.ravel()[:size], ws.ravel()[:size],
            phixs.ravel()[:size], phiys.ravel()[:size])


cdef void cfuvw(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws,
        double u1tx, double u1rx, double u2tx, double u2rx,
        double v1tx, double v1rx, double v2tx, double v2rx,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double u1ty, double u1ry, double u2ty, double u2ry,
        double v1ty, double v1ry, double v2ty, double v2ry,
        double w1ty, double w1ry, double w2ty, double w2ry) nogil:
    cdef int i, j, col, pti
    cdef double x, y, u, v, w, xi, eta
    cdef double *fu
    cdef double *fv
    cdef double *fw
    cdef double *gu
    cdef double *gv
    cdef double *gw

    fu = <double *>malloc(nmax * sizeof(double *))
    gu = <double *>malloc(nmax * sizeof(double *))
    fv = <double *>malloc(nmax * sizeof(double *))
    gv = <double *>malloc(nmax * sizeof(double *))
    fw = <double *>malloc(nmax * sizeof(double *))
    gw = <double *>malloc(nmax * sizeof(double *))

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        calc_vec_f(fu, xi, u1tx, u1rx, u2tx, u2rx)
        calc_vec_f(gu, eta, u1ty, u1ry, u2ty, u2ry)
        calc_vec_f(fv, xi, v1tx, v1rx, v2tx, v2rx)
        calc_vec_f(gv, eta, v1ty, v1ry, v2ty, v2ry)
        calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)

        u = 0
        v = 0
        w = 0

        for j in range(n):
            for i in range(m):
                col = num*(j*m + i)
                u += c[col+0]*fu[i]*gu[j]
                v += c[col+1]*fv[i]*gv[j]
                w += c[col+2]*fw[i]*gw[j]

        us[pti] = u
        vs[pti] = v
        ws[pti] = w

    free(fu)
    free(gu)
    free(fv)
    free(gv)
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
                wx += (2/a)*c[col+2]*fwxi[i]*gw[j]

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
                wy += (2/b)*c[col+2]*fw[i]*gweta[j]

        wys[pti] = wy

    free(fw)
    free(gweta)


def fg(double[:,::1] g, double x, double y, object p):
    if p.__class__.__name__ != 'Panel':
        raise ValueError('A Panel object must be passed')
    a = p.a
    b = p.b
    m = p.m
    n = p.n
    u1tx = p.u1tx ; u1rx = p.u1rx ; u2tx = p.u2tx ; u2rx = p.u2rx
    v1tx = p.v1tx ; v1rx = p.v1rx ; v2tx = p.v2tx ; v2rx = p.v2rx
    w1tx = p.w1tx ; w1rx = p.w1rx ; w2tx = p.w2tx ; w2rx = p.w2rx
    u1ty = p.u1ty ; u1ry = p.u1ry ; u2ty = p.u2ty ; u2ry = p.u2ry
    v1ty = p.v1ty ; v1ry = p.v1ry ; v2ty = p.v2ty ; v2ry = p.v2ry
    w1ty = p.w1ty ; w1ry = p.w1ry ; w2ty = p.w2ty ; w2ry = p.w2ry
    cfg(g, m, n, x, y, a, b,
        u1tx, u1rx, u2tx, u2rx,
        v1tx, v1rx, v2tx, v2rx,
        w1tx, w1rx, w2tx, w2rx,
        u1ty, u1ry, u2ty, u2ry,
        v1ty, v1ry, v2ty, v2ry,
        w1ty, w1ry, w2ty, w2ry)


cdef void cfg(double[:,::1] g, int m, int n,
              double x, double y, double a, double b,
              double u1tx, double u1rx, double u2tx, double u2rx,
              double v1tx, double v1rx, double v2tx, double v2rx,
              double w1tx, double w1rx, double w2tx, double w2rx,
              double u1ty, double u1ry, double u2ty, double u2ry,
              double v1ty, double v1ry, double v2ty, double v2ry,
              double w1ty, double w1ry, double w2ty, double w2ry) nogil:
    cdef int i, j, col
    cdef double xi, eta
    cdef double *fu
    cdef double *fv
    cdef double *fw
    cdef double *gu
    cdef double *gv
    cdef double *gw

    fu = <double *>malloc(nmax * sizeof(double *))
    gu = <double *>malloc(nmax * sizeof(double *))
    fv = <double *>malloc(nmax * sizeof(double *))
    gv = <double *>malloc(nmax * sizeof(double *))
    fw = <double *>malloc(nmax * sizeof(double *))
    gw = <double *>malloc(nmax * sizeof(double *))

    xi = 2*x/a - 1.
    eta = 2*y/b - 1.

    calc_vec_f(fu, xi, u1tx, u1rx, u2tx, u2rx)
    calc_vec_f(gu, eta, u1ty, u1ry, u2ty, u2ry)
    calc_vec_f(fv, xi, v1tx, v1rx, v2tx, v2rx)
    calc_vec_f(gv, eta, v1ty, v1ry, v2ty, v2ry)
    calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
    calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)

    for j in range(n):
        for i in range(m):
            col = num*(j*m + i)
            g[0, col+0] = fu[i]*gu[j]
            g[1, col+1] = fv[i]*gv[j]
            g[2, col+2] = fw[i]*gw[j]

    free(fu)
    free(gu)
    free(fv)
    free(gv)
    free(fw)
    free(gw)
