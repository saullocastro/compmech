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
    double calc_vec_fxixi(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil

DOUBLE = np.float64

cdef int nmax = 30
cdef int num = 3


def fuvw(double [:] c, object p, double [:] xs, double [:] ys, int num_cores=4):
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

    cdef int size_core, pti, j
    cdef double [:, ::1] us, vs, ws, phixs, phiys
    cdef double [:, ::1] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size == num_cores:
        add_size = 0
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

    for pti in range(num_cores):
        for j in range(size_core):
            phixs[pti, j] *= -1.
            phiys[pti, j] *= -1.
    return (np.ravel(us)[:size], np.ravel(vs)[:size], np.ravel(ws)[:size],
            np.ravel(phixs)[:size], np.ravel(phiys)[:size])


def fstrain(double [:] c, object p, double [:] xs, double [:] ys, int num_cores=4, int NLterms=0):
    cdef double a, b, r, alpharad
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry
    a = p.a
    b = p.b
    r = p.r
    alpharad = p.alpharad
    m = p.m
    n = p.n
    u1tx = p.u1tx ; u1rx = p.u1rx ; u2tx = p.u2tx ; u2rx = p.u2rx
    v1tx = p.v1tx ; v1rx = p.v1rx ; v2tx = p.v2tx ; v2rx = p.v2rx
    w1tx = p.w1tx ; w1rx = p.w1rx ; w2tx = p.w2tx ; w2rx = p.w2rx
    u1ty = p.u1ty ; u1ry = p.u1ry ; u2ty = p.u2ty ; u2ry = p.u2ry
    v1ty = p.v1ty ; v1ry = p.v1ry ; v2ty = p.v2ty ; v2ry = p.v2ry
    w1ty = p.w1ty ; w1ry = p.w1ry ; w2ty = p.w2ty ; w2ry = p.w2ry

    cdef int size_core, pti
    cdef double [:, ::1] exxs, eyys, gxys, kxxs, kyys, kxys
    cdef double [:, ::1] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size == num_cores:
        add_size = 0
    new_size = size + add_size

    if (size % num_cores) != 0:
        xs_core = np.ascontiguousarray(np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
    else:
        xs_core = np.ascontiguousarray(np.reshape(xs, (num_cores, -1)), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.reshape(ys, (num_cores, -1)), dtype=DOUBLE)

    size_core = xs_core.shape[1]

    exxs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    eyys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    gxys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kxxs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kyys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kxys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    if alpharad != 0:
        raise NotImplementedError('Conical shells not suported')

    for pti in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfstrain(&c[0], m, n, a, b, r, alpharad,
              &xs_core[pti,0], &ys_core[pti,0], size_core,
              &exxs[pti,0], &eyys[pti,0], &gxys[pti,0],
              &kxxs[pti,0], &kyys[pti,0], &kxys[pti,0],
              u1tx, u1rx, u2tx, u2rx,
              v1tx, v1rx, v2tx, v2rx,
              w1tx, w1rx, w2tx, w2rx,
              u1ty, u1ry, u2ty, u2ry,
              v1ty, v1ry, v2ty, v2ry,
              w1ty, w1ry, w2ty, w2ry, NLterms)

    return (np.ravel(exxs)[:size], np.ravel(eyys)[:size], np.ravel(gxys)[:size],
            np.ravel(kxxs)[:size], np.ravel(kyys)[:size], np.ravel(kxys)[:size])


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


cdef void cfstrain(double *c, int m, int n, double a, double b,
        double r, double alpharad,
        double *xs, double *ys, int size,
        double *exxs, double *eyys, double *gxys,
        double *kxxs, double *kyys, double *kxys,
        double u1tx, double u1rx, double u2tx, double u2rx,
        double v1tx, double v1rx, double v2tx, double v2rx,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double u1ty, double u1ry, double u2ty, double u2ry,
        double v1ty, double v1ry, double v2ty, double v2ry,
        double w1ty, double w1ry, double w2ty, double w2ry, int NLterms) nogil:
    cdef int i, j, col, pti
    cdef double x, y, xi, eta
    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef int flagcyl

    cdef double *fu
    cdef double *fuxi
    cdef double *fv
    cdef double *fvxi
    cdef double *fw
    cdef double *fwxi
    cdef double *fwxixi

    cdef double *gu
    cdef double *gueta
    cdef double *gv
    cdef double *gveta
    cdef double *gw
    cdef double *gweta
    cdef double *gwetaeta

    fu = <double *>malloc(nmax * sizeof(double *))
    fuxi = <double *>malloc(nmax * sizeof(double *))
    gu = <double *>malloc(nmax * sizeof(double *))
    gueta = <double *>malloc(nmax * sizeof(double *))
    fv = <double *>malloc(nmax * sizeof(double *))
    fvxi = <double *>malloc(nmax * sizeof(double *))
    gv = <double *>malloc(nmax * sizeof(double *))
    gveta = <double *>malloc(nmax * sizeof(double *))
    fw = <double *>malloc(nmax * sizeof(double *))
    fwxi = <double *>malloc(nmax * sizeof(double *))
    fwxixi = <double *>malloc(nmax * sizeof(double *))
    gw = <double *>malloc(nmax * sizeof(double *))
    gweta = <double *>malloc(nmax * sizeof(double *))
    gwetaeta = <double *>malloc(nmax * sizeof(double *))

    if r == 0:
        flagcyl = 0
    else:
        flagcyl = 1

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        calc_vec_f(fu, xi, u1tx, u1rx, u2tx, u2rx)
        calc_vec_fxi(fuxi, xi, u1tx, u1rx, u2tx, u2rx)
        calc_vec_f(gu, eta, u1ty, u1ry, u2ty, u2ry)
        calc_vec_fxi(gueta, eta, u1ty, u1ry, u2ty, u2ry)
        calc_vec_f(fv, xi, v1tx, v1rx, v2tx, v2rx)
        calc_vec_fxi(fvxi, xi, v1tx, v1rx, v2tx, v2rx)
        calc_vec_f(gv, eta, v1ty, v1ry, v2ty, v2ry)
        calc_vec_fxi(gveta, eta, v1ty, v1ry, v2ty, v2ry)
        calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_fxi(fwxi, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_fxixi(fwxixi, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)
        calc_vec_fxi(gweta, eta, w1ty, w1ry, w2ty, w2ry)
        calc_vec_fxixi(gwetaeta, eta, w1ty, w1ry, w2ty, w2ry)

        exx = 0
        eyy = 0
        gxy = 0
        kxx = 0
        kyy = 0
        kxy = 0

        for j in range(n):
            for i in range(m):
                col = num*(j*m + i)
                exx += c[col+0]*fuxi[i]*gu[j]*(2/a) + NLterms*2/(a*a)*(c[col+2]*fwxi[i]*gw[j])**2
                if flagcyl == 1:
                    eyy += c[col+1]*fv[i]*gveta[j]*(2/b) + 1/r*c[col+2]*fw[i]*gw[j] + NLterms*2/(b*b)*(c[col+2]*fw[i]*gweta[j])**2
                else:
                    eyy += c[col+1]*fv[i]*gveta[j]*(2/b) + NLterms*2/(b*b)*(c[col+2]*fw[i]*gweta[j])**2
                gxy += c[col+0]*fu[i]*gueta[j]*(2/b) + c[col+1]*fvxi[i]*gv[j]*(2/a) + NLterms*4/(a*b)*c[col+2]*fwxi[i]*gw[j]*c[col+2]*fw[i]*gweta[j]
                kxx += -c[col+2]*fwxixi[i]*gw[j]*4/(a*a)
                kyy += -c[col+2]*fw[i]*gwetaeta[j]*4/(b*b)
                kxy += -2*c[col+2]*fwxi[i]*gweta[j]*4/(a*b)

        exxs[pti] = exx
        eyys[pti] = eyy
        gxys[pti] = gxy
        kxxs[pti] = kxx
        kyys[pti] = kyy
        kxys[pti] = kxy

    free(fu)
    free(fuxi)
    free(gu)
    free(gueta)
    free(fv)
    free(fvxi)
    free(gv)
    free(gveta)
    free(fw)
    free(fwxi)
    free(fwxixi)
    free(gw)
    free(gweta)
    free(gwetaeta)
