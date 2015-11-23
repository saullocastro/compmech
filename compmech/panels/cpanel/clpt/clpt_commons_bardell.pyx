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

include '../../../func/bardell/bardell.pyx'

DOUBLE = np.float64
ctypedef np.double_t cDOUBLE

cdef int num1 = 3
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

        #cfwx(&c[0], m1, n1, &xs_core[i,0], &ys_core[i,0],
             #size_core, a, b, &phixs[i,0])

        #cfwy(&c[0], m1, n1, &xs_core[i,0], &ys_core[i,0],
             #size_core, a, b, &phiys[i,0])

    phixs *= -1.
    phiys *= -1.
    return (us.ravel()[:size], vs.ravel()[:size], ws.ravel()[:size],
            phixs.ravel()[:size], phiys.ravel()[:size])


cdef void cfuvw(double *c, int m1, int n1, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws) nogil:
    cdef int i1, j1, col, i
    cdef double x, y, u, v, w, xi, eta
    cdef double *fxi
    cdef double *feta

    fxi = <double *>malloc(nmax() * sizeof(double *))
    feta = <double *>malloc(nmax() * sizeof(double *))

    for i in range(size):
        x = xs[i]
        y = ys[i]

        xi = (2*x - a)/a
        eta = (2*y - b)/b

        calc_fxi(fxi, xi)
        calc_fxi(feta, eta)

        u = 0
        v = 0
        w = 0

        for i1 in range(m1):
            for j1 in range(n1):
                col = num1*(j1*m1 + i1)
                u += c[col+0]*fxi[i1]*feta[j1]
                v += c[col+1]*fxi[i1]*feta[j1]
                w += c[col+2]*fxi[i1]*feta[j1]

        us[i] = u
        vs[i] = v
        ws[i] = w

    free(fxi)
    free(feta)


def fg(double[:,::1] g, int m1, int n1,
       double x, double y, double a, double b):
    cfg(g, m1, n1, x, y, a, b)


cdef void cfg(double[:,::1] g, int m1, int n1,
              double x, double y, double a, double b) nogil:
    cdef int i1, j1, col
    cdef double xi, eta
    cdef double *fxi
    cdef double *feta

    fxi = <double *>malloc(nmax() * sizeof(double *))
    feta = <double *>malloc(nmax() * sizeof(double *))

    xi = (2*x - a)/a
    eta = (2*y - b)/b

    calc_fxi(fxi, xi)
    calc_fxi(feta, eta)

    for i1 in range(m1):
        for j1 in range(n1):
            col = num1*(j1*m1 + i1)
            g[0, col+0] = fxi[i1]*feta[j1]
            g[1, col+1] = fxi[i1]*feta[j1]
            g[2, col+2] = fxi[i1]*feta[j1]

    free(fxi)
    free(feta)
