#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
#cython: embedsignatures=True
import numpy as np
cimport numpy as np

from cython.parallel import prange

DOUBLE = np.float64

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

ctypedef np.double_t cDOUBLE

ctypedef void *cftype(int m, int n, int num,
                      double *xs, double *thetas, double *a) nogil

cdef double pi = 3.141592653589793
cdef int num_threads = 4


def fa(m0, n0, np.ndarray[cDOUBLE, ndim=1] xs,
               np.ndarray[cDOUBLE, ndim=1] ts, funcnum):
    cdef np.ndarray[cDOUBLE, ndim=2] a
    cdef cftype *cf
    '''Creates the coefficients matrix necessary for the least-squares method.

    '''

    num = xs.shape[0]

    if xs.min() < 0. or xs.max() > 1.:
        raise ValueError('The xs array must be normalized!')

    if funcnum==1:
        size = 2
        cf = &cfa01
    elif funcnum==2:
        size = 2
        cf = &cfa02
    elif funcnum==3:
        size = 4
        cf = &cfa03

    a = np.zeros((num, size*n0*m0), DOUBLE)
    cf(m0, n0, num, &xs[0], &ts[0], &a[0, 0])

    return a


cdef void cfw0x(double *xs, double *ts, int size, double *c0, double L,
                int m0, int n0, double *w0xs, int funcnum) nogil:
    cdef double dsinix, dcosix, sinjt, cosjt, w0x, x, t
    cdef int i, j, col, ix

    if funcnum==1:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0x = 0
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(1, m0+1):
                    dsinix = i*pi/L*cos(i*pi*x/L)
                    col = (i-1)*2 + j*m0*2
                    w0x += c0[col+0]*dsinix*sinjt
                    w0x += c0[col+1]*dsinix*cosjt
            w0xs[ix] = w0x
    elif funcnum==2:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0x = 0
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(m0):
                    dcosix = -i*pi/L*sin(i*pi*x/L)
                    col = i*2 + j*m0*2
                    w0x += c0[col+0]*dcosix*sinjt
                    w0x += c0[col+1]*dcosix*cosjt
            w0xs[ix] = w0x
    elif funcnum==3:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0x = 0
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(m0):
                    dsinix = i*pi/L*cos(i*pi*x/L)
                    dcosix = -i*pi/L*sin(i*pi*x/L)
                    col = i*4 + j*m0*4
                    w0x += c0[col+0]*dsinix*sinjt
                    w0x += c0[col+1]*dsinix*cosjt
                    w0x += c0[col+2]*dcosix*sinjt
                    w0x += c0[col+3]*dcosix*cosjt

            w0xs[ix] = w0x


cdef void cfw0t(double *xs, double *ts, int size, double *c0, double L,
                 int m0, int n0, double *w0ts, int funcnum) nogil:
    cdef double sinix, cosix, sinjt, cosjt, w0t, x, t
    cdef int i, j, col, ix

    if funcnum==1:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0t = 0.
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(1, m0+1):
                    sinix = sin(i*pi*x/L)
                    col = (i-1)*2 + j*m0*2
                    w0t += c0[col+0]*sinix*j*cosjt
                    w0t += c0[col+1]*sinix*j*(-sinjt)
            w0ts[ix] = w0t
    elif funcnum==2:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0t = 0.
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(m0):
                    cosix = cos(i*pi*x/L)
                    col = i*2 + j*m0*2
                    w0t += c0[col+0]*cosix*j*cosjt
                    w0t += c0[col+1]*cosix*j*(-sinjt)
            w0ts[ix] = w0t
    elif funcnum==3:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0t = 0.
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(m0):
                    sinix = sin(i*pi*x/L)
                    cosix = cos(i*pi*x/L)
                    col = i*4 + j*m0*4
                    w0t += c0[col+0]*sinix*j*cosjt
                    w0t += c0[col+1]*sinix*j*(-sinjt)
                    w0t += c0[col+2]*cosix*j*cosjt
                    w0t += c0[col+3]*cosix*j*(-sinjt)

            w0ts[ix] = w0t

cdef void *cfa01(int m0, int n0, int num,
                 double *xs, double *ts, double *a) nogil:
    cdef double x, t, sinix, sinjt, cosjt
    cdef int l, i, j, col

    for l in prange(num, chunksize=num/20, num_threads=num_threads,
                    schedule='static'):
        t = ts[l]
        x = xs[l]
        for j in range(n0):
            sinjt = sin(j*t)
            cosjt = cos(j*t)
            for i in range(1, m0+1):
                sinix = sin(i*pi*x)
                col = (i-1)*2 + j*m0*2
                a[l*(2*m0*n0) + (col+0)] = sinix*sinjt
                a[l*(2*m0*n0) + (col+1)] = sinix*cosjt


cdef void *cfa02(int m0, int n0, int num,
                 double *xs, double *ts, double *a) nogil:
    cdef double x, t, cosix, sinjt, cosjt
    cdef int l, i, j, col

    for l in prange(num, chunksize=num/20, num_threads= num_threads,
                    schedule='static'):
        t = ts[l]
        x = xs[l]
        for j in range(n0):
            sinjt = sin(j*t)
            cosjt = cos(j*t)
            for i in range(m0):
                cosix = cos(i*pi*x)
                col = i*2 + j*m0*2
                a[l*(2*m0*n0) + (col+0)] = cosix*sinjt
                a[l*(2*m0*n0) + (col+1)] = cosix*cosjt


cdef void *cfa03(int m0, int n0, int num,
                 double *xs, double *ts, double *a) nogil:
    cdef double x, t, sinix, cosix, sinjt, cosjt
    cdef int l, i, j, col

    for l in prange(num, chunksize=num/20, num_threads=num_threads,
                    schedule='static'):
        t = ts[l]
        x = xs[l]
        for j in range(n0):
            sinjt = sin(j*t)
            cosjt = cos(j*t)
            for i in range(m0):
                sinix = sin(i*pi*x)
                cosix = cos(i*pi*x)
                col = i*4 + j*m0*4
                a[l*(4*m0*n0) + (col+0)] = sinix*sinjt
                a[l*(4*m0*n0) + (col+1)] = sinix*cosjt
                a[l*(4*m0*n0) + (col+2)] = cosix*sinjt
                a[l*(4*m0*n0) + (col+3)] = cosix*cosjt


def fw0(int m0, int n0,
        np.ndarray[cDOUBLE, ndim=1] c0,
        np.ndarray[cDOUBLE, ndim=1] xs,
        np.ndarray[cDOUBLE, ndim=1] ts, funcnum):
    cdef int ix, i, j, col, size
    cdef double x, t, sinix, cosix, sinjt, cosjt, w0
    cdef np.ndarray[cDOUBLE, ndim=1] w0s
    w0s = np.zeros_like(xs)
    size = np.shape(xs)[0]

    if xs.min() < 0. or xs.max() > 1.:
        raise ValueError('The xs array must be normalized!')

    if funcnum==1:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0 = 0
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(1, m0+1):
                    sinix = sin(i*pi*x)
                    col = (i-1)*2 + j*m0*2
                    w0 += c0[col+0]*sinix*sinjt
                    w0 += c0[col+1]*sinix*cosjt
            w0s[ix] = w0
    elif funcnum==2:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0 = 0
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(m0):
                    cosix = cos(i*pi*x)
                    col = i*2 + j*m0*2
                    w0 += c0[col+0]*cosix*sinjt
                    w0 += c0[col+1]*cosix*cosjt
            w0s[ix] = w0
    elif funcnum==3:
        for ix in range(size):
            x = xs[ix]
            t = ts[ix]
            w0 = 0
            for j in range(n0):
                sinjt = sin(j*t)
                cosjt = cos(j*t)
                for i in range(m0):
                    sinix = sin(i*pi*x)
                    cosix = cos(i*pi*x)
                    col = i*4 + j*m0*4
                    w0 += c0[col+0]*sinix*sinjt
                    w0 += c0[col+1]*sinix*cosjt
                    w0 += c0[col+2]*cosix*sinjt
                    w0 += c0[col+3]*cosix*cosjt
            w0s[ix] = w0

    return w0s
