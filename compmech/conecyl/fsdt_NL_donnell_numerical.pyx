#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

cimport numpy as np
import numpy as np
from scipy.sparse import coo_matrix
from cython.parallel import prange
from libc.stdlib cimport malloc, free

from compmech.conecyl.fsdt_commons cimport cfphix, cfphit, cfN

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

cdef int num0 = 3
cdef int num1 = 7
cdef int num2 = 14
cdef double pi = 3.141592653589793

ctypedef void (*f_type) (int npts, double *xs, double *ts, double *out,
                         double *alphas, double *betas, void *args) nogil

cdef struct cc_attributes:
    double *sina
    double *cosa
    double *tLA
    double *r2
    double *L
    double *F
    int *m1
    int *m2
    int *n2
    double *coeffs

cdef int trapz2d(f_type f, int fdim, np.ndarray[cDOUBLE, ndim=1] out,
                 double xmin, double xmax, int nx,
                 double ymin, double ymax, int ny,
                 void *args, int num_cores):
    cdef int i, j, npts, k, rest
    cdef double c, hx, hy, x, y, alpha, beta
    cdef np.ndarray[cDOUBLE, ndim=1] xs, ys, xs2, ts2, alphas, betas
    cdef np.ndarray[cDOUBLE, ndim=2] outs
    outs = np.zeros((num_cores, out.shape[0]), DOUBLE)
    nx -= 1
    ny -= 1

    xs = np.linspace(xmin, xmax, nx+1).astype(DOUBLE)
    ys = np.linspace(ymin, ymax, ny+1).astype(DOUBLE)

    npts = (nx+1)*(ny+1)
    xs2 = np.zeros(npts, DOUBLE)
    ts2 = np.zeros(npts, DOUBLE)
    alphas = np.zeros(npts, DOUBLE)
    betas = np.zeros(npts, DOUBLE)

    hx = (xmax-xmin)/nx
    hy = (ymax-ymin)/ny
    c = 1/4.*hx*hy

    # building integration points
    k = -1
    for i,j in ((0, 0), (nx, 0), (0, ny), (nx, ny)):
        k += 1
        xs2[k] = xs[i]
        ts2[k] = ys[j]
        alphas[k] = 1*c
        betas[k] = 1
    for i in range(1, nx): # i from 1 to nx-1
        for j in (0, ny):
            k += 1
            xs2[k] = xs[i]
            ts2[k] = ys[j]
            alphas[k] = 2*c
            betas[k] = 1
    for i in (0, nx):
        for j in range(1, ny): # j from 1 to ny-1
            k += 1
            xs2[k] = xs[i]
            ts2[k] = ys[j]
            alphas[k] = 2*c
            betas[k] = 1
    for i in range(1, nx): # i from 1 to nx-1
        for j in range(1, ny): # j from 1 to ny-1
            k += 1
            xs2[k] = xs[i]
            ts2[k] = ys[j]
            alphas[k] = 4*c
            betas[k] = 1

    k = npts/num_cores
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
            schedule='static'):
        f(k, &xs2[k*i], &ts2[k*i], &outs[i, 0], &alphas[k*i], &betas[k*i],
             args=<void *>args)

    rest = npts - k*num_cores
    assert rest >= 0, 'ERROR rest < 0!'
    if rest>0:
        f(rest, &xs2[k*num_cores], &ts2[k*num_cores], &outs[0, 0],
                &alphas[k*num_cores], &betas[k*num_cores], args=<void *>args)

    np.sum(outs, axis=0, out=out)

    return 0

cdef int simps2d(f_type f, int fdim, np.ndarray[cDOUBLE, ndim=1] out,
                 double xmin, double xmax, int nx,
                 double ymin, double ymax, int ny,
                 void *args, int num_cores):
    cdef int i, j, npts, k, rest
    cdef double c, hx, hy, x, y, alpha, beta
    cdef np.ndarray[cDOUBLE, ndim=1] xs, ys, xs2, ts2, alphas, betas
    cdef np.ndarray[cDOUBLE, ndim=2] outs
    outs = np.zeros((num_cores, out.shape[0]), DOUBLE)

    if nx % 2 != 0:
        nx += 1
    if ny % 2 != 0:
        ny += 1

    nx /= 2
    ny /= 2

    xs = np.linspace(xmin, xmax, (2*nx+1)).astype(DOUBLE)
    ys = np.linspace(ymin, ymax, (2*ny+1)).astype(DOUBLE)

    npts = (2*nx + 1)*(2*ny + 1)
    xs2 = np.zeros(npts, DOUBLE)
    ts2 = np.zeros(npts, DOUBLE)
    alphas = np.zeros(npts, DOUBLE)
    betas = np.zeros(npts, DOUBLE)

    hx = (xmax-xmin)/(2*nx)
    hy = (ymax-ymin)/(2*ny)
    c = 1/9.*hx*hy

    # building integration points
    k = -1
    for i,j in ((0, 0), (2*nx, 0), (0, 2*ny), (2*nx, 2*ny)):
        k += 1
        xs2[k] = xs[i]
        ts2[k] = ys[j]
        alphas[k] = 1*c
        betas[k] = 1
    for i in (0, 2*nx):
        for j in range(1, ny+1): # from 1 to ny
            k += 1
            xs2[k] = xs[i]
            ts2[k] = ys[2*j-1]
            alphas[k] = 4*c
            betas[k] = 1
    for i in (0, 2*nx):
        for j in range(1, ny): # from 1 to ny-1
            k += 1
            xs2[k] = xs[i]
            ts2[k] = ys[2*j]
            alphas[k] = 2*c
            betas[k] = 1
    for i in range(1, nx+1): # from 1 to nx
        for j in (0, 2*ny):
            k += 1
            xs2[k] = xs[2*i-1]
            ts2[k] = ys[j]
            alphas[k] = 4*c
            betas[k] = 1
    for i in range(1, nx): # from 1 to nx-1
        for j in (0, 2*ny):
            k += 1
            xs2[k] = xs[2*i]
            ts2[k] = ys[j]
            alphas[k] = 2*c
            betas[k] = 1
    for i in range(1, nx+1): # from 1 to nx
        for j in range(1, ny+1): # from 1 to ny
            k += 1
            xs2[k] = xs[2*i-1]
            ts2[k] = ys[2*j-1]
            alphas[k] = 16*c
            betas[k] = 1
    for i in range(1, nx+1):
        for j in range(1, ny):
            k += 1
            xs2[k] = xs[2*i-1]
            ts2[k] = ys[2*j]
            alphas[k] = 8*c
            betas[k] = 1
    for i in range(1, nx):
        for j in range(1, ny+1):
            k += 1
            xs2[k] = xs[2*i]
            ts2[k] = ys[2*j-1]
            alphas[k] = 8*c
            betas[k] = 1
    for i in range(1, nx):
        for j in range(1, ny):
            k += 1
            xs2[k] = xs[2*i]
            ts2[k] = ys[2*j]
            alphas[k] = 4*c
            betas[k] = 1

    k = npts/num_cores
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
            schedule='static'):
        f(k, &xs2[k*i], &ts2[k*i], &outs[i, 0], &alphas[k*i], &betas[k*i],
             args=<void *>args)

    rest = npts - k*num_cores
    assert rest >= 0, 'ERROR rest < 0!'
    if rest>0:
        f(rest, &xs2[k*num_cores], &ts2[k*num_cores], &outs[0, 0],
                &alphas[k*num_cores], &betas[k*num_cores], args=<void *>args)

    np.sum(outs, axis=0, out=out)

    return 0

def calc_k0L(np.ndarray[cDOUBLE, ndim=1] coeffs,
             double alpharad, double r2, double L, double tLA,
             np.ndarray[cDOUBLE, ndim=2] F,
             int m1, int m2, int n2,
             int nx, int nt, int num_cores, str method='trapz2d'):

    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] k0Lv

    cdef int fdim
    cdef cc_attributes args

    fdim = 9*m1 + 18*m2*n2 + 21*m1**2 + 2*42*m1*m2*n2 + 84*m2**2*n2**2

    k0Lv = np.zeros((fdim,), dtype=DOUBLE)
    rows = np.zeros((fdim,), dtype=INT)
    cols = np.zeros((fdim,), dtype=INT)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    args.sina = &sina
    args.cosa = &cosa
    args.tLA = &tLA
    args.r2 = &r2
    args.L = &L
    args.F = &F[0,0]
    args.m1 = &m1
    args.m2 = &m2
    args.n2 = &n2
    args.coeffs = &coeffs[0]

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    if method=='trapz2d':
        trapz2d(<f_type>cfk0L, fdim, k0Lv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<f_type>cfk0L, fdim, k0Lv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    c = -1

    for k1 in range(1, m1+1):
        col = (k1-1)*num1 + num0

        # k0L_01
        c += 1
        rows[c] = 0
        cols[c] = col+4
        c += 1
        rows[c] = 0
        cols[c] = col+5
        c += 1
        rows[c] = 0
        cols[c] = col+6
        c += 1
        rows[c] = 1
        cols[c] = col+4
        c += 1
        rows[c] = 1
        cols[c] = col+5
        c += 1
        rows[c] = 1
        cols[c] = col+6
        c += 1
        rows[c] = 2
        cols[c] = col+4
        c += 1
        rows[c] = 2
        cols[c] = col+5
        c += 1
        rows[c] = 2
        cols[c] = col+6

    for k2 in range(1, m2+1):
        for l2 in range(1, n2+1):
            col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1

            # k0L_02
            c += 1
            rows[c] = 0
            cols[c] = col+8
            c += 1
            rows[c] = 0
            cols[c] = col+9
            c += 1
            rows[c] = 0
            cols[c] = col+10
            c += 1
            rows[c] = 0
            cols[c] = col+11
            c += 1
            rows[c] = 0
            cols[c] = col+12
            c += 1
            rows[c] = 0
            cols[c] = col+13
            c += 1
            rows[c] = 1
            cols[c] = col+8
            c += 1
            rows[c] = 1
            cols[c] = col+9
            c += 1
            rows[c] = 1
            cols[c] = col+10
            c += 1
            rows[c] = 1
            cols[c] = col+11
            c += 1
            rows[c] = 1
            cols[c] = col+12
            c += 1
            rows[c] = 1
            cols[c] = col+13
            c += 1
            rows[c] = 2
            cols[c] = col+8
            c += 1
            rows[c] = 2
            cols[c] = col+9
            c += 1
            rows[c] = 2
            cols[c] = col+10
            c += 1
            rows[c] = 2
            cols[c] = col+11
            c += 1
            rows[c] = 2
            cols[c] = col+12
            c += 1
            rows[c] = 2
            cols[c] = col+13

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        for k1 in range(1, m1+1):
            col = (k1-1)*num1 + num0

            # k0L_11
            c += 1
            rows[c] = row+0
            cols[c] = col+4
            c += 1
            rows[c] = row+0
            cols[c] = col+5
            c += 1
            rows[c] = row+0
            cols[c] = col+6
            c += 1
            rows[c] = row+1
            cols[c] = col+4
            c += 1
            rows[c] = row+1
            cols[c] = col+5
            c += 1
            rows[c] = row+1
            cols[c] = col+6
            c += 1
            rows[c] = row+2
            cols[c] = col+4
            c += 1
            rows[c] = row+2
            cols[c] = col+5
            c += 1
            rows[c] = row+2
            cols[c] = col+6
            c += 1
            rows[c] = row+3
            cols[c] = col+4
            c += 1
            rows[c] = row+3
            cols[c] = col+5
            c += 1
            rows[c] = row+3
            cols[c] = col+6
            c += 1
            rows[c] = row+4
            cols[c] = col+4
            c += 1
            rows[c] = row+4
            cols[c] = col+5
            c += 1
            rows[c] = row+4
            cols[c] = col+6
            c += 1
            rows[c] = row+5
            cols[c] = col+4
            c += 1
            rows[c] = row+5
            cols[c] = col+5
            c += 1
            rows[c] = row+5
            cols[c] = col+6
            c += 1
            rows[c] = row+6
            cols[c] = col+4
            c += 1
            rows[c] = row+6
            cols[c] = col+5
            c += 1
            rows[c] = row+6
            cols[c] = col+6

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1

                # k0L_12
                c += 1
                rows[c] = row+0
                cols[c] = col+8
                c += 1
                rows[c] = row+0
                cols[c] = col+9
                c += 1
                rows[c] = row+0
                cols[c] = col+10
                c += 1
                rows[c] = row+0
                cols[c] = col+11
                c += 1
                rows[c] = row+0
                cols[c] = col+12
                c += 1
                rows[c] = row+0
                cols[c] = col+13
                c += 1
                rows[c] = row+1
                cols[c] = col+8
                c += 1
                rows[c] = row+1
                cols[c] = col+9
                c += 1
                rows[c] = row+1
                cols[c] = col+10
                c += 1
                rows[c] = row+1
                cols[c] = col+11
                c += 1
                rows[c] = row+1
                cols[c] = col+12
                c += 1
                rows[c] = row+1
                cols[c] = col+13
                c += 1
                rows[c] = row+2
                cols[c] = col+8
                c += 1
                rows[c] = row+2
                cols[c] = col+9
                c += 1
                rows[c] = row+2
                cols[c] = col+10
                c += 1
                rows[c] = row+2
                cols[c] = col+11
                c += 1
                rows[c] = row+2
                cols[c] = col+12
                c += 1
                rows[c] = row+2
                cols[c] = col+13
                c += 1
                rows[c] = row+3
                cols[c] = col+8
                c += 1
                rows[c] = row+3
                cols[c] = col+9
                c += 1
                rows[c] = row+3
                cols[c] = col+10
                c += 1
                rows[c] = row+3
                cols[c] = col+11
                c += 1
                rows[c] = row+3
                cols[c] = col+12
                c += 1
                rows[c] = row+3
                cols[c] = col+13
                c += 1
                rows[c] = row+4
                cols[c] = col+8
                c += 1
                rows[c] = row+4
                cols[c] = col+9
                c += 1
                rows[c] = row+4
                cols[c] = col+10
                c += 1
                rows[c] = row+4
                cols[c] = col+11
                c += 1
                rows[c] = row+4
                cols[c] = col+12
                c += 1
                rows[c] = row+4
                cols[c] = col+13
                c += 1
                rows[c] = row+5
                cols[c] = col+8
                c += 1
                rows[c] = row+5
                cols[c] = col+9
                c += 1
                rows[c] = row+5
                cols[c] = col+10
                c += 1
                rows[c] = row+5
                cols[c] = col+11
                c += 1
                rows[c] = row+5
                cols[c] = col+12
                c += 1
                rows[c] = row+5
                cols[c] = col+13
                c += 1
                rows[c] = row+6
                cols[c] = col+8
                c += 1
                rows[c] = row+6
                cols[c] = col+9
                c += 1
                rows[c] = row+6
                cols[c] = col+10
                c += 1
                rows[c] = row+6
                cols[c] = col+11
                c += 1
                rows[c] = row+6
                cols[c] = col+12
                c += 1
                rows[c] = row+6
                cols[c] = col+13

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            for k1 in range(1, m1+1):
                col = (k1-1)*num1 + num0

                # k0L_21
                c += 1
                rows[c] = row+0
                cols[c] = col+4
                c += 1
                rows[c] = row+0
                cols[c] = col+5
                c += 1
                rows[c] = row+0
                cols[c] = col+6
                c += 1
                rows[c] = row+1
                cols[c] = col+4
                c += 1
                rows[c] = row+1
                cols[c] = col+5
                c += 1
                rows[c] = row+1
                cols[c] = col+6
                c += 1
                rows[c] = row+2
                cols[c] = col+4
                c += 1
                rows[c] = row+2
                cols[c] = col+5
                c += 1
                rows[c] = row+2
                cols[c] = col+6
                c += 1
                rows[c] = row+3
                cols[c] = col+4
                c += 1
                rows[c] = row+3
                cols[c] = col+5
                c += 1
                rows[c] = row+3
                cols[c] = col+6
                c += 1
                rows[c] = row+4
                cols[c] = col+4
                c += 1
                rows[c] = row+4
                cols[c] = col+5
                c += 1
                rows[c] = row+4
                cols[c] = col+6
                c += 1
                rows[c] = row+5
                cols[c] = col+4
                c += 1
                rows[c] = row+5
                cols[c] = col+5
                c += 1
                rows[c] = row+5
                cols[c] = col+6
                c += 1
                rows[c] = row+6
                cols[c] = col+4
                c += 1
                rows[c] = row+6
                cols[c] = col+5
                c += 1
                rows[c] = row+6
                cols[c] = col+6
                c += 1
                rows[c] = row+7
                cols[c] = col+4
                c += 1
                rows[c] = row+7
                cols[c] = col+5
                c += 1
                rows[c] = row+7
                cols[c] = col+6
                c += 1
                rows[c] = row+8
                cols[c] = col+4
                c += 1
                rows[c] = row+8
                cols[c] = col+5
                c += 1
                rows[c] = row+8
                cols[c] = col+6
                c += 1
                rows[c] = row+9
                cols[c] = col+4
                c += 1
                rows[c] = row+9
                cols[c] = col+5
                c += 1
                rows[c] = row+9
                cols[c] = col+6
                c += 1
                rows[c] = row+10
                cols[c] = col+4
                c += 1
                rows[c] = row+10
                cols[c] = col+5
                c += 1
                rows[c] = row+10
                cols[c] = col+6
                c += 1
                rows[c] = row+11
                cols[c] = col+4
                c += 1
                rows[c] = row+11
                cols[c] = col+5
                c += 1
                rows[c] = row+11
                cols[c] = col+6
                c += 1
                rows[c] = row+12
                cols[c] = col+4
                c += 1
                rows[c] = row+12
                cols[c] = col+5
                c += 1
                rows[c] = row+12
                cols[c] = col+6
                c += 1
                rows[c] = row+13
                cols[c] = col+4
                c += 1
                rows[c] = row+13
                cols[c] = col+5
                c += 1
                rows[c] = row+13
                cols[c] = col+6

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1

                    # k0L_22
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+6
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+6
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+6
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+6
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+6
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+6
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+7
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+7
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+7
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+7
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+7
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+7
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+13

    size = num0 + num1*m1 + num2*m2*n2

    k0L = coo_matrix((k0Lv, (rows, cols)), shape=(size, size))

    return k0L

cdef void cfk0L(int npts, double *xs, double *ts, double *out,
                double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2
    cdef int c, i, pos

    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66

    cdef double p0000, p0001, p0002, p0100, p0101, p0102
    cdef double p0200, p0201, p0202, p0300, p0301, p0302
    cdef double p0400, p0401, p0402, p0500, p0501, p0502
    cdef double p0600, p0601, p0602
    cdef double p0700, p0701, p0702, p0800, p0801, p0802,
    cdef double p0900, p0901, p0902, p1000, p1001, p1002,
    cdef double p1100, p1101, p1102, p1200, p1201, p1202
    cdef double p1300, p1301, p1302

    cdef double q0004, q0005, q0106, q0204, q0205, q0206
    cdef double q0008, q0009, q0010, q0011, q0112, q0113
    cdef double q0208, q0209, q0210, q0211, q0212, q0213

    cdef double r, x, t, alpha, beta

    cdef double *F, *coeffs
    cdef double  sina, cosa, tLA, r2, L
    cdef int m1, m2, n2, pti
    cdef double phix, phit

    cdef cc_attributes *args_in=<cc_attributes *>args

    sina = args_in.sina[0]
    cosa = args_in.cosa[0]
    tLA = args_in.tLA[0]
    r2 = args_in.r2[0]
    L = args_in.L[0]
    F = args_in.F
    m1 = args_in.m1[0]
    m2 = args_in.m2[0]
    n2 = args_in.n2[0]
    coeffs = args_in.coeffs

    A11 = F[0]
    A12 = F[1]
    A16 = F[2]
    A22 = F[9]
    A26 = F[10]
    A66 = F[18]
    B11 = F[3]
    B12 = F[4]
    B16 = F[5]
    B22 = F[12]
    B26 = F[13]
    B66 = F[21]

    cdef double sini1x, cosi1x, sink1x, cosk1x
    cdef double sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t

    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))

    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))

    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))

    cdef double *k0Lq_1_q0004 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q0005 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q0106 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q0204 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q0205 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q0206 = <double *>malloc(m1 * sizeof(double))

    cdef double *k0Lq_2_q0008 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0009 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0010 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0011 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0112 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0113 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0208 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0209 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0210 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0211 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0212 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0213 = <double *>malloc(m2*n2 * sizeof(double))

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(1, m1+1):
            vsini1x[i1-1] = sin(pi*i1*x/L)
            vcosi1x[i1-1] = cos(pi*i1*x/L)

        for i2 in range(1, m2+1):
            vsini2x[i2-1] = sin(pi*i2*x/L)
            vcosi2x[i2-1] = cos(pi*i2*x/L)

        for j2 in range(1, n2+1):
            vsinj2t[j2-1] = sin(j2*t)
            vcosj2t[j2-1] = cos(j2*t)

        r = r2 + sina*x

        cfphix(coeffs, m1, m2, n2, L, x, t, &phix)
        cfphit(coeffs, m1, m2, n2, L, x, t, &phit)

        c = -1

        p0000 = (-A11*r + A12*sina*(L - x))/(L*cosa)
        p0001 = (-A12*r + A22*sina*(L - x))/(L*cosa)
        p0002 = (-A16*r + A26*sina*(L - x))/(L*cosa)
        p0100 = -A16*r2*(r + sina*(L - x))/L
        p0101 = -A26*r2*(r + sina*(L - x))/L
        p0102 = -A66*r2*(r + sina*(L - x))/L
        p0200 = (A16*(L - x)*sin(t - tLA) + (A11*r + A12*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        p0201 = (A26*(L - x)*sin(t - tLA) + (A12*r + A22*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        p0202 = (A66*(L - x)*sin(t - tLA) + (A16*r + A26*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)

        for k1 in range(1, m1+1):
            sink1x = vsini1x[k1-1]
            cosk1x = vcosi1x[k1-1]

            q0004 = phix*sink1x
            q0005 = cosk1x*phix
            q0106 = phit*sink1x
            q0204 = phit*sink1x
            q0205 = cosk1x*phit
            q0206 = phix*sink1x

            # k0L_01
            c += 1
            out[c] = beta*out[c] + alpha*(p0000*q0004 + p0002*q0204)
            c += 1
            out[c] = beta*out[c] + alpha*(p0000*q0005 + p0002*q0205)
            c += 1
            out[c] = beta*out[c] + alpha*(p0001*q0106 + p0002*q0206)
            c += 1
            out[c] = beta*out[c] + alpha*(p0100*q0004 + p0102*q0204)
            c += 1
            out[c] = beta*out[c] + alpha*(p0100*q0005 + p0102*q0205)
            c += 1
            out[c] = beta*out[c] + alpha*(p0101*q0106 + p0102*q0206)
            c += 1
            out[c] = beta*out[c] + alpha*(p0200*q0004 + p0202*q0204)
            c += 1
            out[c] = beta*out[c] + alpha*(p0200*q0005 + p0202*q0205)
            c += 1
            out[c] = beta*out[c] + alpha*(p0201*q0106 + p0202*q0206)

            # create buffer
            k0Lq_1_q0004[k1-1] = q0004
            k0Lq_1_q0005[k1-1] = q0005
            k0Lq_1_q0106[k1-1] = q0106
            k0Lq_1_q0204[k1-1] = q0204
            k0Lq_1_q0205[k1-1] = q0205
            k0Lq_1_q0206[k1-1] = q0206

        for k2 in range(1, m2+1):
            sink2x = vsini2x[k2-1]
            cosk2x = vcosi2x[k2-1]
            for l2 in range(1, n2+1):
                sinl2t = vsinj2t[l2-1]
                cosl2t = vcosj2t[l2-1]

                q0008 = phix*sink2x*sinl2t
                q0009 = cosl2t*phix*sink2x
                q0010 = cosk2x*phix*sinl2t
                q0011 = cosk2x*cosl2t*phix
                q0112 = phit*sink2x*sinl2t
                q0113 = cosl2t*phit*sink2x
                q0208 = phit*sink2x*sinl2t
                q0209 = cosl2t*phit*sink2x
                q0210 = cosk2x*phit*sinl2t
                q0211 = cosk2x*cosl2t*phit
                q0212 = phix*sink2x*sinl2t
                q0213 = cosl2t*phix*sink2x

                # k0L_02
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0008 + p0002*q0208)
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0009 + p0002*q0209)
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0010 + p0002*q0210)
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0011 + p0002*q0211)
                c += 1
                out[c] = beta*out[c] + alpha*(p0001*q0112 + p0002*q0212)
                c += 1
                out[c] = beta*out[c] + alpha*(p0001*q0113 + p0002*q0213)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0008 + p0102*q0208)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0009 + p0102*q0209)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0010 + p0102*q0210)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0011 + p0102*q0211)
                c += 1
                out[c] = beta*out[c] + alpha*(p0101*q0112 + p0102*q0212)
                c += 1
                out[c] = beta*out[c] + alpha*(p0101*q0113 + p0102*q0213)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0008 + p0202*q0208)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0009 + p0202*q0209)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0010 + p0202*q0210)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0011 + p0202*q0211)
                c += 1
                out[c] = beta*out[c] + alpha*(p0201*q0112 + p0202*q0212)
                c += 1
                out[c] = beta*out[c] + alpha*(p0201*q0113 + p0202*q0213)

                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                k0Lq_2_q0008[pos] = q0008
                k0Lq_2_q0009[pos] = q0009
                k0Lq_2_q0010[pos] = q0010
                k0Lq_2_q0011[pos] = q0011
                k0Lq_2_q0112[pos] = q0112
                k0Lq_2_q0113[pos] = q0113
                k0Lq_2_q0208[pos] = q0208
                k0Lq_2_q0209[pos] = q0209
                k0Lq_2_q0210[pos] = q0210
                k0Lq_2_q0211[pos] = q0211
                k0Lq_2_q0212[pos] = q0212
                k0Lq_2_q0213[pos] = q0213


        for i1 in range(1, m1+1):
            sini1x = vsini1x[i1-1]
            cosi1x = vcosi1x[i1-1]

            p0000 = pi*A11*cosi1x*i1*r/L + A12*sina*sini1x
            p0001 = pi*A12*cosi1x*i1*r/L + A22*sina*sini1x
            p0002 = pi*A16*cosi1x*i1*r/L + A26*sina*sini1x
            p0100 = -pi*A11*i1*r*sini1x/L + A12*cosi1x*sina
            p0101 = -pi*A12*i1*r*sini1x/L + A22*cosi1x*sina
            p0102 = -pi*A16*i1*r*sini1x/L + A26*cosi1x*sina
            p0200 = -A16*sina*sini1x + pi*A16*cosi1x*i1*r/L
            p0201 = -A26*sina*sini1x + pi*A26*cosi1x*i1*r/L
            p0202 = -A66*sina*sini1x + pi*A66*cosi1x*i1*r/L
            p0300 = A12*cosa*sini1x
            p0301 = A22*cosa*sini1x
            p0302 = A26*cosa*sini1x
            p0400 = pi*B11*cosi1x*i1*r/L + B12*sina*sini1x
            p0401 = pi*B12*cosi1x*i1*r/L + B22*sina*sini1x
            p0402 = pi*B16*cosi1x*i1*r/L + B26*sina*sini1x
            p0500 = -pi*B11*i1*r*sini1x/L + B12*cosi1x*sina
            p0501 = -pi*B12*i1*r*sini1x/L + B22*cosi1x*sina
            p0502 = -pi*B16*i1*r*sini1x/L + B26*cosi1x*sina
            p0600 = -B16*sina*sini1x + pi*B16*cosi1x*i1*r/L
            p0601 = -B26*sina*sini1x + pi*B26*cosi1x*i1*r/L
            p0602 = -B66*sina*sini1x + pi*B66*cosi1x*i1*r/L

            for k1 in range(1, m1+1):
                # access buffer
                q0004 = k0Lq_1_q0004[k1-1]
                q0005 = k0Lq_1_q0005[k1-1]
                q0106 = k0Lq_1_q0106[k1-1]
                q0204 = k0Lq_1_q0204[k1-1]
                q0205 = k0Lq_1_q0205[k1-1]
                q0206 = k0Lq_1_q0206[k1-1]

                # k0L_11
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0004 + p0002*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0005 + p0002*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0001*q0106 + p0002*q0206)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0004 + p0102*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0005 + p0102*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0101*q0106 + p0102*q0206)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0004 + p0202*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0005 + p0202*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0201*q0106 + p0202*q0206)
                c += 1
                out[c] = beta*out[c] + alpha*(p0300*q0004 + p0302*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0300*q0005 + p0302*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0301*q0106 + p0302*q0206)
                c += 1
                out[c] = beta*out[c] + alpha*(p0400*q0004 + p0402*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0400*q0005 + p0402*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0401*q0106 + p0402*q0206)
                c += 1
                out[c] = beta*out[c] + alpha*(p0500*q0004 + p0502*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0500*q0005 + p0502*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0501*q0106 + p0502*q0206)
                c += 1
                out[c] = beta*out[c] + alpha*(p0600*q0004 + p0602*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0600*q0005 + p0602*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0601*q0106 + p0602*q0206)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q0008 = k0Lq_2_q0008[pos]
                    q0009 = k0Lq_2_q0009[pos]
                    q0010 = k0Lq_2_q0010[pos]
                    q0011 = k0Lq_2_q0011[pos]
                    q0112 = k0Lq_2_q0112[pos]
                    q0113 = k0Lq_2_q0113[pos]
                    q0208 = k0Lq_2_q0208[pos]
                    q0209 = k0Lq_2_q0209[pos]
                    q0210 = k0Lq_2_q0210[pos]
                    q0211 = k0Lq_2_q0211[pos]
                    q0212 = k0Lq_2_q0212[pos]
                    q0213 = k0Lq_2_q0213[pos]

                    # k0L_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0008 + p0002*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0009 + p0002*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0010 + p0002*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0011 + p0002*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0001*q0112 + p0002*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0001*q0113 + p0002*q0213)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0008 + p0102*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0009 + p0102*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0010 + p0102*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0011 + p0102*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0101*q0112 + p0102*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0101*q0113 + p0102*q0213)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0008 + p0202*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0009 + p0202*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0010 + p0202*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0011 + p0202*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0201*q0112 + p0202*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0201*q0113 + p0202*q0213)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0008 + p0302*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0009 + p0302*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0010 + p0302*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0011 + p0302*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0301*q0112 + p0302*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0301*q0113 + p0302*q0213)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0008 + p0402*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0009 + p0402*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0010 + p0402*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0011 + p0402*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0401*q0112 + p0402*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0401*q0113 + p0402*q0213)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0008 + p0502*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0009 + p0502*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0010 + p0502*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0011 + p0502*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0501*q0112 + p0502*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0501*q0113 + p0502*q0213)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0008 + p0602*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0009 + p0602*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0010 + p0602*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0011 + p0602*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0601*q0112 + p0602*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0601*q0113 + p0602*q0213)

        for i2 in range(1, m2+1):
            sini2x = vsini2x[i2-1]
            cosi2x = vcosi2x[i2-1]

            for j2 in range(1, n2+1):
                sinj2t = vsinj2t[j2-1]
                cosj2t = vcosj2t[j2-1]

                p0000 = pi*A11*cosi2x*i2*r*sinj2t/L + sini2x*(A12*sina*sinj2t + A16*cosj2t*j2)
                p0001 = pi*A12*cosi2x*i2*r*sinj2t/L + sini2x*(A22*sina*sinj2t + A26*cosj2t*j2)
                p0002 = pi*A16*cosi2x*i2*r*sinj2t/L + sini2x*(A26*sina*sinj2t + A66*cosj2t*j2)
                p0100 = pi*A11*cosi2x*cosj2t*i2*r/L + sini2x*(A12*cosj2t*sina - A16*j2*sinj2t)
                p0101 = pi*A12*cosi2x*cosj2t*i2*r/L + sini2x*(A22*cosj2t*sina - A26*j2*sinj2t)
                p0102 = pi*A16*cosi2x*cosj2t*i2*r/L + sini2x*(A26*cosj2t*sina - A66*j2*sinj2t)
                p0200 = -pi*A11*i2*r*sini2x*sinj2t/L + cosi2x*(A12*sina*sinj2t + A16*cosj2t*j2)
                p0201 = -pi*A12*i2*r*sini2x*sinj2t/L + cosi2x*(A22*sina*sinj2t + A26*cosj2t*j2)
                p0202 = -pi*A16*i2*r*sini2x*sinj2t/L + cosi2x*(A26*sina*sinj2t + A66*cosj2t*j2)
                p0300 = -pi*A11*cosj2t*i2*r*sini2x/L + cosi2x*(A12*cosj2t*sina - A16*j2*sinj2t)
                p0301 = -pi*A12*cosj2t*i2*r*sini2x/L + cosi2x*(A22*cosj2t*sina - A26*j2*sinj2t)
                p0302 = -pi*A16*cosj2t*i2*r*sini2x/L + cosi2x*(A26*cosj2t*sina - A66*j2*sinj2t)
                p0400 = pi*A16*cosi2x*i2*r*sinj2t/L + sini2x*(A12*cosj2t*j2 - A16*sina*sinj2t)
                p0401 = pi*A26*cosi2x*i2*r*sinj2t/L + sini2x*(A22*cosj2t*j2 - A26*sina*sinj2t)
                p0402 = pi*A66*cosi2x*i2*r*sinj2t/L + sini2x*(A26*cosj2t*j2 - A66*sina*sinj2t)
                p0500 = pi*A16*cosi2x*cosj2t*i2*r/L - sini2x*(A12*j2*sinj2t + A16*cosj2t*sina)
                p0501 = pi*A26*cosi2x*cosj2t*i2*r/L - sini2x*(A22*j2*sinj2t + A26*cosj2t*sina)
                p0502 = pi*A66*cosi2x*cosj2t*i2*r/L - sini2x*(A26*j2*sinj2t + A66*cosj2t*sina)
                p0600 = A12*cosa*sini2x*sinj2t
                p0601 = A22*cosa*sini2x*sinj2t
                p0602 = A26*cosa*sini2x*sinj2t
                p0700 = A12*cosa*cosj2t*sini2x
                p0701 = A22*cosa*cosj2t*sini2x
                p0702 = A26*cosa*cosj2t*sini2x
                p0800 = pi*B11*cosi2x*i2*r*sinj2t/L + sini2x*(B12*sina*sinj2t + B16*cosj2t*j2)
                p0801 = pi*B12*cosi2x*i2*r*sinj2t/L + sini2x*(B22*sina*sinj2t + B26*cosj2t*j2)
                p0802 = pi*B16*cosi2x*i2*r*sinj2t/L + sini2x*(B26*sina*sinj2t + B66*cosj2t*j2)
                p0900 = pi*B11*cosi2x*cosj2t*i2*r/L + sini2x*(B12*cosj2t*sina - B16*j2*sinj2t)
                p0901 = pi*B12*cosi2x*cosj2t*i2*r/L + sini2x*(B22*cosj2t*sina - B26*j2*sinj2t)
                p0902 = pi*B16*cosi2x*cosj2t*i2*r/L + sini2x*(B26*cosj2t*sina - B66*j2*sinj2t)
                p1000 = -pi*B11*i2*r*sini2x*sinj2t/L + cosi2x*(B12*sina*sinj2t + B16*cosj2t*j2)
                p1001 = -pi*B12*i2*r*sini2x*sinj2t/L + cosi2x*(B22*sina*sinj2t + B26*cosj2t*j2)
                p1002 = -pi*B16*i2*r*sini2x*sinj2t/L + cosi2x*(B26*sina*sinj2t + B66*cosj2t*j2)
                p1100 = -pi*B11*cosj2t*i2*r*sini2x/L + cosi2x*(B12*cosj2t*sina - B16*j2*sinj2t)
                p1101 = -pi*B12*cosj2t*i2*r*sini2x/L + cosi2x*(B22*cosj2t*sina - B26*j2*sinj2t)
                p1102 = -pi*B16*cosj2t*i2*r*sini2x/L + cosi2x*(B26*cosj2t*sina - B66*j2*sinj2t)
                p1200 = pi*B16*cosi2x*i2*r*sinj2t/L + sini2x*(B12*cosj2t*j2 - B16*sina*sinj2t)
                p1201 = pi*B26*cosi2x*i2*r*sinj2t/L + sini2x*(B22*cosj2t*j2 - B26*sina*sinj2t)
                p1202 = pi*B66*cosi2x*i2*r*sinj2t/L + sini2x*(B26*cosj2t*j2 - B66*sina*sinj2t)
                p1300 = pi*B16*cosi2x*cosj2t*i2*r/L - sini2x*(B12*j2*sinj2t + B16*cosj2t*sina)
                p1301 = pi*B26*cosi2x*cosj2t*i2*r/L - sini2x*(B22*j2*sinj2t + B26*cosj2t*sina)
                p1302 = pi*B66*cosi2x*cosj2t*i2*r/L - sini2x*(B26*j2*sinj2t + B66*cosj2t*sina)

                for k1 in range(1, m1+1):
                    # access buffer
                    q0004 = k0Lq_1_q0004[k1-1]
                    q0005 = k0Lq_1_q0005[k1-1]
                    q0106 = k0Lq_1_q0106[k1-1]
                    q0204 = k0Lq_1_q0204[k1-1]
                    q0205 = k0Lq_1_q0205[k1-1]
                    q0206 = k0Lq_1_q0206[k1-1]

                    # k0L_21
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0004 + p0002*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0005 + p0002*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0001*q0106 + p0002*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0004 + p0102*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0005 + p0102*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0101*q0106 + p0102*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0004 + p0202*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0005 + p0202*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0201*q0106 + p0202*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0004 + p0302*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0005 + p0302*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0301*q0106 + p0302*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0004 + p0402*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0005 + p0402*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0401*q0106 + p0402*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0004 + p0502*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0005 + p0502*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0501*q0106 + p0502*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0004 + p0602*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0005 + p0602*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0601*q0106 + p0602*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0700*q0004 + p0702*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0700*q0005 + p0702*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0701*q0106 + p0702*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0800*q0004 + p0802*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0800*q0005 + p0802*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0801*q0106 + p0802*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0900*q0004 + p0902*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0900*q0005 + p0902*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0901*q0106 + p0902*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1000*q0004 + p1002*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1000*q0005 + p1002*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1001*q0106 + p1002*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1100*q0004 + p1102*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1100*q0005 + p1102*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1101*q0106 + p1102*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1200*q0004 + p1202*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1200*q0005 + p1202*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1201*q0106 + p1202*q0206)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1300*q0004 + p1302*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1300*q0005 + p1302*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p1301*q0106 + p1302*q0206)

                for k2 in range(1, m2+1):
                    for l2 in range(1, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q0008 = k0Lq_2_q0008[pos]
                        q0009 = k0Lq_2_q0009[pos]
                        q0010 = k0Lq_2_q0010[pos]
                        q0011 = k0Lq_2_q0011[pos]
                        q0112 = k0Lq_2_q0112[pos]
                        q0113 = k0Lq_2_q0113[pos]
                        q0208 = k0Lq_2_q0208[pos]
                        q0209 = k0Lq_2_q0209[pos]
                        q0210 = k0Lq_2_q0210[pos]
                        q0211 = k0Lq_2_q0211[pos]
                        q0212 = k0Lq_2_q0212[pos]
                        q0213 = k0Lq_2_q0213[pos]

                        # k0L_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0000*q0008 + p0002*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0000*q0009 + p0002*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0000*q0010 + p0002*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0000*q0011 + p0002*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0001*q0112 + p0002*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0001*q0113 + p0002*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0100*q0008 + p0102*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0100*q0009 + p0102*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0100*q0010 + p0102*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0100*q0011 + p0102*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0101*q0112 + p0102*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0101*q0113 + p0102*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0200*q0008 + p0202*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0200*q0009 + p0202*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0200*q0010 + p0202*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0200*q0011 + p0202*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0201*q0112 + p0202*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0201*q0113 + p0202*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0300*q0008 + p0302*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0300*q0009 + p0302*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0300*q0010 + p0302*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0300*q0011 + p0302*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0301*q0112 + p0302*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0301*q0113 + p0302*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0008 + p0402*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0009 + p0402*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0010 + p0402*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0011 + p0402*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0401*q0112 + p0402*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0401*q0113 + p0402*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0008 + p0502*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0009 + p0502*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0010 + p0502*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0011 + p0502*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0501*q0112 + p0502*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0501*q0113 + p0502*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0600*q0008 + p0602*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0600*q0009 + p0602*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0600*q0010 + p0602*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0600*q0011 + p0602*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0601*q0112 + p0602*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0601*q0113 + p0602*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0700*q0008 + p0702*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0700*q0009 + p0702*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0700*q0010 + p0702*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0700*q0011 + p0702*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0701*q0112 + p0702*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0701*q0113 + p0702*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0008 + p0802*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0009 + p0802*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0010 + p0802*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0011 + p0802*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0801*q0112 + p0802*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0801*q0113 + p0802*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0008 + p0902*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0009 + p0902*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0010 + p0902*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0011 + p0902*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0901*q0112 + p0902*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0901*q0113 + p0902*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0008 + p1002*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0009 + p1002*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0010 + p1002*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0011 + p1002*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1001*q0112 + p1002*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1001*q0113 + p1002*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0008 + p1102*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0009 + p1102*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0010 + p1102*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0011 + p1102*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1101*q0112 + p1102*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1101*q0113 + p1102*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0008 + p1202*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0009 + p1202*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0010 + p1202*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0011 + p1202*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1201*q0112 + p1202*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1201*q0113 + p1202*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0008 + p1302*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0009 + p1302*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0010 + p1302*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0011 + p1302*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1301*q0112 + p1302*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1301*q0113 + p1302*q0213)

    free(vsini1x)
    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)

    free(k0Lq_1_q0004)
    free(k0Lq_1_q0005)
    free(k0Lq_1_q0106)
    free(k0Lq_1_q0204)
    free(k0Lq_1_q0205)
    free(k0Lq_1_q0206)

    free(k0Lq_2_q0008)
    free(k0Lq_2_q0009)
    free(k0Lq_2_q0010)
    free(k0Lq_2_q0011)
    free(k0Lq_2_q0112)
    free(k0Lq_2_q0113)
    free(k0Lq_2_q0208)
    free(k0Lq_2_q0209)
    free(k0Lq_2_q0210)
    free(k0Lq_2_q0211)
    free(k0Lq_2_q0212)
    free(k0Lq_2_q0213)


def calc_kG(np.ndarray[cDOUBLE, ndim=1] coeffs,
              double alpharad, double r2, double L, double tLA,
              np.ndarray[cDOUBLE, ndim=2] F,
              int m1, int m2, int n2,
              int nx, int nt, int num_cores, str method='trapz2d'):

    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kGv
    cdef np.ndarray[cDOUBLE, ndim=2] tmp

    cdef unsigned fdim
    cdef cc_attributes args

    fdim = 9*m1**2/2 + 18*m1*m2*n2 + 36*m2**2*n2**2/2

    rows = np.zeros((fdim,), dtype=INT)
    cols = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    args.sina = &sina
    args.cosa = &cosa
    args.tLA = &tLA
    args.r2 = &r2
    args.L = &L
    args.F = &F[0,0]
    args.m1 = &m1
    args.m2 = &m2
    args.n2 = &n2
    args.coeffs = &coeffs[0]

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    if method=='trapz2d':
        trapz2d(<f_type>cfkG, fdim, kGv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<f_type>cfkG, fdim, kGv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    c = -1

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        #NOTE symmetry
        for k1 in range(i1, m1+1):
            col = (k1-1)*num1 + num0

            # kG_11
            c += 1
            rows[c] = row+4
            cols[c] = col+4
            c += 1
            rows[c] = row+4
            cols[c] = col+5
            c += 1
            rows[c] = row+4
            cols[c] = col+6
            c += 1
            rows[c] = row+5
            cols[c] = col+4
            c += 1
            rows[c] = row+5
            cols[c] = col+5
            c += 1
            rows[c] = row+5
            cols[c] = col+6
            c += 1
            rows[c] = row+6
            cols[c] = col+4
            c += 1
            rows[c] = row+6
            cols[c] = col+5
            c += 1
            rows[c] = row+6
            cols[c] = col+6

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1

                # kG_12
                c += 1
                rows[c] = row+4
                cols[c] = col+8
                c += 1
                rows[c] = row+4
                cols[c] = col+9
                c += 1
                rows[c] = row+4
                cols[c] = col+10
                c += 1
                rows[c] = row+4
                cols[c] = col+11
                c += 1
                rows[c] = row+4
                cols[c] = col+12
                c += 1
                rows[c] = row+4
                cols[c] = col+13
                c += 1
                rows[c] = row+5
                cols[c] = col+8
                c += 1
                rows[c] = row+5
                cols[c] = col+9
                c += 1
                rows[c] = row+5
                cols[c] = col+10
                c += 1
                rows[c] = row+5
                cols[c] = col+11
                c += 1
                rows[c] = row+5
                cols[c] = col+12
                c += 1
                rows[c] = row+5
                cols[c] = col+13
                c += 1
                rows[c] = row+6
                cols[c] = col+8
                c += 1
                rows[c] = row+6
                cols[c] = col+9
                c += 1
                rows[c] = row+6
                cols[c] = col+10
                c += 1
                rows[c] = row+6
                cols[c] = col+11
                c += 1
                rows[c] = row+6
                cols[c] = col+12
                c += 1
                rows[c] = row+6
                cols[c] = col+13

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            #NOTE symmetry
            for k2 in range(i2, m2+1):
                for l2 in range(j2, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1

                    # kG_22
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+13

    size = num0 + num1*m1 + num2*m2*n2

    tmp = coo_matrix((kGv, (rows, cols)), shape=(size, size)).toarray()

    #NOTE symmetry
    for row in range(size):
        for col in range(row, size):
            tmp[col, row] = tmp[row, col]

    kG = coo_matrix(tmp)

    return kG

cdef void cfkG(int npts, double *xs, double *ts, double *out,
                 double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2

    cdef double p0400, p0401, p0500, p0501, p0600, p0601
    cdef double p0800, p0801, p0900, p0901, p1000, p1001
    cdef double p1100, p1101, p1200, p1201, p1300, p1301

    cdef double q0004, q0005, q0008, q0009, q0010, q0011, q0112, q0113, q0106

    cdef double r, x, t, alpha, beta
    cdef int c, i, pos

    cdef double *F, *coeffs
    cdef double r2, L, sina, cosa, tLA
    cdef int m1, m2, n2, pti
    cdef double Nxx, Ntt, Nxt
    cdef double N[6]
    cdef int NL_kinematics=0 # to use cfstrain_donnell in cfN

    cdef cc_attributes *args_in=<cc_attributes *>args

    sina = args_in.sina[0]
    cosa = args_in.cosa[0]
    tLA = args_in.tLA[0]
    r2 = args_in.r2[0]
    L = args_in.L[0]
    F = args_in.F
    m1 = args_in.m1[0]
    m2 = args_in.m2[0]
    n2 = args_in.n2[0]
    coeffs = args_in.coeffs

    cdef double sini1x, cosi1x, sink1x, cosk1x
    cdef double sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t

    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))

    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))

    cdef double *kGq_2_q0008 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q0009 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q0010 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q0011 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q0112 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q0113 = <double *>malloc(m2*n2 * sizeof(double))

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(1, m1+1):
            vsini1x[i1-1] = sin(pi*i1*x/L)
            vcosi1x[i1-1] = cos(pi*i1*x/L)

        for i2 in range(1, m2+1):
            vsini2x[i2-1] = sin(pi*i2*x/L)
            vcosi2x[i2-1] = cos(pi*i2*x/L)

        for j2 in range(1, n2+1):
            vsinj2t[j2-1] = sin(j2*t)
            vcosj2t[j2-1] = cos(j2*t)

        r = r2 + sina*x

        cfN(coeffs, sina, cosa, tLA, x, t, r, r2, L, F, m1, m2, n2,
            &N[0], NL_kinematics)
        Nxx = N[0]
        Ntt = N[1]
        Nxt = N[2]

        c = -1

        for k2 in range(1, m2+1):
            sink2x = vsini2x[k2-1]
            cosk2x = vcosi2x[k2-1]
            for l2 in range(1, n2+1):
                sinl2t = vsinj2t[l2-1]
                cosl2t = vcosj2t[l2-1]

                q0008 = sink2x*sinl2t
                q0009 = cosl2t*sink2x
                q0010 = cosk2x*sinl2t
                q0011 = cosk2x*cosl2t
                q0112 = sink2x*sinl2t
                q0113 = cosl2t*sink2x

                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                kGq_2_q0008[pos] = q0008
                kGq_2_q0009[pos] = q0009
                kGq_2_q0010[pos] = q0010
                kGq_2_q0011[pos] = q0011
                kGq_2_q0112[pos] = q0112
                kGq_2_q0113[pos] = q0113

        for i1 in range(1, m1+1):
            sini1x = vsini1x[k2-1]
            cosi1x = vcosi1x[k2-1]

            p0400 = Nxx*r*sini1x
            p0401 = Nxt*r*sini1x
            p0500 = Nxx*cosi1x*r
            p0501 = Nxt*cosi1x*r
            p0600 = Nxt*r*sini1x
            p0601 = Ntt*r*sini1x

            #NOTE symmetry
            for k1 in range(i1, m1+1):
                sink1x = vsini1x[k1-1]
                cosk1x = vcosi1x[k1-1]

                q0004 = sink1x
                q0005 = cosk1x
                q0106 = sink1x

                # kG_11
                c += 1
                out[c] = beta*out[c] + alpha*(p0400*q0004)
                c += 1
                out[c] = beta*out[c] + alpha*(p0400*q0005)
                c += 1
                out[c] = beta*out[c] + alpha*(p0401*q0106)
                c += 1
                out[c] = beta*out[c] + alpha*(p0500*q0004)
                c += 1
                out[c] = beta*out[c] + alpha*(p0500*q0005)
                c += 1
                out[c] = beta*out[c] + alpha*(p0501*q0106)
                c += 1
                out[c] = beta*out[c] + alpha*(p0600*q0004)
                c += 1
                out[c] = beta*out[c] + alpha*(p0600*q0005)
                c += 1
                out[c] = beta*out[c] + alpha*(p0601*q0106)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q0008 = kGq_2_q0008[pos]
                    q0009 = kGq_2_q0009[pos]
                    q0010 = kGq_2_q0010[pos]
                    q0011 = kGq_2_q0011[pos]
                    q0112 = kGq_2_q0112[pos]
                    q0113 = kGq_2_q0113[pos]

                    # kG_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0008)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0009)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0010)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0011)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0401*q0112)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0401*q0113)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0008)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0009)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0010)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0011)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0501*q0112)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0501*q0113)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0008)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0009)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0010)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0011)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0601*q0112)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0601*q0113)

        for i2 in range(1, m2+1):
            sini2x = vsini2x[i2-1]
            cosi2x = vcosi2x[i2-1]
            for j2 in range(1, n2+1):
                sinj2t = vsinj2t[j2-1]
                cosj2t = vcosj2t[j2-1]

                p0800 = Nxx*r*sini2x*sinj2t
                p0801 = Nxt*r*sini2x*sinj2t
                p0900 = Nxx*cosj2t*r*sini2x
                p0901 = Nxt*cosj2t*r*sini2x
                p1000 = Nxx*cosi2x*r*sinj2t
                p1001 = Nxt*cosi2x*r*sinj2t
                p1100 = Nxx*cosi2x*cosj2t*r
                p1101 = Nxt*cosi2x*cosj2t*r
                p1200 = Nxt*r*sini2x*sinj2t
                p1201 = Ntt*r*sini2x*sinj2t
                p1300 = Nxt*cosj2t*r*sini2x
                p1301 = Ntt*cosj2t*r*sini2x

                #NOTE symmetry
                for k2 in range(i2, m2+1):
                    for l2 in range(j2, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q0008 = kGq_2_q0008[pos]
                        q0009 = kGq_2_q0009[pos]
                        q0010 = kGq_2_q0010[pos]
                        q0011 = kGq_2_q0011[pos]
                        q0112 = kGq_2_q0112[pos]
                        q0113 = kGq_2_q0113[pos]

                        # kG_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0008)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0009)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0010)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0011)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0801*q0112)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0801*q0113)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0008)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0009)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0010)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0011)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0901*q0112)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0901*q0113)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0008)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0009)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0010)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0011)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1001*q0112)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1001*q0113)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0008)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0009)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0010)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0011)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1101*q0112)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1101*q0113)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0008)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0009)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0010)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0011)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1201*q0112)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1201*q0113)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0008)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0009)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0010)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0011)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1301*q0112)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1301*q0113)

    free(vsini1x)
    free(vcosi1x)

    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)

    free(kGq_2_q0008)
    free(kGq_2_q0009)
    free(kGq_2_q0010)
    free(kGq_2_q0011)
    free(kGq_2_q0112)
    free(kGq_2_q0113)


def calc_kLL(np.ndarray[cDOUBLE, ndim=1] coeffs,
             double alpharad, double r2, double L, double tLA,
             np.ndarray[cDOUBLE, ndim=2] F,
             int m1, int m2, int n2,
             int nx, int nt, int num_cores, str method='trapz2d'):

    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kLLv
    cdef np.ndarray[cDOUBLE, ndim=2] tmp

    cdef int fdim
    cdef cc_attributes args

    fdim = 9*m1**2/2 + 18*m1*m2*n2/2 + 36*m2**2*n2**2/2

    rows = np.zeros((fdim,), dtype=INT)
    cols = np.zeros((fdim,), dtype=INT)
    kLLv = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    args.sina = &sina
    args.cosa = &cosa
    args.tLA = &tLA
    args.r2 = &r2
    args.L = &L
    args.F = &F[0,0]
    args.m1 = &m1
    args.m2 = &m2
    args.n2 = &n2
    args.coeffs = &coeffs[0]

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    if method=='trapz2d':
        trapz2d(<f_type>cfkLL, fdim, kLLv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<f_type>cfkLL, fdim, kLLv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    c = -1

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        #NOTE symmetry
        for k1 in range(i1, m1+1):
            col = (k1-1)*num1 + num0

            # kLL_11
            c += 1
            rows[c] = row+4
            cols[c] = col+4
            c += 1
            rows[c] = row+4
            cols[c] = col+5
            c += 1
            rows[c] = row+4
            cols[c] = col+6
            c += 1
            rows[c] = row+5
            cols[c] = col+4
            c += 1
            rows[c] = row+5
            cols[c] = col+5
            c += 1
            rows[c] = row+5
            cols[c] = col+6
            c += 1
            rows[c] = row+6
            cols[c] = col+4
            c += 1
            rows[c] = row+6
            cols[c] = col+5
            c += 1
            rows[c] = row+6
            cols[c] = col+6

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1

                # kLL_12
                c += 1
                rows[c] = row+4
                cols[c] = col+8
                c += 1
                rows[c] = row+4
                cols[c] = col+9
                c += 1
                rows[c] = row+4
                cols[c] = col+10
                c += 1
                rows[c] = row+4
                cols[c] = col+11
                c += 1
                rows[c] = row+4
                cols[c] = col+12
                c += 1
                rows[c] = row+4
                cols[c] = col+13
                c += 1
                rows[c] = row+5
                cols[c] = col+8
                c += 1
                rows[c] = row+5
                cols[c] = col+9
                c += 1
                rows[c] = row+5
                cols[c] = col+10
                c += 1
                rows[c] = row+5
                cols[c] = col+11
                c += 1
                rows[c] = row+5
                cols[c] = col+12
                c += 1
                rows[c] = row+5
                cols[c] = col+13
                c += 1
                rows[c] = row+6
                cols[c] = col+8
                c += 1
                rows[c] = row+6
                cols[c] = col+9
                c += 1
                rows[c] = row+6
                cols[c] = col+10
                c += 1
                rows[c] = row+6
                cols[c] = col+11
                c += 1
                rows[c] = row+6
                cols[c] = col+12
                c += 1
                rows[c] = row+6
                cols[c] = col+13

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            #NOTE symmetry
            for k2 in range(i2, m2+1):
                for l2 in range(j2, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1

                    # kLL_22
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+10
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+11
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+12
                    cols[c] = col+13
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+8
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+9
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+10
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+11
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+12
                    c += 1
                    rows[c] = row+13
                    cols[c] = col+13

    size = num0 + num1*m1 + num2*m2*n2

    tmp = coo_matrix((kLLv, (rows, cols)), shape=(size, size)).toarray()

    #NOTE symmetry
    for row in range(size):
        for col in range(row, size):
            tmp[col, row] = tmp[row, col]

    kLL = coo_matrix(tmp)

    return kLL

cdef void cfkLL(int npts, double *xs, double *ts, double *out,
                double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2
    cdef int c, i, pos

    cdef double A11, A12, A16, A22, A26, A66

    cdef double p0400, p0401, p0402, p0500, p0501, p0502, p0600, p0601, p0602
    cdef double p0800, p0801, p0802, p0900, p0901, p0902
    cdef double p1000, p1001, p1002, p1100, p1101, p1102
    cdef double p1200, p1201, p1202, p1300, p1301, p1302

    cdef double q0004, q0005, q0106, q0204, q0205, q0206
    cdef double q0008, q0009, q0010, q0011, q0112, q0113
    cdef double q0208, q0209, q0210, q0211, q0212, q0213


    cdef double r, x, t, alpha, beta

    cdef double *F, *coeffs
    cdef double sina, r2, L
    cdef int m1, m2, n2, pti
    cdef double phix, phit

    cdef cc_attributes *args_in=<cc_attributes *>args
    sina = args_in.sina[0]
    r2 = args_in.r2[0]
    L = args_in.L[0]
    F = args_in.F
    m1 = args_in.m1[0]
    m2 = args_in.m2[0]
    n2 = args_in.n2[0]
    coeffs = args_in.coeffs

    A11 = F[0]  # F[0,0]
    A12 = F[1]  # F[0,1]
    A16 = F[2]  # F[0,2]
    A22 = F[9]  # F[1,1]
    A26 = F[10] # F[1,2]
    A66 = F[18] # F[2,2]

    cdef double sini1x, cosi1x, sink1x, cosk1x
    cdef double sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t

    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))

    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))

    cdef double *kLLq_2_q0008 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0009 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0010 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0011 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0112 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0113 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0208 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0209 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0210 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0211 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0212 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0213 = <double *>malloc(m2*n2 * sizeof(double))

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(1, m1+1):
            vsini1x[i1-1] = sin(pi*i1*x/L)
            vcosi1x[i1-1] = cos(pi*i1*x/L)

        for i2 in range(1, m2+1):
            vsini2x[i2-1] = sin(pi*i2*x/L)
            vcosi2x[i2-1] = cos(pi*i2*x/L)

        for j2 in range(1, n2+1):
            vsinj2t[j2-1] = sin(j2*t)
            vcosj2t[j2-1] = cos(j2*t)

        r = r2 + sina*x

        cfphix(coeffs, m1, m2, n2, L, x, t, &phix)
        cfphit(coeffs, m1, m2, n2, L, x, t, &phit)

        c = -1

        for k2 in range(1, m2+1):
            sink2x = vsini2x[k2-1]
            cosk2x = vcosi2x[k2-1]
            for l2 in range(1, n2+1):
                sinl2t = vsinj2t[l2-1]
                cosl2t = vcosj2t[l2-1]

                q0008 = phix*sink2x*sinl2t
                q0009 = cosl2t*phix*sink2x
                q0010 = cosk2x*phix*sinl2t
                q0011 = cosk2x*cosl2t*phix
                q0112 = phit*sink2x*sinl2t
                q0113 = cosl2t*phit*sink2x
                q0208 = phit*sink2x*sinl2t
                q0209 = cosl2t*phit*sink2x
                q0210 = cosk2x*phit*sinl2t
                q0211 = cosk2x*cosl2t*phit
                q0212 = phix*sink2x*sinl2t
                q0213 = cosl2t*phix*sink2x

                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                kLLq_2_q0008[pos] = q0008
                kLLq_2_q0009[pos] = q0009
                kLLq_2_q0010[pos] = q0010
                kLLq_2_q0011[pos] = q0011
                kLLq_2_q0112[pos] = q0112
                kLLq_2_q0113[pos] = q0113
                kLLq_2_q0208[pos] = q0208
                kLLq_2_q0209[pos] = q0209
                kLLq_2_q0210[pos] = q0210
                kLLq_2_q0211[pos] = q0211
                kLLq_2_q0212[pos] = q0212
                kLLq_2_q0213[pos] = q0213

        for i1 in range(1, m1+1):
            sini1x = vsini1x[i1-1]
            cosi1x = vcosi1x[i1-1]

            p0400 = r*sini1x*(A11*phix + A16*phit)
            p0401 = r*sini1x*(A12*phix + A26*phit)
            p0402 = r*sini1x*(A16*phix + A66*phit)
            p0500 = cosi1x*r*(A11*phix + A16*phit)
            p0501 = cosi1x*r*(A12*phix + A26*phit)
            p0502 = cosi1x*r*(A16*phix + A66*phit)
            p0600 = r*sini1x*(A12*phit + A16*phix)
            p0601 = r*sini1x*(A22*phit + A26*phix)
            p0602 = r*sini1x*(A26*phit + A66*phix)

            #NOTE symmetry
            for k1 in range(i1, m1+1):
                sink1x = vsini1x[k1-1]
                cosk1x = vcosi1x[k1-1]

                q0004 = phix*sink1x
                q0005 = cosk1x*phix
                q0106 = phit*sink1x
                q0204 = phit*sink1x
                q0205 = cosk1x*phit
                q0206 = phix*sink1x

                # kLL_11
                c += 1
                out[c] = beta*out[c] + alpha*(p0400*q0004 + p0402*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0400*q0005 + p0402*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0401*q0106 + p0402*q0206)
                c += 1
                out[c] = beta*out[c] + alpha*(p0500*q0004 + p0502*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0500*q0005 + p0502*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0501*q0106 + p0502*q0206)
                c += 1
                out[c] = beta*out[c] + alpha*(p0600*q0004 + p0602*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0600*q0005 + p0602*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0601*q0106 + p0602*q0206)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q0008 = kLLq_2_q0008[pos]
                    q0009 = kLLq_2_q0009[pos]
                    q0010 = kLLq_2_q0010[pos]
                    q0011 = kLLq_2_q0011[pos]
                    q0112 = kLLq_2_q0112[pos]
                    q0113 = kLLq_2_q0113[pos]
                    q0208 = kLLq_2_q0208[pos]
                    q0209 = kLLq_2_q0209[pos]
                    q0210 = kLLq_2_q0210[pos]
                    q0211 = kLLq_2_q0211[pos]
                    q0212 = kLLq_2_q0212[pos]
                    q0213 = kLLq_2_q0213[pos]

                    # kLL_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0008 + p0402*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0009 + p0402*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0010 + p0402*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0011 + p0402*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0401*q0112 + p0402*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0401*q0113 + p0402*q0213)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0008 + p0502*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0009 + p0502*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0010 + p0502*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0011 + p0502*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0501*q0112 + p0502*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0501*q0113 + p0502*q0213)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0008 + p0602*q0208)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0009 + p0602*q0209)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0010 + p0602*q0210)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0011 + p0602*q0211)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0601*q0112 + p0602*q0212)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0601*q0113 + p0602*q0213)

        for i2 in range(1, m2+1):
            sini2x = vsini2x[i2-1]
            cosi2x = vcosi2x[i2-1]
            for j2 in range(1, n2+1):
                sinj2t = vsinj2t[j2-1]
                cosj2t = vcosj2t[j2-1]

                p0800 = r*sini2x*sinj2t*(A11*phix + A16*phit)
                p0801 = r*sini2x*sinj2t*(A12*phix + A26*phit)
                p0802 = r*sini2x*sinj2t*(A16*phix + A66*phit)
                p0900 = cosj2t*r*sini2x*(A11*phix + A16*phit)
                p0901 = cosj2t*r*sini2x*(A12*phix + A26*phit)
                p0902 = cosj2t*r*sini2x*(A16*phix + A66*phit)
                p1000 = cosi2x*r*sinj2t*(A11*phix + A16*phit)
                p1001 = cosi2x*r*sinj2t*(A12*phix + A26*phit)
                p1002 = cosi2x*r*sinj2t*(A16*phix + A66*phit)
                p1100 = cosi2x*cosj2t*r*(A11*phix + A16*phit)
                p1101 = cosi2x*cosj2t*r*(A12*phix + A26*phit)
                p1102 = cosi2x*cosj2t*r*(A16*phix + A66*phit)
                p1200 = r*sini2x*sinj2t*(A12*phit + A16*phix)
                p1201 = r*sini2x*sinj2t*(A22*phit + A26*phix)
                p1202 = r*sini2x*sinj2t*(A26*phit + A66*phix)
                p1300 = cosj2t*r*sini2x*(A12*phit + A16*phix)
                p1301 = cosj2t*r*sini2x*(A22*phit + A26*phix)
                p1302 = cosj2t*r*sini2x*(A26*phit + A66*phix)
                #NOTE symmetry
                for k2 in range(i2, m2+1):
                    for l2 in range(j2, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q0008 = kLLq_2_q0008[pos]
                        q0009 = kLLq_2_q0009[pos]
                        q0010 = kLLq_2_q0010[pos]
                        q0011 = kLLq_2_q0011[pos]
                        q0112 = kLLq_2_q0112[pos]
                        q0113 = kLLq_2_q0113[pos]
                        q0208 = kLLq_2_q0208[pos]
                        q0209 = kLLq_2_q0209[pos]
                        q0210 = kLLq_2_q0210[pos]
                        q0211 = kLLq_2_q0211[pos]
                        q0212 = kLLq_2_q0212[pos]
                        q0213 = kLLq_2_q0213[pos]

                        # kLL_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0008 + p0802*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0009 + p0802*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0010 + p0802*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0011 + p0802*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0801*q0112 + p0802*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0801*q0113 + p0802*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0008 + p0902*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0009 + p0902*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0010 + p0902*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0011 + p0902*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0901*q0112 + p0902*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0901*q0113 + p0902*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0008 + p1002*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0009 + p1002*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0010 + p1002*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1000*q0011 + p1002*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1001*q0112 + p1002*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1001*q0113 + p1002*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0008 + p1102*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0009 + p1102*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0010 + p1102*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1100*q0011 + p1102*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1101*q0112 + p1102*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1101*q0113 + p1102*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0008 + p1202*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0009 + p1202*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0010 + p1202*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1200*q0011 + p1202*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1201*q0112 + p1202*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1201*q0113 + p1202*q0213)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0008 + p1302*q0208)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0009 + p1302*q0209)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0010 + p1302*q0210)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1300*q0011 + p1302*q0211)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1301*q0112 + p1302*q0212)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p1301*q0113 + p1302*q0213)

    free(vsini1x)
    free(vcosi1x)

    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)

    free(kLLq_2_q0008)
    free(kLLq_2_q0009)
    free(kLLq_2_q0010)
    free(kLLq_2_q0011)
    free(kLLq_2_q0112)
    free(kLLq_2_q0113)
    free(kLLq_2_q0208)
    free(kLLq_2_q0209)
    free(kLLq_2_q0210)
    free(kLLq_2_q0211)
    free(kLLq_2_q0212)
    free(kLLq_2_q0213)


