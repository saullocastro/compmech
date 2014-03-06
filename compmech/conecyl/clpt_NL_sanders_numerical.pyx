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

from compmech.conecyl.clpt_commons cimport cfv, cfN, cfuvw_x, cfuvw_t

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

cdef int num0 = 3
cdef int num1 = 3
cdef int num2 = 6
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

    npts = (2*nx+1)*(2*ny+1)
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
    #
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] k0Lv

    cdef int fdim
    cdef cc_attributes args

    num_cond_1 = 36
    num_cond_2 = 36
    num_cond_3 = 36
    num_cond_4 = 36
    k22_num = num_cond_1*m2*n2 + num_cond_2*(m2-1)*m2*n2 \
            + num_cond_3*(m2-1)*m2*(n2-1)*n2 + num_cond_4*m2*(n2-1)*n2

    fdim = 9 + 2*9*m1 + 2*18*m2*n2 + 9*m1**2 + 36*m1*m2*n2 + k22_num

    rows = np.zeros((fdim,), dtype=INT)
    cols = np.zeros((fdim,), dtype=INT)
    k0Lv = np.zeros((fdim,), dtype=DOUBLE)

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

    # k0L_00
    c += 1
    rows[c] = 0
    cols[c] = 0
    c += 1
    rows[c] = 0
    cols[c] = 1
    c += 1
    rows[c] = 0
    cols[c] = 2
    c += 1
    rows[c] = 1
    cols[c] = 0
    c += 1
    rows[c] = 1
    cols[c] = 1
    c += 1
    rows[c] = 1
    cols[c] = 2
    c += 1
    rows[c] = 2
    cols[c] = 0
    c += 1
    rows[c] = 2
    cols[c] = 1
    c += 1
    rows[c] = 2
    cols[c] = 2

    for k1 in range(1, m1+1):
        col = (k1-1)*num1 + num0
        # k0L_01
        c += 1
        rows[c] = 0
        cols[c] = col+0
        c += 1
        rows[c] = 0
        cols[c] = col+1
        c += 1
        rows[c] = 0
        cols[c] = col+2
        c += 1
        rows[c] = 1
        cols[c] = col+0
        c += 1
        rows[c] = 1
        cols[c] = col+1
        c += 1
        rows[c] = 1
        cols[c] = col+2
        c += 1
        rows[c] = 2
        cols[c] = col+0
        c += 1
        rows[c] = 2
        cols[c] = col+1
        c += 1
        rows[c] = 2
        cols[c] = col+2

    for k2 in range(1, m2+1):
        for l2 in range(1, n2+1):
            col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
            # k0L_02
            c += 1
            rows[c] = 0
            cols[c] = col+0
            c += 1
            rows[c] = 0
            cols[c] = col+1
            c += 1
            rows[c] = 0
            cols[c] = col+2
            c += 1
            rows[c] = 0
            cols[c] = col+3
            c += 1
            rows[c] = 0
            cols[c] = col+4
            c += 1
            rows[c] = 0
            cols[c] = col+5
            c += 1
            rows[c] = 1
            cols[c] = col+0
            c += 1
            rows[c] = 1
            cols[c] = col+1
            c += 1
            rows[c] = 1
            cols[c] = col+2
            c += 1
            rows[c] = 1
            cols[c] = col+3
            c += 1
            rows[c] = 1
            cols[c] = col+4
            c += 1
            rows[c] = 1
            cols[c] = col+5
            c += 1
            rows[c] = 2
            cols[c] = col+0
            c += 1
            rows[c] = 2
            cols[c] = col+1
            c += 1
            rows[c] = 2
            cols[c] = col+2
            c += 1
            rows[c] = 2
            cols[c] = col+3
            c += 1
            rows[c] = 2
            cols[c] = col+4
            c += 1
            rows[c] = 2
            cols[c] = col+5

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        # k0L_10
        c += 1
        rows[c] = row+0
        cols[c] = 0
        c += 1
        rows[c] = row+0
        cols[c] = 1
        c += 1
        rows[c] = row+0
        cols[c] = 2
        c += 1
        rows[c] = row+1
        cols[c] = 0
        c += 1
        rows[c] = row+1
        cols[c] = 1
        c += 1
        rows[c] = row+1
        cols[c] = 2
        c += 1
        rows[c] = row+2
        cols[c] = 0
        c += 1
        rows[c] = row+2
        cols[c] = 1
        c += 1
        rows[c] = row+2
        cols[c] = 2

        for k1 in range(1, m1+1):
            col = (k1-1)*num1 + num0
            # k0L_11
            c += 1
            rows[c] = row+0
            cols[c] = col+0
            c += 1
            rows[c] = row+0
            cols[c] = col+1
            c += 1
            rows[c] = row+0
            cols[c] = col+2
            c += 1
            rows[c] = row+1
            cols[c] = col+0
            c += 1
            rows[c] = row+1
            cols[c] = col+1
            c += 1
            rows[c] = row+1
            cols[c] = col+2
            c += 1
            rows[c] = row+2
            cols[c] = col+0
            c += 1
            rows[c] = row+2
            cols[c] = col+1
            c += 1
            rows[c] = row+2
            cols[c] = col+2

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                # k0L_12
                c += 1
                rows[c] = row+0
                cols[c] = col+0
                c += 1
                rows[c] = row+0
                cols[c] = col+1
                c += 1
                rows[c] = row+0
                cols[c] = col+2
                c += 1
                rows[c] = row+0
                cols[c] = col+3
                c += 1
                rows[c] = row+0
                cols[c] = col+4
                c += 1
                rows[c] = row+0
                cols[c] = col+5
                c += 1
                rows[c] = row+1
                cols[c] = col+0
                c += 1
                rows[c] = row+1
                cols[c] = col+1
                c += 1
                rows[c] = row+1
                cols[c] = col+2
                c += 1
                rows[c] = row+1
                cols[c] = col+3
                c += 1
                rows[c] = row+1
                cols[c] = col+4
                c += 1
                rows[c] = row+1
                cols[c] = col+5
                c += 1
                rows[c] = row+2
                cols[c] = col+0
                c += 1
                rows[c] = row+2
                cols[c] = col+1
                c += 1
                rows[c] = row+2
                cols[c] = col+2
                c += 1
                rows[c] = row+2
                cols[c] = col+3
                c += 1
                rows[c] = row+2
                cols[c] = col+4
                c += 1
                rows[c] = row+2
                cols[c] = col+5

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            # k0L_20
            c += 1
            rows[c] = row+0
            cols[c] = 0
            c += 1
            rows[c] = row+0
            cols[c] = 1
            c += 1
            rows[c] = row+0
            cols[c] = 2
            c += 1
            rows[c] = row+1
            cols[c] = 0
            c += 1
            rows[c] = row+1
            cols[c] = 1
            c += 1
            rows[c] = row+1
            cols[c] = 2
            c += 1
            rows[c] = row+2
            cols[c] = 0
            c += 1
            rows[c] = row+2
            cols[c] = 1
            c += 1
            rows[c] = row+2
            cols[c] = 2
            c += 1
            rows[c] = row+3
            cols[c] = 0
            c += 1
            rows[c] = row+3
            cols[c] = 1
            c += 1
            rows[c] = row+3
            cols[c] = 2
            c += 1
            rows[c] = row+4
            cols[c] = 0
            c += 1
            rows[c] = row+4
            cols[c] = 1
            c += 1
            rows[c] = row+4
            cols[c] = 2
            c += 1
            rows[c] = row+5
            cols[c] = 0
            c += 1
            rows[c] = row+5
            cols[c] = 1
            c += 1
            rows[c] = row+5
            cols[c] = 2

            for k1 in range(1, m1+1):
                col = (k1-1)*num1 + num0
                # k0L_21
                c += 1
                rows[c] = row+0
                cols[c] = col+0
                c += 1
                rows[c] = row+0
                cols[c] = col+1
                c += 1
                rows[c] = row+0
                cols[c] = col+2
                c += 1
                rows[c] = row+1
                cols[c] = col+0
                c += 1
                rows[c] = row+1
                cols[c] = col+1
                c += 1
                rows[c] = row+1
                cols[c] = col+2
                c += 1
                rows[c] = row+2
                cols[c] = col+0
                c += 1
                rows[c] = row+2
                cols[c] = col+1
                c += 1
                rows[c] = row+2
                cols[c] = col+2
                c += 1
                rows[c] = row+3
                cols[c] = col+0
                c += 1
                rows[c] = row+3
                cols[c] = col+1
                c += 1
                rows[c] = row+3
                cols[c] = col+2
                c += 1
                rows[c] = row+4
                cols[c] = col+0
                c += 1
                rows[c] = row+4
                cols[c] = col+1
                c += 1
                rows[c] = row+4
                cols[c] = col+2
                c += 1
                rows[c] = row+5
                cols[c] = col+0
                c += 1
                rows[c] = row+5
                cols[c] = col+1
                c += 1
                rows[c] = row+5
                cols[c] = col+2

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                    # k0L_22
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+5

    size = num0 + num1*m1 + num2*m2*n2

    k0L = coo_matrix((k0Lv, (rows, cols)), shape=(size, size))

    return k0L

cdef void cfk0L(int npts, double *xs, double *ts, double *out,
                double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2
    cdef int c, i, pos

    cdef double uvw[3]

    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef double p00, p01, p02, p10, p11, p12
    cdef double p20, p21, p22, p30, p31, p32
    cdef double p40, p41, p42, p50, p51, p52
    cdef double q01, q02, q03, q04, q05, q10, q11, q12, q13, q14, q15
    cdef double q20, q21, q22, q23, q24, q25

    cdef double r, x, t, alpha, beta

    cdef double *F, *coeffs
    cdef double  r2, L, sina, cosa, tLA
    cdef int m1, m2, n2, pti
    cdef double wx, wt, ux, ut, vx, v

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

    A11 = F[0]  # F[0,0]
    A12 = F[1]  # F[0,1]
    A16 = F[2]  # F[0,2]
    A22 = F[7]  # F[1,1]
    A26 = F[8]  # F[1,2]
    A66 = F[14] # F[2,2]
    B11 = F[3]  # F[0,3]
    B12 = F[4]  # F[0,4]
    B16 = F[5]  # F[0,5]
    B22 = F[10] # F[1,4]
    B26 = F[11] # F[1,5]
    B66 = F[17] # F[2,5]
    D11 = F[21] # F[3,3]
    D12 = F[22] # F[3.4]
    D16 = F[23] # F[3,5]
    D22 = F[28] # F[4,4]
    D26 = F[29] # F[4,5]
    D66 = F[35] # F[5,5]

    cdef double sini1x, cosi1x, sink1x, cosk1x
    cdef double sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t
    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *k0Lq_1_q01 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q02 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q11 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q20 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q21 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q22 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_2_q02 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q03 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q04 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q05 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q10 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q11 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q12 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q13 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q14 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q15 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q20 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q21 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q22 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q23 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q24 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q25 = <double *>malloc(m2*n2 * sizeof(double))

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

        cfv(coeffs, m1, m2, n2, r2, x, t, L, &v)

        cfuvw_x(coeffs, m1, m2, n2, r2, L, x, t, cosa, tLA, uvw)
        ux = uvw[0]
        vx = uvw[1]
        wx = uvw[2]

        cfuvw_t(coeffs, m1, m2, n2, L, x, t, cosa, tLA, uvw)
        ut = uvw[0]
        wt = uvw[2]

        c = -1

        p00 = (-A11*r + A12*sina*(L - x))/(L*cosa)
        p01 = (-A12*r + A22*sina*(L - x))/(L*cosa)
        p02 = (-A16*r + A26*sina*(L - x))/(L*cosa)
        p10 = -r2*(r + sina*(L - x))*(A16*r + B16)/(L*r)
        p11 = -r2*(r + sina*(L - x))*(A26*r + B26)/(L*r)
        p12 = -r2*(r + sina*(L - x))*(A66*r + B66)/(L*r)
        p20 = (A16*(L - x)*sin(t - tLA) + (A11*r + A12*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        p21 = (A26*(L - x)*sin(t - tLA) + (A12*r + A22*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        p22 = (A66*(L - x)*sin(t - tLA) + (A16*r + A26*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)

        q01 = -r2*vx/L
        q11 = -r2*(L - x)*(cosa*wt + sina*ut - v)/(L*r*r)
        q12 = sina*v*(-L + x)*sin(t - tLA)/(L*cosa*r*r)
        q20 = sina*v/(L*cosa*r)
        q21 = -r2*(L - x)*(cosa*wx + sina*ux)/(L*r)
        q22 = -sina*v*(cos(t - tLA) - 1)/(L*cosa*r)

        # k0L_00
        c += 1
        out[c] = beta*out[c] + alpha*(p02*q20)
        c += 1
        out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p01*q12 + p02*q22)
        c += 1
        out[c] = beta*out[c] + alpha*(p12*q20)
        c += 1
        out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
        c += 1
        out[c] = beta*out[c] + alpha*(p22*q20)
        c += 1
        out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)

        for k1 in range(1, m1+1):
            sink1x = vsini1x[k1-1]
            cosk1x = vcosi1x[k1-1]
            q01 = pi*cosk1x*k1*vx/L
            q02 = pi*cosk1x*k1*wx/L
            q11 = sink1x*(-cosa*wt - sina*ut + v)/(r*r)
            q20 = -pi*cosk1x*k1*sina*v/(L*r)
            q21 = -sink1x*(cosa*wx + sina*ux)/r
            q22 = pi*cosk1x*k1*(-cosa*v + wt)/(L*r)

            # k0L_01
            c += 1
            out[c] = beta*out[c] + alpha*(p02*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p12*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p22*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            # create buffer
            k0Lq_1_q01[k1-1] = q01
            k0Lq_1_q02[k1-1] = q02
            k0Lq_1_q11[k1-1] = q11
            k0Lq_1_q20[k1-1] = q20
            k0Lq_1_q21[k1-1] = q21
            k0Lq_1_q22[k1-1] = q22

        for k2 in range(1, m2+1):
            sink2x = vsini2x[k2-1]
            cosk2x = vcosi2x[k2-1]
            for l2 in range(1, n2+1):
                sinl2t = vsinj2t[l2-1]
                cosl2t = vcosj2t[l2-1]
                q02 = pi*cosk2x*k2*sinl2t*vx/L
                q03 = pi*cosk2x*cosl2t*k2*vx/L
                q04 = pi*cosk2x*k2*sinl2t*wx/L
                q05 = pi*cosk2x*cosl2t*k2*wx/L
                q10 = -cosl2t*l2*sina*sink2x*v/(r*r)
                q11 = l2*sina*sink2x*sinl2t*v/(r*r)
                q12 = sink2x*sinl2t*(-cosa*wt - sina*ut + v)/(r*r)
                q13 = cosl2t*sink2x*(-cosa*wt - sina*ut + v)/(r*r)
                q14 = cosl2t*l2*sink2x*(-cosa*v + wt)/(r*r)
                q15 = l2*sink2x*sinl2t*(cosa*v - wt)/(r*r)
                q20 = -pi*cosk2x*k2*sina*sinl2t*v/(L*r)
                q21 = -pi*cosk2x*cosl2t*k2*sina*v/(L*r)
                q22 = -sink2x*sinl2t*(cosa*wx + sina*ux)/r
                q23 = -cosl2t*sink2x*(cosa*wx + sina*ux)/r
                q24 = (L*cosl2t*l2*sink2x*wx + pi*cosk2x*k2*sinl2t*(-cosa*v + wt))/(L*r)
                q25 = (-L*l2*sink2x*sinl2t*wx + pi*cosk2x*cosl2t*k2*(-cosa*v + wt))/(L*r)

                # k0L_02
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q10 + p02*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q02 + p01*q12 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q03 + p01*q13 + p02*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q10 + p12*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02 + p11*q12 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q03 + p11*q13 + p12*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q10 + p22*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12 + p22*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q03 + p21*q13 + p22*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)

                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                k0Lq_2_q02[pos] = q02
                k0Lq_2_q03[pos] = q03
                k0Lq_2_q04[pos] = q04
                k0Lq_2_q05[pos] = q05
                k0Lq_2_q10[pos] = q10
                k0Lq_2_q11[pos] = q11
                k0Lq_2_q12[pos] = q12
                k0Lq_2_q13[pos] = q13
                k0Lq_2_q14[pos] = q14
                k0Lq_2_q15[pos] = q15
                k0Lq_2_q20[pos] = q20
                k0Lq_2_q21[pos] = q21
                k0Lq_2_q22[pos] = q22
                k0Lq_2_q23[pos] = q23
                k0Lq_2_q24[pos] = q24
                k0Lq_2_q25[pos] = q25

        for i1 in range(1, m1+1):
            sini1x = vsini1x[i1-1]
            cosi1x = vcosi1x[i1-1]
            p00 = pi*A11*cosi1x*i1*r/L + A12*sina*sini1x
            p01 = pi*A12*cosi1x*i1*r/L + A22*sina*sini1x
            p02 = pi*A16*cosi1x*i1*r/L + A26*sina*sini1x
            p10 = (A16*r + B16)*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*r)
            p11 = (A26*r + B26)*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*r)
            p12 = (A66*r + B66)*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*r)
            p20 = (-pi*B12*L*cosi1x*i1*sina + sini1x*(A12*(L*L)*cosa + (pi*pi)*B11*(i1*i1)*r))/(L*L)
            p21 = (-pi*B22*L*cosi1x*i1*sina + sini1x*(A22*(L*L)*cosa + (pi*pi)*B12*(i1*i1)*r))/(L*L)
            p22 = (-pi*B26*L*cosi1x*i1*sina + sini1x*(A26*(L*L)*cosa + (pi*pi)*B16*(i1*i1)*r))/(L*L)

            q01 = -r2*vx/L
            q11 = -r2*(L - x)*(cosa*wt + sina*ut - v)/(L*r*r)
            q12 = sina*v*(-L + x)*sin(t - tLA)/(L*cosa*r*r)
            q20 = sina*v/(L*cosa*r)
            q21 = -r2*(L - x)*(cosa*wx + sina*ux)/(L*r)
            q22 = -sina*v*(cos(t - tLA) - 1)/(L*cosa*r)

            # k0L_10
            c += 1
            out[c] = beta*out[c] + alpha*(p02*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p01*q12 + p02*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p12*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p22*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)

            for k1 in range(1, m1+1):
                # access buffer
                q01 = k0Lq_1_q01[k1-1]
                q02 = k0Lq_1_q02[k1-1]
                q11 = k0Lq_1_q11[k1-1]
                q20 = k0Lq_1_q20[k1-1]
                q21 = k0Lq_1_q21[k1-1]
                q22 = k0Lq_1_q22[k1-1]

                # k0L_11
                c += 1
                out[c] = beta*out[c] + alpha*(p02*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p22*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q02 = k0Lq_2_q02[pos]
                    q03 = k0Lq_2_q03[pos]
                    q04 = k0Lq_2_q04[pos]
                    q05 = k0Lq_2_q05[pos]
                    q10 = k0Lq_2_q10[pos]
                    q11 = k0Lq_2_q11[pos]
                    q12 = k0Lq_2_q12[pos]
                    q13 = k0Lq_2_q13[pos]
                    q14 = k0Lq_2_q14[pos]
                    q15 = k0Lq_2_q15[pos]
                    q20 = k0Lq_2_q20[pos]
                    q21 = k0Lq_2_q21[pos]
                    q22 = k0Lq_2_q22[pos]
                    q23 = k0Lq_2_q23[pos]
                    q24 = k0Lq_2_q24[pos]
                    q25 = k0Lq_2_q25[pos]

                    # k0L_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p01*q10 + p02*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q02 + p01*q12 + p02*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q03 + p01*q13 + p02*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q10 + p12*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q02 + p11*q12 + p12*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q03 + p11*q13 + p12*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q10 + p22*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12 + p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q03 + p21*q13 + p22*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)

        for i2 in range(1, m2+1):
            sini2x = vsini2x[i2-1]
            cosi2x = vcosi2x[i2-1]
            for j2 in range(1, n2+1):
                sinj2t = vsinj2t[j2-1]
                cosj2t = vcosj2t[j2-1]
                p00 = pi*A11*cosi2x*i2*r*sinj2t/L + sini2x*(A12*sina*sinj2t + A16*cosj2t*j2)
                p01 = pi*A12*cosi2x*i2*r*sinj2t/L + sini2x*(A22*sina*sinj2t + A26*cosj2t*j2)
                p02 = pi*A16*cosi2x*i2*r*sinj2t/L + sini2x*(A26*sina*sinj2t + A66*cosj2t*j2)
                p10 = pi*A11*cosi2x*cosj2t*i2*r/L + sini2x*(A12*cosj2t*sina - A16*j2*sinj2t)
                p11 = pi*A12*cosi2x*cosj2t*i2*r/L + sini2x*(A22*cosj2t*sina - A26*j2*sinj2t)
                p12 = pi*A16*cosi2x*cosj2t*i2*r/L + sini2x*(A26*cosj2t*sina - A66*j2*sinj2t)
                p20 = (L*sini2x*(cosj2t*j2*(A12*r + B12) - sina*sinj2t*(A16*r + B16)) + pi*cosi2x*i2*r*sinj2t*(A16*r + B16))/(L*r)
                p21 = (L*sini2x*(cosj2t*j2*(A22*r + B22) - sina*sinj2t*(A26*r + B26)) + pi*cosi2x*i2*r*sinj2t*(A26*r + B26))/(L*r)
                p22 = (L*sini2x*(cosj2t*j2*(A26*r + B26) - sina*sinj2t*(A66*r + B66)) + pi*cosi2x*i2*r*sinj2t*(A66*r + B66))/(L*r)
                p30 = (-L*sini2x*(cosj2t*sina*(A16*r + B16) + j2*sinj2t*(A12*r + B12)) + pi*cosi2x*cosj2t*i2*r*(A16*r + B16))/(L*r)
                p31 = (-L*sini2x*(cosj2t*sina*(A26*r + B26) + j2*sinj2t*(A22*r + B22)) + pi*cosi2x*cosj2t*i2*r*(A26*r + B26))/(L*r)
                p32 = (-L*sini2x*(cosj2t*sina*(A66*r + B66) + j2*sinj2t*(A26*r + B26)) + pi*cosi2x*cosj2t*i2*r*(A66*r + B66))/(L*r)
                p40 = (-pi*L*cosi2x*i2*r*(B12*sina*sinj2t + 2*B16*cosj2t*j2) + sini2x*(B16*(L*L)*cosj2t*j2*sina + sinj2t*(B12*(L*L)*(j2*j2) + r*(A12*(L*L)*cosa + (pi*pi)*B11*(i2*i2)*r))))/((L*L)*r)
                p41 = (-pi*L*cosi2x*i2*r*(B22*sina*sinj2t + 2*B26*cosj2t*j2) + sini2x*(B26*(L*L)*cosj2t*j2*sina + sinj2t*(B22*(L*L)*(j2*j2) + r*(A22*(L*L)*cosa + (pi*pi)*B12*(i2*i2)*r))))/((L*L)*r)
                p42 = (-pi*L*cosi2x*i2*r*(B26*sina*sinj2t + 2*B66*cosj2t*j2) + sini2x*(B66*(L*L)*cosj2t*j2*sina + sinj2t*(B26*(L*L)*(j2*j2) + r*(A26*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/((L*L)*r)
                p50 = (pi*L*cosi2x*i2*r*(-B12*cosj2t*sina + 2*B16*j2*sinj2t) + sini2x*(-B16*(L*L)*j2*sina*sinj2t + cosj2t*(B12*(L*L)*(j2*j2) + r*(A12*(L*L)*cosa + (pi*pi)*B11*(i2*i2)*r))))/((L*L)*r)
                p51 = (pi*L*cosi2x*i2*r*(-B22*cosj2t*sina + 2*B26*j2*sinj2t) + sini2x*(-B26*(L*L)*j2*sina*sinj2t + cosj2t*(B22*(L*L)*(j2*j2) + r*(A22*(L*L)*cosa + (pi*pi)*B12*(i2*i2)*r))))/((L*L)*r)
                p52 = (pi*L*cosi2x*i2*r*(-B26*cosj2t*sina + 2*B66*j2*sinj2t) + sini2x*(-B66*(L*L)*j2*sina*sinj2t + cosj2t*(B26*(L*L)*(j2*j2) + r*(A26*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/((L*L)*r)

                q01 = -r2*vx/L
                q11 = -r2*(L - x)*(cosa*wt + sina*ut - v)/(L*r*r)
                q12 = sina*v*(-L + x)*sin(t - tLA)/(L*cosa*r*r)
                q20 = sina*v/(L*cosa*r)
                q21 = -r2*(L - x)*(cosa*wx + sina*ux)/(L*r)
                q22 = -sina*v*(cos(t - tLA) - 1)/(L*cosa*r)

                # k0L_20
                c += 1
                out[c] = beta*out[c] + alpha*(p02*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q12 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p22*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p32*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p30*q01 + p31*q11 + p32*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p31*q12 + p32*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p42*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p40*q01 + p41*q11 + p42*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p41*q12 + p42*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p52*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p50*q01 + p51*q11 + p52*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p51*q12 + p52*q22)

                for k1 in range(1, m1+1):
                    # access buffer
                    q01 = k0Lq_1_q01[k1-1]
                    q02 = k0Lq_1_q02[k1-1]
                    q11 = k0Lq_1_q11[k1-1]
                    q20 = k0Lq_1_q20[k1-1]
                    q21 = k0Lq_1_q21[k1-1]
                    q22 = k0Lq_1_q22[k1-1]

                    # k0L_21
                    c += 1
                    out[c] = beta*out[c] + alpha*(p02*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p12*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p22*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p32*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p30*q01 + p31*q11 + p32*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p30*q02 + p32*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p42*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p40*q01 + p41*q11 + p42*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p40*q02 + p42*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p52*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p50*q01 + p51*q11 + p52*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p50*q02 + p52*q22)

                for k2 in range(1, m2+1):
                    for l2 in range(1, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q02 = k0Lq_2_q02[pos]
                        q03 = k0Lq_2_q03[pos]
                        q04 = k0Lq_2_q04[pos]
                        q05 = k0Lq_2_q05[pos]
                        q10 = k0Lq_2_q10[pos]
                        q11 = k0Lq_2_q11[pos]
                        q12 = k0Lq_2_q12[pos]
                        q13 = k0Lq_2_q13[pos]
                        q14 = k0Lq_2_q14[pos]
                        q15 = k0Lq_2_q15[pos]
                        q20 = k0Lq_2_q20[pos]
                        q21 = k0Lq_2_q21[pos]
                        q22 = k0Lq_2_q22[pos]
                        q23 = k0Lq_2_q23[pos]
                        q24 = k0Lq_2_q24[pos]
                        q25 = k0Lq_2_q25[pos]

                        # k0L_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p01*q10 + p02*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q02 + p01*q12 + p02*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q03 + p01*q13 + p02*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p11*q10 + p12*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q02 + p11*q12 + p12*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q03 + p11*q13 + p12*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p21*q10 + p22*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12 + p22*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q03 + p21*q13 + p22*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p31*q10 + p32*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p31*q11 + p32*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q02 + p31*q12 + p32*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q03 + p31*q13 + p32*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q04 + p31*q14 + p32*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q05 + p31*q15 + p32*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p41*q10 + p42*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p41*q11 + p42*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q02 + p41*q12 + p42*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q03 + p41*q13 + p42*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14 + p42*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15 + p42*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p51*q10 + p52*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p51*q11 + p52*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q02 + p51*q12 + p52*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q03 + p51*q13 + p52*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q04 + p51*q14 + p52*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q05 + p51*q15 + p52*q25)

    free(vsini1x)
    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
    free(k0Lq_1_q01)
    free(k0Lq_1_q02)
    free(k0Lq_1_q11)
    free(k0Lq_1_q20)
    free(k0Lq_1_q21)
    free(k0Lq_1_q22)
    free(k0Lq_2_q02)
    free(k0Lq_2_q03)
    free(k0Lq_2_q04)
    free(k0Lq_2_q05)
    free(k0Lq_2_q10)
    free(k0Lq_2_q11)
    free(k0Lq_2_q12)
    free(k0Lq_2_q13)
    free(k0Lq_2_q14)
    free(k0Lq_2_q15)
    free(k0Lq_2_q20)
    free(k0Lq_2_q21)
    free(k0Lq_2_q22)
    free(k0Lq_2_q23)
    free(k0Lq_2_q24)
    free(k0Lq_2_q25)


def calc_kG(np.ndarray[cDOUBLE, ndim=1] coeffs,
              double alpharad, double r2, double L, double tLA,
              np.ndarray[cDOUBLE, ndim=2] F,
              int m1, int m2, int n2,
              int nx, int nt, int num_cores, str method='trapz2d'):
    #
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kGv
    cdef np.ndarray[cDOUBLE, ndim=2] tmp

    cdef unsigned fdim
    cdef cc_attributes args

    k11_cond_1 = 6
    k11_cond_2 = 6
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1

    k22_cond_1 = 24
    k22_cond_2 = 24
    k22_cond_3 = 24
    k22_cond_4 = 24
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 5 + 5*m1 + 10*m2*n2 + k11_num + 12*m1*m2*n2 + k22_num

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

    # kG_00
    c += 1
    rows[c] = 0
    cols[c] = 1
    c += 1
    rows[c] = 1
    cols[c] = 0
    c += 1
    rows[c] = 1
    cols[c] = 1
    c += 1
    rows[c] = 1
    cols[c] = 2
    c += 1
    rows[c] = 2
    cols[c] = 1

    for k1 in range(1, m1+1):
        col = (k1-1)*num1 + num0
        # kG_01
        c += 1
        rows[c] = 0
        cols[c] = col+1
        c += 1
        rows[c] = 1
        cols[c] = col+0
        c += 1
        rows[c] = 1
        cols[c] = col+1
        c += 1
        rows[c] = 1
        cols[c] = col+2
        c += 1
        rows[c] = 2
        cols[c] = col+1

    for k2 in range(1, m2+1):
        for l2 in range(1, n2+1):
            col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
            # kG_02
            c += 1
            rows[c] = 0
            cols[c] = col+2
            c += 1
            rows[c] = 0
            cols[c] = col+3
            c += 1
            rows[c] = 1
            cols[c] = col+0
            c += 1
            rows[c] = 1
            cols[c] = col+1
            c += 1
            rows[c] = 1
            cols[c] = col+2
            c += 1
            rows[c] = 1
            cols[c] = col+3
            c += 1
            rows[c] = 1
            cols[c] = col+4
            c += 1
            rows[c] = 1
            cols[c] = col+5
            c += 1
            rows[c] = 2
            cols[c] = col+2
            c += 1
            rows[c] = 2
            cols[c] = col+3

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        #NOTE symmetry
        for k1 in range(i1, m1+1):
            col = (k1-1)*num1 + num0
            # kG_11
            c += 1
            rows[c] = row+0
            cols[c] = col+1
            c += 1
            rows[c] = row+1
            cols[c] = col+0
            c += 1
            rows[c] = row+1
            cols[c] = col+1
            c += 1
            rows[c] = row+1
            cols[c] = col+2
            c += 1
            rows[c] = row+2
            cols[c] = col+1
            c += 1
            rows[c] = row+2
            cols[c] = col+2

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                # kG_12
                c += 1
                rows[c] = row+0
                cols[c] = col+2
                c += 1
                rows[c] = row+0
                cols[c] = col+3
                c += 1
                rows[c] = row+1
                cols[c] = col+0
                c += 1
                rows[c] = row+1
                cols[c] = col+1
                c += 1
                rows[c] = row+1
                cols[c] = col+2
                c += 1
                rows[c] = row+1
                cols[c] = col+3
                c += 1
                rows[c] = row+1
                cols[c] = col+4
                c += 1
                rows[c] = row+1
                cols[c] = col+5
                c += 1
                rows[c] = row+2
                cols[c] = col+2
                c += 1
                rows[c] = row+2
                cols[c] = col+3
                c += 1
                rows[c] = row+2
                cols[c] = col+4
                c += 1
                rows[c] = row+2
                cols[c] = col+5

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            #NOTE symmetry
            for k2 in range(i2, m2+1):
                for l2 in range(j2, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                    # kG_22
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+5

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

    cdef double q00, q01, q02, q10, q11, q12, q14, q20, q21, q22, q23
    cdef double q31, q32, q33, q42, q44, q45, q54, q55
    cdef double p02, p10, p11, p12, p13, p14, p15
    cdef double p20, p21, p22, p23, p24, p25, p30, p31, p32, p33, p34, p35
    cdef double p42, p44, p45, p52, p54, p55

    cdef double r, x, t, alpha, beta
    cdef int c, i, pos

    cdef double *F, *coeffs
    cdef double r2, L, sina, cosa, tLA
    cdef int m1, m2, n2, pti
    cdef double Nxx, Ntt, Nxt
    cdef double N[6]
    cdef int NL_kinematics=1 # to use cfstrain_sanders in cfN

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
    cdef double *kGq_1_q00 = <double *>malloc(m1 * sizeof(double))
    cdef double *kGq_1_q21 = <double *>malloc(m1 * sizeof(double))
    cdef double *kGq_1_q31 = <double *>malloc(m1 * sizeof(double))
    cdef double *kGq_1_q42 = <double *>malloc(m1 * sizeof(double))
    cdef double *kGq_2_q00 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q01 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q10 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q11 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q22 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q23 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q32 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q33 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q44 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q45 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q54 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q55 = <double *>malloc(m2*n2 * sizeof(double))

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

        p02 = Nxt*r*sina/(L*cosa)
        p10 = Nxt*r2*sina*(-L + x)/L
        p11 = Ntt*r2*sina*(-L + x)/L
        p12 = Ntt*r2*(L - x)/L
        p13 = -Nxx*r*r2/L
        p14 = Nxt*cosa*r2*(-L + x)/L
        p15 = Ntt*cosa*r2*(-L + x)/L
        p22 = (Ntt*sina*(-L + x)*sin(t - tLA) - Nxt*r*sina*(cos(t - tLA) - 1))/(L*cosa)

        q00 = -1/(L*cosa)
        q02 = (cos(t - tLA) - 1)/(L*cosa)
        q12 = (L - x)*sin(t - tLA)/(L*cosa*r)
        q21 = r2*(L - x)/(L*r)
        q31 = -r2/L

        # kG_00
        c += 1
        out[c] = beta*out[c] + alpha*(p02*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p10*q00)
        c += 1
        out[c] = beta*out[c] + alpha*(p12*q21 + p13*q31)
        c += 1
        out[c] = beta*out[c] + alpha*(p10*q02 + p11*q12)
        c += 1
        out[c] = beta*out[c] + alpha*(p22*q21)

        for k1 in range(1, m1+1):
            sink1x = vsini1x[k1-1]
            cosk1x = vcosi1x[k1-1]
            q00 = pi*cosk1x*k1/L
            q21 = sink1x/r
            q31 = pi*cosk1x*k1/L
            q42 = pi*cosk1x*k1/L

            # kG_01
            c += 1
            out[c] = beta*out[c] + alpha*(p02*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q00)
            c += 1
            out[c] = beta*out[c] + alpha*(p12*q21 + p13*q31)
            c += 1
            out[c] = beta*out[c] + alpha*(p14*q42)
            c += 1
            out[c] = beta*out[c] + alpha*(p22*q21)

            # create buffer
            kGq_1_q00[k1-1] = q00
            kGq_1_q21[k1-1] = q21
            kGq_1_q31[k1-1] = q31
            kGq_1_q42[k1-1] = q42

        for k2 in range(1, m2+1):
            sink2x = vsini2x[k2-1]
            cosk2x = vcosi2x[k2-1]
            for l2 in range(1, n2+1):
                sinl2t = vsinj2t[l2-1]
                cosl2t = vcosj2t[l2-1]
                q00 = pi*cosk2x*k2*sinl2t/L
                q01 = pi*cosk2x*cosl2t*k2/L
                q10 = cosl2t*l2*sink2x/r
                q11 = -l2*sink2x*sinl2t/r
                q22 = sink2x*sinl2t/r
                q23 = cosl2t*sink2x/r
                q32 = pi*cosk2x*k2*sinl2t/L
                q33 = pi*cosk2x*cosl2t*k2/L
                q44 = pi*cosk2x*k2*sinl2t/L
                q45 = pi*cosk2x*cosl2t*k2/L
                q54 = cosl2t*l2*sink2x/r
                q55 = -l2*sink2x*sinl2t/r

                # kG_02
                c += 1
                out[c] = beta*out[c] + alpha*(p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p02*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q00 + p11*q10)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q22 + p13*q32)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q23 + p13*q33)
                c += 1
                out[c] = beta*out[c] + alpha*(p14*q44 + p15*q54)
                c += 1
                out[c] = beta*out[c] + alpha*(p14*q45 + p15*q55)
                c += 1
                out[c] = beta*out[c] + alpha*(p22*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p22*q23)

                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                kGq_2_q00[pos] = q00
                kGq_2_q01[pos] = q01
                kGq_2_q10[pos] = q10
                kGq_2_q11[pos] = q11
                kGq_2_q22[pos] = q22
                kGq_2_q23[pos] = q23
                kGq_2_q32[pos] = q32
                kGq_2_q33[pos] = q33
                kGq_2_q44[pos] = q44
                kGq_2_q45[pos] = q45
                kGq_2_q54[pos] = q54
                kGq_2_q55[pos] = q55

        for i1 in range(1, m1+1):
            sini1x = vsini1x[i1-1]
            cosi1x = vcosi1x[i1-1]
            p02 = -pi*Nxt*cosi1x*i1*r*sina/L
            p10 = -Nxt*sina*sini1x
            p11 = -Ntt*sina*sini1x
            p12 = Ntt*sini1x
            p13 = pi*Nxx*cosi1x*i1*r/L
            p14 = -Nxt*cosa*sini1x
            p15 = -Ntt*cosa*sini1x
            p22 = -pi*Nxt*cosa*cosi1x*i1*r/L
            p24 = pi*Nxx*cosi1x*i1*r/L
            p25 = pi*Nxt*cosi1x*i1*r/L

            #NOTE symmetry
            for k1 in range(i1, m1+1):
                # access buffer
                q00 = kGq_1_q00[k1-1]
                q21 = kGq_1_q21[k1-1]
                q31 = kGq_1_q31[k1-1]
                q42 = kGq_1_q42[k1-1]

                # kG_11
                c += 1
                out[c] = beta*out[c] + alpha*(p02*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q00)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q21 + p13*q31)
                c += 1
                out[c] = beta*out[c] + alpha*(p14*q42)
                c += 1
                out[c] = beta*out[c] + alpha*(p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p24*q42)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q00 = kGq_2_q00[pos]
                    q01 = kGq_2_q01[pos]
                    q10 = kGq_2_q10[pos]
                    q11 = kGq_2_q11[pos]
                    q22 = kGq_2_q22[pos]
                    q23 = kGq_2_q23[pos]
                    q32 = kGq_2_q32[pos]
                    q33 = kGq_2_q33[pos]
                    q44 = kGq_2_q44[pos]
                    q45 = kGq_2_q45[pos]
                    q54 = kGq_2_q54[pos]
                    q55 = kGq_2_q55[pos]

                    # kG_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p02*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p02*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q00 + p11*q10)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p12*q22 + p13*q32)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p12*q23 + p13*q33)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p14*q44 + p15*q54)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p14*q45 + p15*q55)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p22*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p24*q44 + p25*q54)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p24*q45 + p25*q55)

        for i2 in range(1, m2+1):
            sini2x = vsini2x[i2-1]
            cosi2x = vcosi2x[i2-1]
            for j2 in range(1, n2+1):
                sinj2t = vsinj2t[j2-1]
                cosj2t = vcosj2t[j2-1]
                p02 = -sina*(L*Ntt*cosj2t*j2*sini2x + pi*Nxt*cosi2x*i2*r*sinj2t)/L
                p12 = Ntt*j2*sina*sini2x*sinj2t - pi*Nxt*cosi2x*cosj2t*i2*r*sina/L
                p20 = -Nxt*sina*sini2x*sinj2t
                p21 = -Ntt*sina*sini2x*sinj2t
                p22 = Ntt*sini2x*sinj2t
                p23 = pi*Nxx*cosi2x*i2*r*sinj2t/L
                p24 = -Nxt*cosa*sini2x*sinj2t
                p25 = -Ntt*cosa*sini2x*sinj2t
                p30 = -Nxt*cosj2t*sina*sini2x
                p31 = -Ntt*cosj2t*sina*sini2x
                p32 = Ntt*cosj2t*sini2x
                p33 = pi*Nxx*cosi2x*cosj2t*i2*r/L
                p34 = -Nxt*cosa*cosj2t*sini2x
                p35 = -Ntt*cosa*cosj2t*sini2x
                p42 = -cosa*(L*Ntt*cosj2t*j2*sini2x + pi*Nxt*cosi2x*i2*r*sinj2t)/L
                p44 = Nxt*cosj2t*j2*sini2x + pi*Nxx*cosi2x*i2*r*sinj2t/L
                p45 = Ntt*cosj2t*j2*sini2x + pi*Nxt*cosi2x*i2*r*sinj2t/L
                p52 = Ntt*cosa*j2*sini2x*sinj2t - pi*Nxt*cosa*cosi2x*cosj2t*i2*r/L
                p54 = -Nxt*j2*sini2x*sinj2t + pi*Nxx*cosi2x*cosj2t*i2*r/L
                p55 = -Ntt*j2*sini2x*sinj2t + pi*Nxt*cosi2x*cosj2t*i2*r/L

                #NOTE symmetry
                for k2 in range(i2, m2+1):
                    for l2 in range(j2, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q00 = kGq_2_q00[pos]
                        q01 = kGq_2_q01[pos]
                        q10 = kGq_2_q10[pos]
                        q11 = kGq_2_q11[pos]
                        q22 = kGq_2_q22[pos]
                        q23 = kGq_2_q23[pos]
                        q32 = kGq_2_q32[pos]
                        q33 = kGq_2_q33[pos]
                        q44 = kGq_2_q44[pos]
                        q45 = kGq_2_q45[pos]
                        q54 = kGq_2_q54[pos]
                        q55 = kGq_2_q55[pos]

                        # kG_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p02*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p02*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p12*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p12*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q00 + p21*q10)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p22*q22 + p23*q32)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p22*q23 + p23*q33)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p24*q44 + p25*q54)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p24*q45 + p25*q55)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q00 + p31*q10)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q01 + p31*q11)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p32*q22 + p33*q32)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p32*q23 + p33*q33)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p34*q44 + p35*q54)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p34*q45 + p35*q55)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p42*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p42*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p44*q44 + p45*q54)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p44*q45 + p45*q55)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p52*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p52*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p54*q44 + p55*q54)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p54*q45 + p55*q55)

    free(vsini1x)
    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
    free(kGq_1_q00)
    free(kGq_1_q21)
    free(kGq_1_q31)
    free(kGq_1_q42)
    free(kGq_2_q00)
    free(kGq_2_q01)
    free(kGq_2_q10)
    free(kGq_2_q11)
    free(kGq_2_q22)
    free(kGq_2_q23)
    free(kGq_2_q32)
    free(kGq_2_q33)
    free(kGq_2_q44)
    free(kGq_2_q45)
    free(kGq_2_q54)
    free(kGq_2_q55)

def calc_kLL(np.ndarray[cDOUBLE, ndim=1] coeffs,
             double alpharad, double r2, double L, double tLA,
             np.ndarray[cDOUBLE, ndim=2] F,
             int m1, int m2, int n2,
             int nx, int nt, int num_cores, str method='trapz2d'):
    #
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kLLv
    cdef np.ndarray[cDOUBLE, ndim=2] tmp

    cdef int fdim
    cdef cc_attributes args

    k22_cond_1 = 36
    k22_cond_2 = 36
    k22_cond_3 = 36
    k22_cond_4 = 36
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 9 + 9*m1 + 18*m2*n2 + 9*m1**2 + 18*m1*m2*n2 + k22_num

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

    # kLL_00
    c += 1
    rows[c] = 0
    cols[c] = 0
    c += 1
    rows[c] = 0
    cols[c] = 1
    c += 1
    rows[c] = 0
    cols[c] = 2
    c += 1
    rows[c] = 1
    cols[c] = 0
    c += 1
    rows[c] = 1
    cols[c] = 1
    c += 1
    rows[c] = 1
    cols[c] = 2
    c += 1
    rows[c] = 2
    cols[c] = 0
    c += 1
    rows[c] = 2
    cols[c] = 1
    c += 1
    rows[c] = 2
    cols[c] = 2

    for k1 in range(1, m1+1):
        col = (k1-1)*num1 + num0
        # kLL_01
        c += 1
        rows[c] = 0
        cols[c] = col+0
        c += 1
        rows[c] = 0
        cols[c] = col+1
        c += 1
        rows[c] = 0
        cols[c] = col+2
        c += 1
        rows[c] = 1
        cols[c] = col+0
        c += 1
        rows[c] = 1
        cols[c] = col+1
        c += 1
        rows[c] = 1
        cols[c] = col+2
        c += 1
        rows[c] = 2
        cols[c] = col+0
        c += 1
        rows[c] = 2
        cols[c] = col+1
        c += 1
        rows[c] = 2
        cols[c] = col+2

    for k2 in range(1, m2+1):
        for l2 in range(1, n2+1):
            col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
            # kLL_02
            c += 1
            rows[c] = 0
            cols[c] = col+0
            c += 1
            rows[c] = 0
            cols[c] = col+1
            c += 1
            rows[c] = 0
            cols[c] = col+2
            c += 1
            rows[c] = 0
            cols[c] = col+3
            c += 1
            rows[c] = 0
            cols[c] = col+4
            c += 1
            rows[c] = 0
            cols[c] = col+5
            c += 1
            rows[c] = 1
            cols[c] = col+0
            c += 1
            rows[c] = 1
            cols[c] = col+1
            c += 1
            rows[c] = 1
            cols[c] = col+2
            c += 1
            rows[c] = 1
            cols[c] = col+3
            c += 1
            rows[c] = 1
            cols[c] = col+4
            c += 1
            rows[c] = 1
            cols[c] = col+5
            c += 1
            rows[c] = 2
            cols[c] = col+0
            c += 1
            rows[c] = 2
            cols[c] = col+1
            c += 1
            rows[c] = 2
            cols[c] = col+2
            c += 1
            rows[c] = 2
            cols[c] = col+3
            c += 1
            rows[c] = 2
            cols[c] = col+4
            c += 1
            rows[c] = 2
            cols[c] = col+5

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        #NOTE symmetry
        for k1 in range(i1, m1+1):
            col = (k1-1)*num1 + num0
            # kLL_11
            c += 1
            rows[c] = row+0
            cols[c] = col+0
            c += 1
            rows[c] = row+0
            cols[c] = col+1
            c += 1
            rows[c] = row+0
            cols[c] = col+2
            c += 1
            rows[c] = row+1
            cols[c] = col+0
            c += 1
            rows[c] = row+1
            cols[c] = col+1
            c += 1
            rows[c] = row+1
            cols[c] = col+2
            c += 1
            rows[c] = row+2
            cols[c] = col+0
            c += 1
            rows[c] = row+2
            cols[c] = col+1
            c += 1
            rows[c] = row+2
            cols[c] = col+2

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                # kLL_12
                c += 1
                rows[c] = row+0
                cols[c] = col+0
                c += 1
                rows[c] = row+0
                cols[c] = col+1
                c += 1
                rows[c] = row+0
                cols[c] = col+2
                c += 1
                rows[c] = row+0
                cols[c] = col+3
                c += 1
                rows[c] = row+0
                cols[c] = col+4
                c += 1
                rows[c] = row+0
                cols[c] = col+5
                c += 1
                rows[c] = row+1
                cols[c] = col+0
                c += 1
                rows[c] = row+1
                cols[c] = col+1
                c += 1
                rows[c] = row+1
                cols[c] = col+2
                c += 1
                rows[c] = row+1
                cols[c] = col+3
                c += 1
                rows[c] = row+1
                cols[c] = col+4
                c += 1
                rows[c] = row+1
                cols[c] = col+5
                c += 1
                rows[c] = row+2
                cols[c] = col+0
                c += 1
                rows[c] = row+2
                cols[c] = col+1
                c += 1
                rows[c] = row+2
                cols[c] = col+2
                c += 1
                rows[c] = row+2
                cols[c] = col+3
                c += 1
                rows[c] = row+2
                cols[c] = col+4
                c += 1
                rows[c] = row+2
                cols[c] = col+5

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            #NOTE symmetry
            for k2 in range(i2, m2+1):
                for l2 in range(j2, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                    # kLL_22
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+0
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+1
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+3
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+5
                    cols[c] = col+5

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

    cdef double uvw[3]

    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66

    cdef double p00, p01, p02, p10, p11, p12
    cdef double p20, p21, p22, p30, p31, p32, p40, p41, p42
    cdef double p50, p51, p52
    cdef double q01, q02, q03, q04, q05, q10, q11, q12, q13, q14, q15
    cdef double q20, q21, q22, q23, q24, q25

    cdef double r, x, t, alpha, beta

    cdef double *F, *coeffs
    cdef double r2, L, sina, cosa, tLA
    cdef int m1, m2, n2, pti
    cdef double ux, ut, vx, v, wx, wt

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

    A11 = F[0]  # F[0,0]
    A12 = F[1]  # F[0,1]
    A16 = F[2]  # F[0,2]
    A22 = F[7]  # F[1,1]
    A26 = F[8]  # F[1,2]
    A66 = F[14] # F[2,2]
    B11 = F[3]  # F[0,3]
    B12 = F[4]  # F[0,4]
    B16 = F[5]  # F[0,5]
    B22 = F[10] # F[1,4]
    B26 = F[11] # F[1,5]
    B66 = F[17] # F[2,5]

    cdef double sini1x, cosi1x, sink1x, cosk1x, sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t
    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *kLLq_1_q01 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_1_q02 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_1_q11 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_1_q20 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_1_q21 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_1_q22 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_2_q02 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q03 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q04 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q05 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q10 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q11 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q12 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q13 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q14 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q15 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q20 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q21 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q22 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q23 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q24 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q25 = <double *>malloc(m2*n2 * sizeof(double))

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

        cfv(coeffs, m1, m2, n2, r2, x, t, L, &v)

        cfuvw_x(coeffs, m1, m2, n2, r2, L, x, t, cosa, tLA, uvw)
        ux = uvw[0]
        vx = uvw[1]
        wx = uvw[2]

        cfuvw_t(coeffs, m1, m2, n2, L, x, t, cosa, tLA, uvw)
        ut = uvw[0]
        wt = uvw[2]

        c = -1

        p00 = A16*sina*v/(L*cosa)
        p01 = A26*sina*v/(L*cosa)
        p02 = A66*sina*v/(L*cosa)
        p10 = -r2*(A12*(L - x)*(cosa*wt + sina*ut - v) + r*(A11*r*vx + A16*(L - x)*(cosa*wx + sina*ux)))/(L*r)
        p11 = -r2*(A22*(L - x)*(cosa*wt + sina*ut - v) + r*(A12*r*vx + A26*(L - x)*(cosa*wx + sina*ux)))/(L*r)
        p12 = -r2*(A26*(L - x)*(cosa*wt + sina*ut - v) + r*(A16*r*vx + A66*(L - x)*(cosa*wx + sina*ux)))/(L*r)
        p20 = sina*v*(A12*(-L + x)*sin(t - tLA) - A16*r*cos(t - tLA) + A16*r)/(L*cosa*r)
        p21 = sina*v*(A22*(-L + x)*sin(t - tLA) - A26*r*cos(t - tLA) + A26*r)/(L*cosa*r)
        p22 = sina*v*(A26*(-L + x)*sin(t - tLA) - A66*r*cos(t - tLA) + A66*r)/(L*cosa*r)

        q01 = -r2*vx/L
        q11 = -r2*(L - x)*(cosa*wt + sina*ut - v)/(L*r**2)
        q12 = sina*v*(-L + x)*sin(t - tLA)/(L*cosa*r**2)
        q20 = sina*v/(L*cosa*r)
        q21 = -r2*(L - x)*(cosa*wx + sina*ux)/(L*r)
        q22 = -sina*v*(cos(t - tLA) - 1)/(L*cosa*r)

        # kLL_00
        c += 1
        out[c] = beta*out[c] + alpha*(p02*q20)
        c += 1
        out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p01*q12 + p02*q22)
        c += 1
        out[c] = beta*out[c] + alpha*(p12*q20)
        c += 1
        out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
        c += 1
        out[c] = beta*out[c] + alpha*(p22*q20)
        c += 1
        out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)

        for k1 in range(1, m1+1):
            sink1x = vsini1x[k1-1]
            cosk1x = vcosi1x[k1-1]
            q01 = pi*cosk1x*k1*vx/L
            q02 = pi*cosk1x*k1*wx/L
            q11 = sink1x*(-cosa*wt - sina*ut + v)/(r*r)
            q20 = -pi*cosk1x*k1*sina*v/(L*r)
            q21 = -sink1x*(cosa*wx + sina*ux)/r
            q22 = pi*cosk1x*k1*(-cosa*v + wt)/(L*r)

            # kLL_01
            c += 1
            out[c] = beta*out[c] + alpha*(p02*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p12*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p22*q20)
            c += 1
            out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            # create buffer
            kLLq_1_q01[k1-1] = q01
            kLLq_1_q02[k1-1] = q02
            kLLq_1_q11[k1-1] = q11
            kLLq_1_q20[k1-1] = q20
            kLLq_1_q21[k1-1] = q21
            kLLq_1_q22[k1-1] = q22

        for k2 in range(1, m2+1):
            sink2x = vsini2x[k2-1]
            cosk2x = vcosi2x[k2-1]
            for l2 in range(1, n2+1):
                sinl2t = vsinj2t[l2-1]
                cosl2t = vcosj2t[l2-1]
                q02 = pi*cosk2x*k2*sinl2t*vx/L
                q03 = pi*cosk2x*cosl2t*k2*vx/L
                q04 = pi*cosk2x*k2*sinl2t*wx/L
                q05 = pi*cosk2x*cosl2t*k2*wx/L
                q10 = -cosl2t*l2*sina*sink2x*v/(r*r)
                q11 = l2*sina*sink2x*sinl2t*v/(r*r)
                q12 = sink2x*sinl2t*(-cosa*wt - sina*ut + v)/(r*r)
                q13 = cosl2t*sink2x*(-cosa*wt - sina*ut + v)/(r*r)
                q14 = cosl2t*l2*sink2x*(-cosa*v + wt)/(r*r)
                q15 = l2*sink2x*sinl2t*(cosa*v - wt)/(r*r)
                q20 = -pi*cosk2x*k2*sina*sinl2t*v/(L*r)
                q21 = -pi*cosk2x*cosl2t*k2*sina*v/(L*r)
                q22 = -sink2x*sinl2t*(cosa*wx + sina*ux)/r
                q23 = -cosl2t*sink2x*(cosa*wx + sina*ux)/r
                q24 = (L*cosl2t*l2*sink2x*wx + pi*cosk2x*k2*sinl2t*(-cosa*v + wt))/(L*r)
                q25 = (-L*l2*sink2x*sinl2t*wx + pi*cosk2x*cosl2t*k2*(-cosa*v + wt))/(L*r)

                # kLL_02
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q10 + p02*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q02 + p01*q12 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q03 + p01*q13 + p02*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q10 + p12*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02 + p11*q12 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q03 + p11*q13 + p12*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q10 + p22*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12 + p22*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q03 + p21*q13 + p22*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)

                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                kLLq_2_q02[pos] = q02
                kLLq_2_q03[pos] = q03
                kLLq_2_q04[pos] = q04
                kLLq_2_q05[pos] = q05
                kLLq_2_q10[pos] = q10
                kLLq_2_q11[pos] = q11
                kLLq_2_q12[pos] = q12
                kLLq_2_q13[pos] = q13
                kLLq_2_q14[pos] = q14
                kLLq_2_q15[pos] = q15
                kLLq_2_q20[pos] = q20
                kLLq_2_q21[pos] = q21
                kLLq_2_q22[pos] = q22
                kLLq_2_q23[pos] = q23
                kLLq_2_q24[pos] = q24
                kLLq_2_q25[pos] = q25

        for i1 in range(1, m1+1):
            sini1x = vsini1x[i1-1]
            cosi1x = vcosi1x[i1-1]
            p00 = -pi*A16*cosi1x*i1*sina*v/L
            p01 = -pi*A26*cosi1x*i1*sina*v/L
            p02 = -pi*A66*cosi1x*i1*sina*v/L
            p10 = pi*A11*cosi1x*i1*r*vx/L - sini1x*(A12*(cosa*wt + sina*ut - v) + A16*r*(cosa*wx + sina*ux))/r
            p11 = pi*A12*cosi1x*i1*r*vx/L - sini1x*(A22*(cosa*wt + sina*ut - v) + A26*r*(cosa*wx + sina*ux))/r
            p12 = pi*A16*cosi1x*i1*r*vx/L - sini1x*(A26*(cosa*wt + sina*ut - v) + A66*r*(cosa*wx + sina*ux))/r
            p20 = pi*cosi1x*i1*(A11*r*wx + A16*(-cosa*v + wt))/L
            p21 = pi*cosi1x*i1*(A12*r*wx + A26*(-cosa*v + wt))/L
            p22 = pi*cosi1x*i1*(A16*r*wx + A66*(-cosa*v + wt))/L

            #NOTE symmetry
            for k1 in range(i1, m1+1):
                # access buffer
                q01 = kLLq_1_q01[k1-1]
                q02 = kLLq_1_q02[k1-1]
                q11 = kLLq_1_q11[k1-1]
                q20 = kLLq_1_q20[k1-1]
                q21 = kLLq_1_q21[k1-1]
                q22 = kLLq_1_q22[k1-1]

                # kLL_11
                c += 1
                out[c] = beta*out[c] + alpha*(p02*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q01 + p01*q11 + p02*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q01 + p11*q11 + p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p22*q20)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q01 + p21*q11 + p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q02 = kLLq_2_q02[pos]
                    q03 = kLLq_2_q03[pos]
                    q04 = kLLq_2_q04[pos]
                    q05 = kLLq_2_q05[pos]
                    q10 = kLLq_2_q10[pos]
                    q11 = kLLq_2_q11[pos]
                    q12 = kLLq_2_q12[pos]
                    q13 = kLLq_2_q13[pos]
                    q14 = kLLq_2_q14[pos]
                    q15 = kLLq_2_q15[pos]
                    q20 = kLLq_2_q20[pos]
                    q21 = kLLq_2_q21[pos]
                    q22 = kLLq_2_q22[pos]
                    q23 = kLLq_2_q23[pos]
                    q24 = kLLq_2_q24[pos]
                    q25 = kLLq_2_q25[pos]

                    # kLL_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p01*q10 + p02*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q02 + p01*q12 + p02*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q03 + p01*q13 + p02*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q10 + p12*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q02 + p11*q12 + p12*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q03 + p11*q13 + p12*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q10 + p22*q20)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12 + p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q03 + p21*q13 + p22*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)

        for i2 in range(1, m2+1):
            sini2x = vsini2x[i2-1]
            cosi2x = vcosi2x[i2-1]
            for j2 in range(1, n2+1):
                sinj2t = vsinj2t[j2-1]
                cosj2t = vcosj2t[j2-1]
                p00 = -sina*v*(A12*L*cosj2t*j2*sini2x + pi*A16*cosi2x*i2*r*sinj2t)/(L*r)
                p01 = -sina*v*(A22*L*cosj2t*j2*sini2x + pi*A26*cosi2x*i2*r*sinj2t)/(L*r)
                p02 = -sina*v*(A26*L*cosj2t*j2*sini2x + pi*A66*cosi2x*i2*r*sinj2t)/(L*r)
                p10 = sina*v*(A12*j2*sini2x*sinj2t/r - pi*A16*cosi2x*cosj2t*i2/L)
                p11 = sina*v*(A22*j2*sini2x*sinj2t/r - pi*A26*cosi2x*cosj2t*i2/L)
                p12 = sina*v*(A26*j2*sini2x*sinj2t/r - pi*A66*cosi2x*cosj2t*i2/L)
                p20 = sinj2t*(pi*A11*cosi2x*i2*(r*r)*vx - L*sini2x*(A12*(cosa*wt + sina*ut - v) + A16*r*(cosa*wx + sina*ux)))/(L*r)
                p21 = sinj2t*(pi*A12*cosi2x*i2*(r*r)*vx - L*sini2x*(A22*(cosa*wt + sina*ut - v) + A26*r*(cosa*wx + sina*ux)))/(L*r)
                p22 = sinj2t*(pi*A16*cosi2x*i2*(r*r)*vx - L*sini2x*(A26*(cosa*wt + sina*ut - v) + A66*r*(cosa*wx + sina*ux)))/(L*r)
                p30 = cosj2t*(pi*A11*cosi2x*i2*(r*r)*vx - L*sini2x*(A12*(cosa*wt + sina*ut - v) + A16*r*(cosa*wx + sina*ux)))/(L*r)
                p31 = cosj2t*(pi*A12*cosi2x*i2*(r*r)*vx - L*sini2x*(A22*(cosa*wt + sina*ut - v) + A26*r*(cosa*wx + sina*ux)))/(L*r)
                p32 = cosj2t*(pi*A16*cosi2x*i2*(r*r)*vx - L*sini2x*(A26*(cosa*wt + sina*ut - v) + A66*r*(cosa*wx + sina*ux)))/(L*r)
                p40 = cosj2t*j2*sini2x*(A12*(-cosa*v + wt) + A16*r*wx)/r + pi*cosi2x*i2*sinj2t*(A11*r*wx + A16*(-cosa*v + wt))/L
                p41 = cosj2t*j2*sini2x*(A22*(-cosa*v + wt) + A26*r*wx)/r + pi*cosi2x*i2*sinj2t*(A12*r*wx + A26*(-cosa*v + wt))/L
                p42 = cosj2t*j2*sini2x*(A26*(-cosa*v + wt) + A66*r*wx)/r + pi*cosi2x*i2*sinj2t*(A16*r*wx + A66*(-cosa*v + wt))/L
                p50 = -j2*sini2x*sinj2t*(A12*(-cosa*v + wt) + A16*r*wx)/r + pi*cosi2x*cosj2t*i2*(A11*r*wx + A16*(-cosa*v + wt))/L
                p51 = -j2*sini2x*sinj2t*(A22*(-cosa*v + wt) + A26*r*wx)/r + pi*cosi2x*cosj2t*i2*(A12*r*wx + A26*(-cosa*v + wt))/L
                p52 = -j2*sini2x*sinj2t*(A26*(-cosa*v + wt) + A66*r*wx)/r + pi*cosi2x*cosj2t*i2*(A16*r*wx + A66*(-cosa*v + wt))/L

                #NOTE symmetry
                for k2 in range(i2, m2+1):
                    for l2 in range(j2, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q02 = kLLq_2_q02[pos]
                        q03 = kLLq_2_q03[pos]
                        q04 = kLLq_2_q04[pos]
                        q05 = kLLq_2_q05[pos]
                        q10 = kLLq_2_q10[pos]
                        q11 = kLLq_2_q11[pos]
                        q12 = kLLq_2_q12[pos]
                        q13 = kLLq_2_q13[pos]
                        q14 = kLLq_2_q14[pos]
                        q15 = kLLq_2_q15[pos]
                        q20 = kLLq_2_q20[pos]
                        q21 = kLLq_2_q21[pos]
                        q22 = kLLq_2_q22[pos]
                        q23 = kLLq_2_q23[pos]
                        q24 = kLLq_2_q24[pos]
                        q25 = kLLq_2_q25[pos]

                        # kLL_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p01*q10 + p02*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q02 + p01*q12 + p02*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q03 + p01*q13 + p02*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p11*q10 + p12*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q02 + p11*q12 + p12*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q03 + p11*q13 + p12*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p21*q10 + p22*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12 + p22*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q03 + p21*q13 + p22*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p31*q10 + p32*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p31*q11 + p32*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q02 + p31*q12 + p32*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q03 + p31*q13 + p32*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q04 + p31*q14 + p32*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q05 + p31*q15 + p32*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p41*q10 + p42*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p41*q11 + p42*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q02 + p41*q12 + p42*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q03 + p41*q13 + p42*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14 + p42*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15 + p42*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p51*q10 + p52*q20)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p51*q11 + p52*q21)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q02 + p51*q12 + p52*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q03 + p51*q13 + p52*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q04 + p51*q14 + p52*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q05 + p51*q15 + p52*q25)

    free(vsini1x)
    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
    free(kLLq_1_q01)
    free(kLLq_1_q02)
    free(kLLq_1_q11)
    free(kLLq_1_q20)
    free(kLLq_1_q21)
    free(kLLq_1_q22)
    free(kLLq_2_q02)
    free(kLLq_2_q03)
    free(kLLq_2_q04)
    free(kLLq_2_q05)
    free(kLLq_2_q10)
    free(kLLq_2_q11)
    free(kLLq_2_q12)
    free(kLLq_2_q13)
    free(kLLq_2_q14)
    free(kLLq_2_q15)
    free(kLLq_2_q20)
    free(kLLq_2_q21)
    free(kLLq_2_q22)
    free(kLLq_2_q23)
    free(kLLq_2_q24)
    free(kLLq_2_q25)
