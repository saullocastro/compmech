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

from desicos.clpt_commons cimport cfwx, cfwt, cfN

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
    #
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] k0Lv

    cdef int fdim
    cdef cc_attributes args

    num_cond_1 = 12
    num_cond_2 = 12
    num_cond_3 = 12
    num_cond_4 = 12
    k22_num = num_cond_1*m2*n2 + num_cond_2*(m2-1)*m2*n2 \
            + num_cond_3*(m2-1)*m2*(n2-1)*n2 + num_cond_4*m2*(n2-1)*n2

    fdim = 3*m1 + 6*m2*n2 + 3*m1**2 + 12*m1*m2*n2 + k22_num


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
        cols[c] = col+2
        c += 1
        rows[c] = 1
        cols[c] = col+2
        c += 1
        rows[c] = 2
        cols[c] = col+2

    for k2 in range(1, m2+1):
        for l2 in range(1, n2+1):
            col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
            # k0L_02
            c += 1
            rows[c] = 0
            cols[c] = col+4
            c += 1
            rows[c] = 0
            cols[c] = col+5
            c += 1
            rows[c] = 1
            cols[c] = col+4
            c += 1
            rows[c] = 1
            cols[c] = col+5
            c += 1
            rows[c] = 2
            cols[c] = col+4
            c += 1
            rows[c] = 2
            cols[c] = col+5

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        for k1 in range(1, m1+1):
            col = (k1-1)*num1 + num0
            # k0L_11
            c += 1
            rows[c] = row+0
            cols[c] = col+2
            c += 1
            rows[c] = row+1
            cols[c] = col+2
            c += 1
            rows[c] = row+2
            cols[c] = col+2

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                # k0L_12
                c += 1
                rows[c] = row+0
                cols[c] = col+4
                c += 1
                rows[c] = row+0
                cols[c] = col+5
                c += 1
                rows[c] = row+1
                cols[c] = col+4
                c += 1
                rows[c] = row+1
                cols[c] = col+5
                c += 1
                rows[c] = row+2
                cols[c] = col+4
                c += 1
                rows[c] = row+2
                cols[c] = col+5

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            for k1 in range(1, m1+1):
                col = (k1-1)*num1 + num0
                # k0L_21
                c += 1
                rows[c] = row+0
                cols[c] = col+2
                c += 1
                rows[c] = row+1
                cols[c] = col+2
                c += 1
                rows[c] = row+2
                cols[c] = col+2
                c += 1
                rows[c] = row+3
                cols[c] = col+2
                c += 1
                rows[c] = row+4
                cols[c] = col+2
                c += 1
                rows[c] = row+5
                cols[c] = col+2

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                    # k0L_22
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+3
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+5
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

    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66

    cdef double p00, p01, p02, p10, p11, p12
    cdef double p20, p21, p22, p30, p31, p32
    cdef double p40, p41, p42, p50, p51, p52
    cdef double q02, q04, q05, q14, q15, q22, q24, q25

    cdef double r, x, t, alpha, beta

    cdef double *F, *coeffs
    cdef double  sina, cosa, tLA, r2, L
    cdef int m1, m2, n2, pti
    cdef double wx, wt

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

    cdef double sini1x, cosi1x, cosk1x, sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t
    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *k0Lq_1_q02 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q22 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_2_q04 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q05 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q14 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q15 = <double *>malloc(m2*n2 * sizeof(double))
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

        cfwx(coeffs, m1, m2, n2, x, t, L, &wx)
        cfwt(coeffs, m1, m2, n2, x, t, L, &wt)

        c = -1

        p00 = (-A11*r + A12*sina*(L - x))/(L*cosa)
        p01 = (-A12*r + A22*sina*(L - x))/(L*cosa)
        p02 = (-A16*r + A26*sina*(L - x))/(L*cosa)
        p10 = -A16*r2*(r + sina*(L - x))/L
        p11 = -A26*r2*(r + sina*(L - x))/L
        p12 = -A66*r2*(r + sina*(L - x))/L
        p20 = (A16*(L - x)*sin(t - tLA) + (A11*r + A12*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        p21 = (A26*(L - x)*sin(t - tLA) + (A12*r + A22*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        p22 = (A66*(L - x)*sin(t - tLA) + (A16*r + A26*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)

        for k1 in range(1, m1+1):
            cosk1x = vcosi1x[k1-1]
            q02 = pi*cosk1x*k1*wx/L
            q22 = pi*cosk1x*k1*wt/(L*r)

            # k0L_01
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            # create buffer
            k0Lq_1_q02[k1-1] = q02
            k0Lq_1_q22[k1-1] = q22

        for k2 in range(1, m2+1):
            sink2x = vsini2x[k2-1]
            cosk2x = vcosi2x[k2-1]
            for l2 in range(1, n2+1):
                sinl2t = vsinj2t[l2-1]
                cosl2t = vcosj2t[l2-1]
                q04 = pi*cosk2x*k2*sinl2t*wx/L
                q05 = pi*cosk2x*cosl2t*k2*wx/L
                q14 = cosl2t*l2*sink2x*wt/(r*r)
                q15 = -l2*sink2x*sinl2t*wt/(r*r)
                q24 = (L*cosl2t*l2*sink2x*wx + pi*cosk2x*k2*sinl2t*wt)/(L*r)
                q25 = (-L*l2*sink2x*sinl2t*wx + pi*cosk2x*cosl2t*k2*wt)/(L*r)

                # k0L_02
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)

                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                k0Lq_2_q04[pos] = q04
                k0Lq_2_q05[pos] = q05
                k0Lq_2_q14[pos] = q14
                k0Lq_2_q15[pos] = q15
                k0Lq_2_q24[pos] = q24
                k0Lq_2_q25[pos] = q25

        for i1 in range(1, m1+1):
            sini1x = vsini1x[i1-1]
            cosi1x = vcosi1x[i1-1]
            p00 = pi*A11*i1*r*cosi1x/L + A12*sina*sini1x
            p01 = pi*A12*i1*r*cosi1x/L + A22*sina*sini1x
            p02 = pi*A16*i1*r*cosi1x/L + A26*sina*sini1x
            p10 = -A16*sina*sini1x + pi*A16*i1*r*cosi1x/L
            p11 = -A26*sina*sini1x + pi*A26*i1*r*cosi1x/L
            p12 = -A66*sina*sini1x + pi*A66*i1*r*cosi1x/L
            p20 = (-pi*B12*L*i1*sina*cosi1x + (A12*(L*L)*cosa + (pi*pi)*B11*(i1*i1)*r)*sini1x)/(L*L)
            p21 = (-pi*B22*L*i1*sina*cosi1x + (A22*(L*L)*cosa + (pi*pi)*B12*(i1*i1)*r)*sini1x)/(L*L)
            p22 = (-pi*B26*L*i1*sina*cosi1x + (A26*(L*L)*cosa + (pi*pi)*B16*(i1*i1)*r)*sini1x)/(L*L)

            for k1 in range(1, m1+1):
                # access buffer
                q02 = k0Lq_1_q02[k1-1]
                q22 = k0Lq_1_q22[k1-1]

                # k0L_11
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q04 = k0Lq_2_q04[pos]
                    q05 = k0Lq_2_q05[pos]
                    q14 = k0Lq_2_q14[pos]
                    q15 = k0Lq_2_q15[pos]
                    q24 = k0Lq_2_q24[pos]
                    q25 = k0Lq_2_q25[pos]

                    # k0L_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
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
                p00 = pi*A11*i2*r*sinj2t*cosi2x/L + (A12*sina*sinj2t + A16*j2*cosj2t)*sini2x
                p01 = pi*A12*i2*r*sinj2t*cosi2x/L + (A22*sina*sinj2t + A26*j2*cosj2t)*sini2x
                p02 = pi*A16*i2*r*sinj2t*cosi2x/L + (A26*sina*sinj2t + A66*j2*cosj2t)*sini2x
                p10 = pi*A11*i2*r*cosj2t*cosi2x/L + (A12*sina*cosj2t - A16*j2*sinj2t)*sini2x
                p11 = pi*A12*i2*r*cosj2t*cosi2x/L + (A22*sina*cosj2t - A26*j2*sinj2t)*sini2x
                p12 = pi*A16*i2*r*cosj2t*cosi2x/L + (A26*sina*cosj2t - A66*j2*sinj2t)*sini2x
                p20 = pi*A16*i2*r*sinj2t*cosi2x/L + (A12*j2*cosj2t - A16*sina*sinj2t)*sini2x
                p21 = pi*A26*i2*r*sinj2t*cosi2x/L + (A22*j2*cosj2t - A26*sina*sinj2t)*sini2x
                p22 = pi*A66*i2*r*sinj2t*cosi2x/L + (A26*j2*cosj2t - A66*sina*sinj2t)*sini2x
                p30 = pi*A16*i2*r*cosj2t*cosi2x/L - (A12*j2*sinj2t + A16*sina*cosj2t)*sini2x
                p31 = pi*A26*i2*r*cosj2t*cosi2x/L - (A22*j2*sinj2t + A26*sina*cosj2t)*sini2x
                p32 = pi*A66*i2*r*cosj2t*cosi2x/L - (A26*j2*sinj2t + A66*sina*cosj2t)*sini2x
                p40 = (-pi*L*i2*r*(B12*sina*sinj2t + 2*B16*j2*cosj2t)*cosi2x + (B16*(L*L)*j2*sina*cosj2t + (B12*(L*L)*(j2*j2) + r*(A12*(L*L)*cosa + (pi*pi)*B11*(i2*i2)*r))*sinj2t)*sini2x)/((L*L)*r)
                p41 = (-pi*L*i2*r*(B22*sina*sinj2t + 2*B26*j2*cosj2t)*cosi2x + (B26*(L*L)*j2*sina*cosj2t + (B22*(L*L)*(j2*j2) + r*(A22*(L*L)*cosa + (pi*pi)*B12*(i2*i2)*r))*sinj2t)*sini2x)/((L*L)*r)
                p42 = (-pi*L*i2*r*(B26*sina*sinj2t + 2*B66*j2*cosj2t)*cosi2x + (B66*(L*L)*j2*sina*cosj2t + (B26*(L*L)*(j2*j2) + r*(A26*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))*sinj2t)*sini2x)/((L*L)*r)
                p50 = (pi*L*i2*r*(-B12*sina*cosj2t + 2*B16*j2*sinj2t)*cosi2x + (-B16*(L*L)*j2*sina*sinj2t + (B12*(L*L)*(j2*j2) + r*(A12*(L*L)*cosa + (pi*pi)*B11*(i2*i2)*r))*cosj2t)*sini2x)/((L*L)*r)
                p51 = (pi*L*i2*r*(-B22*sina*cosj2t + 2*B26*j2*sinj2t)*cosi2x + (-B26*(L*L)*j2*sina*sinj2t + (B22*(L*L)*(j2*j2) + r*(A22*(L*L)*cosa + (pi*pi)*B12*(i2*i2)*r))*cosj2t)*sini2x)/((L*L)*r)
                p52 = (pi*L*i2*r*(-B26*sina*cosj2t + 2*B66*j2*sinj2t)*cosi2x + (-B66*(L*L)*j2*sina*sinj2t + (B26*(L*L)*(j2*j2) + r*(A26*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))*cosj2t)*sini2x)/((L*L)*r)

                for k1 in range(1, m1+1):
                    # access buffer
                    q02 = k0Lq_1_q02[k1-1]
                    q22 = k0Lq_1_q22[k1-1]

                    # k0L_21
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p30*q02 + p32*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p40*q02 + p42*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p50*q02 + p52*q22)

                for k2 in range(1, m2+1):
                    for l2 in range(1, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q04 = k0Lq_2_q04[pos]
                        q05 = k0Lq_2_q05[pos]
                        q14 = k0Lq_2_q14[pos]
                        q15 = k0Lq_2_q15[pos]
                        q24 = k0Lq_2_q24[pos]
                        q25 = k0Lq_2_q25[pos]
                        # k0L_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q04 + p31*q14 + p32*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q05 + p31*q15 + p32*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14 + p42*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15 + p42*q25)
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
    free(k0Lq_1_q02)
    free(k0Lq_1_q22)
    free(k0Lq_2_q04)
    free(k0Lq_2_q05)
    free(k0Lq_2_q14)
    free(k0Lq_2_q15)
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

    k11_cond_1 = 1
    k11_cond_2 = 1
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1

    k22_cond_1 = 4
    k22_cond_2 = 4
    k22_cond_3 = 4
    k22_cond_4 = 4
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k11_num + 2*m1*m2*n2 + k22_num

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
            rows[c] = row+2
            cols[c] = col+2

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                # kG_12
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
                    rows[c] = row+4
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+5
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

    cdef double q02, q04, q05, q14, q15
    cdef double p20, p21, p40, p41, p50, p51

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

    cdef double cosk1x, sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *kGq_2_q04 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q05 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q14 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q15 = <double *>malloc(m2*n2 * sizeof(double))

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(1, m1+1):
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
                q04 = pi*k2*sinl2t*cosk2x/L
                q05 = pi*k2*cosl2t*cosk2x/L
                q14 = l2*sink2x*cosl2t/r
                q15 = -l2*sinl2t*sink2x/r
                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                kGq_2_q04[pos] = q04
                kGq_2_q05[pos] = q05
                kGq_2_q14[pos] = q14
                kGq_2_q15[pos] = q15

        for i1 in range(1, m1+1):
            p20 = pi*Nxx*i1*r*cos(pi*i1*x/L)/L
            p21 = pi*Nxt*i1*r*cos(pi*i1*x/L)/L

            #NOTE symmetry
            for k1 in range(i1, m1+1):
                cosk1x = vcosi1x[k1-1]
                q02 = pi*k1*cosk1x/L

                # kG_11
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q04 = kGq_2_q04[pos]
                    q05 = kGq_2_q05[pos]
                    q14 = kGq_2_q14[pos]
                    q15 = kGq_2_q15[pos]
                    # kG_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15)

        for i2 in range(1, m2+1):
            sini2x = vsini2x[i2-1]
            cosi2x = vcosi2x[i2-1]
            for j2 in range(1, n2+1):
                sinj2t = vsinj2t[j2-1]
                cosj2t = vcosj2t[j2-1]
                p40 = Nxt*j2*sini2x*cosj2t + pi*Nxx*i2*r*sinj2t*cosi2x/L
                p41 = Ntt*j2*sini2x*cosj2t + pi*Nxt*i2*r*sinj2t*cosi2x/L
                p50 = -Nxt*j2*sinj2t*sini2x + pi*Nxx*i2*r*cosj2t*cosi2x/L
                p51 = -Ntt*j2*sinj2t*sini2x + pi*Nxt*i2*r*cosj2t*cosi2x/L

                #NOTE symmetry
                for k2 in range(i2, m2+1):
                    for l2 in range(j2, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q04 = kGq_2_q04[pos]
                        q05 = kGq_2_q05[pos]
                        q14 = kGq_2_q14[pos]
                        q15 = kGq_2_q15[pos]
                        # kG_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q04 + p51*q14)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q05 + p51*q15)

    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
    free(kGq_2_q04)
    free(kGq_2_q05)
    free(kGq_2_q14)
    free(kGq_2_q15)

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

    k22_cond_1 = 4
    k22_cond_2 = 4
    k22_cond_3 = 4
    k22_cond_4 = 4
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 1*m1**2 + 2*m1*m2*n2 + k22_num

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
            rows[c] = row+2
            cols[c] = col+2

        for k2 in range(1, m2+1):
            for l2 in range(1, n2+1):
                col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                # kLL_12
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
                    rows[c] = row+4
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+4
                    cols[c] = col+5
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

    cdef double A11, A12, A16, A22, A26, A66

    cdef double p20, p21, p22, p40, p41, p42
    cdef double p50, p51, p52
    cdef double q02, q04, q05, q14, q15, q22, q24, q25

    cdef double r, x, t, alpha, beta

    cdef double *F, *coeffs
    cdef double sina, r2, L
    cdef int m1, m2, n2, pti
    cdef double wx, wt

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
    A22 = F[7]  # F[1,1]
    A26 = F[8]  # F[1,2]
    A66 = F[14] # F[2,2]

    cdef double sini1x, cosi1x, cosk1x, sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *kLLq_2_q04 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q05 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q14 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q15 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q24 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q25 = <double *>malloc(m2*n2 * sizeof(double))

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(1, m1+1):
            vcosi1x[i1-1] = cos(pi*i1*x/L)

        for i2 in range(1, m2+1):
            vsini2x[i2-1] = sin(pi*i2*x/L)
            vcosi2x[i2-1] = cos(pi*i2*x/L)

        for j2 in range(1, n2+1):
            vsinj2t[j2-1] = sin(j2*t)
            vcosj2t[j2-1] = cos(j2*t)

        r = r2 + sina*x

        cfwx(coeffs, m1, m2, n2, x, t, L, &wx)
        cfwt(coeffs, m1, m2, n2, x, t, L, &wt)

        c = -1

        for k2 in range(1, m2+1):
            sink2x = vsini2x[k2-1]
            cosk2x = vcosi2x[k2-1]
            for l2 in range(1, n2+1):
                sinl2t = vsinj2t[l2-1]
                cosl2t = vcosj2t[l2-1]
                q04 = pi*k2*wx*sinl2t*cosk2x/L
                q05 = pi*k2*wx*cosl2t*cosk2x/L
                q14 = l2*wt*sink2x*cosl2t/(r*r)
                q15 = -l2*wt*sinl2t*sink2x/(r*r)
                q24 = (L*l2*wx*sink2x*cosl2t + pi*k2*wt*sinl2t*cosk2x)/(L*r)
                q25 = (-L*l2*wx*sinl2t*sink2x + pi*k2*wt*cosl2t*cosk2x)/(L*r)
                # create buffer
                pos = (k2-1)*n2 + (l2-1)
                kLLq_2_q04[pos] = q04
                kLLq_2_q05[pos] = q05
                kLLq_2_q14[pos] = q14
                kLLq_2_q15[pos] = q15
                kLLq_2_q24[pos] = q24
                kLLq_2_q25[pos] = q25

        for i1 in range(1, m1+1):
            cosi1x = vcosi1x[i1-1]
            p20 = pi*i1*(A11*r*wx + A16*wt)*cosi1x/L
            p21 = pi*i1*(A12*r*wx + A26*wt)*cosi1x/L
            p22 = pi*i1*(A16*r*wx + A66*wt)*cosi1x/L

            #NOTE symmetry
            for k1 in range(i1, m1+1):
                cosk1x = vcosi1x[k1-1]
                q02 = pi*k1*wx*cosk1x/L
                q22 = pi*k1*wt*cosk1x/(L*r)
                # kLL_11
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    # access buffer
                    pos = (k2-1)*n2 + (l2-1)
                    q04 = kLLq_2_q04[pos]
                    q05 = kLLq_2_q05[pos]
                    q14 = kLLq_2_q14[pos]
                    q15 = kLLq_2_q15[pos]
                    q24 = kLLq_2_q24[pos]
                    q25 = kLLq_2_q25[pos]
                    # kLL_12
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
                p40 = j2*(A12*wt + A16*r*wx)*sini2x*cosj2t/r + pi*i2*(A11*r*wx + A16*wt)*sinj2t*cosi2x/L
                p41 = j2*(A22*wt + A26*r*wx)*sini2x*cosj2t/r + pi*i2*(A12*r*wx + A26*wt)*sinj2t*cosi2x/L
                p42 = j2*(A26*wt + A66*r*wx)*sini2x*cosj2t/r + pi*i2*(A16*r*wx + A66*wt)*sinj2t*cosi2x/L
                p50 = -j2*(A12*wt + A16*r*wx)*sinj2t*sini2x/r + pi*i2*(A11*r*wx + A16*wt)*cosj2t*cosi2x/L
                p51 = -j2*(A22*wt + A26*r*wx)*sinj2t*sini2x/r + pi*i2*(A12*r*wx + A26*wt)*cosj2t*cosi2x/L
                p52 = -j2*(A26*wt + A66*r*wx)*sinj2t*sini2x/r + pi*i2*(A16*r*wx + A66*wt)*cosj2t*cosi2x/L
                #NOTE symmetry
                for k2 in range(i2, m2+1):
                    for l2 in range(j2, n2+1):
                        # access buffer
                        pos = (k2-1)*n2 + (l2-1)
                        q04 = kLLq_2_q04[pos]
                        q05 = kLLq_2_q05[pos]
                        q14 = kLLq_2_q14[pos]
                        q15 = kLLq_2_q15[pos]
                        q24 = kLLq_2_q24[pos]
                        q25 = kLLq_2_q25[pos]
                        # kLL_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14 + p42*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15 + p42*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q04 + p51*q14 + p52*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q05 + p51*q15 + p52*q25)

    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
    free(kLLq_2_q04)
    free(kLLq_2_q05)
    free(kLLq_2_q14)
    free(kLLq_2_q15)
    free(kLLq_2_q24)
    free(kLLq_2_q25)


