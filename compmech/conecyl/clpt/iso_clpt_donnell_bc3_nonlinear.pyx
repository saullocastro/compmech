#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from libc.stdlib cimport malloc, free

import numpy as np
from scipy.sparse import coo_matrix

from compmech.conecyl.clpt.clpt_commons_bc3 cimport cfwx, cfwt
from compmech.conecyl.imperfections.mgi cimport cfw0x, cfw0t
from compmech.integrate.integratev cimport integratev


DOUBLE = np.float64
INT = np.int64


cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil


cdef int i0 = 0
cdef int j0 = 1
cdef int num0 = 3
cdef int num1 = 3
cdef int num2 = 6
cdef int e_num = 6
cdef double pi = 3.141592653589793
cdef int funcnum = 2 # to use in the cfw0x and cfw0t functions
cdef int castro = 0


cdef struct cc_attributes:
    double *sina
    double *cosa
    double *tLA
    double *r2
    double *L
    double *F
    double *E11
    double *nu
    double *h
    int *m1
    int *m2
    int *n2
    double *coeffs
    double *c0
    int *m0
    int *n0


cdef int NL_kinematics=0 # to use cfstrain_donnell in cfN


def calc_k0L(double [:] coeffs,
             double alpharad, double r2, double L, double tLA,
             double E11, double nu, double h,
             int m1, int m2, int n2,
             int nx, int nt, int num_cores, str method,
             double [:] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef long [:] rows, cols
    cdef double [:] k0Lv

    cdef int fdim
    cdef cc_attributes args

    fdim = 3*m1 + 6*m2*n2 + 3*m1**2 + 2*6*m1*m2*n2 + 12*m2**2*n2**2

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
    args.E11 = &E11
    args.nu = &nu
    args.h = &h
    args.m1 = &m1
    args.m2 = &m2
    args.n2 = &n2
    args.coeffs = &coeffs[0]
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    integratev(<void *>cfk0L, fdim, &k0Lv[0], xa, xb, nx, ta, tb, nt,
               &args, num_cores, method)

    c = -1

    for k1 in range(i0, m1+i0):
        col = (k1-i0)*num1 + num0
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

    for k2 in range(i0, m2+i0):
        for l2 in range(j0, n2+j0):
            col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
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

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0
        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0
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

        for k2 in range(i0, m2+i0):
            for l2 in range(j0, n2+j0):
                col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
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

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k1 in range(i0, m1+i0):
                col = (k1-i0)*num1 + num0
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

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
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

    cdef double p00, p01, p02, p10, p11, p12
    cdef double p20, p21, p22, p30, p31, p32
    cdef double p40, p41, p50, p51
    cdef double q02, q04, q05, q14, q15, q22, q24, q25

    cdef double r, x, t, alpha, beta

    cdef double E11, nu, h
    cdef double *coeffs
    cdef double *c0
    cdef double sina, cosa, tLA, r2, L
    cdef int m0, n0, m1, m2, n2
    cdef double wx, wt, w0x, w0t

    cdef cc_attributes *args_in=<cc_attributes *>args

    sina = args_in.sina[0]
    cosa = args_in.cosa[0]
    tLA = args_in.tLA[0]
    r2 = args_in.r2[0]
    L = args_in.L[0]
    E11 = args_in.E11[0]
    nu = args_in.nu[0]
    h = args_in.h[0]
    m1 = args_in.m1[0]
    m2 = args_in.m2[0]
    n2 = args_in.n2[0]
    coeffs = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

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
    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfwx(coeffs, m1, m2, n2, xs, ts, npts, L, wxs)
    cfwt(coeffs, m1, m2, n2, xs, ts, npts, L, wts)
    cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    cdef double tick, dt

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        wx = wxs[i]
        wt = wts[i]
        w0x = w0xs[i]
        w0t = w0ts[i]
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(i0, m1+i0):
            vsini1x[i1-i0] = sin(pi*i1*x/L)
            vcosi1x[i1-i0] = cos(pi*i1*x/L)

        for i2 in range(i0, m2+i0):
            vsini2x[i2-i0] = sin(pi*i2*x/L)
            vcosi2x[i2-i0] = cos(pi*i2*x/L)

        for j2 in range(j0, n2+j0):
            vsinj2t[j2-j0] = sin(j2*t)
            vcosj2t[j2-j0] = cos(j2*t)

        r = r2 + sina*x

        c = -1

        # p_0
        p00 = E11*h*(nu*sina*(-L + x) + r)/(L*cosa*(nu**2 - 1))
        p01 = E11*h*(nu*r + sina*(-L + x))/(L*cosa*(nu**2 - 1))
        p12 = -0.5*E11*h*r2*(r + sina*(L - x))/(L*(nu + 1))
        p20 = E11*h*(nu*sina*(-L + x) + r)*cos(t - tLA)/(L*cosa*(nu**2 - 1))
        p21 = E11*h*(nu*r + sina*(-L + x))*cos(t - tLA)/(L*cosa*(nu**2 - 1))
        p22 = 0.5*E11*h*(-L + x)*sin(t - tLA)/(L*cosa*(nu + 1))

        for k1 in range(i0, m1+i0):
            cosk1x = vcosi1x[k1-i0]

            # q_1
            q02 = pi*cosk1x*k1*(w0x + wx)/L
            q22 = pi*cosk1x*k1*(w0t + wt)/(L*r)

            # k0L_01
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q02)
            c += 1
            out[c] = beta*out[c] + alpha*(p12*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            # create buffer q_1
            k0Lq_1_q02[k1-i0] = q02
            k0Lq_1_q22[k1-i0] = q22

        for k2 in range(i0, m2+i0):
            sink2x = vsini2x[k2-i0]
            cosk2x = vcosi2x[k2-i0]
            for l2 in range(j0, n2+j0):
                sinl2t = vsinj2t[l2-j0]
                cosl2t = vcosj2t[l2-j0]

                # q_2
                q04 = pi*cosk2x*k2*sinl2t*(w0x + wx)/L
                q05 = pi*cosk2x*cosl2t*k2*(w0x + wx)/L
                q14 = cosl2t*l2*sink2x*(w0t + wt)/r**2
                q15 = -l2*sink2x*sinl2t*(w0t + wt)/r**2
                q24 = (L*cosl2t*l2*sink2x*(w0x + wx) + pi*cosk2x*k2*sinl2t*(w0t + wt))/(L*r)
                q25 = (-L*l2*sink2x*sinl2t*(w0x + wx) + pi*cosk2x*cosl2t*k2*(w0t + wt))/(L*r)

                # k0L_02
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)

                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                k0Lq_2_q04[pos] = q04
                k0Lq_2_q05[pos] = q05
                k0Lq_2_q14[pos] = q14
                k0Lq_2_q15[pos] = q15
                k0Lq_2_q24[pos] = q24
                k0Lq_2_q25[pos] = q25

        for i1 in range(i0, m1+i0):
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]

            # p_1
            p00 = E11*h*(L*nu*sina*sini1x + pi*cosi1x*i1*r)/(-L*nu**2 + L)
            p01 = E11*h*(L*sina*sini1x + pi*cosi1x*i1*nu*r)/(-L*nu**2 + L)
            p12 = (-E11*L*h*sina*sini1x + pi*E11*cosi1x*h*i1*r)/(2*L*nu + 2*L)
            p20 = E11*cosa*h*nu*sini1x/(-nu**2 + 1)
            p21 = E11*cosa*h*sini1x/(-nu**2 + 1)

            for k1 in range(i0, m1+i0):
                # access buffer q_1
                q02 = k0Lq_1_q02[k1-i0]
                q22 = k0Lq_1_q22[k1-i0]

                # k0L_11
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q02)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02)

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    # access buffer q_2
                    pos = (k2-i0)*n2 + (l2-j0)
                    q04 = k0Lq_2_q04[pos]
                    q05 = k0Lq_2_q05[pos]
                    q14 = k0Lq_2_q14[pos]
                    q15 = k0Lq_2_q15[pos]
                    q24 = k0Lq_2_q24[pos]
                    q25 = k0Lq_2_q25[pos]

                    # k0L_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p12*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p12*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15)

        for i2 in range(i0, m2+i0):
            sini2x = vsini2x[i2-i0]
            cosi2x = vcosi2x[i2-i0]
            for j2 in range(j0, n2+j0):
                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]
                # p_2
                p00 = -E11*h*sinj2t*(L*nu*sina*sini2x + pi*cosi2x*i2*r)/(L*(nu**2 - 1))
                p01 = -E11*h*sinj2t*(L*sina*sini2x + pi*cosi2x*i2*nu*r)/(L*(nu**2 - 1))
                p02 = E11*cosj2t*h*j2*sini2x/(2*nu + 2)
                p10 = -E11*cosj2t*h*(L*nu*sina*sini2x + pi*cosi2x*i2*r)/(L*(nu**2 - 1))
                p11 = -E11*cosj2t*h*(L*sina*sini2x + pi*cosi2x*i2*nu*r)/(L*(nu**2 - 1))
                p12 = -E11*h*j2*sini2x*sinj2t/(2.0*nu + 2.0)
                p20 = E11*cosi2x*cosj2t*h*j2*nu/(-nu**2 + 1)
                p21 = E11*cosi2x*cosj2t*h*j2/(-nu**2 + 1)
                p22 = -0.5*E11*h*sinj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*(nu + 1))
                p30 = E11*cosi2x*h*j2*nu*sinj2t/(nu**2 - 1)
                p31 = E11*cosi2x*h*j2*sinj2t/(nu**2 - 1)
                p32 = -0.5*E11*cosj2t*h*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*(nu + 1))
                p40 = E11*cosa*h*nu*sini2x*sinj2t/(-nu**2 + 1)
                p41 = E11*cosa*h*sini2x*sinj2t/(-nu**2 + 1)
                p50 = E11*cosa*cosj2t*h*nu*sini2x/(-nu**2 + 1)
                p51 = E11*cosa*cosj2t*h*sini2x/(-nu**2 + 1)

                for k1 in range(i0, m1+i0):
                    # access buffer q_1
                    q02 = k0Lq_1_q02[k1-i0]
                    q22 = k0Lq_1_q22[k1-i0]

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
                    out[c] = beta*out[c] + alpha*(p40*q02)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p50*q02)

                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        # access buffer q_2
                        pos = (k2-i0)*n2 + (l2-j0)
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
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q04 + p51*q14)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q05 + p51*q15)

    free(wxs)
    free(wts)
    free(w0xs)
    free(w0ts)

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


def calc_kLL(double [:] coeffs,
             double alpharad, double r2, double L, double tLA,
             double E11, double nu, double h,
             int m1, int m2, int n2,
             int nx, int nt, int num_cores, str method,
             double [:] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef long [:] rows, cols
    cdef double [:] kLLv

    cdef int fdim
    cdef cc_attributes args

    fdim = 1*m1**2 + 2*m1*m2*n2 + 4*m2**2*n2**2

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
    args.E11 = &E11
    args.nu = &nu
    args.h = &h
    args.m1 = &m1
    args.m2 = &m2
    args.n2 = &n2
    args.coeffs = &coeffs[0]
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    integratev(<void *>cfkLL, fdim, &kLLv[0], xa, xb, nx, ta, tb, nt,
               &args, num_cores, method)

    c = -1

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0
        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0

            #NOTE symmetry
            if row > col:
                continue

            # kLL_11
            c += 1
            rows[c] = row+2
            cols[c] = col+2

        for k2 in range(i0, m2+i0):
            for l2 in range(j0, n2+j0):
                col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                # kLL_12
                c += 1
                rows[c] = row+2
                cols[c] = col+4
                c += 1
                rows[c] = row+2
                cols[c] = col+5

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                    #NOTE symmetry
                    if row > col:
                        continue

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

    kLL = coo_matrix((kLLv, (rows, cols)), shape=(size, size))

    return kLL


cdef void cfkLL(int npts, double *xs, double *ts, double *out,
                double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2
    cdef int c, i, pos, row, col

    cdef double p20, p21, p22, p40, p41, p42
    cdef double p50, p51, p52
    cdef double q02, q04, q05, q14, q15, q22, q24, q25

    cdef double r, x, t, alpha, beta

    cdef double E11, nu, h
    cdef double *coeffs
    cdef double *c0
    cdef double sina, r2, L
    cdef int m0, n0, m1, m2, n2
    cdef double wx, wt, w0x, w0t

    cdef cc_attributes *args_in=<cc_attributes *>args
    sina = args_in.sina[0]
    r2 = args_in.r2[0]
    L = args_in.L[0]
    E11 = args_in.E11[0]
    nu = args_in.nu[0]
    h = args_in.h[0]
    m1 = args_in.m1[0]
    m2 = args_in.m2[0]
    n2 = args_in.n2[0]
    coeffs = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

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
    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfwx(coeffs, m1, m2, n2, xs, ts, npts, L, wxs)
    cfwt(coeffs, m1, m2, n2, xs, ts, npts, L, wts)
    cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        wx = wxs[i]
        wt = wts[i]
        w0x = w0xs[i]
        w0t = w0ts[i]
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(i0, m1+i0):
            vcosi1x[i1-i0] = cos(pi*i1*x/L)

        for i2 in range(i0, m2+i0):
            vsini2x[i2-i0] = sin(pi*i2*x/L)
            vcosi2x[i2-i0] = cos(pi*i2*x/L)

        for j2 in range(j0, n2+j0):
            vsinj2t[j2-j0] = sin(j2*t)
            vcosj2t[j2-j0] = cos(j2*t)

        r = r2 + sina*x

        c = -1

        for k2 in range(i0, m2+i0):
            sink2x = vsini2x[k2-i0]
            cosk2x = vcosi2x[k2-i0]
            for l2 in range(j0, n2+j0):
                sinl2t = vsinj2t[l2-j0]
                cosl2t = vcosj2t[l2-j0]

                # q_2
                q04 = pi*cosk2x*k2*sinl2t*(w0x + wx)/L
                q05 = pi*cosk2x*cosl2t*k2*(w0x + wx)/L
                q14 = cosl2t*l2*sink2x*(w0t + wt)/(r*r)
                q15 = -l2*sink2x*sinl2t*(w0t + wt)/(r*r)
                q24 = (L*cosl2t*l2*sink2x*(w0x + wx) + pi*cosk2x*k2*sinl2t*(w0t + wt))/(L*r)
                q25 = (-L*l2*sink2x*sinl2t*(w0x + wx) + pi*cosk2x*cosl2t*k2*(w0t + wt))/(L*r)

                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                kLLq_2_q04[pos] = q04
                kLLq_2_q05[pos] = q05
                kLLq_2_q14[pos] = q14
                kLLq_2_q15[pos] = q15
                kLLq_2_q24[pos] = q24
                kLLq_2_q25[pos] = q25

        for i1 in range(i0, m1+i0):
            row = (i1-i0)*num1 + num0

            cosi1x = vcosi1x[i1-i0]

            # p_1
            p20 = -pi*E11*cosi1x*h*i1*r*(w0x + wx)/(L*(nu**2 - 1))
            p21 = -pi*E11*cosi1x*h*i1*nu*r*(w0x + wx)/(L*(nu**2 - 1))
            p22 = 0.5*pi*E11*cosi1x*h*i1*(w0t + wt)/(L*(nu + 1))

            for k1 in range(i0, m1+i0):
                col = (k1-i0)*num1 + num0

                #NOTE symmetry
                if row > col:
                    continue

                cosk1x = vcosi1x[k1-i0]

                # q_1
                q02 = pi*cosk1x*k1*(w0x + wx)/L
                q22 = pi*cosk1x*k1*(w0t + wt)/(L*r)

                # kLL_11
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    # access buffer q_2
                    pos = (k2-i0)*n2 + (l2-j0)
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

        for i2 in range(i0, m2+i0):
            sini2x = vsini2x[i2-i0]
            cosi2x = vcosi2x[i2-i0]
            for j2 in range(j0, n2+j0):
                row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]

                # p_2
                p40 = -E11*h*(L*cosj2t*j2*nu*sini2x*(w0t + wt) + pi*cosi2x*i2*r**2*sinj2t*(w0x + wx))/(L*r*(nu**2 - 1))
                p41 = -E11*h*(L*cosj2t*j2*sini2x*(w0t + wt) + pi*cosi2x*i2*nu*r**2*sinj2t*(w0x + wx))/(L*r*(nu**2 - 1))
                p42 = 0.5*E11*h*(L*cosj2t*j2*sini2x*(w0x + wx) + pi*cosi2x*i2*sinj2t*(w0t + wt))/(L*(nu + 1))
                p50 = E11*h*(L*j2*nu*sini2x*sinj2t*(w0t + wt) - pi*cosi2x*cosj2t*i2*r**2*(w0x + wx))/(L*r*(nu**2 - 1))
                p51 = E11*h*(L*j2*sini2x*sinj2t*(w0t + wt) - pi*cosi2x*cosj2t*i2*nu*r**2*(w0x + wx))/(L*r*(nu**2 - 1))
                p52 = 0.5*E11*h*(-L*j2*sini2x*sinj2t*(w0x + wx) + pi*cosi2x*cosj2t*i2*(w0t + wt))/(L*(nu + 1))

                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                        #NOTE symmetry
                        if row > col:
                            continue

                        # access buffer q_2
                        pos = (k2-i0)*n2 + (l2-j0)
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

    free(wxs)
    free(wts)
    free(w0xs)
    free(w0ts)

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
