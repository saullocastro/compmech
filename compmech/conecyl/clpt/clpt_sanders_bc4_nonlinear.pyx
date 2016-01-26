#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
include 'clpt_nonlinear_header.pyx'

from compmech.conecyl.clpt.clpt_commons_bc4 cimport cfwx, cfwt, cfN, cfv


cdef int NL_kinematics=1 # to use cfstrain_sanders in cfN


def calc_k0L(np.ndarray[cDOUBLE, ndim=1] coeffs,
             double alpharad, double r2, double L, double tLA,
             np.ndarray[cDOUBLE, ndim=2] F,
             int m1, int m2, int n2,
             int nx, int nt, int num_cores, str method,
             np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] k0Lv

    cdef cc_attributes args

    fdim = (3 + (6+3)*m1 + (12+6)*m2*n2 + 6*m1**2 + 2*12*m1*m2*n2
            + 24*m2**2*n2**2)

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
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    integratev(<void *>cfk0L, fdim, k0Lv, xa, xb, nx, ta, tb, nt,
               &args, num_cores, method)

    c = -1

    # k0L_00
    c += 1
    rows[c] = 0
    cols[c] = 1
    c += 1
    rows[c] = 1
    cols[c] = 1
    c += 1
    rows[c] = 2
    cols[c] = 1

    for k1 in range(i0, m1+i0):
        col = (k1-i0)*num1 + num0
        # k0L_01
        c += 1
        rows[c] = 0
        cols[c] = col+1
        c += 1
        rows[c] = 0
        cols[c] = col+2
        c += 1
        rows[c] = 1
        cols[c] = col+1
        c += 1
        rows[c] = 1
        cols[c] = col+2
        c += 1
        rows[c] = 2
        cols[c] = col+1
        c += 1
        rows[c] = 2
        cols[c] = col+2

    for k2 in range(i0, m2+i0):
        for l2 in range(j0, n2+j0):
            col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
            # k0L_02
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
            c += 1
            rows[c] = 2
            cols[c] = col+4
            c += 1
            rows[c] = 2
            cols[c] = col+5

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0
        # k0L_10
        c += 1
        rows[c] = row+0
        cols[c] = 1
        c += 1
        rows[c] = row+1
        cols[c] = 1
        c += 1
        rows[c] = row+2
        cols[c] = 1

        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0
            # k0L_11
            c += 1
            rows[c] = row+0
            cols[c] = col+1
            c += 1
            rows[c] = row+0
            cols[c] = col+2
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

        for k2 in range(i0, m2+i0):
            for l2 in range(j0, n2+j0):
                col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                # k0L_12
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

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            # k0L_20
            c += 1
            rows[c] = row+0
            cols[c] = 1
            c += 1
            rows[c] = row+1
            cols[c] = 1
            c += 1
            rows[c] = row+2
            cols[c] = 1
            c += 1
            rows[c] = row+3
            cols[c] = 1
            c += 1
            rows[c] = row+4
            cols[c] = 1
            c += 1
            rows[c] = row+5
            cols[c] = 1

            for k1 in range(i0, m1+i0):
                col = (k1-i0)*num1 + num0
                # k0L_21
                c += 1
                rows[c] = row+0
                cols[c] = col+1
                c += 1
                rows[c] = row+0
                cols[c] = col+2
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
                c += 1
                rows[c] = row+3
                cols[c] = col+1
                c += 1
                rows[c] = row+3
                cols[c] = col+2
                c += 1
                rows[c] = row+4
                cols[c] = col+1
                c += 1
                rows[c] = row+4
                cols[c] = col+2
                c += 1
                rows[c] = row+5
                cols[c] = col+1
                c += 1
                rows[c] = row+5
                cols[c] = col+2

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                    # k0L_22
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

    k0L = coo_matrix((k0Lv, (rows, cols)), shape=(size, size))

    return k0L


cdef void cfk0L(int npts, double *xs, double *ts, double *out,
                double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2
    cdef int c, i, pos

    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66

    cdef double p00, p01, p02, p10, p11, p12, p20, p21, p22, p30, p31, p32
    cdef double p40, p41, p42, p50, p51, p52
    cdef double q02, q04, q05, q11, q12, q13, q14, q15
    cdef double q21, q22, q23, q24, q25

    cdef double r, x, t, alpha, beta

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double sina, cosa, tLA, r2, L
    cdef int m0, n0, m1, m2, n2
    cdef double wx, wt, w0x, w0t, v

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
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

    A11 = F[0]
    A12 = F[1]
    A16 = F[2]
    A22 = F[7]
    A26 = F[8]
    A66 = F[14]
    B11 = F[3]
    B12 = F[4]
    B16 = F[5]
    B22 = F[10]
    B26 = F[11]
    B66 = F[17]

    cdef double sini1x, cosi1x, sink1x, cosk1x, sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t
    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *k0Lq_1_q02 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q11 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q21 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q22 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_2_q04 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q05 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q12 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q13 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q14 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q15 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q22 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q23 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q24 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q25 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *vs = <double *>malloc(npts * sizeof(double))
    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfv(coeffs, m1, m2, n2, xs, ts, npts, r2, L, vs)
    cfwx(coeffs, m1, m2, n2, xs, ts, npts, L, wxs)
    cfwt(coeffs, m1, m2, n2, xs, ts, npts, L, wts)
    cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        v = vs[i]
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
        p00 = (-A11*r + A12*sina*(L - x))/(L*cosa)
        p01 = (-A12*r + A22*sina*(L - x))/(L*cosa)
        p02 = (-A16*r + A26*sina*(L - x))/(L*cosa)
        p10 = -r2*(r + sina*(L - x))*(A16*r + B16*cosa)/(L*r)
        p11 = -r2*(r + sina*(L - x))*(A26*r + B26*cosa)/(L*r)
        p12 = -r2*(r + sina*(L - x))*(A66*r + B66*cosa)/(L*r)
        p20 = (A16*(L - x)*sin(t - tLA) + (A11*r + A12*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        p21 = (A26*(L - x)*sin(t - tLA) + (A12*r + A22*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        p22 = (A66*(L - x)*sin(t - tLA) + (A16*r + A26*sina*(-L + x))*(cos(t - tLA) - 1))/(L*cosa)
        # q_0
        q11 = cosa*r2*(L - x)*(cosa*v - w0t - wt)/(L*(r*r))
        q21 = -cosa*r2*(L - x)*(w0x + wx)/(L*r)
        # k0L_00
        c += 1
        out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
        c += 1
        out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)

        for k1 in range(i0, m1+i0):
            sink1x = vsini1x[k1-i0]
            cosk1x = vcosi1x[k1-i0]
            # q_1
            q02 = pi*cosk1x*k1*(w0x + wx)/L
            q11 = cosa*sink1x*(cosa*v - w0t - wt)/(r*r)
            q21 = -cosa*sink1x*(w0x + wx)/r
            q22 = pi*cosk1x*k1*(-cosa*v + w0t + wt)/(L*r)
            # k0L_01
            c += 1
            out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)
            # create buffer q_1
            pos = k1-i0
            k0Lq_1_q02[pos] = q02
            k0Lq_1_q11[pos] = q11
            k0Lq_1_q21[pos] = q21
            k0Lq_1_q22[pos] = q22

        for k2 in range(i0, m2+i0):
            sink2x = vsini2x[k2-i0]
            cosk2x = vcosi2x[k2-i0]
            for l2 in range(j0, n2+j0):
                sinl2t = vsinj2t[l2-j0]
                cosl2t = vcosj2t[l2-j0]
                # q_2
                q04 = pi*cosk2x*k2*sinl2t*(w0x + wx)/L
                q05 = pi*cosk2x*cosl2t*k2*(w0x + wx)/L
                q12 = cosa*cosk2x*sinl2t*(cosa*v - w0t - wt)/(r*r)
                q13 = cosa*cosk2x*cosl2t*(cosa*v - w0t - wt)/(r*r)
                q14 = cosl2t*l2*sink2x*(-cosa*v + w0t + wt)/(r*r)
                q15 = l2*sink2x*sinl2t*(cosa*v - w0t - wt)/(r*r)
                q22 = -cosa*cosk2x*sinl2t*(w0x + wx)/r
                q23 = -cosa*cosk2x*cosl2t*(w0x + wx)/r
                q24 = (L*cosl2t*l2*sink2x*(w0x + wx) + pi*cosk2x*k2*sinl2t*(-cosa*v + w0t + wt))/(L*r)
                q25 = (-L*l2*sink2x*sinl2t*(w0x + wx) + pi*cosk2x*cosl2t*k2*(-cosa*v + w0t + wt))/(L*r)
                # k0L_02
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q12 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q13 + p02*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q13 + p12*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q13 + p22*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)
                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                k0Lq_2_q04[pos] = q04
                k0Lq_2_q05[pos] = q05
                k0Lq_2_q12[pos] = q12
                k0Lq_2_q13[pos] = q13
                k0Lq_2_q14[pos] = q14
                k0Lq_2_q15[pos] = q15
                k0Lq_2_q22[pos] = q22
                k0Lq_2_q23[pos] = q23
                k0Lq_2_q24[pos] = q24
                k0Lq_2_q25[pos] = q25

        for i1 in range(i0, m1+i0):
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]
            # p_1
            p00 = pi*A11*cosi1x*i1*r/L + A12*sina*sini1x
            p01 = pi*A12*cosi1x*i1*r/L + A22*sina*sini1x
            p02 = pi*A16*cosi1x*i1*r/L + A26*sina*sini1x
            p10 = (A16*r + B16*cosa)*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*r)
            p11 = (A26*r + B26*cosa)*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*r)
            p12 = (A66*r + B66*cosa)*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*r)
            p20 = (-pi*B12*L*cosi1x*i1*sina + sini1x*(A12*(L*L)*cosa + (pi*pi)*B11*(i1*i1)*r))/(L*L)
            p21 = (-pi*B22*L*cosi1x*i1*sina + sini1x*(A22*(L*L)*cosa + (pi*pi)*B12*(i1*i1)*r))/(L*L)
            p22 = (-pi*B26*L*cosi1x*i1*sina + sini1x*(A26*(L*L)*cosa + (pi*pi)*B16*(i1*i1)*r))/(L*L)
            # q_0
            q11 = cosa*r2*(L - x)*(cosa*v - w0t - wt)/(L*(r*r))
            q21 = -cosa*r2*(L - x)*(w0x + wx)/(L*r)
            # k0L_10
            c += 1
            out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)

            for k1 in range(i0, m1+i0):
                # access buffer q_1
                pos = k1-i0
                q02 = k0Lq_1_q02[pos]
                q11 = k0Lq_1_q11[pos]
                q21 = k0Lq_1_q21[pos]
                q22 = k0Lq_1_q22[pos]
                # k0L_11
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    # access buffer q_2
                    pos = (k2-i0)*n2 + (l2-j0)
                    q04 = k0Lq_2_q04[pos]
                    q05 = k0Lq_2_q05[pos]
                    q12 = k0Lq_2_q12[pos]
                    q13 = k0Lq_2_q13[pos]
                    q14 = k0Lq_2_q14[pos]
                    q15 = k0Lq_2_q15[pos]
                    q22 = k0Lq_2_q22[pos]
                    q23 = k0Lq_2_q23[pos]
                    q24 = k0Lq_2_q24[pos]
                    q25 = k0Lq_2_q25[pos]
                    # k0L_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p01*q12 + p02*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p01*q13 + p02*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q13 + p12*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q13 + p22*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)

        for i2 in range(i0, m2+i0):
            sini2x = vsini2x[i2-i0]
            cosi2x = vcosi2x[i2-i0]
            for j2 in range(j0, n2+j0):
                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]
                # p_2
                p00 = -pi*A11*i2*r*sini2x*sinj2t/L + cosi2x*(A12*sina*sinj2t + A16*cosj2t*j2)
                p01 = -pi*A12*i2*r*sini2x*sinj2t/L + cosi2x*(A22*sina*sinj2t + A26*cosj2t*j2)
                p02 = -pi*A16*i2*r*sini2x*sinj2t/L + cosi2x*(A26*sina*sinj2t + A66*cosj2t*j2)
                p10 = -pi*A11*cosj2t*i2*r*sini2x/L + cosi2x*(A12*cosj2t*sina - A16*j2*sinj2t)
                p11 = -pi*A12*cosj2t*i2*r*sini2x/L + cosi2x*(A22*cosj2t*sina - A26*j2*sinj2t)
                p12 = -pi*A16*cosj2t*i2*r*sini2x/L + cosi2x*(A26*cosj2t*sina - A66*j2*sinj2t)
                p20 = (L*cosi2x*cosj2t*j2*(A12*r + B12*cosa) - sinj2t*(A16*r + B16*cosa)*(L*cosi2x*sina + pi*i2*r*sini2x))/(L*r)
                p21 = (L*cosi2x*cosj2t*j2*(A22*r + B22*cosa) - sinj2t*(A26*r + B26*cosa)*(L*cosi2x*sina + pi*i2*r*sini2x))/(L*r)
                p22 = (L*cosi2x*cosj2t*j2*(A26*r + B26*cosa) - sinj2t*(A66*r + B66*cosa)*(L*cosi2x*sina + pi*i2*r*sini2x))/(L*r)
                p30 = -(L*cosi2x*j2*sinj2t*(A12*r + B12*cosa) + cosj2t*(A16*r + B16*cosa)*(L*cosi2x*sina + pi*i2*r*sini2x))/(L*r)
                p31 = -(L*cosi2x*j2*sinj2t*(A22*r + B22*cosa) + cosj2t*(A26*r + B26*cosa)*(L*cosi2x*sina + pi*i2*r*sini2x))/(L*r)
                p32 = -(L*cosi2x*j2*sinj2t*(A26*r + B26*cosa) + cosj2t*(A66*r + B66*cosa)*(L*cosi2x*sina + pi*i2*r*sini2x))/(L*r)
                p40 = (-pi*L*cosi2x*i2*r*(B12*sina*sinj2t + 2*B16*cosj2t*j2) + sini2x*(B16*(L*L)*cosj2t*j2*sina + sinj2t*(B12*(L*L)*(j2*j2) + r*(A12*(L*L)*cosa + (pi*pi)*B11*(i2*i2)*r))))/((L*L)*r)
                p41 = (-pi*L*cosi2x*i2*r*(B22*sina*sinj2t + 2*B26*cosj2t*j2) + sini2x*(B26*(L*L)*cosj2t*j2*sina + sinj2t*(B22*(L*L)*(j2*j2) + r*(A22*(L*L)*cosa + (pi*pi)*B12*(i2*i2)*r))))/((L*L)*r)
                p42 = (-pi*L*cosi2x*i2*r*(B26*sina*sinj2t + 2*B66*cosj2t*j2) + sini2x*(B66*(L*L)*cosj2t*j2*sina + sinj2t*(B26*(L*L)*(j2*j2) + r*(A26*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/((L*L)*r)
                p50 = (pi*L*cosi2x*i2*r*(-B12*cosj2t*sina + 2*B16*j2*sinj2t) + sini2x*(-B16*(L*L)*j2*sina*sinj2t + cosj2t*(B12*(L*L)*(j2*j2) + r*(A12*(L*L)*cosa + (pi*pi)*B11*(i2*i2)*r))))/((L*L)*r)
                p51 = (pi*L*cosi2x*i2*r*(-B22*cosj2t*sina + 2*B26*j2*sinj2t) + sini2x*(-B26*(L*L)*j2*sina*sinj2t + cosj2t*(B22*(L*L)*(j2*j2) + r*(A22*(L*L)*cosa + (pi*pi)*B12*(i2*i2)*r))))/((L*L)*r)
                p52 = (pi*L*cosi2x*i2*r*(-B26*cosj2t*sina + 2*B66*j2*sinj2t) + sini2x*(-B66*(L*L)*j2*sina*sinj2t + cosj2t*(B26*(L*L)*(j2*j2) + r*(A26*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/((L*L)*r)
                # q_0
                q11 = cosa*r2*(L - x)*(cosa*v - w0t - wt)/(L*(r*r))
                q21 = -cosa*r2*(L - x)*(w0x + wx)/(L*r)
                # k0L_20
                c += 1
                out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p31*q11 + p32*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p41*q11 + p42*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p51*q11 + p52*q21)

                for k1 in range(i0, m1+i0):
                    # access buffer q_1
                    pos = k1-i0
                    q02 = k0Lq_1_q02[pos]
                    q11 = k0Lq_1_q11[pos]
                    q21 = k0Lq_1_q21[pos]
                    q22 = k0Lq_1_q22[pos]
                    # k0L_21
                    c += 1
                    out[c] = beta*out[c] + alpha*(p01*q11 + p02*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p31*q11 + p32*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p30*q02 + p32*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p41*q11 + p42*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p40*q02 + p42*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p51*q11 + p52*q21)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p50*q02 + p52*q22)

                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        # access buffer q_2
                        pos = (k2-i0)*n2 + (l2-j0)
                        q04 = k0Lq_2_q04[pos]
                        q05 = k0Lq_2_q05[pos]
                        q12 = k0Lq_2_q12[pos]
                        q13 = k0Lq_2_q13[pos]
                        q14 = k0Lq_2_q14[pos]
                        q15 = k0Lq_2_q15[pos]
                        q22 = k0Lq_2_q22[pos]
                        q23 = k0Lq_2_q23[pos]
                        q24 = k0Lq_2_q24[pos]
                        q25 = k0Lq_2_q25[pos]
                        # k0L_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p01*q12 + p02*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p01*q13 + p02*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q04 + p01*q14 + p02*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q05 + p01*q15 + p02*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p11*q13 + p12*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p21*q13 + p22*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p31*q12 + p32*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p31*q13 + p32*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q04 + p31*q14 + p32*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q05 + p31*q15 + p32*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p41*q12 + p42*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p41*q13 + p42*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14 + p42*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15 + p42*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p51*q12 + p52*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p51*q13 + p52*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q04 + p51*q14 + p52*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q05 + p51*q15 + p52*q25)

    free(vs)
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
    free(k0Lq_1_q11)
    free(k0Lq_1_q21)
    free(k0Lq_1_q22)
    free(k0Lq_2_q04)
    free(k0Lq_2_q05)
    free(k0Lq_2_q12)
    free(k0Lq_2_q13)
    free(k0Lq_2_q14)
    free(k0Lq_2_q15)
    free(k0Lq_2_q22)
    free(k0Lq_2_q23)
    free(k0Lq_2_q24)
    free(k0Lq_2_q25)


def calc_kG(np.ndarray[cDOUBLE, ndim=1] coeffs,
            double alpharad, double r2, double L, double tLA,
            np.ndarray[cDOUBLE, ndim=2] F,
            int m1, int m2, int n2,
            int nx, int nt, int num_cores, str method,
            np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kGv

    cdef cc_attributes args

    fdim = 1 + 2*m1 + 4*m2*n2 + 4*m1**2 + 8*m1*m2*n2 + 16*m2**2*n2**2

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
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    integratev(<void *>cfkG, fdim, kGv, xa, xb, nx, ta, tb, nt,
               &args, num_cores, method)

    c = -1

    # kG_00
    c += 1
    rows[c] = 1
    cols[c] = 1

    for k1 in range(i0, m1+i0):
        col = (k1-i0)*num1 + num0
        # kG_01
        c += 1
        rows[c] = 1
        cols[c] = col+1
        c += 1
        rows[c] = 1
        cols[c] = col+2

    for k2 in range(i0, m2+i0):
        for l2 in range(j0, n2+j0):
            col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
            # kG_02
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

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0

        # kG_10
        # not necessary because of the symmetry!

        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0

            #NOTE symmetry
            if row > col:
                continue

            # kG_11
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

        for k2 in range(i0, m2+i0):
            for l2 in range(j0, n2+j0):
                col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                # kG_12
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

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

            # kG_02
            # not necessary because of the symmetry!

            # kG_21
            # not necessary because of the symmetry!

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                    #NOTE symmetry
                    if row > col:
                        continue

                    # kG_22
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

    kG = coo_matrix((kGv, (rows, cols)), shape=(size, size))

    return kG


cdef void cfkG(int npts, double *xs, double *ts, double *out,
               double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2

    cdef double p10, p11, p12, p20, p21, p22, p30, p31, p32
    cdef double p40, p41, p42, p50, p51, p52
    cdef double q02, q04, q05, q14, q15, q21, q22, q23

    cdef double r, x, t, alpha, beta
    cdef int c, i, pos, row, col

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double r2, L, sina, cosa, tLA
    cdef int m0, n0, m1, m2, n2
    cdef double Nxx, Ntt, Nxt

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
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

    cdef double sini1x, cosi1x, sink1x, cosk1x
    cdef double sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t
    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *kGq_1_q02 = <double *>malloc(m1 * sizeof(double))
    cdef double *kGq_1_q21 = <double *>malloc(m1 * sizeof(double))
    cdef double *kGq_2_q04 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q05 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q14 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q15 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q22 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q23 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *Ns = <double *>malloc(e_num*npts * sizeof(double))

    cfN(coeffs, sina, cosa, tLA, xs, ts, npts, r2, L, F, m1, m2, n2,
        c0, m0, n0, funcnum, Ns, NL_kinematics)

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        Nxx = Ns[e_num*i + 0]
        Ntt = Ns[e_num*i + 1]
        Nxt = Ns[e_num*i + 2]
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
        p10 = Nxt*cosa*r2*(-L + x)/L
        p11 = Ntt*cosa*r2*(-L + x)/L
        p12 = Ntt*cosa*r2*(L - x)/L

        # q_0
        q21 = cosa*r2*(L - x)/(L*r)

        # kG_00
        c += 1
        out[c] = beta*out[c] + alpha*(p12*q21)

        for k1 in range(i0, m1+i0):
            sink1x = vsini1x[k1-i0]
            cosk1x = vcosi1x[k1-i0]

            # q_1
            q02 = pi*cosk1x*k1/L
            q21 = cosa*sink1x/r

            # kG_01
            c += 1
            out[c] = beta*out[c] + alpha*(p12*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q02)

            # create buffer q_1
            pos = k1-i0
            kGq_1_q02[pos] = q02
            kGq_1_q21[pos] = q21

        for k2 in range(i0, m2+i0):
            sink2x = vsini2x[k2-i0]
            cosk2x = vcosi2x[k2-i0]
            for l2 in range(j0, n2+j0):
                sinl2t = vsinj2t[l2-j0]
                cosl2t = vcosj2t[l2-j0]

                # q_2
                q04 = pi*cosk2x*k2*sinl2t/L
                q05 = pi*cosk2x*cosl2t*k2/L
                q14 = cosl2t*l2*sink2x/r
                q15 = -l2*sink2x*sinl2t/r
                q22 = cosa*cosk2x*sinl2t/r
                q23 = cosa*cosk2x*cosl2t/r

                # kG_02
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15)

                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                kGq_2_q04[pos] = q04
                kGq_2_q05[pos] = q05
                kGq_2_q14[pos] = q14
                kGq_2_q15[pos] = q15
                kGq_2_q22[pos] = q22
                kGq_2_q23[pos] = q23

        for i1 in range(i0, m1+i0):
            row = (i1-i0)*num1 + num0

            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]

            # p_1
            p10 = -Nxt*cosa*sini1x
            p11 = -Ntt*cosa*sini1x
            p12 = Ntt*cosa*sini1x
            p20 = pi*Nxx*cosi1x*i1*r/L
            p21 = pi*Nxt*cosi1x*i1*r/L
            p22 = -pi*Nxt*cosi1x*i1*r/L

            for k1 in range(i0, m1+i0):
                col = (k1-i0)*num1 + num0

                #NOTE symmetry
                if row > col:
                    continue

                # access buffer q_1
                pos = k1-i0
                q02 = kGq_1_q02[pos]
                q21 = kGq_1_q21[pos]

                # kG_11
                c += 1
                out[c] = beta*out[c] + alpha*(p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02)
                c += 1
                out[c] = beta*out[c] + alpha*(p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02)

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    # access buffer q_2
                    pos = (k2-i0)*n2 + (l2-j0)
                    q04 = kGq_2_q04[pos]
                    q05 = kGq_2_q05[pos]
                    q14 = kGq_2_q14[pos]
                    q15 = kGq_2_q15[pos]
                    q22 = kGq_2_q22[pos]
                    q23 = kGq_2_q23[pos]

                    # kG_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p12*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p12*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p22*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15)

        for i2 in range(i0, m2+i0):
            sini2x = vsini2x[i2-i0]
            cosi2x = vcosi2x[i2-i0]
            for j2 in range(j0, n2+j0):
                row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]

                # p_2
                p20 = -Nxt*cosa*cosi2x*sinj2t
                p21 = -Ntt*cosa*cosi2x*sinj2t
                p22 = Ntt*cosa*cosi2x*sinj2t
                p30 = -Nxt*cosa*cosi2x*cosj2t
                p31 = -Ntt*cosa*cosi2x*cosj2t
                p32 = Ntt*cosa*cosi2x*cosj2t
                p40 = Nxt*cosj2t*j2*sini2x + pi*Nxx*cosi2x*i2*r*sinj2t/L
                p41 = Ntt*cosj2t*j2*sini2x + pi*Nxt*cosi2x*i2*r*sinj2t/L
                p42 = -(L*Ntt*cosj2t*j2*sini2x + pi*Nxt*cosi2x*i2*r*sinj2t)/L
                p50 = -Nxt*j2*sini2x*sinj2t + pi*Nxx*cosi2x*cosj2t*i2*r/L
                p51 = -Ntt*j2*sini2x*sinj2t + pi*Nxt*cosi2x*cosj2t*i2*r/L
                p52 = Ntt*j2*sini2x*sinj2t - pi*Nxt*cosi2x*cosj2t*i2*r/L

                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                        #NOTE symmetry
                        if row > col:
                            continue

                        # access buffer q_2
                        pos = (k2-i0)*n2 + (l2-j0)
                        q04 = kGq_2_q04[pos]
                        q05 = kGq_2_q05[pos]
                        q14 = kGq_2_q14[pos]
                        q15 = kGq_2_q15[pos]
                        q22 = kGq_2_q22[pos]
                        q23 = kGq_2_q23[pos]

                        # kG_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p22*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p22*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p32*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p32*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q04 + p31*q14)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q05 + p31*q15)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p42*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p42*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p52*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p52*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q04 + p51*q14)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q05 + p51*q15)

    free(Ns)
    free(vsini1x)
    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
    free(kGq_1_q02)
    free(kGq_1_q21)
    free(kGq_2_q04)
    free(kGq_2_q05)
    free(kGq_2_q14)
    free(kGq_2_q15)
    free(kGq_2_q22)
    free(kGq_2_q23)


def calc_kLL(np.ndarray[cDOUBLE, ndim=1] coeffs,
             double alpharad, double r2, double L, double tLA,
             np.ndarray[cDOUBLE, ndim=2] F,
             int m1, int m2, int n2,
             int nx, int nt, int num_cores, str method,
             np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kLLv

    cdef cc_attributes args

    fdim = 1 + 2*m1 + 4*m2*n2 + 4*m1**2 + 8*m1*m2*n2 + 16*m2**2*n2**2

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
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    integratev(<void *>cfkLL, fdim, kLLv, xa, xb, nx, ta, tb, nt,
               &args, num_cores, method)

    c = -1

    # kLL_00
    c += 1
    rows[c] = 1
    cols[c] = 1

    for k1 in range(i0, m1+i0):
        col = (k1-i0)*num1 + num0

        # kLL_01
        c += 1
        rows[c] = 1
        cols[c] = col+1
        c += 1
        rows[c] = 1
        cols[c] = col+2

    for k2 in range(i0, m2+i0):
        for l2 in range(j0, n2+j0):
            col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

            # kLL_02
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

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0

        # kLL_10
        # not necessary because of the symmetry!

        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0

            #NOTE symmetry
            if row > col:
                continue

            # kLL_11
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

        for k2 in range(i0, m2+i0):
            for l2 in range(j0, n2+j0):
                col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                # kLL_12
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

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

            # kLL_20
            # not necessary because of the symmetry!

            # kLL_21
            # not necessary because of the symmetry!

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                    #NOTE symmetry
                    if row > col:
                        continue

                    # kLL_22
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

    kLL = coo_matrix((kLLv, (rows, cols)), shape=(size, size))

    return kLL


cdef void cfkLL(int npts, double *xs, double *ts, double *out,
                double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2
    cdef int c, i, pos, row, col

    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66

    cdef double p10, p11, p12, p20, p21, p22, p30, p31, p32
    cdef double p40, p41, p42, p50, p51, p52
    cdef double q02, q04, q05, q11, q12, q13, q14, q15
    cdef double q21, q22, q23, q24, q25
    cdef double r, x, t, alpha, beta

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double sina, cosa, r2, L
    cdef int m0, n0, m1, m2, n2
    cdef double wx, wt, w0x, w0t, v

    cdef cc_attributes *args_in=<cc_attributes *>args
    sina = args_in.sina[0]
    cosa = args_in.cosa[0]
    r2 = args_in.r2[0]
    L = args_in.L[0]
    F = args_in.F
    m1 = args_in.m1[0]
    m2 = args_in.m2[0]
    n2 = args_in.n2[0]
    coeffs = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

    A11 = F[0]
    A12 = F[1]
    A16 = F[2]
    A22 = F[7]
    A26 = F[8]
    A66 = F[14]
    B11 = F[3]
    B12 = F[4]
    B16 = F[5]
    B22 = F[10]
    B26 = F[11]
    B66 = F[17]

    cdef double sini1x, cosi1x, sink1x, cosk1x, sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t
    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *kLLq_1_q02 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_1_q11 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_1_q21 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_1_q22 = <double *>malloc(m1 * sizeof(double))
    cdef double *kLLq_2_q04 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q05 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q12 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q13 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q14 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q15 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q22 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q23 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q24 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q25 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *vs = <double *>malloc(npts * sizeof(double))
    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfv(coeffs, m1, m2, n2, xs, ts, npts, r2, L, vs)
    cfwx(coeffs, m1, m2, n2, xs, ts, npts, L, wxs)
    cfwt(coeffs, m1, m2, n2, xs, ts, npts, L, wts)
    cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        v = vs[i]
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
        p10 = cosa*r2*(L - x)*(A12*(cosa*v - w0t - wt) - A16*r*(w0x + wx))/(L*r)
        p11 = cosa*r2*(L - x)*(A22*(cosa*v - w0t - wt) - A26*r*(w0x + wx))/(L*r)
        p12 = cosa*r2*(L - x)*(A26*(cosa*v - w0t - wt) - A66*r*(w0x + wx))/(L*r)

        # q_0
        q11 = cosa*r2*(L - x)*(cosa*v - w0t - wt)/(L*(r*r))
        q21 = -cosa*r2*(L - x)*(w0x + wx)/(L*r)

        # kLL_00
        c += 1
        out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)

        for k1 in range(i0, m1+i0):
            sink1x = vsini1x[k1-i0]
            cosk1x = vcosi1x[k1-i0]

            # q_1
            q02 = pi*cosk1x*k1*(w0x + wx)/L
            q11 = cosa*sink1x*(cosa*v - w0t - wt)/(r*r)
            q21 = -cosa*sink1x*(w0x + wx)/r
            q22 = pi*cosk1x*k1*(-cosa*v + w0t + wt)/(L*r)

            # kLL_01
            c += 1
            out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)

            # create buffer q_1
            pos = k1-i0
            kLLq_1_q02[pos] = q02
            kLLq_1_q11[pos] = q11
            kLLq_1_q21[pos] = q21
            kLLq_1_q22[pos] = q22

        for k2 in range(i0, m2+i0):
            sink2x = vsini2x[k2-i0]
            cosk2x = vcosi2x[k2-i0]
            for l2 in range(j0, n2+j0):
                sinl2t = vsinj2t[l2-j0]
                cosl2t = vcosj2t[l2-j0]

                # q_2
                q04 = pi*cosk2x*k2*sinl2t*(w0x + wx)/L
                q05 = pi*cosk2x*cosl2t*k2*(w0x + wx)/L
                q12 = cosa*cosk2x*sinl2t*(cosa*v - w0t - wt)/(r*r)
                q13 = cosa*cosk2x*cosl2t*(cosa*v - w0t - wt)/(r*r)
                q14 = cosl2t*l2*sink2x*(-cosa*v + w0t + wt)/(r*r)
                q15 = l2*sink2x*sinl2t*(cosa*v - w0t - wt)/(r*r)
                q22 = -cosa*cosk2x*sinl2t*(w0x + wx)/r
                q23 = -cosa*cosk2x*cosl2t*(w0x + wx)/r
                q24 = (L*cosl2t*l2*sink2x*(w0x + wx) + pi*cosk2x*k2*sinl2t*(-cosa*v + w0t + wt))/(L*r)
                q25 = (-L*l2*sink2x*sinl2t*(w0x + wx) + pi*cosk2x*cosl2t*k2*(-cosa*v + w0t + wt))/(L*r)

                # kLL_02
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q13 + p12*q23)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)

                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                kLLq_2_q04[pos] = q04
                kLLq_2_q05[pos] = q05
                kLLq_2_q12[pos] = q12
                kLLq_2_q13[pos] = q13
                kLLq_2_q14[pos] = q14
                kLLq_2_q15[pos] = q15
                kLLq_2_q22[pos] = q22
                kLLq_2_q23[pos] = q23
                kLLq_2_q24[pos] = q24
                kLLq_2_q25[pos] = q25

        for i1 in range(i0, m1+i0):
            row = (i1-i0)*num1 + num0

            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]

            # p_1
            p10 = cosa*sini1x*(A12*(cosa*v - w0t - wt) - A16*r*(w0x + wx))/r
            p11 = cosa*sini1x*(A22*(cosa*v - w0t - wt) - A26*r*(w0x + wx))/r
            p12 = cosa*sini1x*(A26*(cosa*v - w0t - wt) - A66*r*(w0x + wx))/r
            p20 = pi*cosi1x*i1*(A11*r*(w0x + wx) + A16*(-cosa*v + w0t + wt))/L
            p21 = pi*cosi1x*i1*(A12*r*(w0x + wx) + A26*(-cosa*v + w0t + wt))/L
            p22 = pi*cosi1x*i1*(A16*r*(w0x + wx) + A66*(-cosa*v + w0t + wt))/L

            for k1 in range(i0, m1+i0):
                col = (k1-i0)*num1 + num0

                #NOTE symmetry
                if row > col:
                    continue

                # access buffer q_1
                pos = k1-i0
                q02 = kLLq_1_q02[pos]
                q11 = kLLq_1_q11[pos]
                q21 = kLLq_1_q21[pos]
                q22 = kLLq_1_q22[pos]

                # kLL_11
                c += 1
                out[c] = beta*out[c] + alpha*(p11*q11 + p12*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p21*q11 + p22*q21)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):

                    # access buffer q_2
                    pos = (k2-i0)*n2 + (l2-j0)
                    q04 = kLLq_2_q04[pos]
                    q05 = kLLq_2_q05[pos]
                    q12 = kLLq_2_q12[pos]
                    q13 = kLLq_2_q13[pos]
                    q14 = kLLq_2_q14[pos]
                    q15 = kLLq_2_q15[pos]
                    q22 = kLLq_2_q22[pos]
                    q23 = kLLq_2_q23[pos]
                    q24 = kLLq_2_q24[pos]
                    q25 = kLLq_2_q25[pos]

                    # kLL_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q12 + p12*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p11*q13 + p12*q23)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q04 + p11*q14 + p12*q24)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p10*q05 + p11*q15 + p12*q25)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p21*q13 + p22*q23)
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
                p20 = cosa*cosi2x*sinj2t*(A12*(cosa*v - w0t - wt) - A16*r*(w0x + wx))/r
                p21 = cosa*cosi2x*sinj2t*(A22*(cosa*v - w0t - wt) - A26*r*(w0x + wx))/r
                p22 = cosa*cosi2x*sinj2t*(A26*(cosa*v - w0t - wt) - A66*r*(w0x + wx))/r
                p30 = cosa*cosi2x*cosj2t*(A12*(cosa*v - w0t - wt) - A16*r*(w0x + wx))/r
                p31 = cosa*cosi2x*cosj2t*(A22*(cosa*v - w0t - wt) - A26*r*(w0x + wx))/r
                p32 = cosa*cosi2x*cosj2t*(A26*(cosa*v - w0t - wt) - A66*r*(w0x + wx))/r
                p40 = (L*cosj2t*j2*sini2x*(A12*(-cosa*v + w0t + wt) + A16*r*(w0x + wx)) + pi*cosi2x*i2*r*sinj2t*(A11*r*(w0x + wx) + A16*(-cosa*v + w0t + wt)))/(L*r)
                p41 = (L*cosj2t*j2*sini2x*(A22*(-cosa*v + w0t + wt) + A26*r*(w0x + wx)) + pi*cosi2x*i2*r*sinj2t*(A12*r*(w0x + wx) + A26*(-cosa*v + w0t + wt)))/(L*r)
                p42 = (L*cosj2t*j2*sini2x*(A26*(-cosa*v + w0t + wt) + A66*r*(w0x + wx)) + pi*cosi2x*i2*r*sinj2t*(A16*r*(w0x + wx) + A66*(-cosa*v + w0t + wt)))/(L*r)
                p50 = (-L*j2*sini2x*sinj2t*(A12*(-cosa*v + w0t + wt) + A16*r*(w0x + wx)) + pi*cosi2x*cosj2t*i2*r*(A11*r*(w0x + wx) + A16*(-cosa*v + w0t + wt)))/(L*r)
                p51 = (-L*j2*sini2x*sinj2t*(A22*(-cosa*v + w0t + wt) + A26*r*(w0x + wx)) + pi*cosi2x*cosj2t*i2*r*(A12*r*(w0x + wx) + A26*(-cosa*v + w0t + wt)))/(L*r)
                p52 = (-L*j2*sini2x*sinj2t*(A26*(-cosa*v + w0t + wt) + A66*r*(w0x + wx)) + pi*cosi2x*cosj2t*i2*r*(A16*r*(w0x + wx) + A66*(-cosa*v + w0t + wt)))/(L*r)

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
                        q12 = kLLq_2_q12[pos]
                        q13 = kLLq_2_q13[pos]
                        q14 = kLLq_2_q14[pos]
                        q15 = kLLq_2_q15[pos]
                        q22 = kLLq_2_q22[pos]
                        q23 = kLLq_2_q23[pos]
                        q24 = kLLq_2_q24[pos]
                        q25 = kLLq_2_q25[pos]

                        # kLL_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p21*q12 + p22*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p21*q13 + p22*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q04 + p21*q14 + p22*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q05 + p21*q15 + p22*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p31*q12 + p32*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p31*q13 + p32*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q04 + p31*q14 + p32*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p30*q05 + p31*q15 + p32*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p41*q12 + p42*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p41*q13 + p42*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q04 + p41*q14 + p42*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p40*q05 + p41*q15 + p42*q25)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p51*q12 + p52*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p51*q13 + p52*q23)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q04 + p51*q14 + p52*q24)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p50*q05 + p51*q15 + p52*q25)

    free(vs)
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
    free(kLLq_1_q02)
    free(kLLq_1_q11)
    free(kLLq_1_q21)
    free(kLLq_1_q22)
    free(kLLq_2_q04)
    free(kLLq_2_q05)
    free(kLLq_2_q12)
    free(kLLq_2_q13)
    free(kLLq_2_q14)
    free(kLLq_2_q15)
    free(kLLq_2_q22)
    free(kLLq_2_q23)
    free(kLLq_2_q24)
    free(kLLq_2_q25)


def calc_fint_0L_L0_LL(np.ndarray[cDOUBLE, ndim=1] coeffs,
              double alpharad, double r2, double L, double tLA,
              np.ndarray[cDOUBLE, ndim=2] F,
              int m1, int m2, int n2,
              int nx, int nt, int num_cores, str method,
              np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef cc_attributes args
    cdef double sina, cosa

    fdim = num0 + num1*m1 + num2*m2*n2
    fint = np.zeros((fdim,), dtype=DOUBLE)

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
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = 0.
    xb = L
    ta = 0.
    tb = 2*pi

    # numerical integration
    integratev(<void *>cffint, fdim, fint, xa, xb, nx, ta, tb, nt,
               &args, num_cores, method)

    return fint


cdef void cffint(int npts, double *xs, double *ts, double *fint,
                 double *alphas, double *betas, void *args) nogil:
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double Nxx0, Ntt0, Nxt0, Mxx0, Mtt0, Mxt0
    cdef double NxxL, NttL, NxtL, MxxL, MttL, MxtL
    cdef double exx0, ett0, gxt0, kxx0, ktt0, kxt0
    cdef double exxL, ettL, gxtL, kxxL, kttL, kxtL
    cdef double x, t, wx, wt, w0x, w0t, w0, v
    cdef double sini1x, cosi1x, sini2x, dsini2x, cosi2x, sinj2t, cosj2t

    cdef double alpha, beta

    cdef double *F
    cdef double *c
    cdef double *c0
    cdef double sina, cosa, tLA, r, r2, L
    cdef int m0, n0, m1, m2, n2
    cdef int i1, i2, j2, i, col

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
    c = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

    A11 = F[0]
    A12 = F[1]
    A16 = F[2]
    A22 = F[7]
    A26 = F[8]
    A66 = F[14]
    B11 = F[3]
    B12 = F[4]
    B16 = F[5]
    B22 = F[10]
    B26 = F[11]
    B66 = F[17]
    D11 = F[18]
    D12 = F[19]
    D16 = F[20]
    D22 = F[25]
    D26 = F[26]
    D66 = F[32]

    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))
    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))

    cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        w0x = w0xs[i]
        w0t = w0ts[i]
        alpha = alphas[i]
        beta = betas[i]
        r = r2 + x*sina

        for i1 in range(i0, m1+i0):
            vsini1x[i1-i0] = sin(pi*i1*x/L)
            vcosi1x[i1-i0] = cos(pi*i1*x/L)

        for i2 in range(i0, m2+i0):
            vsini2x[i2-i0] = sin(pi*i2*x/L)
            vcosi2x[i2-i0] = cos(pi*i2*x/L)

        for j2 in range(j0, n2+j0):
            vsinj2t[j2-j0] = sin(j2*t)
            vcosj2t[j2-j0] = cos(j2*t)

        # v, wx and wt
        v = ((L-x)*r2/L)*c[1]
        wx = 0.
        wt = 0.
        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            v += c[col+1]*vsini1x[i1-i0]
            wx += c[col+2]*i1*pi/L*vcosi1x[i1-i0]
        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                dsini2x = i2*pi/L*vcosi2x[i2-i0]
                sini2x = vsini2x[i2-i0]
                cosi2x = vcosi2x[i2-i0]
                v += c[col+2]*cosi2x*sinj2t
                v += c[col+3]*cosi2x*cosj2t
                wx += c[col+4]*dsini2x*sinj2t
                wx += c[col+5]*dsini2x*cosj2t
                wt += c[col+4]*sini2x*(j2*cosj2t)
                wt += c[col+5]*sini2x*(-j2*sinj2t)

        # strains
        exx0 = 0.
        ett0 = 0.
        gxt0 = 0.
        kxx0 = 0.
        ktt0 = 0.
        kxt0 = 0.

        exxL = 0.
        ettL = 0.
        gxtL = 0.
        kxxL = 0.
        kttL = 0.
        kxtL = 0.

        #TODO if castro=1
        w0 = 0.

        exx0 = (-c[0]/(L*cosa)
                -c[2]*cos(t - tLA)/(L*cosa))
        exxL = 0.5*castro*w0x*w0x

        ett0 = (c[0]*sina*(L - x)/(L*cosa*r)
                +c[2]*sina*(L - x)*cos(t - tLA)/(L*cosa*r))
        ettL = (c[1]*(0.5*cosa*r2*(L - x)*(cosa*v/r - wt/r)/(L*r) - cosa*r2*w0t*(L - x)/(L*(r*r)))
                +castro*(cosa*w0/r + 0.5*w0t*w0t/(r*r)))

        gxt0 = (-c[1]*r2*(r + sina*(L - x))/(L*r)
                +c[2]*(-L + x)*sin(t - tLA)/(L*cosa*r))

        gxtL = (c[1]*(-cosa*r2*w0x*(L - x)/(L*r) - 0.5*cosa*r2*wx*(L - x)/(L*r))
                +castro*w0t*w0x/r)

        kxt0 = -c[1]*cosa*r2*(r + sina*(L - x))/(L*(r*r))

        for i1 in range(i0, m1+i0):
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]
            col = (i1-i0)*num1 + num0

            exx0 += pi*c[col+0]*cosi1x*i1/L
            exxL += c[col+2]*(pi*cosi1x*i1*w0x/L + 0.5*pi*cosi1x*i1*wx/L)

            ett0 += (c[col+0]*sina*sini1x/r
                     +c[col+2]*cosa*sini1x/r)
            ettL += c[col+1]*(0.5*cosa*sini1x*(cosa*v/r - wt/r)/r - cosa*sini1x*w0t/(r*r))

            gxt0 += c[col+1]*(-sina*sini1x/r + pi*cosi1x*i1/L)

            gxtL += (c[col+1]*(-cosa*sini1x*w0x/r - 0.5*cosa*sini1x*wx/r)
                     +c[col+2]*(0.5*pi*cosi1x*i1*(-cosa*v/r + wt/r)/L + pi*cosi1x*i1*w0t/(L*r)))

            kxx0 += (pi*pi)*c[col+2]*(i1*i1)*sini1x/(L*L)

            ktt0 += -pi*c[col+2]*cosi1x*i1*sina/(L*r)

            kxt0 += c[col+1]*cosa*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*(r*r))

        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                sini2x = vsini2x[i2-i0]
                cosi2x = vcosi2x[i2-i0]
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                exx0 += (-pi*c[col+0]*i2*sini2x*sinj2t/L
                         -pi*c[col+1]*cosj2t*i2*sini2x/L)
                exxL += (c[col+4]*(pi*cosi2x*i2*sinj2t*w0x/L + 0.5*pi*cosi2x*i2*sinj2t*wx/L)
                         +c[col+5]*(pi*cosi2x*cosj2t*i2*w0x/L + 0.5*pi*cosi2x*cosj2t*i2*wx/L))

                ett0 += (c[col+0]*cosi2x*sina*sinj2t/r
                         +c[col+1]*cosi2x*cosj2t*sina/r
                         +c[col+2]*cosi2x*cosj2t*j2/r
                         -c[col+3]*cosi2x*j2*sinj2t/r
                         +c[col+4]*cosa*sini2x*sinj2t/r
                         +c[col+5]*cosa*cosj2t*sini2x/r)
                ettL += (c[col+2]*(0.5*cosa*cosi2x*sinj2t*(cosa*v/r - wt/r)/r - cosa*cosi2x*sinj2t*w0t/(r*r))
                         +c[col+3]*(0.5*cosa*cosi2x*cosj2t*(cosa*v/r - wt/r)/r - cosa*cosi2x*cosj2t*w0t/(r*r))
                         +c[col+4]*(0.5*cosj2t*j2*sini2x*(-cosa*v/r + wt/r)/r + cosj2t*j2*sini2x*w0t/(r*r))
                         +c[col+5]*(-0.5*j2*sini2x*sinj2t*(-cosa*v/r + wt/r)/r - j2*sini2x*sinj2t*w0t/(r*r)))

                gxt0 += (c[col+0]*cosi2x*cosj2t*j2/r
                         -c[col+1]*cosi2x*j2*sinj2t/r
                         -c[col+2]*sinj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*r)
                         -c[col+3]*cosj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*r))
                gxtL += (c[col+2]*(-cosa*cosi2x*sinj2t*w0x/r - 0.5*cosa*cosi2x*sinj2t*wx/r)
                         +c[col+3]*(-cosa*cosi2x*cosj2t*w0x/r - 0.5*cosa*cosi2x*cosj2t*wx/r)
                         +c[col+4]*(cosj2t*j2*sini2x*w0x/r + 0.5*cosj2t*j2*sini2x*wx/r + 0.5*pi*cosi2x*i2*sinj2t*(-cosa*v/r + wt/r)/L + pi*cosi2x*i2*sinj2t*w0t/(L*r))
                         +c[col+5]*(-j2*sini2x*sinj2t*w0x/r - 0.5*j2*sini2x*sinj2t*wx/r + 0.5*pi*cosi2x*cosj2t*i2*(-cosa*v/r + wt/r)/L + pi*cosi2x*cosj2t*i2*w0t/(L*r)))

                kxx0 += ((pi*pi)*c[col+4]*(i2*i2)*sini2x*sinj2t/(L*L)
                         +(pi*pi)*c[col+5]*cosj2t*(i2*i2)*sini2x/(L*L))

                ktt0 += (c[col+2]*cosa*cosi2x*cosj2t*j2/(r*r)
                         -c[col+3]*cosa*cosi2x*j2*sinj2t/(r*r)
                         +c[col+4]*sinj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r))
                         +c[col+5]*cosj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r)))

                kxt0 += (-c[col+2]*cosa*sinj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*(r*r))
                         -c[col+3]*cosa*cosj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*(r*r))
                         +c[col+4]*cosj2t*j2*(L*sina*sini2x - 2*pi*cosi2x*i2*r)/(L*(r*r))
                         +c[col+5]*j2*sinj2t*(-L*sina*sini2x + 2*pi*cosi2x*i2*r)/(L*(r*r)))

        # stresses
        Nxx0 = A11*exx0 + A12*ett0 + A16*gxt0 + B11*kxx0 + B12*ktt0 + B16*kxt0
        Ntt0 = A12*exx0 + A22*ett0 + A26*gxt0 + B12*kxx0 + B22*ktt0 + B26*kxt0
        Nxt0 = A16*exx0 + A26*ett0 + A66*gxt0 + B16*kxx0 + B26*ktt0 + B66*kxt0
        Mxx0 = B11*exx0 + B12*ett0 + B16*gxt0 + D11*kxx0 + D12*ktt0 + D16*kxt0
        Mtt0 = B12*exx0 + B22*ett0 + B26*gxt0 + D12*kxx0 + D22*ktt0 + D26*kxt0
        Mxt0 = B16*exx0 + B26*ett0 + B66*gxt0 + D16*kxx0 + D26*ktt0 + D66*kxt0

        NxxL = A11*exxL + A12*ettL + A16*gxtL + B11*kxxL + B12*kttL + B16*kxtL
        NttL = A12*exxL + A22*ettL + A26*gxtL + B12*kxxL + B22*kttL + B26*kxtL
        NxtL = A16*exxL + A26*ettL + A66*gxtL + B16*kxxL + B26*kttL + B66*kxtL
        MxxL = B11*exxL + B12*ettL + B16*gxtL + D11*kxxL + D12*kttL + D16*kxtL
        MttL = B12*exxL + B22*ettL + B26*gxtL + D12*kxxL + D22*kttL + D26*kxtL
        MxtL = B16*exxL + B26*ettL + B66*gxtL + D16*kxxL + D26*kttL + D66*kxtL

        fint[0] = beta*(fint[0]) + alpha*((NttL*sina*(L - x) - NxxL*r)/(L*cosa))
        fint[1] = beta*(fint[1]) + alpha*(r2*(-NxtL*r*(r + sina*(L - x)) + (cosa*cosa)*v*(L - x)*(Ntt0 + NttL) - cosa*(MxtL*(r + sina*(L - x)) + (L - x)*(Ntt0*(w0t + wt) + NttL*(w0t + wt) + r*(Nxt0 + NxtL)*(w0x + wx))))/(L*r))
        fint[2] = beta*(fint[2]) + alpha*((NxtL*(-L + x)*sin(t - tLA) - (NttL*sina*(-L + x) + NxxL*r)*cos(t - tLA))/(L*cosa))

        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]
            fint[col+0] = beta*(fint[col+0]) + alpha*(NttL*sina*sini1x + pi*NxxL*cosi1x*i1*r/L)
            fint[col+1] = beta*(fint[col+1]) + alpha*((L*sini1x*(-NxtL*r*sina + (cosa*cosa)*v*(Ntt0 + NttL) - cosa*(MxtL*sina + Ntt0*(w0t + wt) + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx)) + pi*cosi1x*i1*r*(MxtL*cosa + NxtL*r))/(L*r))
            fint[col+2] = beta*(fint[col+2]) + alpha*((-pi*L*cosi1x*i1*(MttL*sina - Nxt0*w0t - Nxt0*wt - NxtL*w0t - NxtL*wt - Nxx0*r*w0x - Nxx0*r*wx - NxxL*r*w0x - NxxL*r*wx + cosa*v*(Nxt0 + NxtL)) + sini1x*((L*L)*NttL*cosa + (pi*pi)*MxxL*(i1*i1)*r))/(L*L))

        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                sini2x = vsini2x[i2-i0]
                cosi2x = vcosi2x[i2-i0]
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                fint[col+0] = beta*(fint[col+0]) + alpha*(NxtL*cosi2x*cosj2t*j2 + sinj2t*(L*NttL*cosi2x*sina - pi*NxxL*i2*r*sini2x)/L)
                fint[col+1] = beta*(fint[col+1]) + alpha*(-NxtL*cosi2x*j2*sinj2t + cosj2t*(NttL*cosi2x*sina - pi*NxxL*i2*r*sini2x/L))
                fint[col+2] = beta*(fint[col+2]) + alpha*((L*cosi2x*cosj2t*j2*(MttL*cosa + NttL*r) + sinj2t*(L*cosi2x*(-NxtL*r*sina + (cosa*cosa)*v*(Ntt0 + NttL) - cosa*(MxtL*sina + Ntt0*(w0t + wt) + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx)) - pi*i2*r*sini2x*(MxtL*cosa + NxtL*r)))/(L*r))
                fint[col+3] = beta*(fint[col+3]) + alpha*((-L*cosi2x*j2*sinj2t*(MttL*cosa + NttL*r) + cosj2t*(L*cosi2x*(-NxtL*r*sina + (cosa*cosa)*v*(Ntt0 + NttL) - cosa*(MxtL*sina + Ntt0*(w0t + wt) + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx)) - pi*i2*r*sini2x*(MxtL*cosa + NxtL*r)))/(L*r))
                fint[col+4] = beta*(fint[col+4]) + alpha*((L*cosj2t*j2*(L*sini2x*(MxtL*sina + Ntt0*w0t + Ntt0*wt + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx - cosa*v*(Ntt0 + NttL)) - 2*pi*MxtL*cosi2x*i2*r) + sinj2t*(pi*L*cosi2x*i2*r*(-MttL*sina + Nxt0*w0t + Nxt0*wt + NxtL*w0t + NxtL*wt + Nxx0*r*w0x + Nxx0*r*wx + NxxL*r*w0x + NxxL*r*wx - cosa*v*(Nxt0 + NxtL)) + sini2x*((L*L)*MttL*(j2*j2) + r*((L*L)*NttL*cosa + (pi*pi)*MxxL*(i2*i2)*r))))/((L*L)*r))
                fint[col+5] = beta*(fint[col+5]) + alpha*((-L*j2*sinj2t*(L*sini2x*(MxtL*sina + Ntt0*w0t + Ntt0*wt + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx - cosa*v*(Ntt0 + NttL)) - 2*pi*MxtL*cosi2x*i2*r) + cosj2t*(pi*L*cosi2x*i2*r*(-MttL*sina + Nxt0*w0t + Nxt0*wt + NxtL*w0t + NxtL*wt + Nxx0*r*w0x + Nxx0*r*wx + NxxL*r*w0x + NxxL*r*wx - cosa*v*(Nxt0 + NxtL)) + sini2x*((L*L)*MttL*(j2*j2) + r*((L*L)*NttL*cosa + (pi*pi)*MxxL*(i2*i2)*r))))/((L*L)*r))

    free(w0xs)
    free(w0ts)
    free(vsini1x)
    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
