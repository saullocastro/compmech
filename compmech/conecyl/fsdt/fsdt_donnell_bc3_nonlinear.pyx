#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
include 'fsdt_nonlinear_header.pyx'

from compmech.conecyl.fsdt.fsdt_commons_bc3 cimport cfwx, cfwt, cfN


def calc_k0L(double [:] coeffs,
             double alpharad, double r2, double L, double tLA,
             double [:, ::1] F,
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

    fdim = 3*m1 + 6*m2*n2 + 5*m1**2 + 2*10*m1*m2*n2 + 20*m2**2*n2**2

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
            c += 1
            rows[c] = row+3
            cols[c] = col+2
            c += 1
            rows[c] = row+4
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
                c += 1
                rows[c] = row+6
                cols[c] = col+2
                c += 1
                rows[c] = row+7
                cols[c] = col+2
                c += 1
                rows[c] = row+8
                cols[c] = col+2
                c += 1
                rows[c] = row+9
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
                    c += 1
                    rows[c] = row+6
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+6
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+7
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+7
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+8
                    cols[c] = col+5
                    c += 1
                    rows[c] = row+9
                    cols[c] = col+4
                    c += 1
                    rows[c] = row+9
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

    cdef double p0000, p0001, p0002, p0100, p0101, p0102
    cdef double p0200, p0201, p0202, p0300, p0301, p0302
    cdef double p0400, p0401, p0402, p0500, p0501, p0502
    cdef double p0600, p0601, p0602, p0700, p0701, p0702
    cdef double p0800, p0801, p0802, p0900, p0901, p0902

    cdef double q0002, q0004, q0005
    cdef double q0104, q0105
    cdef double q0202, q0204, q0205

    cdef double r, x, t, alpha, beta

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double  sina, cosa, tLA, r2, L
    cdef int m0, n0, m1, m2, n2, pti
    cdef double wx, wt, w0x, w0t

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
    A22 = F[9]
    A26 = F[10]
    A66 = F[18]
    B11 = F[3]
    B12 = F[4]
    B16 = F[5]
    B22 = F[12]
    B26 = F[13]
    B66 = F[21]

    cdef double sini1x, cosi1x, cosk1x
    cdef double sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t

    cdef double *vsini1x = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))

    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))

    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))

    cdef double *k0Lq_1_q0002 = <double *>malloc(m1 * sizeof(double))
    cdef double *k0Lq_1_q0202 = <double *>malloc(m1 * sizeof(double))

    cdef double *k0Lq_2_q0004 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0005 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0104 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0105 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0204 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *k0Lq_2_q0205 = <double *>malloc(m2*n2 * sizeof(double))

    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfwx(coeffs, m1, m2, n2, L, xs, ts, npts, wxs)
    cfwt(coeffs, m1, m2, n2, L, xs, ts, npts, wts)
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
        p0000 = (-A11*r + A12*sina*(L - x))/(L*cosa)
        p0001 = (-A12*r + A22*sina*(L - x))/(L*cosa)
        p0002 = (-A16*r + A26*sina*(L - x))/(L*cosa)
        p0100 = -A16*r2*(r + sina*(L - x))/L
        p0101 = -A26*r2*(r + sina*(L - x))/L
        p0102 = -A66*r2*(r + sina*(L - x))/L
        p0200 = (A16*(-L + x)*sin(t - tLA) - (A11*r + A12*sina*(-L + x))*cos(t - tLA))/(L*cosa)
        p0201 = (A26*(-L + x)*sin(t - tLA) - (A12*r + A22*sina*(-L + x))*cos(t - tLA))/(L*cosa)
        p0202 = (A66*(-L + x)*sin(t - tLA) - (A16*r + A26*sina*(-L + x))*cos(t - tLA))/(L*cosa)

        for k1 in range(i0, m1+i0):
            cosk1x = vcosi1x[k1-i0]

            # q_1
            q0002 = pi*cosk1x*k1*(w0x + wx)/L
            q0202 = pi*cosk1x*k1*(w0t + wt)/(L*r)

            # k0L_01
            c += 1
            out[c] = beta*out[c] + alpha*(p0000*q0002 + p0002*q0202)
            c += 1
            out[c] = beta*out[c] + alpha*(p0100*q0002 + p0102*q0202)
            c += 1
            out[c] = beta*out[c] + alpha*(p0200*q0002 + p0202*q0202)

            # create buffer q_1
            k0Lq_1_q0002[k1-i0] = q0002
            k0Lq_1_q0202[k1-i0] = q0202

        for k2 in range(i0, m2+i0):
            sink2x = vsini2x[k2-i0]
            cosk2x = vcosi2x[k2-i0]
            for l2 in range(j0, n2+j0):
                sinl2t = vsinj2t[l2-j0]
                cosl2t = vcosj2t[l2-j0]

                # q_2
                q0004 = pi*cosk2x*k2*sinl2t*(w0x + wx)/L
                q0005 = pi*cosk2x*cosl2t*k2*(w0x + wx)/L
                q0104 = cosl2t*l2*sink2x*(w0t + wt)/(r*r)
                q0105 = -l2*sink2x*sinl2t*(w0t + wt)/(r*r)
                q0204 = (L*cosl2t*l2*sink2x*(w0x + wx) + pi*cosk2x*k2*sinl2t*(w0t + wt))/(L*r)
                q0205 = (-L*l2*sink2x*sinl2t*(w0x + wx) + pi*cosk2x*cosl2t*k2*(w0t + wt))/(L*r)

                # k0L_02
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0004 + p0001*q0104 + p0002*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0005 + p0001*q0105 + p0002*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0004 + p0101*q0104 + p0102*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0005 + p0101*q0105 + p0102*q0205)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0004 + p0201*q0104 + p0202*q0204)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0005 + p0201*q0105 + p0202*q0205)

                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                k0Lq_2_q0004[pos] = q0004
                k0Lq_2_q0005[pos] = q0005
                k0Lq_2_q0104[pos] = q0104
                k0Lq_2_q0105[pos] = q0105
                k0Lq_2_q0204[pos] = q0204
                k0Lq_2_q0205[pos] = q0205

        for i1 in range(i0, m1+i0):
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]

            # p_1
            p0000 = pi*A11*cosi1x*i1*r/L + A12*sina*sini1x
            p0001 = pi*A12*cosi1x*i1*r/L + A22*sina*sini1x
            p0002 = pi*A16*cosi1x*i1*r/L + A26*sina*sini1x
            p0100 = -A16*sina*sini1x + pi*A16*cosi1x*i1*r/L
            p0101 = -A26*sina*sini1x + pi*A26*cosi1x*i1*r/L
            p0102 = -A66*sina*sini1x + pi*A66*cosi1x*i1*r/L
            p0200 = A12*cosa*sini1x
            p0201 = A22*cosa*sini1x
            p0202 = A26*cosa*sini1x
            p0300 = -pi*B11*i1*r*sini1x/L + B12*cosi1x*sina
            p0301 = -pi*B12*i1*r*sini1x/L + B22*cosi1x*sina
            p0302 = -pi*B16*i1*r*sini1x/L + B26*cosi1x*sina
            p0400 = -B16*sina*sini1x + pi*B16*cosi1x*i1*r/L
            p0401 = -B26*sina*sini1x + pi*B26*cosi1x*i1*r/L
            p0402 = -B66*sina*sini1x + pi*B66*cosi1x*i1*r/L

            for k1 in range(i0, m1+i0):
                # access buffer q_1
                pos = k1-i0
                q0002 = k0Lq_1_q0002[pos]
                q0202 = k0Lq_1_q0202[pos]

                # k0L_11
                c += 1
                out[c] = beta*out[c] + alpha*(p0000*q0002 + p0002*q0202)
                c += 1
                out[c] = beta*out[c] + alpha*(p0100*q0002 + p0102*q0202)
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0002 + p0202*q0202)
                c += 1
                out[c] = beta*out[c] + alpha*(p0300*q0002 + p0302*q0202)
                c += 1
                out[c] = beta*out[c] + alpha*(p0400*q0002 + p0402*q0202)

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    # access buffer q_2
                    pos = (k2-i0)*n2 + (l2-j0)
                    q0004 = k0Lq_2_q0004[pos]
                    q0005 = k0Lq_2_q0005[pos]
                    q0104 = k0Lq_2_q0104[pos]
                    q0105 = k0Lq_2_q0105[pos]
                    q0204 = k0Lq_2_q0204[pos]
                    q0205 = k0Lq_2_q0205[pos]

                    # k0L_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0004 + p0001*q0104 + p0002*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0005 + p0001*q0105 + p0002*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0004 + p0101*q0104 + p0102*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0005 + p0101*q0105 + p0102*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0004 + p0201*q0104 + p0202*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0005 + p0201*q0105 + p0202*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0004 + p0301*q0104 + p0302*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0005 + p0301*q0105 + p0302*q0205)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0004 + p0401*q0104 + p0402*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0005 + p0401*q0105 + p0402*q0205)


        for i2 in range(i0, m2+i0):
            sini2x = vsini2x[i2-i0]
            cosi2x = vcosi2x[i2-i0]

            for j2 in range(j0, n2+j0):
                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]

                # p_2
                p0000 = pi*A11*cosi2x*i2*r*sinj2t/L + sini2x*(A12*sina*sinj2t + A16*cosj2t*j2)
                p0001 = pi*A12*cosi2x*i2*r*sinj2t/L + sini2x*(A22*sina*sinj2t + A26*cosj2t*j2)
                p0002 = pi*A16*cosi2x*i2*r*sinj2t/L + sini2x*(A26*sina*sinj2t + A66*cosj2t*j2)
                p0100 = pi*A11*cosi2x*cosj2t*i2*r/L + sini2x*(A12*cosj2t*sina - A16*j2*sinj2t)
                p0101 = pi*A12*cosi2x*cosj2t*i2*r/L + sini2x*(A22*cosj2t*sina - A26*j2*sinj2t)
                p0102 = pi*A16*cosi2x*cosj2t*i2*r/L + sini2x*(A26*cosj2t*sina - A66*j2*sinj2t)
                p0200 = -pi*A16*i2*r*sini2x*sinj2t/L + cosi2x*(A12*cosj2t*j2 - A16*sina*sinj2t)
                p0201 = -pi*A26*i2*r*sini2x*sinj2t/L + cosi2x*(A22*cosj2t*j2 - A26*sina*sinj2t)
                p0202 = -pi*A66*i2*r*sini2x*sinj2t/L + cosi2x*(A26*cosj2t*j2 - A66*sina*sinj2t)
                p0300 = -pi*A16*cosj2t*i2*r*sini2x/L - cosi2x*(A12*j2*sinj2t + A16*cosj2t*sina)
                p0301 = -pi*A26*cosj2t*i2*r*sini2x/L - cosi2x*(A22*j2*sinj2t + A26*cosj2t*sina)
                p0302 = -pi*A66*cosj2t*i2*r*sini2x/L - cosi2x*(A26*j2*sinj2t + A66*cosj2t*sina)
                p0400 = A12*cosa*sini2x*sinj2t
                p0401 = A22*cosa*sini2x*sinj2t
                p0402 = A26*cosa*sini2x*sinj2t
                p0500 = A12*cosa*cosj2t*sini2x
                p0501 = A22*cosa*cosj2t*sini2x
                p0502 = A26*cosa*cosj2t*sini2x
                p0600 = -pi*B11*i2*r*sini2x*sinj2t/L + cosi2x*(B12*sina*sinj2t + B16*cosj2t*j2)
                p0601 = -pi*B12*i2*r*sini2x*sinj2t/L + cosi2x*(B22*sina*sinj2t + B26*cosj2t*j2)
                p0602 = -pi*B16*i2*r*sini2x*sinj2t/L + cosi2x*(B26*sina*sinj2t + B66*cosj2t*j2)
                p0700 = -pi*B11*cosj2t*i2*r*sini2x/L + cosi2x*(B12*cosj2t*sina - B16*j2*sinj2t)
                p0701 = -pi*B12*cosj2t*i2*r*sini2x/L + cosi2x*(B22*cosj2t*sina - B26*j2*sinj2t)
                p0702 = -pi*B16*cosj2t*i2*r*sini2x/L + cosi2x*(B26*cosj2t*sina - B66*j2*sinj2t)
                p0800 = pi*B16*cosi2x*i2*r*sinj2t/L + sini2x*(B12*cosj2t*j2 - B16*sina*sinj2t)
                p0801 = pi*B26*cosi2x*i2*r*sinj2t/L + sini2x*(B22*cosj2t*j2 - B26*sina*sinj2t)
                p0802 = pi*B66*cosi2x*i2*r*sinj2t/L + sini2x*(B26*cosj2t*j2 - B66*sina*sinj2t)
                p0900 = pi*B16*cosi2x*cosj2t*i2*r/L - sini2x*(B12*j2*sinj2t + B16*cosj2t*sina)
                p0901 = pi*B26*cosi2x*cosj2t*i2*r/L - sini2x*(B22*j2*sinj2t + B26*cosj2t*sina)
                p0902 = pi*B66*cosi2x*cosj2t*i2*r/L - sini2x*(B26*j2*sinj2t + B66*cosj2t*sina)

                for k1 in range(i0, m1+i0):
                    # access buffer q_1
                    pos = k1-i0
                    q0002 = k0Lq_1_q0002[pos]
                    q0202 = k0Lq_1_q0202[pos]

                    # k0L_21
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0000*q0002 + p0002*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0100*q0002 + p0102*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0002 + p0202*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0300*q0002 + p0302*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0400*q0002 + p0402*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0500*q0002 + p0502*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0600*q0002 + p0602*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0700*q0002 + p0702*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0800*q0002 + p0802*q0202)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0900*q0002 + p0902*q0202)

                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        # access buffer q_2
                        pos = (k2-i0)*n2 + (l2-j0)
                        q0004 = k0Lq_2_q0004[pos]
                        q0005 = k0Lq_2_q0005[pos]
                        q0104 = k0Lq_2_q0104[pos]
                        q0105 = k0Lq_2_q0105[pos]
                        q0204 = k0Lq_2_q0204[pos]
                        q0205 = k0Lq_2_q0205[pos]

                        # k0L_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0000*q0004 + p0001*q0104 + p0002*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0000*q0005 + p0001*q0105 + p0002*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0100*q0004 + p0101*q0104 + p0102*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0100*q0005 + p0101*q0105 + p0102*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0200*q0004 + p0201*q0104 + p0202*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0200*q0005 + p0201*q0105 + p0202*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0300*q0004 + p0301*q0104 + p0302*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0300*q0005 + p0301*q0105 + p0302*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0004 + p0401*q0104 + p0402*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0005 + p0401*q0105 + p0402*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0004 + p0501*q0104 + p0502*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0005 + p0501*q0105 + p0502*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0600*q0004 + p0601*q0104 + p0602*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0600*q0005 + p0601*q0105 + p0602*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0700*q0004 + p0701*q0104 + p0702*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0700*q0005 + p0701*q0105 + p0702*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0004 + p0801*q0104 + p0802*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0800*q0005 + p0801*q0105 + p0802*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0004 + p0901*q0104 + p0902*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0900*q0005 + p0901*q0105 + p0902*q0205)

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

    free(k0Lq_1_q0002)
    free(k0Lq_1_q0202)

    free(k0Lq_2_q0004)
    free(k0Lq_2_q0005)
    free(k0Lq_2_q0104)
    free(k0Lq_2_q0105)
    free(k0Lq_2_q0204)
    free(k0Lq_2_q0205)


def calc_kG(double [:] coeffs,
            double alpharad, double r2, double L, double tLA,
            double [:, ::1] F,
            int m1, int m2, int n2,
            int nx, int nt, int num_cores, str method,
            double [:] c0, int m0, int n0):

    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef long [:] rows, cols
    cdef double [:] kGv

    cdef unsigned fdim
    cdef cc_attributes args

    fdim = 1*m1**2 + 2*m1*m2*n2 + 4*m2**2*n2**2

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
    integratev(<void *>cfkG, fdim, &kGv[0], xa, xb, nx, ta, tb, nt,
               &args, num_cores, method)

    c = -1

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0
        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0

            #NOTE symmetry
            if row > col:
                continue

            # kG_11
            c += 1
            rows[c] = row+2
            cols[c] = col+2


        for k2 in range(i0, m2+i0):
            for l2 in range(j0, n2+j0):
                col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                # kG_12
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

    kG = coo_matrix((kGv, (rows, cols)), shape=(size, size))

    return kG


cdef void cfkG(int npts, double *xs, double *ts, double *out,
               double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2

    cdef double p0200, p0201, p0202, p0400, p0401, p0402, p0500, p0501, p0502

    cdef double q0002, q0004, q0005, q0104, q0105, q0204, q0205

    cdef double r, x, t, alpha, beta
    cdef int c, i, pos, row, col

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double r2, L, sina, cosa, tLA
    cdef int m0, n0, m1, m2, n2, pti
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

    cdef double cosi1x, cosk1x
    cdef double sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t

    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))

    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))

    cdef double *kGq_2_q0004 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q0005 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q0104 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kGq_2_q0105 = <double *>malloc(m2*n2 * sizeof(double))

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
                q0004 = pi*cosk2x*k2*sinl2t/L
                q0005 = pi*cosk2x*cosl2t*k2/L
                q0104 = cosl2t*l2*sink2x/r
                q0105 = -l2*sink2x*sinl2t/r

                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                kGq_2_q0004[pos] = q0004
                kGq_2_q0005[pos] = q0005
                kGq_2_q0104[pos] = q0104
                kGq_2_q0105[pos] = q0105

        for i1 in range(i0, m1+i0):
            row = (i1-i0)*num1 + num0

            cosi1x = vcosi1x[i1-i0]

            # p_1
            p0200 = pi*Nxx*cosi1x*i1*r/L
            p0201 = pi*Nxt*cosi1x*i1*r/L

            for k1 in range(i0, m1+i0):
                col = (k1-i0)*num1 + num0

                #NOTE symmetry
                if row > col:
                    continue

                cosk1x = vcosi1x[k1-i0]

                # q_1
                q0002 = pi*cosk1x*k1/L

                # kG_11
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0002)

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    # access buffer q_2
                    pos = (k2-i0)*n2 + (l2-j0)
                    q0004 = kGq_2_q0004[pos]
                    q0005 = kGq_2_q0005[pos]
                    q0104 = kGq_2_q0104[pos]
                    q0105 = kGq_2_q0105[pos]

                    # kG_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0004 + p0201*q0104)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0005 + p0201*q0105)

        for i2 in range(i0, m2+i0):
            sini2x = vsini2x[i2-i0]
            cosi2x = vcosi2x[i2-i0]
            for j2 in range(j0, n2+j0):
                row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]

                # p_2
                p0400 = Nxt*cosj2t*j2*sini2x + pi*Nxx*cosi2x*i2*r*sinj2t/L
                p0401 = Ntt*cosj2t*j2*sini2x + pi*Nxt*cosi2x*i2*r*sinj2t/L
                p0500 = -Nxt*j2*sini2x*sinj2t + pi*Nxx*cosi2x*cosj2t*i2*r/L
                p0501 = -Ntt*j2*sini2x*sinj2t + pi*Nxt*cosi2x*cosj2t*i2*r/L

                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                        #NOTE symmetry
                        if row > col:
                            continue

                        # access buffer q_2
                        pos = (k2-i0)*n2 + (l2-j0)
                        q0004 = kGq_2_q0004[pos]
                        q0005 = kGq_2_q0005[pos]
                        q0104 = kGq_2_q0104[pos]
                        q0105 = kGq_2_q0105[pos]

                        # kG_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0004 + p0401*q0104)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0005 + p0401*q0105)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0004 + p0501*q0104)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0005 + p0501*q0105)

    free(Ns)

    free(vcosi1x)

    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)

    free(kGq_2_q0004)
    free(kGq_2_q0005)
    free(kGq_2_q0104)
    free(kGq_2_q0105)


def calc_kLL(double [:] coeffs,
             double alpharad, double r2, double L, double tLA,
             double [:, ::1] F,
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

    cdef double A11, A12, A16, A22, A26, A66

    cdef double p0200, p0201, p0202
    cdef double p0300, p0301, p0302, p0400, p0401, p0402
    cdef double p0500, p0501, p0502, p0600, p0601, p0602

    cdef double q0002, q0004, q0005
    cdef double q0104, q0105, q0202, q0204, q0205

    cdef double r, x, t, alpha, beta

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double sina, r2, L
    cdef int m0, n0, m1, m2, n2, pti
    cdef double wx, wt, w0x, w0t

    cdef cc_attributes *args_in=<cc_attributes *>args
    sina = args_in.sina[0]
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

    A11 = F[0]  # F[0,0]
    A12 = F[1]  # F[0,1]
    A16 = F[2]  # F[0,2]
    A22 = F[9]  # F[1,1]
    A26 = F[10] # F[1,2]
    A66 = F[18] # F[2,2]

    cdef double cosi1x, cosk1x
    cdef double sini2x, cosi2x, sink2x, cosk2x
    cdef double sinl2t, cosl2t, sinj2t, cosj2t

    cdef double *vcosi1x = <double *>malloc(m1 * sizeof(double))

    cdef double *vsini2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vcosi2x = <double *>malloc(m2 * sizeof(double))
    cdef double *vsinj2t = <double *>malloc(n2 * sizeof(double))
    cdef double *vcosj2t = <double *>malloc(n2 * sizeof(double))

    cdef double *kLLq_2_q0004 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0005 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0104 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0105 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0204 = <double *>malloc(m2*n2 * sizeof(double))
    cdef double *kLLq_2_q0205 = <double *>malloc(m2*n2 * sizeof(double))

    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfwx(coeffs, m1, m2, n2, L, xs, ts, npts, wxs)
    cfwt(coeffs, m1, m2, n2, L, xs, ts, npts, wts)
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
                q0004 = pi*cosk2x*k2*sinl2t*(w0x + wx)/L
                q0005 = pi*cosk2x*cosl2t*k2*(w0x + wx)/L
                q0104 = cosl2t*l2*sink2x*(w0t + wt)/(r*r)
                q0105 = -l2*sink2x*sinl2t*(w0t + wt)/(r*r)
                q0204 = (L*cosl2t*l2*sink2x*(w0x + wx) + pi*cosk2x*k2*sinl2t*(w0t + wt))/(L*r)
                q0205 = (-L*l2*sink2x*sinl2t*(w0x + wx) + pi*cosk2x*cosl2t*k2*(w0t + wt))/(L*r)

                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                kLLq_2_q0004[pos] = q0004
                kLLq_2_q0005[pos] = q0005
                kLLq_2_q0104[pos] = q0104
                kLLq_2_q0105[pos] = q0105
                kLLq_2_q0204[pos] = q0204
                kLLq_2_q0205[pos] = q0205

        for i1 in range(i0, m1+i0):
            row = (i1-i0)*num1 + num0

            cosi1x = vcosi1x[i1-i0]

            # p_1
            p0200 = pi*cosi1x*i1*(A11*r*(w0x + wx) + A16*(w0t + wt))/L
            p0201 = pi*cosi1x*i1*(A12*r*(w0x + wx) + A26*(w0t + wt))/L
            p0202 = pi*cosi1x*i1*(A16*r*(w0x + wx) + A66*(w0t + wt))/L

            for k1 in range(i0, m1+i0):
                col = (k1-i0)*num1 + num0

                #NOTE symmetry
                if row > col:
                    continue

                cosk1x = vcosi1x[k1-i0]

                # q_1
                q0002 = pi*cosk1x*k1*(w0x + wx)/L
                q0202 = pi*cosk1x*k1*(w0t + wt)/(L*r)

                # kLL_11
                c += 1
                out[c] = beta*out[c] + alpha*(p0200*q0002 + p0202*q0202)

            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    # access buffer q_2
                    pos = (k2-i0)*n2 + (l2-j0)
                    q0004 = kLLq_2_q0004[pos]
                    q0005 = kLLq_2_q0005[pos]
                    q0104 = kLLq_2_q0104[pos]
                    q0105 = kLLq_2_q0105[pos]
                    q0204 = kLLq_2_q0204[pos]
                    q0205 = kLLq_2_q0205[pos]

                    # kLL_12
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0004 + p0201*q0104 + p0202*q0204)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p0200*q0005 + p0201*q0105 + p0202*q0205)

        for i2 in range(i0, m2+i0):
            sini2x = vsini2x[i2-i0]
            cosi2x = vcosi2x[i2-i0]
            for j2 in range(j0, n2+j0):
                row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]

                # p_2
                p0400 = cosj2t*j2*sini2x*(A12*(w0t + wt) + A16*r*(w0x + wx))/r + pi*cosi2x*i2*sinj2t*(A11*r*(w0x + wx) + A16*(w0t + wt))/L
                p0401 = cosj2t*j2*sini2x*(A22*(w0t + wt) + A26*r*(w0x + wx))/r + pi*cosi2x*i2*sinj2t*(A12*r*(w0x + wx) + A26*(w0t + wt))/L
                p0402 = cosj2t*j2*sini2x*(A26*(w0t + wt) + A66*r*(w0x + wx))/r + pi*cosi2x*i2*sinj2t*(A16*r*(w0x + wx) + A66*(w0t + wt))/L
                p0500 = -j2*sini2x*sinj2t*(A12*(w0t + wt) + A16*r*(w0x + wx))/r + pi*cosi2x*cosj2t*i2*(A11*r*(w0x + wx) + A16*(w0t + wt))/L
                p0501 = -j2*sini2x*sinj2t*(A22*(w0t + wt) + A26*r*(w0x + wx))/r + pi*cosi2x*cosj2t*i2*(A12*r*(w0x + wx) + A26*(w0t + wt))/L
                p0502 = -j2*sini2x*sinj2t*(A26*(w0t + wt) + A66*r*(w0x + wx))/r + pi*cosi2x*cosj2t*i2*(A16*r*(w0x + wx) + A66*(w0t + wt))/L

                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                        #NOTE symmetry
                        if row > col:
                            continue

                        # access buffer q_2
                        pos = (k2-i0)*n2 + (l2-j0)
                        q0004 = kLLq_2_q0004[pos]
                        q0005 = kLLq_2_q0005[pos]
                        q0104 = kLLq_2_q0104[pos]
                        q0105 = kLLq_2_q0105[pos]
                        q0204 = kLLq_2_q0204[pos]
                        q0205 = kLLq_2_q0205[pos]

                        # kLL_22
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0004 + p0401*q0104 + p0402*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0400*q0005 + p0401*q0105 + p0402*q0205)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0004 + p0501*q0104 + p0502*q0204)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p0500*q0005 + p0501*q0105 + p0502*q0205)

    free(wxs)
    free(wts)
    free(w0xs)
    free(w0ts)

    free(vcosi1x)

    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)

    free(kLLq_2_q0004)
    free(kLLq_2_q0005)
    free(kLLq_2_q0104)
    free(kLLq_2_q0105)
    free(kLLq_2_q0204)
    free(kLLq_2_q0205)


def calc_fint_0L_L0_LL(double [:] coeffs,
              double alpharad, double r2, double L, double tLA,
                       double [:, ::1] F,
              int m1, int m2, int n2,
              int nx, int nt, int num_cores, str method,
                       double [:] c0, int m0, int n0):
    cdef cc_attributes args
    cdef double sina, cosa
    cdef double [:] fint

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
    integratev(<void *>cffint, fdim, &fint[0], xa, xb, nx, ta, tb, nt,
               &args, num_cores, method)

    return fint


cdef void cffint(int npts, double *xs, double *ts, double *fint,
                 double *alphas, double *betas, void *args) nogil:
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double Nxx0, Ntt0, Nxt0, Mxx0, Mtt0, Mxt0, Qt0, Qx0
    cdef double NxxL, NttL, NxtL, MxxL, MttL, MxtL, QtL, QxL
    cdef double exx0, ett0, gxt0, kxx0, ktt0, kxt0, gtz0, gxz0
    cdef double exxL, ettL, gxtL, kxxL, kttL, kxtL, gtzL, gxzL
    cdef double x, t, wx, wt, w0x, w0t, w0
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
    A22 = F[9]
    A26 = F[10]
    A66 = F[18]
    B11 = F[3]
    B12 = F[4]
    B16 = F[5]
    B22 = F[12]
    B26 = F[13]
    B66 = F[21]
    D11 = F[27]
    D12 = F[28]
    D16 = F[29]
    D22 = F[36]
    D26 = F[37]
    D66 = F[45]
    A44 = F[54]
    A45 = F[55]
    A55 = F[63]

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

        # wx and wt
        wx = 0.
        wt = 0.
        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            wx += c[col+2]*(i1*pi/L)*vcosi1x[i1-i0]
        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                dsini2x = (i2*pi/L)*vcosi2x[i2-i0]
                sini2x = vsini2x[i2-i0]
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
        gtz0 = 0.
        gxz0 = 0.

        exxL = 0.
        ettL = 0.
        gxtL = 0.
        kxxL = 0.
        kttL = 0.
        kxtL = 0.
        gtzL = 0.
        gxzL = 0.

        #TODO if castro==1
        w0 = 0.

        exx0 = (-c[0]/(L*cosa)
                -c[2]*cos(t - tLA)/(L*cosa))
        exxL = 0.5*castro*w0x*w0x

        ett0 = (c[0]*sina*(L - x)/(L*cosa*r)
                +c[2]*sina*(L - x)*cos(t - tLA)/(L*cosa*r))
        ettL = 0.5*castro*(2*cosa*r*w0 + w0t*w0t)/(r*r)

        gxt0 = (-c[1]*r2*(r + sina*(L - x))/(L*r)
               + c[2]*(-L + x)*sin(t - tLA)/(L*cosa*r))
        gxtL = castro*w0t*w0x/r

        gtz0 = c[1]*cosa*r2*(-L + x)/(L*r)

        for i1 in range(i0, m1+i0):
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]
            col = (i1-i0)*num1 + num0

            exx0 += (+pi*c[col+0]*cosi1x*i1/L
                     +pi*c[col+1]*cosi1x*i1/L)
            exxL += -pi*c[col+3]*i1*sini1x*(w0x + 0.5*wx)/L

            ett0 += (c[col+0]*sina*sini1x/r
                     +c[col+1]*sina*sini1x/r
                     +c[col+3]*cosa*cosi1x/r)

            gxt0 += c[col+2]*(-sina*sini1x/r + pi*cosi1x*i1/L)
            gxtL += -pi*c[col+3]*i1*sini1x*(w0t + 0.5*wt)/(L*r)

            kxx0 += pi*c[col+4]*cosi1x*i1/L

            ktt0 += c[col+4]*sina*sini1x/r

            gtz0 += -c[col+2]*cosa*sini1x/r

            gxz0 += (-pi*c[col+3]*i1*sini1x/L
                     +c[col+4]*sini1x)

        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                sini2x = vsini2x[i2-i0]
                cosi2x = vcosi2x[i2-i0]
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                exx0 += (pi*c[col+0]*cosi2x*i2*sinj2t/L
                         +pi*c[col+1]*cosi2x*cosj2t*i2/L)
                exxL += (c[col+4]*(pi*cosi2x*i2*sinj2t*w0x/L + 0.5*pi*cosi2x*i2*sinj2t*wx/L)
                         +c[col+5]*(pi*cosi2x*cosj2t*i2*w0x/L + 0.5*pi*cosi2x*cosj2t*i2*wx/L))

                ett0 += (c[col+0]*sina*sini2x*sinj2t/r
                         +c[col+1]*cosj2t*sina*sini2x/r
                         +c[col+2]*cosi2x*cosj2t*j2/r
                         -c[col+3]*cosi2x*j2*sinj2t/r
                         +c[col+4]*cosa*sini2x*sinj2t/r
                         +c[col+5]*cosa*cosj2t*sini2x/r)
                ettL += (c[col+4]*(cosj2t*j2*sini2x*w0t/(r*r) + 0.5*cosj2t*j2*sini2x*wt/(r*r))
                         +c[col+5]*(-j2*sini2x*sinj2t*w0t/(r*r) - 0.5*j2*sini2x*sinj2t*wt/(r*r)))

                gxt0 += (c[col+0]*cosj2t*j2*sini2x/r
                         -c[col+1]*j2*sini2x*sinj2t/r
                         -c[col+2]*sinj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*r)
                         -c[col+3]*cosj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*r))
                gxtL += (c[col+4]*(cosj2t*j2*sini2x*w0x/r + 0.5*cosj2t*j2*sini2x*wx/r + pi*cosi2x*i2*sinj2t*w0t/(L*r) + 0.5*pi*cosi2x*i2*sinj2t*wt/(L*r))
                         +c[col+5]*(-j2*sini2x*sinj2t*w0x/r - 0.5*j2*sini2x*sinj2t*wx/r + pi*cosi2x*cosj2t*i2*w0t/(L*r) + 0.5*pi*cosi2x*cosj2t*i2*wt/(L*r)))

                kxx0 += (-pi*c[col+6]*i2*sini2x*sinj2t/L
                         -pi*c[col+7]*cosj2t*i2*sini2x/L)

                ktt0 += (c[col+6]*cosi2x*sina*sinj2t/r
                         +c[col+7]*cosi2x*cosj2t*sina/r
                         +c[col+8]*cosj2t*j2*sini2x/r
                         -c[col+9]*j2*sini2x*sinj2t/r)

                kxt0 += (c[col+6]*cosi2x*cosj2t*j2/r
                         -c[col+7]*cosi2x*j2*sinj2t/r
                         +c[col+8]*sinj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                         +c[col+9]*cosj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r))

                gtz0 += (-c[col+2]*cosa*cosi2x*sinj2t/r
                         -c[col+3]*cosa*cosi2x*cosj2t/r
                         +c[col+4]*cosj2t*j2*sini2x/r
                         -c[col+5]*j2*sini2x*sinj2t/r
                         +c[col+8]*sini2x*sinj2t
                         +c[col+9]*cosj2t*sini2x)

                gxz0 += (pi*c[col+4]*cosi2x*i2*sinj2t/L
                         +pi*c[col+5]*cosi2x*cosj2t*i2/L
                         +c[col+6]*cosi2x*sinj2t
                         +c[col+7]*cosi2x*cosj2t)

        # stresses
        Nxx0 = A11*exx0 + A12*ett0 + A16*gxt0 + B11*kxx0 + B12*ktt0 + B16*kxt0
        Ntt0 = A12*exx0 + A22*ett0 + A26*gxt0 + B12*kxx0 + B22*ktt0 + B26*kxt0
        Nxt0 = A16*exx0 + A26*ett0 + A66*gxt0 + B16*kxx0 + B26*ktt0 + B66*kxt0
        Mxx0 = B11*exx0 + B12*ett0 + B16*gxt0 + D11*kxx0 + D12*ktt0 + D16*kxt0
        Mtt0 = B12*exx0 + B22*ett0 + B26*gxt0 + D12*kxx0 + D22*ktt0 + D26*kxt0
        Mxt0 = B16*exx0 + B26*ett0 + B66*gxt0 + D16*kxx0 + D26*ktt0 + D66*kxt0
        Qt0 = A44*gtz0 + A45*gxz0
        Qx0 = A45*gtz0 + A55*gxz0

        NxxL = A11*exxL + A12*ettL + A16*gxtL + B11*kxxL + B12*kttL + B16*kxtL
        NttL = A12*exxL + A22*ettL + A26*gxtL + B12*kxxL + B22*kttL + B26*kxtL
        NxtL = A16*exxL + A26*ettL + A66*gxtL + B16*kxxL + B26*kttL + B66*kxtL
        MxxL = B11*exxL + B12*ettL + B16*gxtL + D11*kxxL + D12*kttL + D16*kxtL
        MttL = B12*exxL + B22*ettL + B26*gxtL + D12*kxxL + D22*kttL + D26*kxtL
        MxtL = B16*exxL + B26*ettL + B66*gxtL + D16*kxxL + D26*kttL + D66*kxtL
        QtL = A44*gtzL + A45*gxzL
        QxL = A45*gtzL + A55*gxzL

        fint[0] = beta*(fint[0]) + alpha*((NttL*sina*(L - x) - NxxL*r)/(L*cosa))
        fint[1] = beta*(fint[1]) + alpha*(-r2*(NxtL*(r + sina*(L - x)) + QtL*cosa*(L - x))/L)
        fint[2] = beta*(fint[2]) + alpha*((NxtL*(-L + x)*sin(t - tLA) - (NttL*sina*(-L + x) + NxxL*r)*cos(t - tLA))/(L*cosa))

        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]
            fint[col+0] = beta*(fint[col+0]) + alpha*(NttL*sina*sini1x + pi*NxxL*cosi1x*i1*r/L)
            fint[col+1] = beta*(fint[col+1]) + alpha*(NttL*sina*sini1x + pi*NxxL*cosi1x*i1*r/L)
            fint[col+2] = beta*(fint[col+2]) + alpha*((-L*sini1x*(NxtL*sina + QtL*cosa) + pi*NxtL*cosi1x*i1*r)/L)
            fint[col+3] = beta*(fint[col+3]) + alpha*((L*NttL*cosa*cosi1x - pi*i1*sini1x*(Nxt0*(w0t + wt) + NxtL*w0t + NxtL*wt + Nxx0*r*w0x + Nxx0*r*wx + NxxL*r*w0x + NxxL*r*wx + QxL*r))/L)
            fint[col+4] = beta*(fint[col+4]) + alpha*((L*sini1x*(MttL*sina + QxL*r) + pi*MxxL*cosi1x*i1*r)/L)

        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                sini2x = vsini2x[i2-i0]
                cosi2x = vcosi2x[i2-i0]
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                fint[col+0] = beta*(fint[col+0]) + alpha*((L*sini2x*(NttL*sina*sinj2t + NxtL*cosj2t*j2) + pi*NxxL*cosi2x*i2*r*sinj2t)/L)
                fint[col+1] = beta*(fint[col+1]) + alpha*(-NxtL*j2*sini2x*sinj2t + cosj2t*(NttL*sina*sini2x + pi*NxxL*cosi2x*i2*r/L))
                fint[col+2] = beta*(fint[col+2]) + alpha*((L*NttL*cosi2x*cosj2t*j2 - sinj2t*(L*cosi2x*(NxtL*sina + QtL*cosa) + pi*NxtL*i2*r*sini2x))/L)
                fint[col+3] = beta*(fint[col+3]) + alpha*(-(L*NttL*cosi2x*j2*sinj2t + cosj2t*(L*cosi2x*(NxtL*sina + QtL*cosa) + pi*NxtL*i2*r*sini2x))/L)
                fint[col+4] = beta*(fint[col+4]) + alpha*((L*sini2x*(NttL*cosa*r*sinj2t + cosj2t*j2*(Ntt0*(w0t + wt) + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx + QtL*r)) + pi*cosi2x*i2*r*sinj2t*(Nxt0*(w0t + wt) + NxtL*w0t + NxtL*wt + Nxx0*r*w0x + Nxx0*r*wx + NxxL*r*w0x + NxxL*r*wx + QxL*r))/(L*r))
                fint[col+5] = beta*(fint[col+5]) + alpha*((-L*j2*sini2x*sinj2t*(Ntt0*(w0t + wt) + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx + QtL*r) + cosj2t*r*(L*NttL*cosa*sini2x + pi*cosi2x*i2*(Nxt0*(w0t + wt) + NxtL*w0t + NxtL*wt + Nxx0*r*w0x + Nxx0*r*wx + NxxL*r*w0x + NxxL*r*wx + QxL*r)))/(L*r))
                fint[col+6] = beta*(fint[col+6]) + alpha*((L*MxtL*cosi2x*cosj2t*j2 + sinj2t*(L*cosi2x*(MttL*sina + QxL*r) - pi*MxxL*i2*r*sini2x))/L)
                fint[col+7] = beta*(fint[col+7]) + alpha*((-L*MxtL*cosi2x*j2*sinj2t + cosj2t*(L*cosi2x*(MttL*sina + QxL*r) - pi*MxxL*i2*r*sini2x))/L)
                fint[col+8] = beta*(fint[col+8]) + alpha*((L*sini2x*(MttL*cosj2t*j2 + sinj2t*(-MxtL*sina + QtL*r)) + pi*MxtL*cosi2x*i2*r*sinj2t)/L)
                fint[col+9] = beta*(fint[col+9]) + alpha*((-L*MttL*j2*sini2x*sinj2t + cosj2t*(L*sini2x*(-MxtL*sina + QtL*r) + pi*MxtL*cosi2x*i2*r))/L)

    free(w0xs)
    free(w0ts)
    free(vsini1x)
    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
