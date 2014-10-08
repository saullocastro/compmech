#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
include 'clpt_nonlinear_header.pyx'

from compmech.conecyl.clpt.clpt_commons_bc1 cimport cfwx, cfwt, cfN

cdef extern from "clpt_donnell_bc1_nonlinear_clean.h":
    void cfk0L_clean(double *wxs, double *wts, double *w0xs, double *w0ts,
                     int npts, double *xs, double *ts,
                     double *out, double *alphas, double *betas, void *args) nogil

cdef int NL_kinematics=0 # to use cfstrain_donnell in cfN

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

    fdim = 3*m1 + 3*2*m2*n2 + 3*m1**2 + 2*6*m1*m2*n2 + 12*m2**2*n2**2

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
    if method=='trapz2d':
        trapz2d(<void *>cfk0L, fdim, k0Lv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<void *>cfk0L, fdim, k0Lv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

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

    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66

    cdef double p00, p01, p02, p10, p11, p12
    cdef double p20, p21, p22, p30, p31, p32
    cdef double p40, p41, p42, p50, p51, p52
    cdef double q02, q04, q05, q14, q15, q22, q24, q25

    cdef double r, x, t, alpha, beta

    cdef double *F
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
        p00 = (-A11*r + A12*sina*(L - x))/(L*cosa)
        p01 = (-A12*r + A22*sina*(L - x))/(L*cosa)
        p02 = (-A16*r + A26*sina*(L - x))/(L*cosa)
        p10 = -A16*r2*(r + sina*(L - x))/L
        p11 = -A26*r2*(r + sina*(L - x))/L
        p12 = -A66*r2*(r + sina*(L - x))/L
        p20 = (A16*(-L + x)*sin(t - tLA) - (A11*r + A12*sina*(-L + x))*cos(t - tLA))/(L*cosa)
        p21 = (A26*(-L + x)*sin(t - tLA) - (A12*r + A22*sina*(-L + x))*cos(t - tLA))/(L*cosa)
        p22 = (A66*(-L + x)*sin(t - tLA) - (A16*r + A26*sina*(-L + x))*cos(t - tLA))/(L*cosa)

        for k1 in range(i0, m1+i0):
            cosk1x = vcosi1x[k1-i0]
            # q_1
            q02 = pi*cosk1x*k1*(w0x + wx)/L
            q22 = pi*cosk1x*k1*(w0t + wt)/(L*r)

            # k0L_01
            c += 1
            out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
            c += 1
            out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
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
                q14 = cosl2t*l2*sink2x*(w0t + wt)/(r*r)
                q15 = -l2*sink2x*sinl2t*(w0t + wt)/(r*r)
                q24 = (L*cosl2t*l2*sink2x*(w0x + wx) + pi*cosk2x*k2*sinl2t*(w0t + wt))/(L*r)
                q25 = (-L*l2*sink2x*sinl2t*(w0x + wx) + pi*cosk2x*cosl2t*k2*(w0t + wt))/(L*r)

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
            p00 = pi*A11*i1*r*cosi1x/L + A12*sina*sini1x
            p01 = pi*A12*i1*r*cosi1x/L + A22*sina*sini1x
            p02 = pi*A16*i1*r*cosi1x/L + A26*sina*sini1x
            p10 = -A16*sina*sini1x + pi*A16*i1*r*cosi1x/L
            p11 = -A26*sina*sini1x + pi*A26*i1*r*cosi1x/L
            p12 = -A66*sina*sini1x + pi*A66*i1*r*cosi1x/L
            p20 = (-pi*B12*L*i1*sina*cosi1x + (A12*(L*L)*cosa + (pi*pi)*B11*(i1*i1)*r)*sini1x)/(L*L)
            p21 = (-pi*B22*L*i1*sina*cosi1x + (A22*(L*L)*cosa + (pi*pi)*B12*(i1*i1)*r)*sini1x)/(L*L)
            p22 = (-pi*B26*L*i1*sina*cosi1x + (A26*(L*L)*cosa + (pi*pi)*B16*(i1*i1)*r)*sini1x)/(L*L)

            for k1 in range(i0, m1+i0):
                # access buffer q_1
                q02 = k0Lq_1_q02[k1-i0]
                q22 = k0Lq_1_q22[k1-i0]

                # k0L_11
                c += 1
                out[c] = beta*out[c] + alpha*(p00*q02 + p02*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p10*q02 + p12*q22)
                c += 1
                out[c] = beta*out[c] + alpha*(p20*q02 + p22*q22)

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

        for i2 in range(i0, m2+i0):
            sini2x = vsini2x[i2-i0]
            cosi2x = vcosi2x[i2-i0]
            for j2 in range(j0, n2+j0):
                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]
                # p_2
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
                    out[c] = beta*out[c] + alpha*(p40*q02 + p42*q22)
                    c += 1
                    out[c] = beta*out[c] + alpha*(p50*q02 + p52*q22)

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

cdef void cfk0L2(int npts, double *xs, double *ts, double *out,
                 double *alphas, double *betas, void *args) nogil:
    cdef int i1, k1, i2, j2, k2, l2
    cdef int c, i, pos

    cdef double r, x, t, alpha, beta

    cdef double *coeffs
    cdef double *c0
    cdef double L
    cdef int m0, n0, m1, m2, n2
    cdef double wx, wt, w0x, w0t

    cdef cc_attributes *args_in=<cc_attributes *>args

    L = args_in.L[0]
    m1 = args_in.m1[0]
    m2 = args_in.m2[0]
    n2 = args_in.n2[0]
    coeffs = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]


    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfwx(coeffs, m1, m2, n2, xs, ts, npts, L, wxs)
    cfwt(coeffs, m1, m2, n2, xs, ts, npts, L, wts)
    cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    cfk0L_clean(wxs, wts, w0xs, w0ts, npts, xs, ts, out,
                alphas, betas, args)

    free(wxs)
    free(wts)
    free(w0xs)
    free(w0ts)


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
    cdef np.ndarray[cDOUBLE, ndim=2] tmp

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
    if method=='trapz2d':
        trapz2d(<void *>cfkG, fdim, kGv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<void *>cfkG, fdim, kGv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    c = -1

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0
        #NOTE symmetry
        for k1 in range(i1, m1+i0):
            col = (k1-i0)*num1 + num0
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
            #NOTE symmetry
            for k2 in range(i2, m2+i0):
                for l2 in range(j2, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
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

    cdef double cosi1x, cosk1x, sini2x, cosi2x, sink2x, cosk2x
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
                q04 = pi*k2*sinl2t*cosk2x/L
                q05 = pi*k2*cosl2t*cosk2x/L
                q14 = l2*sink2x*cosl2t/r
                q15 = -l2*sinl2t*sink2x/r
                # create buffer q_2
                pos = (k2-i0)*n2 + (l2-j0)
                kGq_2_q04[pos] = q04
                kGq_2_q05[pos] = q05
                kGq_2_q14[pos] = q14
                kGq_2_q15[pos] = q15

        for i1 in range(i0, m1+i0):
            cosi1x = vcosi1x[i1-i0]
            # p_1
            p20 = pi*Nxx*i1*r*cosi1x/L
            p21 = pi*Nxt*i1*r*cosi1x/L

            #NOTE symmetry
            for k1 in range(i1, m1+i0):
                cosk1x = vcosi1x[k1-i0]
                # q_1
                q02 = pi*k1*cosk1x/L

                # kG_11
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
                    # kG_12
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
                p40 = Nxt*j2*sini2x*cosj2t + pi*Nxx*i2*r*sinj2t*cosi2x/L
                p41 = Ntt*j2*sini2x*cosj2t + pi*Nxt*i2*r*sinj2t*cosi2x/L
                p50 = -Nxt*j2*sinj2t*sini2x + pi*Nxx*i2*r*cosj2t*cosi2x/L
                p51 = -Ntt*j2*sinj2t*sini2x + pi*Nxt*i2*r*cosj2t*cosi2x/L

                #NOTE symmetry
                for k2 in range(i2, m2+i0):
                    for l2 in range(j2, n2+j0):
                        # access buffer q_2
                        pos = (k2-i0)*n2 + (l2-j0)
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

    free(Ns)

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
             int nx, int nt, int num_cores, str method,
             np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, k1, i2, j2, k2, l2
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kLLv
    cdef np.ndarray[cDOUBLE, ndim=2] tmp

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
    if method=='trapz2d':
        trapz2d(<void *>cfkLL, fdim, kLLv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<void *>cfkLL, fdim, kLLv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    c = -1

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0
        #NOTE symmetry
        for k1 in range(i1, m1+i0):
            col = (k1-i0)*num1 + num0
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
            #NOTE symmetry
            for k2 in range(i2, m2+i0):
                for l2 in range(j2, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
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

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double sina, r2, L
    cdef int m0, n0, m1, m2, n2
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

    A11 = F[0]
    A12 = F[1]
    A16 = F[2]
    A22 = F[7]
    A26 = F[8]
    A66 = F[14]

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
            cosi1x = vcosi1x[i1-i0]
            # p_1
            p20 = pi*cosi1x*i1*(A11*r*(w0x + wx) + A16*(w0t + wt))/L
            p21 = pi*cosi1x*i1*(A12*r*(w0x + wx) + A26*(w0t + wt))/L
            p22 = pi*cosi1x*i1*(A16*r*(w0x + wx) + A66*(w0t + wt))/L

            #NOTE symmetry
            for k1 in range(i1, m1+i0):
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
                sinj2t = vsinj2t[j2-j0]
                cosj2t = vcosj2t[j2-j0]
                # p_2
                p40 = (L*cosj2t*j2*sini2x*(A12*(w0t + wt) + A16*r*(w0x + wx)) + pi*cosi2x*i2*r*sinj2t*(A11*r*(w0x + wx) + A16*(w0t + wt)))/(L*r)
                p41 = (L*cosj2t*j2*sini2x*(A22*(w0t + wt) + A26*r*(w0x + wx)) + pi*cosi2x*i2*r*sinj2t*(A12*r*(w0x + wx) + A26*(w0t + wt)))/(L*r)
                p42 = (L*cosj2t*j2*sini2x*(A26*(w0t + wt) + A66*r*(w0x + wx)) + pi*cosi2x*i2*r*sinj2t*(A16*r*(w0x + wx) + A66*(w0t + wt)))/(L*r)
                p50 = (-L*j2*sini2x*sinj2t*(A12*(w0t + wt) + A16*r*(w0x + wx)) + pi*cosi2x*cosj2t*i2*r*(A11*r*(w0x + wx) + A16*(w0t + wt)))/(L*r)
                p51 = (-L*j2*sini2x*sinj2t*(A22*(w0t + wt) + A26*r*(w0x + wx)) + pi*cosi2x*cosj2t*i2*r*(A12*r*(w0x + wx) + A26*(w0t + wt)))/(L*r)
                p52 = (-L*j2*sini2x*sinj2t*(A26*(w0t + wt) + A66*r*(w0x + wx)) + pi*cosi2x*cosj2t*i2*r*(A16*r*(w0x + wx) + A66*(w0t + wt)))/(L*r)
                #NOTE symmetry
                for k2 in range(i2, m2+i0):
                    for l2 in range(j2, n2+j0):
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
    if method=='trapz2d':
        trapz2d(<void *>cffint, fdim, fint, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<void *>cffint, fdim, fint, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

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

        # wx and wt
        wx = 0.
        wt = 0.
        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            wx += c[col+2]*i1*pi/L*vcosi1x[i1-i0]
        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                dsini2x = i2*pi/L*vcosi2x[i2-i0]
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
                + c[2]*sina*(L - x)*cos(t - tLA)/(L*cosa*r))
        ettL = 0.5*castro*(2*cosa*r*w0 + w0t*w0t)/(r*r)

        gxt0 = (-c[1]*r2*(r + sina*(L - x))/(L*r)
               + c[2]*(-L + x)*sin(t - tLA)/(L*cosa*r))
        gxtL = castro*w0t*w0x/r

        for i1 in range(i0, m1+i0):
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]
            col = (i1-i0)*num1 + num0

            exx0 += pi*c[col+0]*cosi1x*i1/L
            exxL += pi*c[col+2]*cosi1x*i1*(w0x + 0.5*wx)/L

            ett0 += (c[col+0]*sina*sini1x/r
                     + c[col+2]*cosa*sini1x/r)

            gxt0 += c[col+1]*(-sina*sini1x/r + pi*cosi1x*i1/L)
            gxtL += pi*c[col+2]*cosi1x*i1*(w0t + 0.5*wt)/(L*r)

            kxx0 += (pi*pi)*c[col+2]*(i1*i1)*sini1x/(L*L)

            ktt0 += -pi*c[col+2]*cosi1x*i1*sina/(L*r)

        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                sini2x = vsini2x[i2-i0]
                cosi2x = vcosi2x[i2-i0]
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                exx0 += (pi*c[col+0]*cosi2x*i2*sinj2t/L
                         + pi*c[col+1]*cosi2x*cosj2t*i2/L)
                exxL += (pi*c[col+4]*cosi2x*i2*sinj2t*(w0x + 0.5*wx)/L
                         + pi*c[col+5]*cosi2x*cosj2t*i2*(w0x + 0.5*wx)/L)

                ett0 += (c[col+0]*sina*sini2x*sinj2t/r
                         + c[col+1]*cosj2t*sina*sini2x/r
                         + c[col+2]*cosj2t*j2*sini2x/r
                         -c[col+3]*j2*sini2x*sinj2t/r
                         + 0.5*c[col+4]*sini2x*2*cosa*r*sinj2t/(r*r)
                         + 0.5*c[col+5]*sini2x*2*cosa*cosj2t*r/(r*r))
                ettL += (c[col+4]*sini2x*cosj2t*j2*(w0t + 0.5*wt)/(r*r)
                         - c[col+5]*sini2x*j2*sinj2t*(w0t + 0.5*wt)/(r*r))

                gxt0 += (c[col+0]*cosj2t*j2*sini2x/r
                        -c[col+1]*j2*sini2x*sinj2t/r
                        + c[col+2]*sinj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                        + c[col+3]*cosj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r))
                gxtL += (c[col+4]*(L*cosj2t*j2*sini2x*(w0x + 0.5*wx) + pi*cosi2x*i2*sinj2t*(w0t + 0.5*wt))/(L*r)
                        + c[col+5]*(-L*j2*sini2x*sinj2t*(w0x + 0.5*wx) + pi*cosi2x*cosj2t*i2*(w0t + 0.5*wt))/(L*r))

                kxx0 += ((pi*pi)*c[col+4]*(i2*i2)*sini2x*sinj2t/(L*L)
                         + (pi*pi)*c[col+5]*cosj2t*(i2*i2)*sini2x/(L*L))

                ktt0 += (c[col+4]*sinj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r))
                         + c[col+5]*cosj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r)))

                kxt0 += (c[col+4]*cosj2t*j2*(L*sina*sini2x - 2*pi*cosi2x*i2*r)/(L*(r*r))
                         + c[col+5]*j2*sinj2t*(-L*sina*sini2x + 2*pi*cosi2x*i2*r)/(L*(r*r)))

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
        fint[1] = beta*(fint[1]) + alpha*(-NxtL*r2*(r + sina*(L - x))/L)
        fint[2] = beta*(fint[2]) + alpha*((NxtL*(-L + x)*sin(t - tLA) - (NttL*sina*(-L + x) + NxxL*r)*cos(t - tLA))/(L*cosa))

        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            sini1x = vsini1x[i1-i0]
            cosi1x = vcosi1x[i1-i0]
            fint[col+0] = beta*(fint[col+0]) + alpha*(NttL*sina*sini1x + pi*NxxL*cosi1x*i1*r/L)
            fint[col+1] = beta*(fint[col+1]) + alpha*(-NxtL*sina*sini1x + pi*NxtL*cosi1x*i1*r/L)
            fint[col+2] = beta*(fint[col+2]) + alpha*((-pi*L*cosi1x*i1*(MttL*sina - Nxt0*(w0t + wt) - NxtL*w0t - NxtL*wt - Nxx0*r*w0x - Nxx0*r*wx - NxxL*r*w0x - NxxL*r*wx) + sini1x*((L*L)*NttL*cosa + (pi*pi)*MxxL*(i1*i1)*r))/(L*L))

        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[j2-j0]
            cosj2t = vcosj2t[j2-j0]
            for i2 in range(i0, m2+i0):
                sini2x = vsini2x[i2-i0]
                cosi2x = vcosi2x[i2-i0]
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                fint[col+0] = beta*(fint[col+0]) + alpha*((L*sini2x*(NttL*sina*sinj2t + NxtL*cosj2t*j2) + pi*NxxL*cosi2x*i2*r*sinj2t)/L)
                fint[col+1] = beta*(fint[col+1]) + alpha*(-NxtL*j2*sini2x*sinj2t + cosj2t*(NttL*sina*sini2x + pi*NxxL*cosi2x*i2*r/L))
                fint[col+2] = beta*(fint[col+2]) + alpha*((L*sini2x*(NttL*cosj2t*j2 - NxtL*sina*sinj2t) + pi*NxtL*cosi2x*i2*r*sinj2t)/L)
                fint[col+3] = beta*(fint[col+3]) + alpha*( -NttL*j2*sini2x*sinj2t + NxtL*cosj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/L)
                fint[col+4] = beta*(fint[col+4]) + alpha*((L*cosj2t*j2*(L*sini2x*(MxtL*sina + Ntt0*(w0t + wt) + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx) - 2*pi*MxtL*cosi2x*i2*r) + sinj2t*(pi*L*cosi2x*i2*r*(-MttL*sina + Nxt0*(w0t + wt) + NxtL*w0t + NxtL*wt + Nxx0*r*w0x + Nxx0*r*wx + NxxL*r*w0x + NxxL*r*wx) + sini2x*((L*L)*MttL*(j2*j2) + r*((L*L)*NttL*cosa + (pi*pi)*MxxL*(i2*i2)*r))))/((L*L)*r))
                fint[col+5] = beta*(fint[col+5]) + alpha*((-L*j2*sinj2t*(L*sini2x*(MxtL*sina + Ntt0*(w0t + wt) + NttL*w0t + NttL*wt + Nxt0*r*w0x + Nxt0*r*wx + NxtL*r*w0x + NxtL*r*wx) - 2*pi*MxtL*cosi2x*i2*r) + cosj2t*(pi*L*cosi2x*i2*r*(-MttL*sina + Nxt0*(w0t + wt) + NxtL*w0t + NxtL*wt + Nxx0*r*w0x + Nxx0*r*wx + NxxL*r*w0x + NxxL*r*wx) + sini2x*((L*L)*MttL*(j2*j2) + r*((L*L)*NttL*cosa + (pi*pi)*MxxL*(i2*i2)*r))))/((L*L)*r))

    free(w0xs)
    free(w0ts)
    free(vsini1x)
    free(vcosi1x)
    free(vsini2x)
    free(vcosi2x)
    free(vsinj2t)
    free(vcosj2t)
