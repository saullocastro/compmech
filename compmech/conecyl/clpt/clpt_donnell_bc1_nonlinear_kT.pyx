def calc_kT(np.ndarray[cDOUBLE, ndim=1] coeffs,
            double alpharad, double r2, double L, double tLA,
            np.ndarray[cDOUBLE, ndim=2] F,
            int m1, int m2, int n2,
            int nx, int nt, int num_cores, str method,
            np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):

    cdef double sina, cosa, xa, xb, ta, tb
    cdef np.ndarray[cDOUBLE, ndim=2] kT

    cdef cc_attributes args

    size = num0 + num1*m1 + num2*m2*n2
    kT = np.zeros((size, size), dtype=DOUBLE)

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
        trapz2d(<f_type>cfkT, size, kT.ravel(), xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<f_type>cfkT, size, kT.ravel(), xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    fint = calc_fint(coeffs, alpharad, r2, L, tLA, F, m1, m2, n2, nx, nt,
                     num_cores, method, c0, m0, n0)
    kT -= (np.tile(fint, size)/h).reshape(size, size)

    return kT




cdef void cfkT(int npts, double *xs, double *ts, double *kTv,
               double *alphas, double *betas, void *args):
    cdef np.ndarray[cDOUBLE, ndim=1] vsini1x, vcosi1x, xtmp
    cdef np.ndarray[cDOUBLE, ndim=1] vsini2x, vcosi2x, vsinj2t, vcosj2t
    cdef np.ndarray[cDOUBLE, ndim=1] dgwdx, dgwdt, cnp, e, tmp1d
    cdef np.ndarray[cINT, ndim=1] valid
    cdef np.ndarray[cDOUBLE, ndim=2] B0, BL, Fnp, dedc
    cdef np.ndarray[cDOUBLE, ndim=3] dBLdc, dedcidcj

    cdef double x, t, wx, wt, w0x, w0t
    cdef double sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t
    cdef double dsini2

    cdef double alpha, beta

    cdef double *F
    cdef double *c
    cdef double *c0
    cdef double  sina, cosa, tLA, r, r2, L
    cdef int m0, n0, m1, m2, n2, size
    cdef int i1, i2, j2, i, ic, jc, col, valid_shape

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

    size = num0 + num1*m1 + num2*m2*n2

    cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cvfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    cvfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    dBLdc = np.zeros((e_num, size, size), DOUBLE)
    dedc = np.zeros((e_num, size), DOUBLE)
    dgwdx = np.zeros(size, DOUBLE)
    dgwdt = np.zeros(size, DOUBLE)
    Fnp = np.zeros((e_num, e_num), DOUBLE)
    cnp = np.zeros(size, DOUBLE)
    tmp1d = Fnp.ravel()
    for ic in range(e_num*e_num):
        tmp1d[ic] = F[ic]
    tmp1d = cnp.ravel()
    for ic in range(size):
        tmp1d[ic] = c[ic]
    B0 = np.zeros((e_num, size), DOUBLE)
    BL = np.zeros((e_num, size), DOUBLE)
    vsini1x = np.zeros(m1*npts, DOUBLE)
    vcosi1x = np.zeros(m1*npts, DOUBLE)
    vsini2x = np.zeros(m2*npts, DOUBLE)
    vcosi2x = np.zeros(m2*npts, DOUBLE)
    vsinj2t = np.zeros(n2*npts, DOUBLE)
    vcosj2t = np.zeros(n2*npts, DOUBLE)

    xtmp = np.zeros(m1*npts, DOUBLE)
    for i in range(npts):
        for i1 in range(i0, m1+i0):
            xtmp[n2*i + i1-i0] = i1*xs[i]*pi/L
    np.sin(xtmp, out=vsini1x)
    np.cos(xtmp, out=vcosi1x)

    xtmp = np.zeros(m2*npts, DOUBLE)
    for i in range(npts):
        for i2 in range(i0, m2+i0):
            xtmp[n2*i + i1-i0] = i2*xs[i]*pi/L
    np.sin(xtmp, out=vsini2x)
    np.cos(xtmp, out=vcosi2x)

    xtmp = np.zeros(n2*npts, DOUBLE)
    for i in range(npts):
        for j2 in range(j0, n2+j0):
            xtmp[n2*i + j2-j0] = j2*ts[i]
    np.sin(xtmp, out=vsinj2t)
    np.cos(xtmp, out=vcosj2t)

    tmp = []
    for i1 in range(i0, m1+i0):
        col = (i1-i0)*num1 + num0
        tmp.append(col+2)
    for j2 in range(j0, n2+j0):
        for i2 in range(i0, m2+i0):
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            tmp.append(col+4)
            tmp.append(col+5)
    valid = np.array(tmp, INT)
    valid_shape = np.shape(valid)[0]

    for i in range(npts):
        x = xs[i]
        t = ts[i]
        w0x = w0xs[i]
        w0t = w0ts[i]
        alpha = alphas[i]
        beta = betas[i]
        r = r2 + x*sina

        # wx and wt
        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            dgwdx[col+2] = i1*pi/L*vcosi1x[m1*i + i1-i0]
        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[n2*i + j2-j0]
            cosj2t = vcosj2t[n2*i + j2-j0]
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                sini2x = vsini2x[m2*i + i2-i0]
                dsini2 = i2*pi/L*vcosi2x[m2*i + i2-i0]
                dgwdx[col+4] = dsini2*sinj2t
                dgwdx[col+5] = dsini2*cosj2t
                dgwdt[col+4] = sini2x*(j2*cosj2t)
                dgwdt[col+5] = sini2x*(-j2*sinj2t)
        wx = dgwdx.dot(cnp)
        wt = dgwdt.dot(cnp)

        print 'DEBUG start B0, BL'

        B0[0,0] = -1/(L*cosa)
        B0[0,2] = -(-cos(t - tLA) + 1)/(L*cosa)
        B0[1,0] = sina*(L - x)/(L*cosa*r)
        B0[1,2] = sina*(L - x)*(-cos(t - tLA) + 1)/(L*cosa*r)
        B0[2,1] = -r2/L - r2*sina*(L - x)/(L*r)
        B0[2,2] = (L - x)*sin(t - tLA)/(L*cosa*r)

        for i1 in range(i0, m1+i0):
            sini1x = vsini1x[m1*i + i1-i0]
            cosi1x = vcosi1x[m1*i + i1-i0]
            col = (i1-i0)*num1 + num0
            B0[0,col+0] = pi*cosi1x*i1/L
            B0[1,col+0] = sina*sini1x/r
            B0[1,col+2] = cosa*sini1x/r
            B0[2,col+1] = -sina*sini1x/r + pi*cosi1x*i1/L
            B0[3,col+2] = (pi*pi)*(i1*i1)*sini1x/(L*L)
            B0[4,col+2] = -pi*cosi1x*i1*sina/(L*r)

            BL[0,col+2] = pi*cosi1x*i1*(w0x + 0.5*wx)/L
            BL[2,col+2] = pi*cosi1x*i1*(w0t + 0.5*wt)/(L*r)

            for jc in range(valid_shape):
                ic = valid[jc]
                dBLdc[0, col+2, ic] = pi*cosi1x*i1*0.5/L * dgwdx[ic]
                dBLdc[2, col+2, ic] = pi*cosi1x*i1*0.5/(L*r) * dgwdt[ic]

        for j2 in range(j0, n2+j0):
            sinj2t = vsinj2t[n2*i + j2-j0]
            cosj2t = vcosj2t[n2*i + j2-j0]
            for i2 in range(i0, m2+i0):
                sini2x = vsini2x[m2*i + i2-i0]
                cosi2x = vcosi2x[m2*i + i2-i0]
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                B0[0,col+0] = pi*cosi2x*i2*sinj2t/L
                B0[0,col+1] = pi*cosi2x*cosj2t*i2/L
                B0[1,col+0] = sina*sini2x*sinj2t/r
                B0[1,col+1] = cosj2t*sina*sini2x/r
                B0[1,col+2] = cosj2t*j2*sini2x/r
                B0[1,col+3] = -j2*sini2x*sinj2t/r
                B0[1,col+4] = cosa*sini2x*sinj2t/r
                B0[1,col+5] = cosa*cosj2t*sini2x/r
                B0[2,col+0] = cosj2t*j2*sini2x/r
                B0[2,col+1] = -j2*sini2x*sinj2t/r
                B0[2,col+2] = -sina*sini2x*sinj2t/r + pi*cosi2x*i2*sinj2t/L
                B0[2,col+3] = -cosj2t*sina*sini2x/r + pi*cosi2x*cosj2t*i2/L
                B0[3,col+4] = (pi*pi)*(i2*i2)*sini2x*sinj2t/(L*L)
                B0[3,col+5] = (pi*pi)*cosj2t*(i2*i2)*sini2x/(L*L)
                B0[4,col+4] = -(-(j2*j2)*sini2x*sinj2t/r + pi*cosi2x*i2*sina*sinj2t/L)/r
                B0[4,col+5] = -(-cosj2t*(j2*j2)*sini2x/r + pi*cosi2x*cosj2t*i2*sina/L)/r
                B0[5,col+4] = -(-cosj2t*j2*sina*sini2x/r + 2*pi*cosi2x*cosj2t*i2*j2/L)/r
                B0[5,col+5] = -(j2*sina*sini2x*sinj2t/r - 2*pi*cosi2x*i2*j2*sinj2t/L)/r

                BL[0,col+4] = pi*cosi2x*i2*sinj2t*(w0x + 0.5*wx)/L
                BL[0,col+5] = pi*cosi2x*cosj2t*i2*(w0x + 0.5*wx)/L
                BL[1,col+4] = cosj2t*j2*sini2x*(w0t + 0.5*wt)/(r*r)
                BL[1,col+5] = -j2*sini2x*sinj2t*(w0t + 0.5*wt)/(r*r)
                BL[2,col+4] = cosj2t*j2*sini2x*(w0x + 0.5*wx)/r + pi*cosi2x*i2*sinj2t*(w0t + 0.5*wt)/(L*r)
                BL[2,col+5] = -j2*sini2x*sinj2t*(w0x + 0.5*wx)/r + pi*cosi2x*cosj2t*i2*(w0t + 0.5*wt)/(L*r)

                for jc in range(valid_shape):
                    ic = valid[jc]
                    dBLdc[0, col+4, ic] = pi*cosi2x*i2*sinj2t*(0.5*dgwdx[ic])/L
                    dBLdc[0, col+5, ic] = pi*cosi2x*cosj2t*i2*(0.5*dgwdx[ic])/L
                    dBLdc[1, col+4, ic] = cosj2t*j2*sini2x*(0.5*dgwdt[ic])/(r*r)
                    dBLdc[1, col+5, ic] = -j2*sini2x*sinj2t*(0.5*dgwdt[ic])/(r*r)
                    dBLdc[2, col+4, ic] = cosj2t*j2*sini2x*(0.5*dgwdx[ic])/r + pi*cosi2x*i2*sinj2t*(0.5*dgwdt[ic])/(L*r)
                    dBLdc[2, col+5, ic] = -j2*sini2x*sinj2t*(0.5*dgwdx[ic])/r + pi*cosi2x*cosj2t*i2*(0.5*dgwdt[ic])/(L*r)

        print 'DEBUG start dedc'
        dedc = B0 + 0.5*BL + 0.5*dBLdc.dot(cnp)
        e = (B0 + 0.5*BL).dot(cnp)

        print 'DEBUG start dedcidcj'
        dedcidcj = dBLdc
        tmp1d = e.dot(Fnp)
        dedcidcj[0, :] *= tmp1d[0]
        dedcidcj[1, :] *= tmp1d[1]
        dedcidcj[2, :] *= tmp1d[2]
        dedcidcj[3, :] *= tmp1d[3]
        dedcidcj[4, :] *= tmp1d[4]
        dedcidcj[5, :] *= tmp1d[5]

        print 'DEBUG start tmp1d'
        tmp1d = (2*dedcidcj.sum(axis=0) + 2*(dedc.T.dot(Fnp)).dot(dedc)).ravel()
        print 'DEBUG end 1'
        for ic in range(size*size):
            kTv[ic] = beta*(kTv[ic]) + alpha*tmp1d[ic]
        print 'DEBUG end 2'

    free(w0xs)
    free(w0ts)


