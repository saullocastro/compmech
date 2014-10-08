#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
include 'fsdt_commons_include_header.pyx'
include 'fsdt_commons_include_fstrain.pyx'
include 'fsdt_commons_include_cfN.pyx'
include 'fsdt_commons_include_fuvw.pyx'

cdef void cfuvw(double *c, int m1, int m2, int n2, double r2, double L,
                double *xs, double *ts, int size,
                double cosa, double tLA,
                double *us, double *vs, double *ws,
                double *phixs, double *phits) nogil:
    cdef int i1, i2, j2, col, i
    cdef double sinbi, cosbi, sinbj, cosbj, u, v, w, phix, phit, x, t

    for i in range(size):
        x = xs[i]
        t = ts[i]
        u = (c[0]*((L-x)/(cosa*L))
             +c[2]*(L - x)/(L*cosa)*cos(t - tLA))
        v = c[1]*((L-x)*r2/L)
        w = 0
        phix = 0
        phit = 0
        for i1 in range(i0, m1+i0):
            sinbi = sin(i1*pi*x/L)
            cosbi = cos(i1*pi*x/L)
            col = (i1-i0)*num1 + num0
            u += c[col+0]*sinbi
            v += c[col+1]*sinbi
            w += c[col+2]*sinbi
            phix += c[col+3]*cosbi
            phit += c[col+4]*sinbi
        for j2 in range(j0, n2+j0):
            sinbj = sin(j2*t)
            cosbj = cos(j2*t)
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                sinbi = sin(i2*pi*x/L)
                cosbi = cos(i2*pi*x/L)
                u += c[col+0]*cosbi*sinbj
                u += c[col+1]*cosbi*cosbj

                v += c[col+2]*sinbi*sinbj
                v += c[col+3]*sinbi*cosbj

                w += c[col+4]*sinbi*sinbj
                w += c[col+5]*sinbi*cosbj

                phix += c[col+6]*cosbi*sinbj
                phix += c[col+7]*cosbi*cosbj

                phit += c[col+8]*sinbi*sinbj
                phit += c[col+9]*sinbi*cosbj

        us[i] = u
        vs[i] = v
        ws[i] = w
        phixs[i] = phix
        phits[i] = phit

cdef void cfwx(double *c, int m1, int m2, int n2, double L,
               double *xs, double *ts, int size, double *wxs) nogil:
    cdef int i1, i2, j2, col, i
    cdef double sinbj, cosbj, dsinbi, wx, x, t

    for i in range(size):
        x = xs[i]
        t = ts[i]
        wx = 0
        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            wx += c[col+2]*(i1*pi/L)*cos(i1*pi*x/L)
        for j2 in range(j0, n2+j0):
            sinbj = sin(j2*t)
            cosbj = cos(j2*t)
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                dsinbi = (i2*pi/L)*cos(i2*pi*x/L)
                wx += c[col+4]*dsinbi*sinbj
                wx += c[col+5]*dsinbi*cosbj
        wxs[i] = wx


cdef void cfwt(double *c, int m1, int m2, int n2, double L,
               double *xs, double *ts, int size, double *wts) nogil:
    cdef int i1, i2, j2, col, i
    cdef double sinbi, dsinbj, dcosbj, wt, x, t

    for i in range(size):
        x = xs[i]
        t = ts[i]
        wt = 0
        for j2 in range(j0, n2+j0):
            dsinbj = j2*cos(j2*t)
            dcosbj = -j2*sin(j2*t)
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                sinbi = sin(i2*pi*x/L)
                wt += c[col+4]*sinbi*dsinbj
                wt += c[col+5]*sinbi*dcosbj
        wts[i] = wt

def fg(double[:,::1] g, int m1, int m2, int n2,
       double r2, double x, double t, double L, double cosa, double tLA):
    cfg(g, m1, m2, n2, r2, x, t, L, cosa, tLA)

cdef cfg(double[:, ::1] g, int m1, int m2, int n2,
         double r2, double x, double t, double L, double cosa, double tLA):
    cdef double sinbi, cosbi, sinbj, cosbj
    cdef int i1, i2, j2, col
    g[0,0] = (L-x)/(L*cosa)
    g[1,1] = (L-x)*r2/L
    g[0,2] = (L - x)/(L*cosa)*cos(t - tLA)

    for i1 in range(i0, m1+i0):
        sinbi = sin(i1*pi*x/L)
        cosbi = cos(i1*pi*x/L)
        col = (i1-i0)*num1 + num0
        g[0, col+0] = sinbi
        g[1, col+1] = sinbi
        g[2, col+2] = sinbi
        g[3, col+3] = cosbi
        g[4, col+4] = sinbi

    for i2 in range(i0, m2+i0):
        sinbi = sin(i2*pi*x/L)
        cosbi = cos(i2*pi*x/L)
        for j2 in range(j0, n2+j0):
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            sinbj = sin(j2*t)
            cosbj = cos(j2*t)
            g[0, col+0] = cosbi*sinbj
            g[0, col+1] = cosbi*cosbj

            g[1, col+2] = sinbi*sinbj
            g[1, col+3] = sinbi*cosbj

            g[2, col+4] = sinbi*sinbj
            g[2, col+5] = sinbi*cosbj

            g[3, col+6] = cosbi*sinbj
            g[3, col+7] = cosbi*cosbj

            g[4, col+8] = sinbi*sinbj
            g[4, col+9] = sinbi*cosbj


cdef void *cfstrain_donnell(double *c, double sina, double cosa, double tLA,
                            double *xs, double *ts, int size,
                            double r2, double L,
                            int m1, int m2, int n2,
                            double *c0, int m0, int n0, int funcnum,
                            double *es) nogil:
    cdef int i1, i2, j2, col, i
    cdef double wx, wt, w0x, w0t, x, t, r, w0
    cdef double exx, ett, gxt, kxx, ktt, kxt, gtz, gxz
    cdef double sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t

    cdef double *wxs = <double *>malloc(size * sizeof(double))
    cdef double *wts = <double *>malloc(size * sizeof(double))
    cdef double *w0xs = <double *>malloc(size * sizeof(double))
    cdef double *w0ts = <double *>malloc(size * sizeof(double))

    #TODO
    w0 = 0.

    cfwx(c, m1, m2, n2, L, xs, ts, size, wxs)
    cfwt(c, m1, m2, n2, L, xs, ts, size, wts)

    cfw0x(xs, ts, size, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, size, c0, L, m0, n0, w0ts, funcnum)

    for i in range(size):
        x = xs[i]
        t = ts[i]
        wx = wxs[i]
        wt = wts[i]
        w0x = w0xs[i]
        w0t = w0ts[i]
        r = r2 + x*sina

        exx = 0
        ett = 0
        gxt = 0
        kxx = 0
        ktt = 0
        kxt = 0
        gtz = 0
        gxz = 0

        exx = (-c[0]/(L*cosa)
               -c[2]*cos(t - tLA)/(L*cosa)
               +0.5*castro*w0x*w0x)

        ett = (c[0]*sina*(L - x)/(L*cosa*r)
               +c[2]*sina*(L - x)*cos(t - tLA)/(L*cosa*r)
               +0.5*castro*(2*cosa*r*w0 + w0t*w0t)/(r*r))

        gxt = (-c[1]*r2*(r + sina*(L - x))/(L*r)
               +c[2]*(-L + x)*sin(t - tLA)/(L*cosa*r)
               +castro*w0t*w0x/r)

        gtz = c[1]*cosa*r2*(-L + x)/(L*r)


        for i1 in range(i0, m1+i0):
            sini1x = sin(pi*i1*x/L)
            cosi1x = cos(pi*i1*x/L)
            col = (i1-i0)*num1 + num0

            exx += (pi*c[col+0]*cosi1x*i1/L
                    + pi*c[col+1]*cosi1x*i1/L
                    -0.5*pi*c[col+3]*i1*sini1x*(2*w0x + wx)/L)

            ett += (c[col+0]*sina*sini1x/r
                    + c[col+1]*sina*sini1x/r
                    + c[col+3]*cosa*cosi1x/r)

            gxt += (c[col+2]*(-sina*sini1x/r + pi*cosi1x*i1/L)
                    -0.5*pi*c[col+3]*i1*sini1x*(2*w0t + wt)/(L*r))

            kxx += pi*c[col+4]*cosi1x*i1/L

            ktt += c[col+4]*sina*sini1x/r

            gtz += -c[col+2]*cosa*sini1x/r

            gxz += (-pi*c[col+3]*i1*sini1x/L
                    + c[col+4]*sini1x)


        for j2 in range(j0, n2+j0):
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            for i2 in range(i0, m2+i0):
                sini2x = sin(pi*i2*x/L)
                cosi2x = cos(pi*i2*x/L)
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                exx += (-pi*c[col+0]*i2*sini2x*sinj2t/L
                        -pi*c[col+1]*cosj2t*i2*sini2x/L
                        + 0.5*pi*c[col+4]*cosi2x*i2*sinj2t*(2*w0x + wx)/L
                        + 0.5*pi*c[col+5]*cosi2x*cosj2t*i2*(2*w0x + wx)/L)

                ett += (c[col+0]*cosi2x*sina*sinj2t/r
                        + c[col+1]*cosi2x*cosj2t*sina/r
                        + c[col+2]*cosj2t*j2*sini2x/r
                        -c[col+3]*j2*sini2x*sinj2t/r
                        + 0.5*c[col+4]*sini2x*(2*cosa*r*sinj2t + cosj2t*j2*(2*w0t + wt))/r**2
                        + 0.5*c[col+5]*sini2x*(2*cosa*cosj2t*r - j2*sinj2t*(2*w0t + wt))/r**2)

                gxt += (c[col+0]*cosi2x*cosj2t*j2/r
                        -c[col+1]*cosi2x*j2*sinj2t/r
                        + c[col+2]*sinj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                        + c[col+3]*cosj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                        + 0.5*c[col+4]*(L*cosj2t*j2*sini2x*(2*w0x + wx) + pi*cosi2x*i2*sinj2t*(2*w0t + wt))/(L*r)
                        + 0.5*c[col+5]*(-L*j2*sini2x*sinj2t*(2*w0x + wx) + pi*cosi2x*cosj2t*i2*(2*w0t + wt))/(L*r))

                kxx += (-pi*c[col+6]*i2*sini2x*sinj2t/L
                        -pi*c[col+7]*cosj2t*i2*sini2x/L)

                ktt += (c[col+6]*cosi2x*sina*sinj2t/r
                        + c[col+7]*cosi2x*cosj2t*sina/r
                        + c[col+8]*cosj2t*j2*sini2x/r
                        -c[col+9]*j2*sini2x*sinj2t/r)

                kxt += (c[col+6]*cosi2x*cosj2t*j2/r
                        -c[col+7]*cosi2x*j2*sinj2t/r
                        + c[col+8]*sinj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                        + c[col+9]*cosj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r))

                gtz += (-c[col+2]*cosa*sini2x*sinj2t/r
                        -c[col+3]*cosa*cosj2t*sini2x/r
                        + c[col+4]*cosj2t*j2*sini2x/r
                        -c[col+5]*j2*sini2x*sinj2t/r
                        + c[col+8]*sini2x*sinj2t
                        + c[col+9]*cosj2t*sini2x)

                gxz += (pi*c[col+4]*cosi2x*i2*sinj2t/L
                        + pi*c[col+5]*cosi2x*cosj2t*i2/L
                        + c[col+6]*cosi2x*sinj2t
                        + c[col+7]*cosi2x*cosj2t)

        es[e_num*i + 0] = exx
        es[e_num*i + 1] = ett
        es[e_num*i + 2] = gxt
        es[e_num*i + 3] = kxx
        es[e_num*i + 4] = ktt
        es[e_num*i + 5] = kxt
        es[e_num*i + 6] = gtz
        es[e_num*i + 7] = gxz

    free(wxs)
    free(wts)
    free(w0xs)
    free(w0ts)
