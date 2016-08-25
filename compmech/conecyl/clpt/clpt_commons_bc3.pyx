#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
include 'clpt_commons_include_header.pxi'
include 'clpt_commons_include_fstrain.pxi'
include 'clpt_commons_include_fuvw.pxi'
include 'clpt_commons_include_cfN.pxi'

cdef void cfuvw(double *c, int m1, int m2, int n2, double r2, double L,
                double *xs, double *ts, int size,
                double cosa, double tLA,
                double *us, double *vs, double *ws) nogil:
    cdef int i1, i2, j2, col, i
    cdef double sini1x, sini2x, cosi2x, sinj2t, cosj2t, x, t, u, v, w

    for i in range(size):
        x = xs[i]
        t = ts[i]
        u = (c[0]*((L - x)/(L*cosa))
             + c[2]*(L - x)/(L*cosa)*cos(t - tLA))
        v = ((L-x)*r2/L)*c[1]
        w = 0
        for i1 in range(i0, m1+i0):
            sini1x = sin(i1*pi*x/L)
            col = (i1-i0)*num1 + num0
            u += c[col+0]*sini1x
            v += c[col+1]*sini1x
            w += c[col+2]*sini1x
        for j2 in range(j0, n2+j0):
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                sini2x = sin(i2*pi*x/L)
                cosi2x = cos(i2*pi*x/L)
                u += c[col+0]*sini2x*sinj2t
                u += c[col+1]*sini2x*cosj2t

                v += c[col+2]*cosi2x*sinj2t
                v += c[col+3]*cosi2x*cosj2t

                w += c[col+4]*sini2x*sinj2t
                w += c[col+5]*sini2x*cosj2t

        us[i] = u
        vs[i] = v
        ws[i] = w

cdef void cfv(double *c, int m1, int m2, int n2, double *xs, double *ts,
              int size, double r2, double L, double *vs) nogil:
    cdef int i1, i2, j2, col, i
    cdef double cosi2x, sinj2t, cosj2t, x, t, v

    for i in range(size):
        x = xs[i]
        t = ts[i]

        v = c[1]*(L-x)*r2/L
        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            v += sin(i1*pi*x/L)*c[col+1]
        for j2 in range(j0, n2+j0):
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                cosi2x = cos(i2*pi*x/L)
                v += cosi2x*sinj2t*c[col+2]
                v += cosi2x*cosj2t*c[col+3]
        vs[i] = v

cdef void cfwx(double *c, int m1, int m2, int n2, double *xs, double *ts,
               int size, double L, double *outwx) nogil:
    cdef double dsini2x, sinj2t, cosj2t, wx, x, t
    cdef int i1, i2, j2, col, i
    for i in range(size):
        x = xs[i]
        t = ts[i]
        wx = 0.
        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            wx += i1*pi/L*cos(i1*pi*x/L)*c[col+2]
        for j2 in range(j0, n2+j0):
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                dsini2x = i2*pi/L*cos(i2*pi*x/L)
                wx += dsini2x*sinj2t*c[col+4]
                wx += dsini2x*cosj2t*c[col+5]
        outwx[i] = wx

cdef void cfwt(double *c, int m1, int m2, int n2, double *xs, double *ts,
               int size, double L, double *outwt) nogil:
    cdef double sini2x, sinj2t, cosj2t, wt, x, t
    cdef int i2, j2, col, i

    for i in range(size):
        x = xs[i]
        t = ts[i]
        wt = 0.
        for j2 in range(j0, n2+j0):
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            for i2 in range(i0, m2+i0):
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                sini2x = sin(i2*pi*x/L)
                wt += sini2x*(j2*cosj2t)*c[col+4]
                wt += sini2x*(-j2*sinj2t)*c[col+5]
        outwt[i] = wt

def fg(double[:,::1] g, int m1, int m2, int n2,
       double r2, double x, double t, double L, double cosa, double tLA):
    cfg(g, m1, m2, n2, r2, x, t, L, cosa, tLA)

cdef cfg(double[:,::1] g, int m1, int m2, int n2,
           double r2, double x, double t, double L, double cosa, double tLA):
    cdef double sini1x, sini2x, cosi2x, sinj2t, cosj2t
    cdef int i1, i2, j2, col
    g[0, 0] = (L-x)/(L*cosa)
    g[1, 1] = (L-x)*r2/L
    g[0, 2] = (L-x)/(L*cosa)*cos(t - tLA)

    for i1 in range(i0, m1+i0):
        sini1x = sin(i1*pi*x/L)
        col = (i1-i0)*num1 + num0
        g[0, col+0] = sini1x
        g[1, col+1] = sini1x
        g[2, col+2] = sini1x

    for i2 in range(i0, m2+i0):
        sini2x = sin(i2*pi*x/L)
        cosi2x = cos(i2*pi*x/L)
        for j2 in range(j0, n2+j0):
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            g[0, col+0] = sini2x*sinj2t
            g[0, col+1] = sini2x*cosj2t

            g[1, col+2] = cosi2x*sinj2t
            g[1, col+3] = cosi2x*cosj2t

            g[2, col+4] = sini2x*sinj2t
            g[2, col+5] = sini2x*cosj2t

cdef void *cfstrain_donnell(double *c, double sina, double cosa, double tLA,
                            double *xs, double *ts, int size,
                            double r2, double L,
                            int m1, int m2, int n2,
                            double *c0, int m0, int n0, int funcnum,
                            double *es) nogil:
    cdef int i, i1, i2, j2, col
    cdef double wx, wt, w0x, w0t, x, t, r, w0
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef double sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t

    cdef double *wxs = <double *>malloc(size * sizeof(double))
    cdef double *wts = <double *>malloc(size * sizeof(double))
    cdef double *w0xs = <double *>malloc(size * sizeof(double))
    cdef double *w0ts = <double *>malloc(size * sizeof(double))

    cfwx(c, m1, m2, n2, xs, ts, size, L, wxs)
    cfwt(c, m1, m2, n2, xs, ts, size, L, wts)

    cfw0x(xs, ts, size, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, size, c0, L, m0, n0, w0ts, funcnum)

    #TODO
    w0 = 0.

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

        exx = (-c[0]/(L*cosa)
               - c[2]*cos(t - tLA)/(L*cosa)
               + 0.5*castro*w0x*w0x)

        ett = (c[0]*sina*(L - x)/(L*cosa*r)
               + c[2]*sina*(L - x)*cos(t - tLA)/(L*cosa*r)
               +0.5*castro*(2*cosa*r*w0 + w0t*w0t)/(r*r))

        gxt = (castro*w0t*w0x/r
               -c[1]*r2*(r + sina*(L - x))/(L*r)
               + c[2]*(-L + x)*sin(t - tLA)/(L*cosa*r))

        for i1 in range(i0, m1+i0):
            sini1x = sin(pi*i1*x/L)
            cosi1x = cos(pi*i1*x/L)
            col = (i1-i0)*num1 + num0

            exx += (pi*c[col+0]*cosi1x*i1/L
                    + 0.5*pi*c[col+2]*cosi1x*i1*(2*w0x + wx)/L)

            ett += (c[col+0]*sina*sini1x/r
                    + c[col+2]*cosa*sini1x/r)

            gxt += (c[col+1]*(-sina*sini1x/r + pi*cosi1x*i1/L)
                    + 0.5*pi*c[col+2]*cosi1x*i1*(2*w0t + wt)/(L*r))

            kxx += (pi*pi)*c[col+2]*(i1*i1)*sini1x/(L*L)

            ktt += -pi*c[col+2]*cosi1x*i1*sina/(L*r)

        for j2 in range(j0, n2+j0):
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            for i2 in range(i0, m2+i0):
                sini2x = sin(pi*i2*x/L)
                cosi2x = cos(pi*i2*x/L)
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                exx += (pi*c[col+0]*cosi2x*i2*sinj2t/L
                        + pi*c[col+1]*cosi2x*cosj2t*i2/L
                        + 0.5*pi*c[col+4]*cosi2x*i2*sinj2t*(2*w0x + wx)/L
                        + 0.5*pi*c[col+5]*cosi2x*cosj2t*i2*(2*w0x + wx)/L)

                ett += (c[col+0]*sina*sini2x*sinj2t/r
                        + c[col+1]*cosj2t*sina*sini2x/r
                        + c[col+2]*cosi2x*cosj2t*j2/r
                        -c[col+3]*cosi2x*j2*sinj2t/r
                        + 0.5*c[col+4]*sini2x*(2*cosa*r*sinj2t + cosj2t*j2*(2*w0t + wt))/(r*r)
                        + 0.5*c[col+5]*sini2x*(2*cosa*cosj2t*r - j2*sinj2t*(2*w0t + wt))/(r*r))

                gxt += (c[col+0]*cosj2t*j2*sini2x/r
                        -c[col+1]*j2*sini2x*sinj2t/r
                        -c[col+2]*sinj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*r)
                        -c[col+3]*cosj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*r)
                        +0.5*c[col+4]*(L*cosj2t*j2*sini2x*(2*w0x + wx) + pi*cosi2x*i2*sinj2t*(2*w0t + wt))/(L*r)
                        +0.5*c[col+5]*(-L*j2*sini2x*sinj2t*(2*w0x + wx) + pi*cosi2x*cosj2t*i2*(2*w0t + wt))/(L*r))

                kxx += ((pi*pi)*c[col+4]*(i2*i2)*sini2x*sinj2t/(L*L)
                        + (pi*pi)*c[col+5]*cosj2t*(i2*i2)*sini2x/(L*L))

                ktt += (c[col+4]*sinj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r))
                        + c[col+5]*cosj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r)))

                kxt += (c[col+4]*cosj2t*j2*(L*sina*sini2x - 2*pi*cosi2x*i2*r)/(L*(r*r))
                        + c[col+5]*j2*sinj2t*(-L*sina*sini2x + 2*pi*cosi2x*i2*r)/(L*(r*r)))

        es[e_num*i + 0] = exx
        es[e_num*i + 1] = ett
        es[e_num*i + 2] = gxt
        es[e_num*i + 3] = kxx
        es[e_num*i + 4] = ktt
        es[e_num*i + 5] = kxt

    free(wxs)
    free(wts)
    free(w0xs)
    free(w0ts)


cdef void *cfstrain_sanders(double *c, double sina, double cosa, double tLA,
                            double *xs, double *ts, int size,
                            double r2, double L,
                            int m1, int m2, int n2,
                            double *c0, int m0, int n0, int funcnum,
                            double *es) nogil:
    cdef int i, i1, i2, j2, col
    cdef double v, wx, wt, w0x, w0t, w0, x, t, r
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef double sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t
    cdef double *vs = <double *>malloc(size * sizeof(double))
    cdef double *wxs = <double *>malloc(size * sizeof(double))
    cdef double *wts = <double *>malloc(size * sizeof(double))
    cdef double *w0xs = <double *>malloc(size * sizeof(double))
    cdef double *w0ts = <double *>malloc(size * sizeof(double))

    #TODO
    w0 = 0.

    cfv(c, m1, m2, n2, xs, ts, size, r2, L, vs)
    cfwx(c, m1, m2, n2, xs, ts, size, L, wxs)
    cfwt(c, m1, m2, n2, xs, ts, size, L, wts)

    cfw0x(xs, ts, size, c0, L, m0, n0, w0xs, funcnum)
    cfw0t(xs, ts, size, c0, L, m0, n0, w0ts, funcnum)

    for i in range(size):
        x = xs[i]
        t = ts[i]
        wx = wxs[i]
        wt = wts[i]
        w0x = w0xs[i]
        w0t = w0ts[i]
        v = vs[i]

        r = r2 + x*sina

        exx = 0
        ett = 0
        gxt = 0
        kxx = 0
        ktt = 0
        kxt = 0

        exx = (-c[0]/(L*cosa)
               -c[2]*cos(t - tLA)/(L*cosa)
               + 0.5*castro*w0x*w0x)

        ett = (c[0]*sina*(L - x)/(L*cosa*r)
               + 0.5*c[1]*cosa*r2*(L - x)*(cosa*v - 2*w0t - wt)/(L*(r*r))
               + c[2]*sina*(L - x)*cos(t - tLA)/(L*cosa*r)
               + 0.5*castro*(2*cosa*r*w0 + w0t*w0t)/(r*r))

        gxt = (-0.5*c[1]*r2*(2*r + (L - x)*(cosa*(2*w0x + wx) + 2*sina))/(L*r)
               + c[2]*(-L + x)*sin(t - tLA)/(L*cosa*r)
               + castro*w0t*w0x/r)

        kxt = -c[1]*cosa*r2*(r + sina*(L - x))/(L*(r*r))

        for i1 in range(i0, m1+i0):
            sini1x = sin(pi*i1*x/L)
            cosi1x = cos(pi*i1*x/L)
            col = (i1-i0)*num1 + num0

            exx += (pi*c[col+0]*cosi1x*i1/L
                    + 0.5*pi*c[col+2]*cosi1x*i1*(2*w0x + wx)/L)

            ett += (c[col+0]*sina*sini1x/r
                    + 0.5*c[col+1]*cosa*sini1x*(cosa*v - 2*w0t - wt)/(r*r)
                    + c[col+2]*cosa*sini1x/r)

            gxt += (c[col+1]*(-0.5*sini1x*(cosa*(2*w0x + wx) + 2*sina)/r + pi*cosi1x*i1/L)
                    + 0.5*pi*c[col+2]*cosi1x*i1*(-cosa*v + 2*w0t + wt)/(L*r))

            kxx += (pi*pi)*c[col+2]*(i1*i1)*sini1x/(L*L)

            ktt += -pi*c[col+2]*cosi1x*i1*sina/(L*r)

            kxt += c[col+1]*cosa*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*(r*r))

        for j2 in range(j0, n2+j0):
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            for i2 in range(i0, m2+i0):
                sini2x = sin(pi*i2*x/L)
                cosi2x = cos(pi*i2*x/L)
                col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

                exx += (pi*c[col+0]*cosi2x*i2*sinj2t/L
                        + pi*c[col+1]*cosi2x*cosj2t*i2/L
                        + 0.5*pi*c[col+4]*cosi2x*i2*sinj2t*(2*w0x + wx)/L
                        + 0.5*pi*c[col+5]*cosi2x*cosj2t*i2*(2*w0x + wx)/L)

                ett += (c[col+0]*sina*sini2x*sinj2t/r
                        + c[col+1]*cosj2t*sina*sini2x/r
                        + 0.5*c[col+2]*cosi2x*(cosa*sinj2t*(cosa*v - 2*w0t - wt) + 2*cosj2t*j2*r)/(r*r)
                        + 0.5*c[col+3]*cosi2x*(cosa*cosj2t*(cosa*v - 2*w0t - wt) - 2*j2*r*sinj2t)/(r*r)
                        + 0.5*c[col+4]*sini2x*(2*cosa*r*sinj2t + cosj2t*j2*(-cosa*v + 2*w0t + wt))/(r*r)
                        + 0.5*c[col+5]*sini2x*(2*cosa*cosj2t*r + j2*sinj2t*(cosa*v - 2*w0t - wt))/(r*r))

                gxt += (-c[col+1]*j2*sini2x*sinj2t/r
                        -0.5*c[col+2]*sinj2t*(L*cosi2x*(cosa*(2*w0x + wx) + 2*sina) + 2*pi*i2*r*sini2x)/(L*r)
                        -0.5*c[col+3]*cosj2t*(L*cosi2x*(cosa*(2*w0x + wx) + 2*sina) + 2*pi*i2*r*sini2x)/(L*r)
                        +0.5*c[col+4]*(L*cosj2t*j2*sini2x*(2*w0x + wx) + pi*cosi2x*i2*sinj2t*(-cosa*v + 2*w0t + wt))/(L*r)
                        +0.5*c[col+5]*(-L*j2*sini2x*sinj2t*(2*w0x + wx) + pi*cosi2x*cosj2t*i2*(-cosa*v + 2*w0t + wt))/(L*r))

                kxx += ((pi*pi)*c[col+4]*(i2*i2)*sini2x*sinj2t/(L*L)
                        + (pi*pi)*c[col+5]*cosj2t*(i2*i2)*sini2x/(L*L))

                ktt += (c[col+2]*cosa*cosi2x*cosj2t*j2/(r*r)
                        -c[col+3]*cosa*cosi2x*j2*sinj2t/(r*r)
                        + c[col+4]*sinj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r))
                        + c[col+5]*cosj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r)))

                kxt += (-c[col+2]*cosa*sinj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*(r*r))
                        -c[col+3]*cosa*cosj2t*(L*cosi2x*sina + pi*i2*r*sini2x)/(L*(r*r))
                        + c[col+4]*cosj2t*j2*(L*sina*sini2x - 2*pi*cosi2x*i2*r)/(L*(r*r))
                        + c[col+5]*j2*sinj2t*(-L*sina*sini2x + 2*pi*cosi2x*i2*r)/(L*(r*r)))

        es[e_num*i + 0] = exx
        es[e_num*i + 1] = ett
        es[e_num*i + 2] = gxt
        es[e_num*i + 3] = kxx
        es[e_num*i + 4] = ktt
        es[e_num*i + 5] = kxt

    free(vs)
    free(wxs)
    free(wts)
    free(w0xs)
    free(w0ts)

