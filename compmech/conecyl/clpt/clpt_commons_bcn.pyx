#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
include 'clpt_commons_include_header.pyx'
include 'clpt_commons_include_fstrain.pyx'
include 'clpt_commons_include_fuvw.pyx'
include 'clpt_commons_include_cfN.pyx'

cdef void cfuvw(double *c, int m1, int m2, int n2, double r2, double L,
                double *xs, double *ts, int size,
                double cosa, double tLA,
                double *us, double *vs, double *ws) nogil:
    cdef int i1, i2, j2, col, i
    cdef double sini1x, cosi2x, sinj2t, cosj2t, x, t, u, v, w

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
                sini1x = sin(i2*pi*x/L)
                cosi2x = cos(i2*pi*x/L)
                u += c[col+0]*cosi2x*sinj2t
                u += c[col+1]*cosi2x*cosj2t

                v += c[col+2]*cosi2x*sinj2t
                v += c[col+3]*cosi2x*cosj2t

                w += c[col+4]*sini1x*sinj2t
                w += c[col+5]*sini1x*cosj2t
                w += c[col+6]*cosi2x*sinj2t
                w += c[col+7]*cosi2x*cosj2t

        us[i] = u
        vs[i] = v
        ws[i] = w


cdef void cfv(double *c, int m1, int m2, int n2, double *xs, double *ts,
              int size, double r2, double L, double *vs) nogil:
    cdef int i1, i2, j2, col, i
    cdef double cosi2, sinj2t, cosj2t, x, t, v

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
                cosi2 = cos(i2*pi*x/L)
                v += cosi2*sinj2t*c[col+2]
                v += cosi2*cosj2t*c[col+3]
        vs[i] = v

cdef void cfwx(double *c, int m1, int m2, int n2, double *xs, double *ts,
               int size, double L, double *outwx) nogil:
    cdef double dsini2, dcosi2, sinj2t, cosj2t, wx, x, t
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
                dsini2 = i2*pi/L*cos(i2*pi*x/L)
                dcosi2 = -i2*pi/L*sin(i2*pi*x/L)
                wx += dsini2*sinj2t*c[col+4]
                wx += dsini2*cosj2t*c[col+5]
                wx += dcosi2*sinj2t*c[col+6]
                wx += dcosi2*cosj2t*c[col+7]
        outwx[i] = wx

cdef void cfwt(double *c, int m1, int m2, int n2, double *xs, double *ts,
               int size, double L, double *outwt) nogil:
    cdef double sini1x, cosi2x, sinj2t, cosj2t, wt, x, t
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
                sini1x = sin(i2*pi*x/L)
                cosi2x = cos(i2*pi*x/L)
                wt += sini1x*(j2*cosj2t)*c[col+4]
                wt += sini1x*(-j2*sinj2t)*c[col+5]
                wt += cosi2x*(j2*cosj2t)*c[col+6]
                wt += cosi2x*(-j2*sinj2t)*c[col+7]
        outwt[i] = wt

def fg(double[:,::1] gss, int m1, int m2, int n2,
       double r2, double x, double t, double L, double cosa, double tLA):
    cfgss(gss, m1, m2, n2, r2, x, t, L, cosa, tLA)

cdef cfgss(double[:,::1] gss, int m1, int m2, int n2,
           double r2, double x, double t, double L, double cosa, double tLA):
    cdef double sini1x, cosi2x, sinj2t, cosj2t
    cdef int i1, i2, j2, col
    gss[0, 0] = (L-x)/(L*cosa)
    gss[1, 1] = (L-x)*r2/L
    gss[0, 2] = (L-x)/(L*cosa)*cos(t - tLA)

    for i1 in range(i0, m1+i0):
        sini1x = sin(i1*pi*x/L)
        col = (i1-i0)*num1 + num0
        gss[0, col+0] = sini1x
        gss[1, col+1] = sini1x
        gss[2, col+2] = sini1x

    for i2 in range(i0, m2+i0):
        sini1x = sin(i2*pi*x/L)
        cosi2x = cos(i2*pi*x/L)
        for j2 in range(j0, n2+j0):
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            gss[0, col+0] = cosi2x*sinj2t
            gss[0, col+1] = cosi2x*cosj2t

            gss[1, col+2] = cosi2x*sinj2t
            gss[1, col+3] = cosi2x*cosj2t

            gss[2, col+4] = sini1x*sinj2t
            gss[2, col+5] = sini1x*cosj2t
            gss[2, col+6] = cosi2x*sinj2t
            gss[2, col+7] = cosi2x*cosj2t

cdef void *cfstrain_donnell(double *c, double sina, double cosa, double tLA,
                            double *xs, double *ts, int size,
                            double r2, double L,
                            int m1, int m2, int n2,
                            double *c0, int m0, int n0, int funcnum,
                            double *es) nogil:
    with gil:
        raise NotImplementedError

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

        with gil:
            raise NotImplementedError

        exx = (-c[0]/(L*cosa)
                + c[2]*(cos(t - tLA) - 1)/(L*cosa))

        ett = (c[0]*sina*(L - x)/(L*cosa*r)
               -c[2]*sina*(L - x)*(cos(t - tLA) - 1)/(L*cosa*r))

        gxt = (-c[1]*r2*(r + sina*(L - x))/(L*r)
                + c[2]*(L - x)*sin(t - tLA)/(L*cosa*r))

        for i1 in range(i0, m1+i0):
            sini1x = sin(pi*i1*x/L)
            cosi1x = cos(pi*i1*x/L)
            col = (i1-i0)*num1 + num0

            exx += (pi*c[col+0]*cosi1x*i1/L
                    + 0.5*pi*c[col+2]*cosi1x*i1*wx/L)

            ett += (c[col+0]*sina*sini1x/r
                    + c[col+2]*cosa*sini1x/r)

            gxt += (c[col+1]*(-sina*sini1x/r + pi*cosi1x*i1/L)
                    + 0.5*pi*c[col+2]*cosi1x*i1*wt/(L*r))

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
                        + 0.5*pi*c[col+4]*cosi2x*i2*sinj2t*wx/L
                        + 0.5*pi*c[col+5]*cosi2x*cosj2t*i2*wx/L)

                ett += (c[col+0]*sina*sini2x*sinj2t/r
                        + c[col+1]*cosj2t*sina*sini2x/r
                        + c[col+2]*cosj2t*j2*sini2x/r
                        -c[col+3]*j2*sini2x*sinj2t/r
                        + 0.5*c[col+4]*sini2x*(2*cosa*r*sinj2t + cosj2t*j2*wt)/(r*r)
                        + 0.5*c[col+5]*sini2x*(2*cosa*cosj2t*r - j2*sinj2t*wt)/(r*r))

                gxt += (c[col+0]*cosj2t*j2*sini2x/r
                        -c[col+1]*j2*sini2x*sinj2t/r
                        + c[col+2]*sinj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                        + c[col+3]*cosj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                        + 0.5*c[col+4]*(L*cosj2t*j2*sini2x*wx + pi*cosi2x*i2*sinj2t*wt)/(L*r)
                        + 0.5*c[col+5]*(-L*j2*sini2x*sinj2t*wx + pi*cosi2x*cosj2t*i2*wt)/(L*r))

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
    with gil:
        raise NotImplementedError

    cdef int i1, i2, j2, col
    cdef double v, ux, vx, wx, ut, wt
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef double sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t

    exx = 0
    ett = 0
    gxt = 0
    kxx = 0
    ktt = 0
    kxt = 0

    exx = (-c[0]/(L*cosa)
           -0.5*c[1]*r2*vx/L
           + c[2]*(cos(t - tLA) - 1)/(L*cosa))

    ett = (c[0]*sina*(L - x)/(L*cosa*r)
           -0.5*c[1]*r2*(L - x)*(cosa*wt + sina*ut - v)/(L*(r*r))
           -0.5*c[2]*sina*(L - x)*(2*r*(cos(t - tLA) - 1) + v*sin(t - tLA))/(L*cosa*(r*r)))

    gxt = (0.5*c[0]*sina*v/(L*cosa*r)
           -0.5*c[1]*r2*(2*r + (L - x)*(cosa*wx + sina*(ux + 2)))/(L*r)
           + 0.5*c[2]*(-sina*v*cos(t - tLA) + sina*v + (2*L - 2*x)*sin(t - tLA))/(L*cosa*r))

    kxt = -c[1]*r2*(r + sina*(L - x))/(L*(r*r))

    for i1 in range(i0, m1+i0):
        sini1x = sin(pi*i1*x/L)
        cosi1x = cos(pi*i1*x/L)
        col = (i1-i0)*num1 + num0

        exx += (pi*c[col+0]*cosi1x*i1/L
                + 0.5*pi*c[col+1]*cosi1x*i1*vx/L
                + 0.5*pi*c[col+2]*cosi1x*i1*wx/L)

        ett += (c[col+0]*sina*sini1x/r
                -0.5*c[col+1]*sini1x*(cosa*wt + sina*ut - v)/(r*r)
                + c[col+2]*cosa*sini1x/r)

        gxt += (-0.5*pi*c[col+0]*cosi1x*i1*sina*v/(L*r)
                + c[col+1]*(-0.5*sini1x*(cosa*wx + sina*(ux + 2))/r + pi*cosi1x*i1/L)
                + 0.5*pi*c[col+2]*cosi1x*i1*(-cosa*v + wt)/(L*r))

        kxx += (pi*pi)*c[col+2]*(i1*i1)*sini1x/(L*L)

        ktt += -pi*c[col+2]*cosi1x*i1*sina/(L*r)

        kxt += c[col+1]*(-L*sina*sini1x + pi*cosi1x*i1*r)/(L*(r*r))

    for j2 in range(j0, n2+j0):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(i0, m2+i0):
            sini2x = sin(pi*i2*x/L)
            cosi2x = cos(pi*i2*x/L)
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1

            exx += (pi*c[col+0]*cosi2x*i2*sinj2t/L
                    + pi*c[col+1]*cosi2x*cosj2t*i2/L
                    + 0.5*pi*c[col+2]*cosi2x*i2*sinj2t*vx/L
                    + 0.5*pi*c[col+3]*cosi2x*cosj2t*i2*vx/L
                    + 0.5*pi*c[col+4]*cosi2x*i2*sinj2t*wx/L
                    + 0.5*pi*c[col+5]*cosi2x*cosj2t*i2*wx/L)

            ett += (0.5*c[col+0]*sina*sini2x*(-cosj2t*j2*v + 2*r*sinj2t)/(r*r)
                    + 0.5*c[col+1]*sina*sini2x*(2*cosj2t*r + j2*sinj2t*v)/(r*r)
                    + 0.5*c[col+2]*sini2x*(2*cosj2t*j2*r + sinj2t*(-cosa*wt - sina*ut + v))/(r*r)
                    -0.5*c[col+3]*sini2x*(cosj2t*(cosa*wt + sina*ut - v) + 2*j2*r*sinj2t)/(r*r)
                    + 0.5*c[col+4]*sini2x*(2*cosa*r*sinj2t + cosj2t*j2*(-cosa*v + wt))/(r*r)
                    + 0.5*c[col+5]*sini2x*(2*cosa*cosj2t*r + j2*sinj2t*(cosa*v - wt))/(r*r)
                    )

            gxt += (0.5*c[col+0]*(2*cosj2t*j2*sini2x - pi*cosi2x*i2*sina*sinj2t*v/L)/r
                    -0.5*c[col+1]*(2*L*j2*sini2x*sinj2t + pi*cosi2x*cosj2t*i2*sina*v)/(L*r)
                    + 0.5*c[col+2]*sinj2t*(-L*sini2x*(cosa*wx + sina*(ux + 2)) + 2*pi*cosi2x*i2*r)/(L*r)
                    + 0.5*c[col+3]*cosj2t*(-L*sini2x*(cosa*wx + sina*(ux + 2)) + 2*pi*cosi2x*i2*r)/(L*r)
                    + 0.5*c[col+4]*(L*cosj2t*j2*sini2x*wx + pi*cosi2x*i2*sinj2t*(-cosa*v + wt))/(L*r)
                    -0.5*c[col+5]*(L*j2*sini2x*sinj2t*wx + pi*cosi2x*cosj2t*i2*(cosa*v - wt))/(L*r))

            kxx += ((pi*pi)*c[col+4]*(i2*i2)*sini2x*sinj2t/(L*L)
                    + (pi*pi)*c[col+5]*cosj2t*(i2*i2)*sini2x/(L*L))

            ktt += (c[col+2]*cosj2t*j2*sini2x/(r*r)
                    -c[col+3]*j2*sini2x*sinj2t/(r*r)
                    + c[col+4]*sinj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r))
                    + c[col+5]*cosj2t*(L*(j2*j2)*sini2x - pi*cosi2x*i2*r*sina)/(L*(r*r)))

            kxt += (c[col+2]*sinj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*(r*r))
                    + c[col+3]*cosj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*(r*r))
                    + c[col+4]*cosj2t*j2*(L*sina*sini2x - 2*pi*cosi2x*i2*r)/(L*(r*r))
                    + c[col+5]*j2*sinj2t*(-L*sina*sini2x + 2*pi*cosi2x*i2*r)/(L*(r*r)))

    e[0] = exx
    e[1] = ett
    e[2] = gxt
    e[3] = kxx
    e[4] = ktt
    e[5] = kxt

