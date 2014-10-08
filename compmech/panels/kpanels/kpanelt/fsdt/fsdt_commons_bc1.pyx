#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange

DOUBLE = np.float64
INT = np.int64
ctypedef np.double_t cDOUBLE
ctypedef np.int64_t cINT

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

cdef int num0 = 4
cdef int num1 = 2
cdef int num2 = 2
cdef int num3 = 2
cdef int num4 = 5
cdef int e_num = 8
cdef double pi=3.141592653589793

include 'fsdt_commons_include_fstrain.pyx'
include 'fsdt_commons_include_fuvw.pyx'
include 'fsdt_commons_include_cfN.pyx'

cdef void cfuvw(double *c, int m2, int n3, int m4, int n4,
        double L, double tmin, double tmax,
        double *xs, double *ts, int size, double *us, double *vs, double *ws,
        double *phixs, double *phits) nogil:
    cdef int i2, j3, i4, j4, col, i
    cdef double cosi2bx, cosj3bt, sini4bx, sinj4bt, cosi4bx, cosj4bt
    cdef double x, t, u, v, w, phix, phit, bx, bt

    for i in range(size):
        x = xs[i]
        t = ts[i]
        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        u = c[0]*bx + c[1]*bt
        v = c[2]*bx + c[3]*bt
        w = 0.
        phix = c[4]
        phit = c[5]

        for i2 in range(1, m2+1):
            cosi2bx = cos(i2*pi*bx)
            col = num0 + num1 + num2*(i2-1)
            phix += c[col+0]*cosi2bx
            phit += c[col+1]*cosi2bx

        for j3 in range(1, n3+1):
            cosj3bt = cos(j3*pi*bt)
            col = num0 + num1 + num2*m2 + num3*(j3-1)
            phix += c[col+0]*cosj3bt
            phit += c[col+1]*cosj3bt

        for j4 in range(1, n4+1):
            sinj4bt = sin(j4*pi*bt)
            cosj4bt = cos(j4*pi*bt)
            for i4 in range(1, m4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 + (j4-1)*num4*m4 +
                        (i4-1)*num4)
                sini4bx = sin(i4*pi*bx)
                cosi4bx = cos(i4*pi*bx)
                u += c[col+0]*sini4bx*sinj4bt
                v += c[col+1]*sini4bx*sinj4bt
                w += c[col+2]*sini4bx*sinj4bt
                phix += c[col+3]*cosi4bx*cosj4bt
                phit += c[col+4]*cosi4bx*cosj4bt

        us[i] = u
        vs[i] = v
        ws[i] = w
        phixs[i] = phix
        phits[i] = phit


cdef void cfwx(double *c, int m2, int n3, int m4, int n4, double *xs, double
    *ts, int size, double L, double tmin, double tmax, double *outwx) nogil:
    cdef double dsini4bx, sinj4bt, wx, x, t, bx, bt
    cdef int i2, j3, i4, j4, col, i

    for i in range(size):
        x = xs[i]
        t = ts[i]
        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax-tmin)

        wx = 0.

        for j4 in range(1, n4+1):
            sinj4bt = sin(j4*pi*bt)
            for i4 in range(1, m4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 + (j4-1)*num4*m4 +
                        (i4-1)*num4)
                dsini4bx = i4*pi/L*cos(i4*pi*bx)
                wx += dsini4bx*sinj4bt*c[col+2]

        outwx[i] = wx


cdef void cfwt(double *c, int m2, int n3, int m4, int n4, double *xs, double
    *ts, int size, double L, double tmin, double tmax, double *outwt) nogil:
    cdef double sini4bx, dsinj4bt, wt, x, t, bx, bt
    cdef int i2, j3, i4, j4, col, i

    for i in range(size):
        x = xs[i]
        t = ts[i]
        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax-tmin)

        wt = 0.

        for j4 in range(1, n4+1):
            dsinj4bt = j4*pi/(tmax-tmin)*cos(j4*pi*bt)
            for i4 in range(1, m4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 + (j4-1)*num4*m4 +
                        (i4-1)*num4)
                sini4bx = sin(i4*pi*bx)
                wt += sini4bx*dsinj4bt*c[col+2]

        outwt[i] = wt

def fg(double[:,::1] g, int m2, int n3, int m4, int n4,
       double x, double t, double L, double tmin, double tmax):
    cfg(g, m2, n3, m4, n4, x, t, L, tmin, tmax)

cdef void cfg(double[:,::1] g, int m2, int n3, int m4, int n4,
              double x, double t, double L, double tmin, double tmax) nogil:
    cdef int i2, j3, i4, j4, col, i
    cdef double cosi2bx, cosj3bt, sini4bx, sinj4bt, cosi4bx, cosj4bt
    cdef double u, v, w, phix, phit, bx, bt

    bx = (x + L/2.)/L
    bt = (t - tmin)/(tmax - tmin)

    g[0, 0] = bx
    g[0, 1] = bt
    g[1, 2] = bx
    g[1, 3] = bt
    g[3, 4] = 1.
    g[4, 5] = 1.

    for i2 in range(1, m2+1):
        cosi2bx = cos(i2*pi*bx)
        col = num0 + num1 + num2*(i2-1)
        g[3, col+0] = cosi2bx
        g[4, col+1] = cosi2bx

    for j3 in range(1, n3+1):
        cosj3bt = cos(j3*pi*bt)
        col = num0 + num1 + num2*m2 + num3*(j3-1)
        g[3, col+0] = cosj3bt
        g[4, col+1] = cosj3bt

    for j4 in range(1, n4+1):
        sinj4bt = sin(j4*pi*bt)
        cosj4bt = cos(j4*pi*bt)
        for i4 in range(1, m4+1):
            col = (num0 + num1 + num2*m2 + num3*n3 + (j4-1)*num4*m4 +
                    (i4-1)*num4)
            sini4bx = sin(i4*pi*bx)
            cosi4bx = cos(i4*pi*bx)
            g[0, col+0] = sini4bx*sinj4bt
            g[1, col+1] = sini4bx*sinj4bt
            g[2, col+2] = sini4bx*sinj4bt
            g[3, col+3] = cosi4bx*cosj4bt
            g[4, col+4] = cosi4bx*cosj4bt


cdef void *cfstrain_donnell(double *c, double sina, double cosa, double tLA,
                            double *xs, double *ts, int size,
                            double r1, double L,
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

    #TODO
    w0 = 0.

    #cfwx(c, m1, m2, n2, xs, ts, size, L, wxs)
    #cfwt(c, m1, m2, n2, xs, ts, size, L, wts)

    for i in range(size):
        x = xs[i]
        t = ts[i]
        wx = wxs[i]
        wt = wts[i]
        w0x = w0xs[i]
        w0t = w0ts[i]
        r = r1 + sina*(x + L/2.)

        exx = 0
        ett = 0
        gxt = 0
        kxx = 0
        ktt = 0
        kxt = 0

        exx = (-c[0]/(L*cosa)
               - c[2]*cos(t - tLA)/(L*cosa)
               + 0.5*0*w0x*w0x)

        ett = (c[0]*sina*(L - x)/(L*cosa*r)
               + c[2]*sina*(L - x)*cos(t - tLA)/(L*cosa*r)
               + 0.5*0*(2*cosa*r*w0 + w0t*w0t)/(r*r))

        gxt = (0)

        for i1 in range(1, m1+1):
            sini1x = sin(pi*i1*x/L)
            cosi1x = cos(pi*i1*x/L)
            col = (i1-1)*num1 + num0

            exx += (pi*c[col+0]*cosi1x*i1/L
                    + pi*c[col+2]*cosi1x*i1*(w0x + 0.5*wx)/L)

            ett += (c[col+0]*sina*sini1x/r
                    + c[col+2]*cosa*sini1x/r)

            gxt += (c[col+1]*(-sina*sini1x/r + pi*cosi1x*i1/L)
                    + pi*c[col+2]*cosi1x*i1*(w0t + 0.5*wt)/(L*r))

            kxx += (pi*pi)*c[col+2]*(i1*i1)*sini1x/(L*L)

            ktt += -pi*c[col+2]*cosi1x*i1*sina/(L*r)

        for j2 in range(1, n2+1):
            sinj2t = sin(j2*t)
            cosj2t = cos(j2*t)
            for i2 in range(1, m2+1):
                sini2x = sin(pi*i2*x/L)
                cosi2x = cos(pi*i2*x/L)
                col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1

                exx += (pi*c[col+0]*cosi2x*i2*sinj2t/L
                        + pi*c[col+1]*cosi2x*cosj2t*i2/L
                        + pi*c[col+4]*cosi2x*i2*sinj2t*(w0x + 0.5*wx)/L
                        + pi*c[col+5]*cosi2x*cosj2t*i2*(w0x + 0.5*wx)/L)

                ett += (c[col+0]*sina*sini2x*sinj2t/r
                        + c[col+1]*cosj2t*sina*sini2x/r
                        + c[col+2]*cosj2t*j2*sini2x/r
                        -c[col+3]*j2*sini2x*sinj2t/r
                        + 0.5*c[col+4]*sini2x*(2*cosa*r*sinj2t + cosj2t*j2*(2*w0t + wt))/(r*r)
                        + 0.5*c[col+5]*sini2x*(2*cosa*cosj2t*r - j2*sinj2t*(2*w0t + wt))/(r*r))

                gxt += (c[col+0]*cosj2t*j2*sini2x/r
                        -c[col+1]*j2*sini2x*sinj2t/r
                        + c[col+2]*sinj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                        + c[col+3]*cosj2t*(-L*sina*sini2x + pi*cosi2x*i2*r)/(L*r)
                        + 0.5*c[col+4]*(L*cosj2t*j2*sini2x*(2*w0x + wx) + pi*cosi2x*i2*sinj2t*(2*w0t + wt))/(L*r)
                        + 0.5*c[col+5]*(-L*j2*sini2x*sinj2t*(2*w0x + wx) + pi*cosi2x*cosj2t*i2*(2*w0t + wt))/(L*r))

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

