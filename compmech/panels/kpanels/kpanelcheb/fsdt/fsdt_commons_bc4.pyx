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

cdef int num0 = 0
cdef int num1 = 5
cdef int e_num = 6
cdef double pi=3.141592653589793

include 'fsdt_commons_include_fuvw.pyx'
#include 'fsdt_commons_include_fstrain.pyx'
#include 'fsdt_commons_include_cfN.pyx'

cdef void cfuvw(double *c, int m1, int n1, double L, double tmin, double tmax,
        double *xs, double *ts, int size, double *us, double *vs,
        double *ws, double *phixs, double *phits) nogil:
    cdef int i1, j1, col, i
    cdef double sini1bx, sinj1bt
    cdef double cosi1bx, cosj1bt
    cdef double x, t, u, v, w, phix, phit, bx, bt

    for i in range(size):
        x = xs[i]
        t = ts[i]
        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        u = 0
        v = 0
        w = 0
        phix = 0
        phit = 0

        for j1 in range(n1):
            sinj1bt = sin(j1*pi*bt)
            cosj1bt = cos(j1*pi*bt)
            for i1 in range(m1):
                col = num0 + num1*((j1)*m1 + (i1))
                sini1bx = sin(i1*pi*bx)
                cosi1bx = cos(i1*pi*bx)
                u += c[col+0]*cosi1bx*cosj1bt
                v += c[col+1]*cosi1bx*cosj1bt
                w += c[col+2]*sini1bx*sinj1bt
                phix += c[col+3]*cosi1bx*cosj1bt
                phit += c[col+4]*cosi1bx*cosj1bt

        us[i] = u
        vs[i] = v
        ws[i] = w
        phixs[i] = phix
        phits[i] = phit


cdef void cfwx(double *c, int m1, int n1, double *xs, double *ts, int size,
        double L, double tmin, double tmax, double *outwx) nogil:
    cdef double dsini1bx, sinj3bt, sinj1bt, wx, x, t, bx, bt
    cdef int i1, j1, col, i

    for i in range(size):
        x = xs[i]
        t = ts[i]
        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        wx = 0.

        for j1 in range(n1):
            sinj1bt = sin(j1*pi*bt)
            for i1 in range(m1):
                col = num0 + num1*((j1)*m1 + (i1))
                dsini1bx = i1*pi/L*cos(i1*pi*bx)
                wx += c[col+2]*dsini1bx*sinj1bt

        outwx[i] = wx


cdef void cfwt(double *c, int m1, int n1, double *xs, double *ts, int size,
        double L, double tmin, double tmax, double *outwt) nogil:
    cdef double sini1bx, dsinj1bt, wt, x, t, bx, bt
    cdef int i1, j1, col, i

    for i in range(size):
        x = xs[i]
        t = ts[i]
        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        wt = 0.

        for j1 in range(n1):
            dsinj1bt = j1*pi/(tmax-tmin)*cos(j1*pi*bt)
            for i1 in range(m1):
                col = num0 + num1*((j1)*m1 + (i1))
                sini1bx = sin(i1*pi*bx)
                wt += c[col+2]*sini1bx*dsinj1bt

        outwt[i] = wt


def fg(double[:,::1] g, int m1, int n1,
       double x, double t, double L, double tmin, double tmax):
    cfg(g, m1, n1, x, t, L, tmin, tmax)

cdef void cfg(double[:,::1] g, int m1, int n1,
              double x, double t, double L, double tmin, double tmax) nogil:
    cdef int i1, j1, col, i
    cdef double sini1bx, sinj1bt
    cdef double cosi1bx, cosj1bt
    cdef double bx, bt

    bx = (x + L/2.)/L
    bt = (t - tmin)/(tmax - tmin)

    for j1 in range(n1):
        sinj1bt = sin(j1*pi*bt)
        cosj1bt = cos(j1*pi*bt)
        for i1 in range(m1):
            col = num0 + num1*((j1)*m1 + (i1))
            sini1bx = sin(i1*pi*bx)
            cosi1bx = cos(i1*pi*bx)
            g[0, col+0] = cosi1bx*cosj1bt
            g[1, col+1] = cosi1bx*cosj1bt
            g[2, col+2] = sini1bx*sinj1bt
            g[3, col+3] = cosi1bx*cosj1bt
            g[4, col+4] = cosi1bx*cosj1bt


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
        r = r1 - sina*(x + L/2.)

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

        for i1 in range(m1):
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
            for i2 in range(m2):
                sini2x = sin(pi*i2*x/L)
                cosi2x = cos(pi*i2*x/L)

                col = 0

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

