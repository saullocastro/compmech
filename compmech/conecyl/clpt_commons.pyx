#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

cimport numpy as np
import numpy as np

DOUBLE = np.float64
INT = np.int64
ctypedef np.double_t cDOUBLE
ctypedef np.int64_t cINT

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

cdef int num0 = 3
cdef int num1 = 3
cdef int num2 = 6
cdef double pi=3.141592653589793

def fstrain(np.ndarray[cDOUBLE, ndim=1] c,
            double sina, double cosa, double tLA,
            np.ndarray[cDOUBLE, ndim=1] xvec,
            np.ndarray[cDOUBLE, ndim=1] tvec,
            double r2, double L, int m1, int m2, int n2, int NL_kinematics):
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef double x, t, r
    cdef int ie, ix
    cdef np.ndarray[cDOUBLE, ndim=1] evec
    cdef double e[6]
    cdef cfstraintype *cfstrain
    if NL_kinematics==0:
        cfstrain = &cfstrain_donnell
    elif NL_kinematics==1:
        cfstrain = &cfstrain_sanders
    evec = np.zeros((xvec.shape[0]*6), dtype=DOUBLE)
    for ix in range(xvec.shape[0]):
        x = xvec[ix]
        t = tvec[ix]
        r = r2 + x*sina
        cfstrain(&c[0], sina, cosa, tLA, x, t, r, r2, L, m1, m2, n2, e)
        for ie in range(6):
            evec[ix*6 + ie] = e[ie]
    return evec

cdef void cfN(double *c, double sina, double cosa, double tLA,
              double x, double t, double r, double r2, double L, double *F,
              int m1, int m2, int n2, double *N, int NL_kinematics) nogil:
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef cfstraintype *cfstrain
    if NL_kinematics==0:
        cfstrain = &cfstrain_donnell
    elif NL_kinematics==1:
        cfstrain = &cfstrain_sanders
    cfstrain(c, sina, cosa, tLA, x, t, r, r2, L, m1, m2, n2, N)
    #NOTE using array N to transfer values of strains
    exx = N[0]
    ett = N[1]
    gxt = N[2]
    kxx = N[3]
    ktt = N[4]
    kxt = N[5]
    N[0] =  F[0]*exx  + F[1]*ett  + F[2]*gxt  + F[3]*kxx  + F[4]*ktt  + F[5]*kxt
    N[1] =  F[6]*exx  + F[7]*ett  + F[8]*gxt  + F[9]*kxx  + F[10]*ktt + F[11]*kxt
    N[2] =  F[12]*exx + F[13]*ett + F[14]*gxt + F[15]*kxx + F[16]*ktt + F[17]*kxt
    N[3] =  F[18]*exx + F[19]*ett + F[20]*gxt + F[21]*kxx + F[22]*ktt + F[23]*kxt
    N[4] =  F[24]*exx + F[25]*ett + F[26]*gxt + F[27]*kxx + F[28]*ktt + F[29]*kxt
    N[5] =  F[30]*exx + F[31]*ett + F[32]*gxt + F[33]*kxx + F[34]*ktt + F[35]*kxt

cdef void cfuvw(double *c, int m1, int m2, int n2, double r2, double L,
                double x, double t,
                double cosa, double tLA, double *uvw) nogil:
    cdef int i1, i2, j2, col
    cdef double sinbi, sinbj, cosbj, u, v, w
    u = (c[0]*((L - x)/(L*cosa))
         + c[2]*(L - x)/(L*cosa)*(1 - cos(t - tLA)))
    v = ((L-x)*r2/L)*c[1]
    w = 0
    for i1 in range(1, m1+1):
        sinbi = sin(i1*pi*x/L)
        col = (i1-1)*num1 + num0
        u += c[col+0]*sinbi
        v += c[col+1]*sinbi
        w += c[col+2]*sinbi

    for j2 in range(1, n2+1):
        sinbj = sin(j2*t)
        cosbj = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbi = sin(i2*pi*x/L)
            u += c[col+0]*sinbi*sinbj
            u += c[col+1]*sinbi*cosbj
            v += c[col+2]*sinbi*sinbj
            v += c[col+3]*sinbi*cosbj
            w += c[col+4]*sinbi*sinbj
            w += c[col+5]*sinbi*cosbj

    uvw[0] = u
    uvw[1] = v
    uvw[2] = w


cdef void cfuvw_x(double *c, int m1, int m2, int n2, double r2, double L,
                  double x, double t,
                  double cosa, double tLA, double *uvw_x) nogil:
    cdef int i1, i2, j2, col
    cdef double expr, sinbj, cosbj, ux, vx, wx

    ux = -1/(cosa*L)*c[0] - 1/(L*cosa)*(1 - cos(t - tLA))*c[2]
    vx = -r2/L*c[1]
    wx = 0
    for i1 in range(1, m1+1):
        expr = i1*pi/L*cos(i1*pi*x/L)
        col = (i1-1)*num1 + num0
        ux += c[col+0]*expr
        vx += c[col+1]*expr
        wx += c[col+2]*expr
    for j2 in range(1, n2+1):
        sinbj = sin(j2*t)
        cosbj = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            expr = i2*pi/L*cos(i2*pi*x/L)
            ux += c[col+0]*expr*sinbj
            ux += c[col+1]*expr*cosbj
            vx += c[col+2]*expr*sinbj
            vx += c[col+3]*expr*cosbj
            wx += c[col+4]*expr*sinbj
            wx += c[col+5]*expr*cosbj

    uvw_x[0] = ux
    uvw_x[1] = vx
    uvw_x[2] = wx

cdef void cfuvw_t(double *c, int m1, int m2, int n2, double L,
                  double x, double t,
                  double cosa, double tLA, double *uvw_t) nogil:
    cdef int i2, j2, col
    cdef double sinbi, expr1, expr2, ut, vt, wt
    ut = c[2]*(L-x)*sin(t-tLA)/(cosa*L)
    vt = 0
    wt = 0
    for j2 in range(1, n2+1):
        expr1 = j2*cos(j2*t)
        expr2 = -j2*sin(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbi = sin(i2*pi*x/L)
            ut += c[col+0]*sinbi*expr1
            ut += c[col+1]*sinbi*expr2
            vt += c[col+2]*sinbi*expr1
            vt += c[col+3]*sinbi*expr2
            wt += c[col+4]*sinbi*expr1
            wt += c[col+5]*sinbi*expr2

    uvw_t[0] = ut
    uvw_t[1] = vt
    uvw_t[2] = wt

cdef void cfvx(double *c, int m1, int m2, int n2, double r2,
               double x, double t, double L, double *refvx) nogil:
    cdef double expr, sinj2t, cosj2t, vx
    cdef int i1, i2, j2, col
    vx = -c[1]*r2/L
    for i1 in range(1, m1+1):
        col = (i1-1)*num1 + num0
        vx += i1*pi/L*cos(i1*pi*x/L)*c[col+1]
    for j2 in range(1, n2+1):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            expr = i2*pi/L*cos(i2*pi*x/L)
            vx += expr*sinj2t*c[col+2]
            vx += expr*cosj2t*c[col+3]

    refvx[0] = vx

cdef void cfvt(double *c, int m1, int m2, int n2,
               double x, double t, double L, double *refvt) nogil:
    cdef double sinbi, sinj2t, cosj2t, vt
    cdef int i2, j2, col
    vt = 0.
    for j2 in range(1, n2+1):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbi = sin(i2*pi*x/L)
            vt += sinbi*(j2*cosj2t)*c[col+2]
            vt += sinbi*(-j2*sinj2t)*c[col+3]

    refvt[0] = vt

cdef void cfv(double *c, int m1, int m2, int n2, double r2,
              double x, double t, double L, double *refv) nogil:
    cdef double expr, sinj2t, cosj2t, v
    cdef int i1, i2, j2, col
    v = c[1]*(L-x)*r2/L
    for i1 in range(1, m1+1):
        col = (i1-1)*num1 + num0
        v += sin(i1*pi*x/L)*c[col+1]
    for j2 in range(1, n2+1):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            expr = sin(i2*pi*x/L)
            v += expr*sinj2t*c[col+2]
            v += expr*cosj2t*c[col+3]

    refv[0] = v

cdef void cfw(double *c, int m1, int m2, int n2,
              double x, double t, double L, double *refw) nogil:
    cdef double expr, sinj2t, cosj2t, w
    cdef int i1, i2, j2, col
    w = 0.
    for i1 in range(1, m1+1):
        col = (i1-1)*num1 + num0
        w += sin(i1*pi*x/L)*c[col+2]
    for j2 in range(1, n2+1):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            expr = sin(i2*pi*x/L)
            w += expr*sinj2t*c[col+4]
            w += expr*cosj2t*c[col+5]

    refw[0] = w

cdef void cfwx(double *c, int m1, int m2, int n2,
               double x, double t, double L, double *refwx) nogil:
    cdef double expr, sinj2t, cosj2t, wx
    cdef int i1, i2, j2, col
    wx = 0.
    for i1 in range(1, m1+1):
        col = (i1-1)*num1 + num0
        wx += i1*pi/L*cos(i1*pi*x/L)*c[col+2]
    for j2 in range(1, n2+1):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            expr = i2*pi/L*cos(i2*pi*x/L)
            wx += expr*sinj2t*c[col+4]
            wx += expr*cosj2t*c[col+5]

    refwx[0] = wx

cdef void cfwt(double *c, int m1, int m2, int n2,
               double x, double t, double L, double *refwt) nogil:
    cdef double sinbi, sinj2t, cosj2t, wt
    cdef int i2, j2, col
    wt = 0.
    for j2 in range(1, n2+1):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbi = sin(i2*pi*x/L)
            wt += sinbi*(j2*cosj2t)*c[col+4]
            wt += sinbi*(-j2*sinj2t)*c[col+5]

    refwt[0] = wt

def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m1, int m2, int n2,
         double alpharad, double r2, double L, double tLA,
         np.ndarray[cDOUBLE, ndim=1] xvec,
         np.ndarray[cDOUBLE, ndim=1] tvec):
    cdef int ix
    cdef double x, t, sina, cosa
    cdef double uvw[3]
    cdef np.ndarray[cDOUBLE, ndim=1] u, v, w

    sina = sin(alpharad)
    cosa = cos(alpharad)

    u = np.zeros(np.shape(xvec), dtype=DOUBLE)
    v = np.zeros(np.shape(xvec), dtype=DOUBLE)
    w = np.zeros(np.shape(xvec), dtype=DOUBLE)

    for ix in range(xvec.shape[0]):
        x = xvec[ix]
        t = tvec[ix]
        cfuvw(&c[0], m1, m2, n2, r2, L, x, t, cosa, tLA, &uvw[0])
        u[ix] = uvw[0]
        v[ix] = uvw[1]
        w[ix] = uvw[2]
    return u, v, w

def fgss(double[:,::1] gss, int m1, int m2, int n2,
         double r2, double x, double t, double L, double cosa, double tLA):
    cfgss(gss, m1, m2, n2, r2, x, t, L, cosa, tLA)

cdef cfgss(double[:,::1] gss, int m1, int m2, int n2,
           double r2, double x, double t, double L, double cosa, double tLA):
    cdef double sinbi, sinbj, cosbj
    cdef int i1, i2, j2, col
    gss[0, 0] = (L-x)/(L*cosa)
    gss[1, 1] = (L-x)*r2/L
    gss[0, 2] = (L-x)/(L*cosa)*(1 - cos(t - tLA))

    for i1 in range(1, m1+1):
        sinbi = sin(i1*pi*x/L)
        col = (i1-1)*num1 + num0
        gss[0, col+0] = sinbi

        gss[1, col+1] = sinbi

        gss[2, col+2] = sinbi

    for i2 in range(1, m2+1):
        sinbi = sin(i2*pi*x/L)
        for j2 in range(1, n2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbj = sin(j2*t)
            cosbj = cos(j2*t)
            gss[0, col+0] = sinbi*sinbj
            gss[0, col+1] = sinbi*cosbj

            gss[1, col+2] = sinbi*sinbj
            gss[1, col+3] = sinbi*cosbj

            gss[2, col+4] = sinbi*sinbj
            gss[2, col+5] = sinbi*cosbj

cdef void *cfstrain_donnell(double *c, double sina, double cosa, double tLA,
                            double x, double t, double r, double r2, double L,
                            int m1, int m2, int n2, double *e) nogil:
    cdef int i1, i2, j2, col
    cdef double wx, wt
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef double sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t

    cfwx(c, m1, m2, n2, x, t, L, &wx)
    cfwt(c, m1, m2, n2, x, t, L, &wt)

    exx = 0
    ett = 0
    gxt = 0
    kxx = 0
    ktt = 0
    kxt = 0

    exx = (-c[0]/(L*cosa)
            + c[2]*(cos(t - tLA) - 1)/(L*cosa))

    ett = (c[0]*sina*(L - x)/(L*cosa*r)
           -c[2]*sina*(L - x)*(cos(t - tLA) - 1)/(L*cosa*r))

    gxt = (-c[1]*r2*(r + sina*(L - x))/(L*r)
            + c[2]*(L - x)*sin(t - tLA)/(L*cosa*r))

    for i1 in range(1, m1+1):
        sini1x = sin(pi*i1*x/L)
        cosi1x = cos(pi*i1*x/L)
        col = (i1-1)*num1 + num0

        exx += (pi*c[col+0]*cosi1x*i1/L
                + 0.5*pi*c[col+2]*cosi1x*i1*wx/L)

        ett += (c[col+0]*sina*sini1x/r
                + c[col+2]*cosa*sini1x/r)

        gxt += (c[col+1]*(-sina*sini1x/r + pi*cosi1x*i1/L)
                + 0.5*pi*c[col+2]*cosi1x*i1*wt/(L*r))

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

    e[0] = exx
    e[1] = ett
    e[2] = gxt
    e[3] = kxx
    e[4] = ktt
    e[5] = kxt


cdef void *cfstrain_sanders(double *c, double sina, double cosa, double tLA,
                            double x, double t, double r, double r2, double L,
                            int m1, int m2, int n2, double *e) nogil:
    cdef int i1, i2, j2, col
    cdef double v, ux, vx, wx, ut, wt
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef double sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t

    cfuvw(c, m1, m2, n2, r2, L, x, t, cosa, tLA, e)
    v = e[1]

    cfuvw_x(c, m1, m2, n2, r2, L, x, t, cosa, tLA, e)
    ux = e[0]
    vx = e[1]
    wx = e[2]

    cfuvw_t(c, m1, m2, n2, L, x, t, cosa, tLA, e)
    ut = e[0]
    wt = e[2]

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

    for i1 in range(1, m1+1):
        sini1x = sin(pi*i1*x/L)
        cosi1x = cos(pi*i1*x/L)
        col = (i1-1)*num1 + num0

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

    for j2 in range(1, n2+1):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(1, m2+1):
            sini2x = sin(pi*i2*x/L)
            cosi2x = cos(pi*i2*x/L)
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1

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

