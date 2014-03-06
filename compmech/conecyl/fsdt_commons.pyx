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
cdef int num1 = 7
cdef int num2 = 14
cdef double pi=3.141592653589793

def fstrain(np.ndarray[cDOUBLE, ndim=1] c, double sina, double cosa,
            np.ndarray[cDOUBLE, ndim=1] xvec,
            np.ndarray[cDOUBLE, ndim=1] tvec,
            double r2, double L, int m1, int m2, int n2, int NL_kinematics):
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef double x, t, r
    cdef int ie, ix
    cdef np.ndarray[cDOUBLE, ndim=1] evec
    cdef double e[8]
    cdef cfstraintype *cfstrain
    if NL_kinematics==0:
        cfstrain = &cfstrain_donnell
    elif NL_kinematics==1:
        cfstrain = &cfstrain_sanders
    evec = np.zeros((xvec.shape[0]*8), dtype=DOUBLE)
    for ix in range(xvec.shape[0]):
        x = xvec[ix]
        t = tvec[ix]
        r = r2 + x*sina
        cfstrain(&c[0], sina, cosa, x, t, r, r2, L, m1, m2, n2, e)
        for ie in range(8):
            evec[ix*8 + ie] = e[ie]
    return evec

cdef void cfN(double *c, double sina, double cosa,
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
    cfstrain(c, sina, cosa, x, t, r, r2, L, m1, m2, n2, N)
    #NOTE using array N to transfer values of strains
    exx = N[0]
    ett = N[1]
    gxt = N[2]
    kxx = N[3]
    ktt = N[4]
    kxt = N[5]
    exz = N[6]
    etz = N[7]
    N[0] = F[0]*exx + F[1]*ett + F[2]*gxt + F[3]*kxx + F[4]*ktt + F[5]*kxt
    N[1] = F[8]*exx + F[9]*ett + F[10]*gxt + F[11]*kxx + F[12]*ktt + F[13]*kxt
    N[2] = F[16]*exx + F[17]*ett + F[18]*gxt + F[19]*kxx + F[20]*ktt + F[21]*kxt
    N[3] = F[24]*exx + F[25]*ett + F[26]*gxt + F[27]*kxx + F[28]*ktt + F[29]*kxt
    N[4] = F[32]*exx + F[33]*ett + F[34]*gxt + F[35]*kxx + F[36]*ktt + F[37]*kxt
    N[5] = F[40]*exx + F[41]*ett + F[42]*gxt + F[43]*kxx + F[44]*ktt + F[45]*kxt
    N[6] = F[54]*exz + F[55]*etz
    N[7] = F[62]*exz + F[63]*etz


cdef void cfuvw(double *c, int m1, int m2, int n2, double r2,
                double L, double x, double t,
                double cosa, double tLA, double *uvw) nogil:
    cdef int i1, i2, j2, col
    cdef double sinbi, cosbi, sinbj, cosbj, u, v, w, phix, phit
    u = (c[0]*((L-x)/(cosa*L))
         -c[2]*(1/cosa)*(L - x)/L*(-(1 - cos(t - tLA))))
    v = c[1]*((L-x)*r2/L)
    w = 0
    phix = 0
    phit = 0
    for i1 in range(1, m1+1):
        sinbi = sin(i1*pi*x/L)
        cosbi = cos(i1*pi*x/L)
        col = (i1-1)*num1 + num0
        u += c[col+0]*sinbi
        u += c[col+1]*cosbi
        v += c[col+2]*sinbi
        w += c[col+3]*sinbi
        phix += c[col+4]*sinbi
        phix += c[col+5]*cosbi
        phit += c[col+6]*sinbi

    for j2 in range(1, n2+1):
        sinbj = sin(j2*t)
        cosbj = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbi = sin(i2*pi*x/L)
            cosbi = cos(i2*pi*x/L)
            u += c[col+0]*sinbi*sinbj
            u += c[col+1]*sinbi*cosbj
            u += c[col+2]*cosbi*sinbj
            u += c[col+3]*cosbi*cosbj
            v += c[col+4]*sinbi*sinbj
            v += c[col+5]*sinbi*cosbj
            w += c[col+6]*sinbi*sinbj
            w += c[col+7]*sinbi*cosbj
            phix += c[col+8]*sinbi*sinbj
            phix += c[col+9]*sinbi*cosbj
            phix += c[col+10]*cosbi*sinbj
            phix += c[col+11]*cosbi*cosbj
            phit += c[col+12]*sinbi*sinbj
            phit += c[col+13]*sinbi*cosbj

    uvw[0] = u
    uvw[1] = v
    uvw[2] = w
    uvw[3] = phix
    uvw[4] = phit


cdef void cfphix(double *c, int m1, int m2, int n2, double L,
                 double x, double t, double *refphix) nogil:
    cdef int i1, i2, j2, col
    cdef double sinbi, cosbi, sinbj, cosbj, phix
    phix = 0
    for i1 in range(1, m1+1):
        sinbi = sin(i1*pi*x/L)
        cosbi = cos(i1*pi*x/L)
        col = (i1-1)*num1 + num0
        phix += c[col+4]*sinbi
        phix += c[col+5]*cosbi

    for j2 in range(1, n2+1):
        sinbj = sin(j2*t)
        cosbj = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbi = sin(i2*pi*x/L)
            cosbi = cos(i2*pi*x/L)
            phix += c[col+8]*sinbi*sinbj
            phix += c[col+9]*sinbi*cosbj
            phix += c[col+10]*cosbi*sinbj
            phix += c[col+11]*cosbi*cosbj

    refphix[0] = phix


cdef void cfphit(double *c, int m1, int m2, int n2, double L,
                 double x, double t, double *refphit) nogil:
    cdef int i1, i2, j2, col
    cdef double sinbi, cosbi, sinbj, cosbj, phit
    phit = 0
    for i1 in range(1, m1+1):
        sinbi = sin(i1*pi*x/L)
        cosbi = cos(i1*pi*x/L)
        col = (i1-1)*num1 + num0
        phit += c[col+6]*sinbi
        phit += c[col+7]*cosbi

    for j2 in range(1, n2+1):
        sinbj = sin(j2*t)
        cosbj = cos(j2*t)
        for i2 in range(1, m2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbi = sin(i2*pi*x/L)
            cosbi = cos(i2*pi*x/L)
            phit += c[col+12]*sinbi*sinbj
            phit += c[col+13]*sinbi*cosbj

    refphit[0] = phit

def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m1, int m2, int n2,
         double alpharad, double r2, double L, double tLA,
         np.ndarray[cDOUBLE, ndim=1] xvec,
         np.ndarray[cDOUBLE, ndim=1] tvec, ss):
    cdef int ix
    cdef double x, t, sina, cosa
    cdef double uvw[5]
    cdef np.ndarray[cDOUBLE, ndim=1] u, v, w, phix, phit

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
        phix[ix] = uvw[3]
        phit[ix] = uvw[4]

    return u, v, w, phix, phit

def fg(double[:,::1] gss, int m1, int m2, int n2,
       double r2, double x, double t, double L, double cosa, double tLA):
    cfg(gss, m1, m2, n2, r2, x, t, L, cosa, tLA)

cdef cfg(double[:, ::1] gss, int m1, int m2, int n2,
         double r2, double x, double t, double L, double cosa, double tLA):
    cdef double sinbi, cosbi, sinbj, cosbj
    cdef int i1, i2, j2, col
    gss[0,0] = (L-x)/(L*cosa)
    gss[1,1] = (L-x)*r2/L
    gss[0,2] = (L - x)/(L*cosa)*(1 - cos(t - tLA))

    for i1 in range(1, m1+1):
        sinbi = sin(i1*pi*x/L)
        cosbi = cos(i1*pi*x/L)
        col = (i1-1)*num1 + num0
        gss[0, col+0] = sinbi
        gss[0, col+1] = cosbi
        gss[1, col+2] = sinbi
        gss[2, col+3] = sinbi
        gss[3, col+4] = sinbi
        gss[3, col+5] = cosbi
        gss[4, col+6] = sinbi

    for i2 in range(1, m2+1):
        sinbi = sin(i2*pi*x/L)
        cosbi = cos(i2*pi*x/L)
        for j2 in range(1, n2+1):
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            sinbj = sin(j2*t)
            cosbj = cos(j2*t)
            gss[0, col+0] = sinbi*sinbj
            gss[0, col+1] = sinbi*cosbj
            gss[0, col+2] = cosbi*sinbj
            gss[0, col+3] = cosbi*cosbj

            gss[1, col+4] = sinbi*sinbj
            gss[1, col+5] = sinbi*cosbj

            gss[2, col+6] = sinbi*sinbj
            gss[2, col+7] = sinbi*cosbj

            gss[3, col+8] = sinbi*sinbj
            gss[3, col+9] = sinbi*cosbj
            gss[3, col+10] = cosbi*sinbj
            gss[3, col+11] = cosbi*cosbj

            gss[4, col+12] = sinbi*sinbj
            gss[4, col+13] = sinbi*cosbj


cdef void *cfstrain_donnell(double *c, double sina, double cosa,
        double x, double t, double r, double r2, double L,
        int m1, int m2, int n2, double *e) nogil:
    cdef int i1, i2, j2, col
    cdef double phix, phit
    cdef double exx, ett, gxt, kxx, ktt, kxt, exz, etz
    cdef double sini1x, cosi1x, sini2x, cosi2x, sinj2t, cosj2t

    cfphix(c, m1, m2, n2, L, x, t, &phix)
    cfphit(c, m1, m2, n2, L, x, t, &phit)

    exx = 0
    ett = 0
    gxt = 0
    kxx = 0
    ktt = 0
    kxt = 0
    exz = 0
    etz = 0

    exx =
    ett =
    gxt =

    for i1 in range(1, m1+1):
        sini1x = sin(pi*i1*x/L)
        cosi1x = cos(pi*i1*x/L)
        col = (i1-1)*num1 + num0

        exx += (

                )

        ett += (

                )

        gxt += (

                )

        kxx += (

                )

        ktt += (

                )

    for j2 in range(1, n2+1):
        sinj2t = sin(j2*t)
        cosj2t = cos(j2*t)
        for i2 in range(1, m2+1):
            sini2x = sin(pi*i2*x/L)
            cosi2x = cos(pi*i2*x/L)
            col = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1

            exx += (

                    )

            ett += (

                    )

            gxt += (

                    )

            kxx += (

                    )

            ktt += (

                    )

            kxt += (

                    )

    e[0] = exx
    e[1] = ett
    e[2] = gxt
    e[3] = kxx
    e[4] = ktt
    e[5] = kxt
    e[6] = exz
    e[7] = etz
