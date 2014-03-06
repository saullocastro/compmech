#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
from __future__ import division

cdef extern from "math.h":
    float cos(float t) nogil
    float sin(float t) nogil

cdef void scfN(float *c, float sina, float cosa,
               float x, float t, float r, float L, float *F,
               int m1, int m2, int n2, int pdoff, float c00, float *N,
               int NL_kinematics) nogil:
    # NL_kinematics = 0 donnell
    cdef float exx, ett, gxt, kxx, ktt, kxt
    cdef scfstraintype *cfstrain
    if NL_kinematics==0:
        cfstrain = &scfstrain_donnell
    cfstrain(c, sina, cosa, x, t, r, L, m1, m2, n2, pdoff, c00, N)
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

cdef void scfwx(float *c, int m1, int m2, int n2,
          float x, float t, float L, int pdoff, float *wx) nogil:
    cdef float pi=3.141592653589793
    cdef float expr
    cdef int i1, i2, j2, col
    wx[0] = 0.
    for i1 in range(1, m1+1):
        col = (i1-1)*3 + pdoff
        wx[0] += i1*pi/L*cos(i1*pi*x/L)*c[col+2]
    for i2 in range(1, m2+1):
        expr = i2*pi/L*cos(i2*pi*x/L)
        for j2 in range(1, n2+1):
            col = (i2-1)*6 + (j2-1)*6*m2 + pdoff + 3*m1
            wx[0] += expr*sin(j2*t)*c[col+4]
            wx[0] += expr*cos(j2*t)*c[col+5]

cdef void scfwt(float *c, int m1, int m2, int n2,
               float x, float t, float L, int pdoff, float *wt) nogil:
    cdef float pi=3.141592653589793
    cdef float sinbi
    cdef int i2, j2, col
    wt[0] = 0.
    for i2 in range(1, m2+1):
        sinbi = sin(i2*pi*x/L)
        for j2 in range(1, n2+1):
            col = (i2-1)*6 + (j2-1)*6*m2 + pdoff + 3*m1
            wt[0] += sinbi*(j2*cos(j2*t))*c[col+4]
            wt[0] += sinbi*(-j2*sin(j2*t))*c[col+5]

cdef void *scfstrain_donnell(float *c, float sina, float cosa,
        float x, float t, float r, float L,
        int m1, int m2, int n2, int pdoff, float c00, float *e) nogil:
    cdef float pi = 3.141592653589793
    cdef int k1, k2, l2, col
    cdef float wx, wt
    cdef float exx, ett, gxt, kxx, ktt, kxt
    #
    scfwx(c, m1, m2, n2, x, t, L, pdoff, &wx)
    scfwt(c, m1, m2, n2, x, t, L, pdoff, &wt)
    #
    exx = 0
    ett = 0
    gxt = 0
    kxx = 0
    ktt = 0
    kxt = 0
    #
    if pdoff==0:
        c00 = c00
    elif pdoff==1:
        c00 = c[0]
    exx = -cosa**(-1)*L**(-1)*c00
    ett = (r**(-1)*sina*cosa**(-1) - r**(-1)*sina*cosa**(-1)*x*L**(-1))*c00
    for k1 in range(1, m1+1):
        col = (k1-1)*3 + pdoff
        # from B0
        exx += c[col+0]*pi*cos(pi*k1*x*L**(-1))*k1*L**(-1)
        ett += c[col+0]*r**(-1)*sina*sin(pi*k1*x*L**(-1))
        ett += c[col+2]*r**(-1)*cosa*sin(pi*k1*x*L**(-1))
        gxt += c[col+1]*(-sina*r**(-1)*sin(pi*k1*x*L**(-1)) + pi*cos(pi*k1*x*L**(-1))*k1*L**(-1))
        kxx += c[col+2]*pi**2*sin(pi*k1*x*L**(-1))*k1*L**(-1)*k1*L**(-1)
        ktt += c[col+2]*(-1)*pi*r**(-1)*sina*cos(pi*k1*x*L**(-1))*k1*L**(-1)
        # from BNL
        exx += c[col+2]*pi*wx*cos(pi*k1*x*L**(-1))*k1*L**(-1)
        gxt += c[col+2]*pi*r**(-1)*wt*cos(pi*k1*x*L**(-1))*k1*L**(-1)

    for k2 in range(1, m2+1):
        for l2 in range(1, n2+1):
            col = (k2-1)*6 + (l2-1)*6*m2 + pdoff + 3*m1
            # from B0
            exx += c[col+0]*pi*cos(pi*k2*x*L**(-1))*k2*L**(-1)*sin(l2*t)
            exx += c[col+1]*pi*cos(pi*k2*x*L**(-1))*k2*L**(-1)*cos(l2*t)
            ett += c[col+0]*r**(-1)*sina*sin(pi*k2*x*L**(-1))*sin(l2*t)
            ett += c[col+1]*r**(-1)*sina*sin(pi*k2*x*L**(-1))*cos(l2*t)
            ett += c[col+2]*r**(-1)*sin(pi*k2*x*L**(-1))*cos(l2*t)*l2
            ett += c[col+3]*(-1)*r**(-1)*sin(pi*k2*x*L**(-1))*sin(l2*t)*l2
            ett += c[col+4]*r**(-1)*cosa*sin(pi*k2*x*L**(-1))*sin(l2*t)
            ett += c[col+5]*r**(-1)*cosa*sin(pi*k2*x*L**(-1))*cos(l2*t)
            gxt += c[col+0]*r**(-1)*sin(pi*k2*x*L**(-1))*cos(l2*t)*l2
            gxt += c[col+1]*(-1)*r**(-1)*sin(pi*k2*x*L**(-1))*sin(l2*t)*l2
            gxt += c[col+2]*(-sina*r**(-1)*sin(pi*k2*x*L**(-1))*sin(l2*t) + pi*cos(pi*k2*x*L**(-1))*k2*L**(-1)*sin(l2*t))
            gxt += c[col+3]*(-sina*r**(-1)*sin(pi*k2*x*L**(-1))*cos(l2*t) + pi*cos(pi*k2*x*L**(-1))*k2*L**(-1)*cos(l2*t))
            kxx += c[col+4]*pi**2*sin(pi*k2*x*L**(-1))*k2*L**(-1)*k2*L**(-1)*sin(l2*t)
            kxx += c[col+5]*pi**2*sin(pi*k2*x*L**(-1))*k2*L**(-1)*k2*L**(-1)*cos(l2*t)
            ktt += c[col+4]*(r**(-2)*sin(pi*k2*x*L**(-1))*sin(l2*t)*l2**2 - pi*r**(-1)*sina*cos(pi*k2*x*L**(-1))*k2*L**(-1)*sin(l2*t))
            ktt += c[col+5]*(r**(-2)*sin(pi*k2*x*L**(-1))*cos(l2*t)*l2**2 - pi*r**(-1)*sina*cos(pi*k2*x*L**(-1))*k2*L**(-1)*cos(l2*t))
            kxt += c[col+4]*(r**(-2)*sina*sin(pi*k2*x*L**(-1))*cos(l2*t)*l2 - 2*pi*r**(-1)*cos(pi*k2*x*L**(-1))*k2*L**(-1)*cos(l2*t)*l2)
            kxt += c[col+5]*(-r**(-2)*sina*sin(pi*k2*x*L**(-1))*sin(l2*t)*l2 + 2*pi*r**(-1)*cos(pi*k2*x*L**(-1))*k2*L**(-1)*sin(l2*t)*l2)
            # from BNL
            exx += c[col+4]*pi*wx*cos(pi*k2*x*L**(-1))*k2*L**(-1)*sin(l2*t)
            exx += c[col+5]*pi*wx*cos(pi*k2*x*L**(-1))*k2*L**(-1)*cos(l2*t)
            ett += c[col+4]*r**(-1)*wt*r**(-1)*sin(pi*k2*x*L**(-1))*cos(l2*t)*l2
            ett += c[col+5]*(-1)*r**(-1)*wt*r**(-1)*sin(pi*k2*x*L**(-1))*sin(l2*t)*l2
            gxt += c[col+4]*(pi*r**(-1)*wt*cos(pi*k2*x*L**(-1))*k2*L**(-1)*sin(l2*t) + wx*r**(-1)*sin(pi*k2*x*L**(-1))*cos(l2*t)*l2)
            gxt += c[col+5]*(pi*r**(-1)*wt*cos(pi*k2*x*L**(-1))*k2*L**(-1)*cos(l2*t) - wx*r**(-1)*sin(pi*k2*x*L**(-1))*sin(l2*t)*l2)

    e[0] = exx
    e[1] = ett
    e[2] = gxt
    e[3] = kxx
    e[4] = ktt
    e[5] = kxt

