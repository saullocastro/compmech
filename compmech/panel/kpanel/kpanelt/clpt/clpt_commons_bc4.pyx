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
cdef int num1 = 3
cdef int e_num = 6
cdef double pi=3.141592653589793


def fstrain(np.ndarray[cDOUBLE, ndim=1] c,
            double sina, double cosa,
            np.ndarray[cDOUBLE, ndim=1] xs,
            np.ndarray[cDOUBLE, ndim=1] ts,
            double r1, double L, double tmin, double tmax, int m1, int n1,
            np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0, int funcnum,
            int NL_kinematics, int num_cores=4):
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int size_core, i
    cdef np.ndarray[cDOUBLE, ndim=2] es
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ts_core
    cdef cfstraintype *cfstrain

    if NL_kinematics==0:
        cfstrain = &cfstrain_donnell
    else:
        raise NotImplementedError('only NL_kinematics=0 is implemented')
    #elif NL_kinematics==1:
        #cfstrain = &cfstrain_sanders

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ts_core = np.hstack((ts, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ts_core = ts.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    es = np.zeros((num_cores, size_core*e_num), dtype=DOUBLE)
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfstrain(&c[0], sina, cosa, &xs_core[i,0], &ts_core[i,0],
                size_core, r1, L, tmin, tmax, m1, n1, &c0[0], m0, n0, funcnum,
                &es[i,0])
    return es.ravel()[:size*e_num]


def fstress(np.ndarray[cDOUBLE, ndim=1] c,
       np.ndarray[cDOUBLE, ndim=2] F,
       double sina, double cosa,
       np.ndarray[cDOUBLE, ndim=1] xs,
       np.ndarray[cDOUBLE, ndim=1] ts,
       double r1, double L, double tmin, double tmax, int m1, int n1,
       np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0, int funcnum,
       int NL_kinematics, int num_cores=4):
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int size_core, i
    cdef np.ndarray[cDOUBLE, ndim=2] Ns
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ts_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ts_core = np.hstack((ts, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ts_core = ts.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    Ns = np.zeros((num_cores, size_core*e_num), dtype=DOUBLE)
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfN(&c[0], sina, cosa, &xs_core[i,0], &ts_core[i,0], size_core, r1, L,
                tmin, tmax, &F[0,0], m1, n1, &c0[0], m0, n0, funcnum,
                &Ns[i,0], NL_kinematics)
    return Ns.ravel()[:size*e_num]


def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m1, int n1, double L, double tmin,
        double tmax,
        np.ndarray[cDOUBLE, ndim=1] xs, np.ndarray[cDOUBLE, ndim=1] ts,
        double r1, double alpharad, int num_cores=4):
    cdef int size_core, i
    cdef double sina
    cdef np.ndarray[cDOUBLE, ndim=2] us, vs, ws, phixs, phits
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ts_core

    sina = sin(alpharad)

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ts_core = np.hstack((ts, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ts_core = ts.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phits = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m1, n1, L, tmin, tmax, &xs_core[i,0],
                &ts_core[i,0], size_core, &us[i,0], &vs[i,0], &ws[i,0])

        cfwx(&c[0], m1, n1, &xs_core[i,0], &ts_core[i,0],
             size_core, L, tmin, tmax, &phixs[i,0])

        cfwt(&c[0], m1, n1, &xs_core[i,0], &ts_core[i,0],
             size_core, L, tmin, tmax, &phits[i,0])

    phixs *= -1.
    r = r1 - sina*(xs_core + L/2.)
    phits *= -1./r
    return (us.ravel()[:size], vs.ravel()[:size], ws.ravel()[:size],
            phixs.ravel()[:size], phits.ravel()[:size])


cdef void cfuvw(double *c, int m1, int n1, double L, double tmin, double tmax,
        double *xs, double *ts, int size,
        double *us, double *vs, double *ws) nogil:
    cdef int i1, j1, col, i
    cdef double sini1bx, sinj1bt
    cdef double cosi1bx, cosj1bt
    cdef double x, t, u, v, w, bx, bt

    for i in range(size):
        x = xs[i]
        t = ts[i]
        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        u = 0
        v = 0
        w = 0

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

        us[i] = u
        vs[i] = v
        ws[i] = w


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


cdef void cfN(double *c, double sina, double cosa, double *xs, double *ts, int
        size, double r1, double L, double tmin, double tmax, double *F, int
        m1, int n1, double *c0, int m0, int n0, int funcnum, double *Ns, int
        NL_kinematics) nogil:
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int i
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef double *es = <double *>malloc(size*e_num * sizeof(double))
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef cfstraintype *cfstrain

    if NL_kinematics==0:
        cfstrain = &cfstrain_donnell
    else:
        with gil:
            raise NotImplementedError('only NL_kinematics=0 is implemented')
    #elif NL_kinematics==1:
        #cfstrain = &cfstrain_sanders

    cfstrain(c, sina, cosa, xs, ts, size, r1, L, tmin, tmax, m1, n1,
             c0, m0, n0, funcnum, es)

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

    for i in range(size):
        exx = es[e_num*i + 0]
        ett = es[e_num*i + 1]
        gxt = es[e_num*i + 2]
        kxx = es[e_num*i + 3]
        ktt = es[e_num*i + 4]
        kxt = es[e_num*i + 5]
        Ns[e_num*i + 0] = A11*exx + A12*ett + A16*gxt + B11*kxx + B12*ktt + B16*kxt
        Ns[e_num*i + 1] = A12*exx + A22*ett + A26*gxt + B12*kxx + B22*ktt + B26*kxt
        Ns[e_num*i + 2] = A16*exx + A26*ett + A66*gxt + B16*kxx + B26*ktt + B66*kxt
        Ns[e_num*i + 3] = B11*exx + B12*ett + B16*gxt + D11*kxx + D12*ktt + D16*kxt
        Ns[e_num*i + 4] = B12*exx + B22*ett + B26*gxt + D12*kxx + D22*ktt + D26*kxt
        Ns[e_num*i + 5] = B16*exx + B26*ett + B66*gxt + D16*kxx + D26*ktt + D66*kxt
    free(es)


cdef void *cfstrain_donnell(double *c, double sina, double cosa,
                            double *xs, double *ts, int size,
                            double r1, double L, double tmin, double tmax,
                            int m1, int n1,
                            double *c0, int m0, int n0, int funcnum,
                            double *es) nogil:
    cdef int i, i1, j1, col
    cdef double wx, wt, w0x, w0t, x, t, r
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef double sini1bx, cosi1bx, sinj1bt, cosj1_bt

    cdef double *wxs = <double *>malloc(size * sizeof(double))
    cdef double *wts = <double *>malloc(size * sizeof(double))
    #cdef double *w0xs = <double *>malloc(size * sizeof(double))
    #cdef double *w0ts = <double *>malloc(size * sizeof(double))

    cfwx(c, m1, n1, xs, ts, size, L, tmin, tmax, wxs)
    cfwt(c, m1, n1, xs, ts, size, L, tmin, tmax, wts)

    w0x = 0
    w0t = 0

    for i in range(size):
        x = xs[i]
        t = ts[i]
        wx = wxs[i]
        wt = wts[i]
        #w0x = w0xs[i]
        #w0t = w0ts[i]
        r = r1 - sina*(x + L/2.)

        exx = 0
        ett = 0
        gxt = 0
        kxx = 0
        ktt = 0
        kxt = 0

        for j1 in range(n1):
            sinj1bt = sin(pi*j1*(t - tmin)/(tmax - tmin))
            cosj1_bt =  cos(pi*j1*(t - tmin)/(-tmax + tmin))
            for i1 in range(m1):
                sini1bx = sin(pi*i1*(0.5*L + x)/L)
                cosi1bx = cos(pi*i1*(0.5*L + x)/L)

                col = num0 + num1*((j1)*m1 + (i1))

                exx += (-pi*c[col+0]*cosj1_bt*i1*sini1bx/L
                        + 0.5*pi*c[col+2]*cosi1bx*i1*sinj1bt*(2*w0x + wx)/L)

                ett += (c[col+0]*cosi1bx*cosj1_bt*sina/r
                        -pi*c[col+1]*cosi1bx*j1*sinj1bt/(r*tmax - r*tmin)
                        + 0.5*c[col+2]*sini1bx*(2*cosa*r*sinj1bt*(tmax - tmin) + pi*cosj1_bt*j1*(2*w0t + wt))/(r**2*(tmax - tmin)))

                gxt += (-pi*c[col+0]*cosi1bx*j1*sinj1bt/(r*tmax - r*tmin)
                        -c[col+1]*cosj1_bt*(L*cosi1bx*sina + pi*i1*r*sini1bx)/(L*r)
                        +0.5*pi*c[col+2]*(L*cosj1_bt*j1*sini1bx*(2*w0x + wx) + cosi1bx*i1*sinj1bt*(tmax - tmin)*(2*w0t + wt))/(L*r*(tmax - tmin)))

                kxx += pi**2*c[col+2]*i1**2*sini1bx*sinj1bt/L**2

                ktt += pi*c[col+2]*sinj1bt*(pi*L*j1**2*sini1bx - cosi1bx*i1*r*sina*(tmax - tmin)**2)/(L*r**2*(tmax - tmin)**2)

                kxt += pi*c[col+2]*cosj1_bt*j1*(L*sina*sini1bx - 2*pi*cosi1bx*i1*r)/(L*r**2*(tmax - tmin))

        es[e_num*i + 0] = exx
        es[e_num*i + 1] = ett
        es[e_num*i + 2] = gxt
        es[e_num*i + 3] = kxx
        es[e_num*i + 4] = ktt
        es[e_num*i + 5] = kxt

    free(wxs)
    free(wts)
    #free(w0xs)
    #free(w0ts)

