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
    double cos(double theta) nogil
    double sin(double theta) nogil


cdef int num0 = 2
cdef int num1 = 4
cdef int e_num = 6
cdef double pi=3.141592653589793


def fstrain(np.ndarray[cDOUBLE, ndim=1] c,
            np.ndarray[cDOUBLE, ndim=1] xs,
            np.ndarray[cDOUBLE, ndim=1] ys,
            double a, double b, int m1, int n1,
            np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0, int funcnum,
            int NL_kinematics, int num_cores=4):
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int size_core, i
    cdef np.ndarray[cDOUBLE, ndim=2] es
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ys_core
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
        ys_core = np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ys_core = ys.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    es = np.zeros((num_cores, size_core*e_num), dtype=DOUBLE)
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfstrain(&c[0], &xs_core[i,0], &ys_core[i,0],
                size_core, a, b, m1, n1, &c0[0], m0, n0, funcnum,
                &es[i,0])
    return es.ravel()[:size*e_num]


def fstress(np.ndarray[cDOUBLE, ndim=1] c,
       np.ndarray[cDOUBLE, ndim=2] F,
       np.ndarray[cDOUBLE, ndim=1] xs,
       np.ndarray[cDOUBLE, ndim=1] ys,
       double a, double b, int m1, int n1,
       np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0, int funcnum,
       int NL_kinematics, int num_cores=4):
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int size_core, i
    cdef np.ndarray[cDOUBLE, ndim=2] Ns
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ys_core = np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ys_core = ys.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    Ns = np.zeros((num_cores, size_core*e_num), dtype=DOUBLE)
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfN(&c[0], &xs_core[i,0], &ys_core[i,0], size_core, a,
                b, &F[0,0], m1, n1, &c0[0], m0, n0, funcnum,
                &Ns[i,0], NL_kinematics)
    return Ns.ravel()[:size*e_num]


def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m1, int n1, double a, double b,
        np.ndarray[cDOUBLE, ndim=1] xs, np.ndarray[cDOUBLE, ndim=1] ys,
        int num_cores=4):
    cdef int size_core, i
    cdef np.ndarray[cDOUBLE, ndim=2] us, vs, ws, phixs, phiys
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ys_core = np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ys_core = ys.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phiys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m1, n1, a, b, &xs_core[i,0],
              &ys_core[i,0], size_core, &us[i,0], &vs[i,0], &ws[i,0])

        cfwx(&c[0], m1, n1, &xs_core[i,0], &ys_core[i,0],
             size_core, a, b, &phixs[i,0])

        cfwy(&c[0], m1, n1, &xs_core[i,0], &ys_core[i,0],
             size_core, a, b, &phiys[i,0])

    phixs *= -1.
    phiys *= -1.
    return (us.ravel()[:size], vs.ravel()[:size], ws.ravel()[:size],
            phixs.ravel()[:size], phiys.ravel()[:size])


cdef void cfuvw(double *c, int m1, int n1, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws) nogil:
    cdef int i1, j1, col, i
    cdef double sini1bx, sinj1by
    cdef double cosi1bx, cosj1by
    cdef double x, y, u, v, w, bx, by

    for i in range(size):
        x = xs[i]
        y = ys[i]

        bx = (x + a/2.)/a
        by = (y + b/2.)/b

        u = bx*c[0]
        v = by*c[1]
        w = 0

        for j1 in range(1, n1+1):
            sinj1by = sin(j1*pi*by)
            cosj1by = cos(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num0 + num1*((j1-1)*m1 + (i1-1))
                sini1bx = sin(i1*pi*bx)
                cosi1bx = cos(i1*pi*bx)
                u += c[col+0]*cosi1bx*cosj1by
                v += c[col+1]*cosi1bx*cosj1by
                w += c[col+2]*sini1bx*sinj1by
                w += c[col+3]*cosi1bx*cosj1by

        us[i] = u
        vs[i] = v
        ws[i] = w


cdef void cfwx(double *c, int m1, int n1, double *xs, double *ys, int size,
        double a, double b, double *outwx) nogil:
    cdef double dcosi1bx, dsini1bx, cosj1by, sinj1by, wx, x, y, bx, by
    cdef int i1, j1, col, i

    for i in range(size):
        x = xs[i]
        y = ys[i]
        bx = (x + a/2.)/a
        by = (y + b/2.)/b

        wx = 0.

        for j1 in range(1, n1+1):
            cosj1by = cos(j1*pi*by)
            sinj1by = sin(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num0 + num1*((j1-1)*m1 + (i1-1))
                dcosi1bx = -i1*pi/a*sin(i1*pi*bx)
                dsini1bx = i1*pi/a*cos(i1*pi*bx)
                wx += c[col+2]*dsini1bx*sinj1by
                wx += c[col+3]*dcosi1bx*cosj1by

        outwx[i] = wx


cdef void cfwy(double *c, int m1, int n1, double *xs, double *ys, int size,
        double a, double b, double *outwt) nogil:
    cdef double cosi1bx, sini1bx, dcosj1by, dsinj1by, wy, x, y, bx, by
    cdef int i1, j1, col, i

    for i in range(size):
        x = xs[i]
        y = ys[i]
        bx = (x + a/2.)/a
        by = (y + b/2.)/b

        wy = 0.

        for j1 in range(1, n1+1):
            dcosj1by = -j1*pi/b*sin(j1*pi*by)
            dsinj1by = j1*pi/b*cos(j1*pi*by)
            for i1 in range(1, m1+1):
                col = num0 + num1*((j1-1)*m1 + (i1-1))
                cosi1bx = cos(i1*pi*bx)
                sini1bx = sin(i1*pi*bx)
                wy += c[col+2]*sini1bx*dsinj1by
                wy += c[col+3]*cosi1bx*dcosj1by

        outwt[i] = wy


def fg(double[:,::1] g, int m1, int n1,
       double x, double y, double a, double b):
    cfg(g, m1, n1, x, y, a, b)


cdef void cfg(double[:,::1] g, int m1, int n1,
              double x, double y, double a, double b) nogil:
    cdef int i1, j1, col, i
    cdef double sini1bx, sinj1by
    cdef double cosi1bx, cosj1by
    cdef double bx, by

    bx = (x + a/2.)/a
    by = (y + b/2.)/b

    g[0, 0] = bx
    g[1, 1] = by

    for j1 in range(1, n1+1):
        sinj1by = sin(j1*pi*by)
        cosj1by = cos(j1*pi*by)
        for i1 in range(1, m1+1):
            col = num0 + num1*((j1-1)*m1 + (i1-1))
            sini1bx = sin(i1*pi*bx)
            cosi1bx = cos(i1*pi*bx)
            g[0, col+0] = cosi1bx*cosj1by
            g[1, col+1] = cosi1bx*cosj1by
            g[2, col+2] = sini1bx*sinj1by
            g[2, col+3] = cosi1bx*cosj1by


cdef void cfN(double *c, double *xs, double *ys, int size, double a, double b,
        double *F, int m1, int n1, double *c0, int m0, int n0, int funcnum,
        double *Ns, int NL_kinematics) nogil:
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int i
    cdef double exx, eyy, gxy, kxx, kyy, kxy
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

    cfstrain(c, xs, ys, size, a, b, m1, n1,
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
        eyy = es[e_num*i + 1]
        gxy = es[e_num*i + 2]
        kxx = es[e_num*i + 3]
        kyy = es[e_num*i + 4]
        kxy = es[e_num*i + 5]
        Ns[e_num*i + 0] = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
        Ns[e_num*i + 1] = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
        Ns[e_num*i + 2] = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy
        Ns[e_num*i + 3] = B11*exx + B12*eyy + B16*gxy + D11*kxx + D12*kyy + D16*kxy
        Ns[e_num*i + 4] = B12*exx + B22*eyy + B26*gxy + D12*kxx + D22*kyy + D26*kxy
        Ns[e_num*i + 5] = B16*exx + B26*eyy + B66*gxy + D16*kxx + D26*kyy + D66*kxy
    free(es)


cdef void *cfstrain_donnell(double *c, double *xs, double *ys, int size,
        double a, double b, int m1, int n1, double *c0, int m0, int n0,
        int funcnum, double *es) nogil:
    cdef int i, i1, j1, col
    cdef double wx, wy, x, y
    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double sini1bx, cosi1bx, sinj1by

    cdef double *wxs = <double *>malloc(size * sizeof(double))
    cdef double *wys = <double *>malloc(size * sizeof(double))

    cfwx(c, m1, n1, xs, ys, size, a, b, wxs)
    cfwy(c, m1, n1, xs, ys, size, a, b, wys)

    for i in range(size):
        x = xs[i]
        y = ys[i]
        wx = wxs[i]
        wy = wys[i]

        exx = 0
        eyy = 0
        gxy = 0
        kxx = 0
        kyy = 0
        kxy = 0

        for j1 in range(1, n1+1):
            for i1 in range(1, m1+1):
                col = num0 + num1*((j1-1)*m1 + (i1-1))

                exx += (0)

                eyy += (0)

                gxy += (0)

                kxx += 0

                kyy += 0


                kxy += 0


        es[e_num*i + 0] = exx
        es[e_num*i + 1] = eyy
        es[e_num*i + 2] = gxy
        es[e_num*i + 3] = kxx
        es[e_num*i + 4] = kyy
        es[e_num*i + 5] = kxy

    free(wxs)
    free(wys)

