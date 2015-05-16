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

include 'clpt_commons_include_fuvw.pyx'
#include 'clpt_commons_include_fstrain.pyx'
#include 'clpt_commons_include_cfN.pyx'

cdef void cfuvw(double *c, int m1, int n1, double L, double tmin, double tmax,
        double *xs, double *ts, int size,
        double *us, double *vs, double *ws) nogil:
    cdef int i1, j1, col, i
    cdef double x, t, u, v, w, xi, eta
    cdef double f[64]

    for i in range(size):
        x = xs[i]
        t = ts[i]

        xi = x/(L/2.)
        eta = 2.*(t - tmin)/(tmax - tmin) - 1.

        f[0] = 1
        f[1] = 2*xi
        f[2] = -1 + 4*xi**2
        f[3] = -4*xi + 8*xi**3
        f[4] = 1 - 12*xi**2 + 16*xi**4
        f[5] = 6*xi - 32*xi**3 + 32*xi**5
        f[6] = -1 + 24*xi**2 - 80*xi**4 + 64*xi**6
        f[7] = -8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7
        f[8] = 2*eta
        f[9] = 4*eta*xi
        f[10] = 2*eta*(-1 + 4*xi**2)
        f[11] = 2*eta*(-4*xi + 8*xi**3)
        f[12] = 2*eta*(1 - 12*xi**2 + 16*xi**4)
        f[13] = 2*eta*(6*xi - 32*xi**3 + 32*xi**5)
        f[14] = 2*eta*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[15] = 2*eta*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[16] = -1 + 4*eta**2
        f[17] = 2*(-1 + 4*eta**2)*xi
        f[18] = (-1 + 4*eta**2)*(-1 + 4*xi**2)
        f[19] = (-1 + 4*eta**2)*(-4*xi + 8*xi**3)
        f[20] = (-1 + 4*eta**2)*(1 - 12*xi**2 + 16*xi**4)
        f[21] = (-1 + 4*eta**2)*(6*xi - 32*xi**3 + 32*xi**5)
        f[22] = (-1 + 4*eta**2)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[23] = (-1 + 4*eta**2)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[24] = -4*eta + 8*eta**3
        f[25] = 2*(-4*eta + 8*eta**3)*xi
        f[26] = (-4*eta + 8*eta**3)*(-1 + 4*xi**2)
        f[27] = (-4*eta + 8*eta**3)*(-4*xi + 8*xi**3)
        f[28] = (-4*eta + 8*eta**3)*(1 - 12*xi**2 + 16*xi**4)
        f[29] = (-4*eta + 8*eta**3)*(6*xi - 32*xi**3 + 32*xi**5)
        f[30] = (-4*eta + 8*eta**3)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[31] = (-4*eta + 8*eta**3)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[32] = 1 - 12*eta**2 + 16*eta**4
        f[33] = 2*(1 - 12*eta**2 + 16*eta**4)*xi
        f[34] = (1 - 12*eta**2 + 16*eta**4)*(-1 + 4*xi**2)
        f[35] = (1 - 12*eta**2 + 16*eta**4)*(-4*xi + 8*xi**3)
        f[36] = (1 - 12*eta**2 + 16*eta**4)*(1 - 12*xi**2 + 16*xi**4)
        f[37] = (1 - 12*eta**2 + 16*eta**4)*(6*xi - 32*xi**3 + 32*xi**5)
        f[38] = (1 - 12*eta**2 + 16*eta**4)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[39] = (1 - 12*eta**2 + 16*eta**4)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[40] = 6*eta - 32*eta**3 + 32*eta**5
        f[41] = 2*(6*eta - 32*eta**3 + 32*eta**5)*xi
        f[42] = (6*eta - 32*eta**3 + 32*eta**5)*(-1 + 4*xi**2)
        f[43] = (6*eta - 32*eta**3 + 32*eta**5)*(-4*xi + 8*xi**3)
        f[44] = (6*eta - 32*eta**3 + 32*eta**5)*(1 - 12*xi**2 + 16*xi**4)
        f[45] = (6*eta - 32*eta**3 + 32*eta**5)*(6*xi - 32*xi**3 + 32*xi**5)
        f[46] = (6*eta - 32*eta**3 + 32*eta**5)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[47] = (6*eta - 32*eta**3 + 32*eta**5)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[48] = -1 + 24*eta**2 - 80*eta**4 + 64*eta**6
        f[49] = 2*(-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*xi
        f[50] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-1 + 4*xi**2)
        f[51] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-4*xi + 8*xi**3)
        f[52] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(1 - 12*xi**2 + 16*xi**4)
        f[53] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(6*xi - 32*xi**3 + 32*xi**5)
        f[54] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[55] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[56] = -8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7
        f[57] = 2*(-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*xi
        f[58] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-1 + 4*xi**2)
        f[59] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-4*xi + 8*xi**3)
        f[60] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(1 - 12*xi**2 + 16*xi**4)
        f[61] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(6*xi - 32*xi**3 + 32*xi**5)
        f[62] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[63] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)

        u = 0
        v = 0
        w = 0

        for j1 in range(n1):
            for i1 in range(m1):
                col = num0 + num1*((j1)*m1 + (i1))
                u += c[col+0]*f[i1 + j1*m1]
                v += c[col+1]*f[i1 + j1*m1]
                w += c[col+2]*f[i1 + j1*m1]

        us[i] = u
        vs[i] = v
        ws[i] = w


cdef void cfwx(double *c, int m1, int n1, double *xs, double *ts, int size,
        double L, double tmin, double tmax, double *outwx) nogil:
    cdef int i1, j1, col, i, j
    cdef double wx, x, t, xi, eta
    cdef double f[64]

    for i in range(size):
        x = xs[i]
        t = ts[i]

        xi = x/(L/2.)
        eta = 2.*(t - tmin)/(tmax - tmin) - 1.

        f[0] = 0
        f[1] = 2
        f[2] = 8*xi
        f[3] = -4 + 24*xi**2
        f[4] = -24*xi + 64*xi**3
        f[5] = 6 - 96*xi**2 + 160*xi**4
        f[6] = 48*xi - 320*xi**3 + 384*xi**5
        f[7] = -8 + 240*xi**2 - 960*xi**4 + 896*xi**6
        f[8] = 0
        f[9] = 4*eta
        f[10] = 16*eta*xi
        f[11] = 2*eta*(-4 + 24*xi**2)
        f[12] = 2*eta*(-24*xi + 64*xi**3)
        f[13] = 2*eta*(6 - 96*xi**2 + 160*xi**4)
        f[14] = 2*eta*(48*xi - 320*xi**3 + 384*xi**5)
        f[15] = 2*eta*(-8 + 240*xi**2 - 960*xi**4 + 896*xi**6)
        f[16] = 0
        f[17] = 2*(-1 + 4*eta**2)
        f[18] = 8*(-1 + 4*eta**2)*xi
        f[19] = (-1 + 4*eta**2)*(-4 + 24*xi**2)
        f[20] = (-1 + 4*eta**2)*(-24*xi + 64*xi**3)
        f[21] = (-1 + 4*eta**2)*(6 - 96*xi**2 + 160*xi**4)
        f[22] = (-1 + 4*eta**2)*(48*xi - 320*xi**3 + 384*xi**5)
        f[23] = (-1 + 4*eta**2)*(-8 + 240*xi**2 - 960*xi**4 + 896*xi**6)
        f[24] = 0
        f[25] = 2*(-4*eta + 8*eta**3)
        f[26] = 8*(-4*eta + 8*eta**3)*xi
        f[27] = (-4*eta + 8*eta**3)*(-4 + 24*xi**2)
        f[28] = (-4*eta + 8*eta**3)*(-24*xi + 64*xi**3)
        f[29] = (-4*eta + 8*eta**3)*(6 - 96*xi**2 + 160*xi**4)
        f[30] = (-4*eta + 8*eta**3)*(48*xi - 320*xi**3 + 384*xi**5)
        f[31] = (-4*eta + 8*eta**3)*(-8 + 240*xi**2 - 960*xi**4 + 896*xi**6)
        f[32] = 0
        f[33] = 2*(1 - 12*eta**2 + 16*eta**4)
        f[34] = 8*(1 - 12*eta**2 + 16*eta**4)*xi
        f[35] = (1 - 12*eta**2 + 16*eta**4)*(-4 + 24*xi**2)
        f[36] = (1 - 12*eta**2 + 16*eta**4)*(-24*xi + 64*xi**3)
        f[37] = (1 - 12*eta**2 + 16*eta**4)*(6 - 96*xi**2 + 160*xi**4)
        f[38] = (1 - 12*eta**2 + 16*eta**4)*(48*xi - 320*xi**3 + 384*xi**5)
        f[39] = (1 - 12*eta**2 + 16*eta**4)*(-8 + 240*xi**2 - 960*xi**4 + 896*xi**6)
        f[40] = 0
        f[41] = 2*(6*eta - 32*eta**3 + 32*eta**5)
        f[42] = 8*(6*eta - 32*eta**3 + 32*eta**5)*xi
        f[43] = (6*eta - 32*eta**3 + 32*eta**5)*(-4 + 24*xi**2)
        f[44] = (6*eta - 32*eta**3 + 32*eta**5)*(-24*xi + 64*xi**3)
        f[45] = (6*eta - 32*eta**3 + 32*eta**5)*(6 - 96*xi**2 + 160*xi**4)
        f[46] = (6*eta - 32*eta**3 + 32*eta**5)*(48*xi - 320*xi**3 + 384*xi**5)
        f[47] = (6*eta - 32*eta**3 + 32*eta**5)*(-8 + 240*xi**2 - 960*xi**4 + 896*xi**6)
        f[48] = 0
        f[49] = 2*(-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)
        f[50] = 8*(-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*xi
        f[51] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-4 + 24*xi**2)
        f[52] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-24*xi + 64*xi**3)
        f[53] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(6 - 96*xi**2 + 160*xi**4)
        f[54] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(48*xi - 320*xi**3 + 384*xi**5)
        f[55] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-8 + 240*xi**2 - 960*xi**4 + 896*xi**6)
        f[56] = 0
        f[57] = 2*(-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)
        f[58] = 8*(-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*xi
        f[59] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-4 + 24*xi**2)
        f[60] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-24*xi + 64*xi**3)
        f[61] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(6 - 96*xi**2 + 160*xi**4)
        f[62] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(48*xi - 320*xi**3 + 384*xi**5)
        f[63] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-8 + 240*xi**2 - 960*xi**4 + 896*xi**6)

        wx = 0.
        for j1 in range(n1):
            for i1 in range(m1):
                col = num0 + num1*((j1)*m1 + (i1))
                wx += c[col+2]*f[i1 + j1*m1]

        outwx[i] = wx


cdef void cfwt(double *c, int m1, int n1, double *xs, double *ts, int size,
        double L, double tmin, double tmax, double *outwt) nogil:
    cdef double wt, x, t, xi, eta
    cdef int i1, j1, col, i
    cdef double f[64]

    for i in range(size):
        x = xs[i]
        t = ts[i]

        xi = x/(L/2.)
        eta = 2.*(t - tmin)/(tmax - tmin) - 1.

        f[0] = 0
        f[1] = 0
        f[2] = 0
        f[3] = 0
        f[4] = 0
        f[5] = 0
        f[6] = 0
        f[7] = 0
        f[8] = 2
        f[9] = 4*xi
        f[10] = 2*(-1 + 4*xi**2)
        f[11] = 2*(-4*xi + 8*xi**3)
        f[12] = 2*(1 - 12*xi**2 + 16*xi**4)
        f[13] = 2*(6*xi - 32*xi**3 + 32*xi**5)
        f[14] = 2*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[15] = 2*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[16] = 8*eta
        f[17] = 16*eta*xi
        f[18] = 8*eta*(-1 + 4*xi**2)
        f[19] = 8*eta*(-4*xi + 8*xi**3)
        f[20] = 8*eta*(1 - 12*xi**2 + 16*xi**4)
        f[21] = 8*eta*(6*xi - 32*xi**3 + 32*xi**5)
        f[22] = 8*eta*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[23] = 8*eta*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[24] = -4 + 24*eta**2
        f[25] = 2*(-4 + 24*eta**2)*xi
        f[26] = (-4 + 24*eta**2)*(-1 + 4*xi**2)
        f[27] = (-4 + 24*eta**2)*(-4*xi + 8*xi**3)
        f[28] = (-4 + 24*eta**2)*(1 - 12*xi**2 + 16*xi**4)
        f[29] = (-4 + 24*eta**2)*(6*xi - 32*xi**3 + 32*xi**5)
        f[30] = (-4 + 24*eta**2)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[31] = (-4 + 24*eta**2)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[32] = -24*eta + 64*eta**3
        f[33] = 2*(-24*eta + 64*eta**3)*xi
        f[34] = (-24*eta + 64*eta**3)*(-1 + 4*xi**2)
        f[35] = (-24*eta + 64*eta**3)*(-4*xi + 8*xi**3)
        f[36] = (-24*eta + 64*eta**3)*(1 - 12*xi**2 + 16*xi**4)
        f[37] = (-24*eta + 64*eta**3)*(6*xi - 32*xi**3 + 32*xi**5)
        f[38] = (-24*eta + 64*eta**3)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[39] = (-24*eta + 64*eta**3)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[40] = 6 - 96*eta**2 + 160*eta**4
        f[41] = 2*(6 - 96*eta**2 + 160*eta**4)*xi
        f[42] = (6 - 96*eta**2 + 160*eta**4)*(-1 + 4*xi**2)
        f[43] = (6 - 96*eta**2 + 160*eta**4)*(-4*xi + 8*xi**3)
        f[44] = (6 - 96*eta**2 + 160*eta**4)*(1 - 12*xi**2 + 16*xi**4)
        f[45] = (6 - 96*eta**2 + 160*eta**4)*(6*xi - 32*xi**3 + 32*xi**5)
        f[46] = (6 - 96*eta**2 + 160*eta**4)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[47] = (6 - 96*eta**2 + 160*eta**4)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[48] = 48*eta - 320*eta**3 + 384*eta**5
        f[49] = 2*(48*eta - 320*eta**3 + 384*eta**5)*xi
        f[50] = (48*eta - 320*eta**3 + 384*eta**5)*(-1 + 4*xi**2)
        f[51] = (48*eta - 320*eta**3 + 384*eta**5)*(-4*xi + 8*xi**3)
        f[52] = (48*eta - 320*eta**3 + 384*eta**5)*(1 - 12*xi**2 + 16*xi**4)
        f[53] = (48*eta - 320*eta**3 + 384*eta**5)*(6*xi - 32*xi**3 + 32*xi**5)
        f[54] = (48*eta - 320*eta**3 + 384*eta**5)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[55] = (48*eta - 320*eta**3 + 384*eta**5)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
        f[56] = -8 + 240*eta**2 - 960*eta**4 + 896*eta**6
        f[57] = 2*(-8 + 240*eta**2 - 960*eta**4 + 896*eta**6)*xi
        f[58] = (-8 + 240*eta**2 - 960*eta**4 + 896*eta**6)*(-1 + 4*xi**2)
        f[59] = (-8 + 240*eta**2 - 960*eta**4 + 896*eta**6)*(-4*xi + 8*xi**3)
        f[60] = (-8 + 240*eta**2 - 960*eta**4 + 896*eta**6)*(1 - 12*xi**2 + 16*xi**4)
        f[61] = (-8 + 240*eta**2 - 960*eta**4 + 896*eta**6)*(6*xi - 32*xi**3 + 32*xi**5)
        f[62] = (-8 + 240*eta**2 - 960*eta**4 + 896*eta**6)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
        f[63] = (-8 + 240*eta**2 - 960*eta**4 + 896*eta**6)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)

        wt = 0.

        for j1 in range(n1):
            for i1 in range(m1):
                col = num0 + num1*((j1)*m1 + (i1))
                wt += c[col+2]*f[i1 + j1*m1]

        outwt[i] = wt


def fg(double[:,::1] g, int m1, int n1,
       double x, double t, double L, double tmin, double tmax):
    cfg(g, m1, n1, x, t, L, tmin, tmax)


cdef void cfg(double[:,::1] g, int m1, int n1,
              double x, double t, double L, double tmin, double tmax) nogil:
    cdef int i1, j1, col, i
    cdef double xi, eta, f[64]

    xi = x/(L/2.)
    eta = 2.*(t - tmin)/(tmax - tmin) - 1.

    f[0] = 1
    f[1] = 2*xi
    f[2] = -1 + 4*xi**2
    f[3] = -4*xi + 8*xi**3
    f[4] = 1 - 12*xi**2 + 16*xi**4
    f[5] = 6*xi - 32*xi**3 + 32*xi**5
    f[6] = -1 + 24*xi**2 - 80*xi**4 + 64*xi**6
    f[7] = -8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7
    f[8] = 2*eta
    f[9] = 4*eta*xi
    f[10] = 2*eta*(-1 + 4*xi**2)
    f[11] = 2*eta*(-4*xi + 8*xi**3)
    f[12] = 2*eta*(1 - 12*xi**2 + 16*xi**4)
    f[13] = 2*eta*(6*xi - 32*xi**3 + 32*xi**5)
    f[14] = 2*eta*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
    f[15] = 2*eta*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
    f[16] = -1 + 4*eta**2
    f[17] = 2*(-1 + 4*eta**2)*xi
    f[18] = (-1 + 4*eta**2)*(-1 + 4*xi**2)
    f[19] = (-1 + 4*eta**2)*(-4*xi + 8*xi**3)
    f[20] = (-1 + 4*eta**2)*(1 - 12*xi**2 + 16*xi**4)
    f[21] = (-1 + 4*eta**2)*(6*xi - 32*xi**3 + 32*xi**5)
    f[22] = (-1 + 4*eta**2)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
    f[23] = (-1 + 4*eta**2)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
    f[24] = -4*eta + 8*eta**3
    f[25] = 2*(-4*eta + 8*eta**3)*xi
    f[26] = (-4*eta + 8*eta**3)*(-1 + 4*xi**2)
    f[27] = (-4*eta + 8*eta**3)*(-4*xi + 8*xi**3)
    f[28] = (-4*eta + 8*eta**3)*(1 - 12*xi**2 + 16*xi**4)
    f[29] = (-4*eta + 8*eta**3)*(6*xi - 32*xi**3 + 32*xi**5)
    f[30] = (-4*eta + 8*eta**3)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
    f[31] = (-4*eta + 8*eta**3)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
    f[32] = 1 - 12*eta**2 + 16*eta**4
    f[33] = 2*(1 - 12*eta**2 + 16*eta**4)*xi
    f[34] = (1 - 12*eta**2 + 16*eta**4)*(-1 + 4*xi**2)
    f[35] = (1 - 12*eta**2 + 16*eta**4)*(-4*xi + 8*xi**3)
    f[36] = (1 - 12*eta**2 + 16*eta**4)*(1 - 12*xi**2 + 16*xi**4)
    f[37] = (1 - 12*eta**2 + 16*eta**4)*(6*xi - 32*xi**3 + 32*xi**5)
    f[38] = (1 - 12*eta**2 + 16*eta**4)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
    f[39] = (1 - 12*eta**2 + 16*eta**4)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
    f[40] = 6*eta - 32*eta**3 + 32*eta**5
    f[41] = 2*(6*eta - 32*eta**3 + 32*eta**5)*xi
    f[42] = (6*eta - 32*eta**3 + 32*eta**5)*(-1 + 4*xi**2)
    f[43] = (6*eta - 32*eta**3 + 32*eta**5)*(-4*xi + 8*xi**3)
    f[44] = (6*eta - 32*eta**3 + 32*eta**5)*(1 - 12*xi**2 + 16*xi**4)
    f[45] = (6*eta - 32*eta**3 + 32*eta**5)*(6*xi - 32*xi**3 + 32*xi**5)
    f[46] = (6*eta - 32*eta**3 + 32*eta**5)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
    f[47] = (6*eta - 32*eta**3 + 32*eta**5)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
    f[48] = -1 + 24*eta**2 - 80*eta**4 + 64*eta**6
    f[49] = 2*(-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*xi
    f[50] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-1 + 4*xi**2)
    f[51] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-4*xi + 8*xi**3)
    f[52] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(1 - 12*xi**2 + 16*xi**4)
    f[53] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(6*xi - 32*xi**3 + 32*xi**5)
    f[54] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
    f[55] = (-1 + 24*eta**2 - 80*eta**4 + 64*eta**6)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)
    f[56] = -8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7
    f[57] = 2*(-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*xi
    f[58] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-1 + 4*xi**2)
    f[59] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-4*xi + 8*xi**3)
    f[60] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(1 - 12*xi**2 + 16*xi**4)
    f[61] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(6*xi - 32*xi**3 + 32*xi**5)
    f[62] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-1 + 24*xi**2 - 80*xi**4 + 64*xi**6)
    f[63] = (-8*eta + 80*eta**3 - 192*eta**5 + 128*eta**7)*(-8*xi + 80*xi**3 - 192*xi**5 + 128*xi**7)

    for j1 in range(n1):
        for i1 in range(m1):
            col = num0 + num1*((j1)*m1 + (i1))
            g[0, col+0] = f[i1 + j1*m1]
            g[1, col+1] = f[i1 + j1*m1]
            g[2, col+2] = f[i1 + j1*m1]


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

