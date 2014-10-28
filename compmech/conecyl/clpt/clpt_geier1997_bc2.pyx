#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

cimport numpy as np
import numpy as np

from scipy.sparse import coo_matrix


DOUBLE = np.float64
INT = np.int64
ctypedef np.double_t cDOUBLE
ctypedef np.int64_t cINT


cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil


cdef int i0 = 0
cdef int j0 = 0
cdef int num0 = 0
cdef int num1 = 0
cdef int num2 = 3
cdef double pi=3.141592653589793


cdef void cfuvw(double *c, int m1, int m2, int n2, double r2,
                double L, double x, double t,
                double cosa, double tLA, double *uvw) nogil:
    cdef int i1, i2, j2, col
    cdef double sinbi, cosbi, sinbj, cosbj, u, v, w
    u = 0
    v = 0
    w = 0

    for j2 in range(j0, n2+j0):
        sinbj = sin(j2*t)
        cosbj = cos(j2*t)
        for i2 in range(i0, m2+i0):
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            sinbi = sin(i2*pi*x/L)
            cosbi = cos(i2*pi*x/L)
            u += c[col+0]*cosbi*cosbj
            v += c[col+1]*sinbi*sinbj
            w += c[col+2]*sinbi*cosbj

    uvw[0] = u
    uvw[1] = v
    uvw[2] = w


def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m1, int m2, int n2,
         double alpharad, double r2, double L, double tLA,
         np.ndarray[cDOUBLE, ndim=1] xvec,
         np.ndarray[cDOUBLE, ndim=1] tvec):
    cdef int ix
    cdef double x, t, r, sina, cosa, wx, wt
    cdef double uvw[5]
    cdef np.ndarray[cDOUBLE, ndim=1] u, v, w, phix, phit

    sina = sin(alpharad)
    cosa = cos(alpharad)

    u = np.zeros(np.shape(xvec), dtype=DOUBLE)
    v = np.zeros(np.shape(xvec), dtype=DOUBLE)
    w = np.zeros(np.shape(xvec), dtype=DOUBLE)
    phix = np.zeros(np.shape(xvec), dtype=DOUBLE)
    phit = np.zeros(np.shape(xvec), dtype=DOUBLE)

    for ix in range(xvec.shape[0]):
        x = xvec[ix]
        t = tvec[ix]
        r = r2 + sina*x
        cfuvw(&c[0], m1, m2, n2, r2, L, x, t, cosa, tLA, &uvw[0])
        u[ix] = uvw[0]
        v[ix] = uvw[1]
        w[ix] = uvw[2]
        cfwx(&c[0], m1, m2, n2, x, t, L, &wx)
        phix[ix] = -wx
        cfwt(&c[0], m1, m2, n2, x, t, L, &wt)
        phit[ix] = -wt/r

    return u, v, w, phix, phit


cdef void cfwx(double *c, int m1, int m2, int n2,
               double x, double t, double L, double *refwx) nogil:
    cdef double dsini2, cosj2t, wx
    cdef int i1, i2, j2, col
    wx = 0.
    for i2 in range(i0, m2+i0):
        dsini2 = i2*pi/L*cos(i2*pi*x/L)
        for j2 in range(j0, n2+j0):
            cosj2t = cos(j2*t)
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            wx += c[col+2]*dsini2*cosj2t

    refwx[0] = wx


cdef void cfwt(double *c, int m1, int m2, int n2,
               double x, double t, double L, double *refwt) nogil:
    cdef double sinbi, sinj2t, wt
    cdef int i2, j2, col
    wt = 0.
    for i2 in range(i0, m2+i0):
        sinbi = sin(i2*pi*x/L)
        for j2 in range(j0, n2+j0):
            sinj2t = sin(j2*t)
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            wt += c[col+2]*sinbi*(-j2*sinj2t)

    refwt[0] = wt


def fg(double[:,::1] g, int m1, int m2, int n2,
       double r2, double x, double t, double L, double cosa, double tLA):
    cfg(g, m1, m2, n2, r2, x, t, L, cosa, tLA)


cdef cfg(double[:, ::1] g, int m1, int m2, int n2,
         double r2, double x, double t, double L, double cosa, double tLA):
    cdef double sinbi, cosbi, sinbj, cosbj
    cdef int i1, i2, j2, col

    for i2 in range(i0, m2+i0):
        sinbi = sin(i2*pi*x/L)
        cosbi = cos(i2*pi*x/L)
        for j2 in range(j0, n2+j0):
            col = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            sinbj = sin(j2*t)
            cosbj = cos(j2*t)
            g[0, col+0] = cosbi*cosbj
            g[1, col+1] = sinbi*sinbj
            g[2, col+2] = sinbi*cosbj


def fk0(double alpharad, double r2, double L, np.ndarray[cDOUBLE, ndim=2] F,
           int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r, sina, cosa, xa, xb
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    sina = sin(alpharad)
    cosa = cos(alpharad)

    # sparse parameters
    k22_cond_1 = 9
    k22_cond_2 = 9
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k22_num

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    A11 = F[0,0]
    A12 = F[0,1]
    A16 = F[0,2]
    A22 = F[1,1]
    A26 = F[1,2]
    A66 = F[2,2]
    A44 = F[6,6]
    A45 = F[6,7]
    A55 = F[7,7]

    B11 = F[0,3]
    B12 = F[0,4]
    B16 = F[0,5]
    B22 = F[1,4]
    B26 = F[1,5]
    B66 = F[2,5]

    D11 = F[3,3]
    D12 = F[3,4]
    D16 = F[3,5]
    D22 = F[4,4]
    D26 = F[4,5]
    D66 = F[5,5]

    for section in range(s):
        c = -1

        xa = L*float(section)/s
        xb = L*float(section+1)/s

        r = r2 + sina*((xa+xb)/2.)

        for i2 in range(i0, m2+i0):
            for j2 in range(j0, n2+j0):
                row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                        #NOTE symmetry
                        if row > col:
                            continue

                        if k2==i2 and l2==j2:
                            if i2!=0:
                                # k0_22 cond_1
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += 0.25*(2*L*(2*pi*A12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (-pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+1
                                k0v[c] += 0.25*j2*(L**2*sina*(A22 + A66)*cos(2*pi*i2*xa/L) - L**2*sina*(A22 + A66)*cos(2*pi*i2*xb/L) + pi*i2*r*(L*(A12 - A66)*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) + pi*i2*(2*A12 + 2*A66)*(xa - xb)))/(L*i2*r)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += 0.25*(2*L*(-L*sina*(A22*L**2*cosa*r + 2*pi**2*B12*i2**2*r**2 + L**2*j2**2*(B22 + B66))*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(A12*L**2*cosa*r + pi**2*B11*i2**2*r**2 + B12*L**2*j2**2 - B22*L**2*sina**2 - 2*B66*L**2*j2**2)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(A12*L**2*cosa*r + pi**2*B11*i2**2*r**2 + B12*L**2*j2**2 + B22*L**2*sina**2 + 2*B66*L**2*j2**2))/(L**3*i2*r**2)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+0
                                k0v[c] += 0.25*j2*(L**2*sina*(A22 + A66)*cos(2*pi*i2*xa/L) - L**2*sina*(A22 + A66)*cos(2*pi*i2*xb/L) + pi*i2*r*(L*(A12 - A66)*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) + pi*i2*(2*A12 + 2*A66)*(xa - xb)))/(L*i2*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += 0.25*(2*L*(2*pi*A66*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) + (-pi**2*A66*i2**2*r**2 + L**2*(A22*j2**2 + A66*sina**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(pi**2*A66*i2**2*r**2 + L**2*(A22*j2**2 + A66*sina**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += 0.25*j2*(2*L*(pi*L*i2*r*sina*(B22 + 3*B66)*sin(pi*i2*(xa + xb)/L) + (B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*i2**2*r*(B12 - 2*B66)))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(A22*L**2*cosa*r + B22*L**2*j2**2 + B66*L**2*sina**2 + pi**2*i2**2*r**2*(B12 + 2*B66)))/(L**2*i2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += 0.25*(2*L*(-L*sina*(A22*L**2*cosa*r + 2*pi**2*B12*i2**2*r**2 + L**2*j2**2*(B22 + B66))*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(A12*L**2*cosa*r + pi**2*B11*i2**2*r**2 + B12*L**2*j2**2 - B22*L**2*sina**2 - 2*B66*L**2*j2**2)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(A12*L**2*cosa*r + pi**2*B11*i2**2*r**2 + B12*L**2*j2**2 + B22*L**2*sina**2 + 2*B66*L**2*j2**2))/(L**3*i2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += 0.25*j2*(2*L*(pi*L*i2*r*sina*(B22 + 3*B66)*sin(pi*i2*(xa + xb)/L) + (B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*i2**2*r*(B12 - 2*B66)))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(A22*L**2*cosa*r + B22*L**2*j2**2 + B66*L**2*sina**2 + pi**2*i2**2*r**2*(B12 + 2*B66)))/(L**2*i2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += 0.25*(2*L*(2*pi*L*i2*r*sina*(B22*L**2*cosa*r + pi**2*D12*i2**2*r**2 + L**2*j2**2*(D22 + 2*D66))*sin(pi*i2*(xa + xb)/L) + (D22*(L**4*j2**4 - pi**2*L**2*i2**2*r**2*sina**2) + D66*L**4*j2**2*sina**2 + r*(2*B22*L**4*cosa*j2**2 + r*(A22*L**4*cosa**2 + pi**2*i2**2*(2*B12*L**2*cosa*r + pi**2*D11*i2**2*r**2 + L**2*j2**2*(2*D12 - 4*D66)))))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D22*(L**4*j2**4 + pi**2*L**2*i2**2*r**2*sina**2) + D66*L**4*j2**2*sina**2 + r*(2*B22*L**4*cosa*j2**2 + r*(A22*L**4*cosa**2 + pi**2*i2**2*(2*B12*L**2*cosa*r + pi**2*D11*i2**2*r**2 + L**2*j2**2*(2*D12 + 4*D66))))))/(L**4*i2*r**3)

                            else:
                                # k0_22 cond_5
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += pi*(-xa + xb)*(A22*sina**2 + A66*j2**2)/r

                        elif k2!=i2 and l2==j2:
                            # k0_22 cond_2
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(pi**2*A11*k2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2)) + (pi*A12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*A12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += j2*(i2*(-L*sina*(A22 + A66)*sin(pi*k2*xa/L) + pi*k2*r*(A12 + A66)*cos(pi*k2*xa/L))*sin(pi*i2*xa/L) + k2*(L*sina*(A22 + A66)*cos(pi*i2*xb/L) - pi*i2*r*(A12 + A66)*sin(pi*i2*xb/L))*cos(pi*k2*xb/L) + (L*i2*sina*(A22 + A66)*sin(pi*i2*xb/L) + pi*r*(A12*i2**2 + A66*k2**2)*cos(pi*i2*xb/L))*sin(pi*k2*xb/L) + (-L*k2*sina*(A22 + A66)*cos(pi*k2*xa/L) - pi*r*(A12*i2**2 + A66*k2**2)*sin(pi*k2*xa/L))*cos(pi*i2*xa/L))/(r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += (i2*(-L**3*sina*(A22*cosa*r + j2**2*(B22 + B66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + pi*k2*r*(A12*L**2*cosa*r + pi**2*B11*k2**2*r**2 + B12*L**2*j2**2 + B22*L**2*sina**2 + 2*B66*L**2*j2**2)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + (L**3*sina*(A22*cosa*r + j2**2*(B22 + B66))*sin(pi*k2*xb/L) - pi*k2*r*(A12*L**2*cosa*r + pi**2*B11*k2**2*r**2 + B12*L**2*j2**2 + B22*L**2*sina**2 + 2*B66*L**2*j2**2)*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)) - (L*k2*sina*(A22*L**2*cosa*r + pi**2*B12*r**2*(-i2**2 + k2**2) + L**2*j2**2*(B22 + B66))*cos(pi*k2*xa/L) + pi*r*(B12*L**2*i2**2*j2**2 + B22*L**2*k2**2*sina**2 + 2*B66*L**2*j2**2*k2**2 + i2**2*r*(A12*L**2*cosa + pi**2*B11*k2**2*r))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(A22*L**2*cosa*r + pi**2*B12*r**2*(-i2**2 + k2**2) + L**2*j2**2*(B22 + B66))*cos(pi*k2*xb/L) + pi*r*(B12*L**2*i2**2*j2**2 + B22*L**2*k2**2*sina**2 + 2*B66*L**2*j2**2*k2**2 + i2**2*r*(A12*L**2*cosa + pi**2*B11*k2**2*r))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L**2*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += j2*(-L*k2*sina*(A22 + A66)*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + i2*(L*sina*(A22 + A66)*cos(pi*k2*xa/L) - pi*k2*r*(A12 + A66)*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + i2*(-L*sina*(A22 + A66)*cos(pi*k2*xb/L) + pi*k2*r*(A12 + A66)*sin(pi*k2*xb/L))*cos(pi*i2*xb/L) - pi*r*(A12*k2**2 + A66*i2**2)*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (L*k2*sina*(A22 + A66)*sin(pi*k2*xa/L) + pi*r*(A12*k2**2 + A66*i2**2)*cos(pi*k2*xa/L))*sin(pi*i2*xa/L))/(r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += (-k2*(pi**2*A66*i2**2*r**2 + L**2*(A22*j2**2 + A66*sina**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + k2*(pi**2*A66*i2**2*r**2 + L**2*(A22*j2**2 + A66*sina**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (pi*A66*L*r*sina*(i2 - k2)*(i2 + k2)*sin(pi*i2*xa/L) + i2*(pi**2*A66*k2**2*r**2 + L**2*(A22*j2**2 + A66*sina**2))*cos(pi*i2*xa/L))*sin(pi*k2*xa/L) - (pi*A66*L*r*sina*(i2 - k2)*(i2 + k2)*sin(pi*i2*xb/L) + i2*(pi**2*A66*k2**2*r**2 + L**2*(A22*j2**2 + A66*sina**2))*cos(pi*i2*xb/L))*sin(pi*k2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += j2*(pi*L*r*sina*(-B66*i2**2 + k2**2*(B22 + 2*B66))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + i2*(-pi*L*k2*r*sina*(B22 + B66)*cos(pi*k2*xa/L) + (B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*k2**2*r*(B12 + 2*B66)))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + i2*(pi*L*k2*r*sina*(B22 + B66)*cos(pi*k2*xb/L) - (B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*k2**2*r*(B12 + 2*B66)))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L) + k2*(B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*r*(B12*k2**2 + 2*B66*i2**2)))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (pi*L*r*sina*(B66*i2**2 - k2**2*(B22 + 2*B66))*sin(pi*k2*xa/L) - k2*(B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*r*(B12*k2**2 + 2*B66*i2**2)))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += (-L**3*k2*sina*(A22*cosa*r + j2**2*(B22 + B66))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + i2*(L*sina*(A22*L**2*cosa*r + pi**2*B12*r**2*(i2 - k2)*(i2 + k2) + L**2*j2**2*(B22 + B66))*cos(pi*k2*xa/L) - pi*k2*r*(A12*L**2*cosa*r + pi**2*B11*i2**2*r**2 + B12*L**2*j2**2 + B22*L**2*sina**2 + 2*B66*L**2*j2**2)*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + i2*(-L*sina*(A22*L**2*cosa*r + pi**2*B12*r**2*(i2 - k2)*(i2 + k2) + L**2*j2**2*(B22 + B66))*cos(pi*k2*xb/L) + pi*k2*r*(A12*L**2*cosa*r + pi**2*B11*i2**2*r**2 + B12*L**2*j2**2 + B22*L**2*sina**2 + 2*B66*L**2*j2**2)*sin(pi*k2*xb/L))*cos(pi*i2*xb/L) - pi*r*(B22*L**2*i2**2*sina**2 + 2*B66*L**2*i2**2*j2**2 + k2**2*(A12*L**2*cosa*r + pi**2*B11*i2**2*r**2 + B12*L**2*j2**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (L**3*k2*sina*(A22*cosa*r + j2**2*(B22 + B66))*sin(pi*k2*xa/L) + pi*r*(B22*L**2*i2**2*sina**2 + 2*B66*L**2*i2**2*j2**2 + k2**2*(A12*L**2*cosa*r + pi**2*B11*i2**2*r**2 + B12*L**2*j2**2))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L))/(L**2*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += j2*(pi*L*r*sina*(B66*k2**2 - i2**2*(B22 + 2*B66))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + i2*(pi*L*k2*r*sina*(B22 + B66)*cos(pi*k2*xa/L) + (B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*r*(B12*i2**2 + 2*B66*k2**2)))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + i2*(-pi*L*k2*r*sina*(B22 + B66)*cos(pi*k2*xb/L) - (B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*r*(B12*i2**2 + 2*B66*k2**2)))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L) + k2*(B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*i2**2*r*(B12 + 2*B66)))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (pi*L*r*sina*(-B66*k2**2 + i2**2*(B22 + 2*B66))*sin(pi*k2*xa/L) - k2*(B22*L**2*j2**2 + B66*L**2*sina**2 + r*(A22*L**2*cosa + pi**2*i2**2*r*(B12 + 2*B66)))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += (pi*L**3*r*sina*(-i2**2 + k2**2)*(B22*cosa*r + j2**2*(D22 + 2*D66))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + i2*(pi**3*D12*L*k2*r**3*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + (D22*(L**4*j2**4 + pi**2*L**2*k2**2*r**2*sina**2) + D66*L**4*j2**2*sina**2 + r*(2*B22*L**4*cosa*j2**2 + r*(A22*L**4*cosa**2 + pi**2*(B12*L**2*cosa*r*(i2**2 + k2**2) + pi**2*D11*i2**2*k2**2*r**2 + L**2*j2**2*(D12*i2**2 + k2**2*(D12 + 4*D66))))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + i2*(pi**3*D12*L*k2*r**3*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - (D22*(L**4*j2**4 + pi**2*L**2*k2**2*r**2*sina**2) + D66*L**4*j2**2*sina**2 + r*(2*B22*L**4*cosa*j2**2 + r*(A22*L**4*cosa**2 + pi**2*(B12*L**2*cosa*r*(i2**2 + k2**2) + pi**2*D11*i2**2*k2**2*r**2 + L**2*j2**2*(D12*i2**2 + k2**2*(D12 + 4*D66))))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L) + k2*(D22*(L**4*j2**4 + pi**2*L**2*i2**2*r**2*sina**2) + D66*L**4*j2**2*sina**2 + r*(2*B22*L**4*cosa*j2**2 + r*(A22*L**4*cosa**2 + pi**2*(B12*L**2*cosa*r*(i2**2 + k2**2) + pi**2*D11*i2**2*k2**2*r**2 + L**2*j2**2*(D12*k2**2 + i2**2*(D12 + 4*D66))))))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (pi*L**3*r*sina*(i2 - k2)*(i2 + k2)*(B22*cosa*r + j2**2*(D22 + 2*D66))*sin(pi*k2*xa/L) - k2*(D22*(L**4*j2**4 + pi**2*L**2*i2**2*r**2*sina**2) + D66*L**4*j2**2*sina**2 + r*(2*B22*L**4*cosa*j2**2 + r*(A22*L**4*cosa**2 + pi**2*(B12*L**2*cosa*r*(i2**2 + k2**2) + pi**2*D11*i2**2*k2**2*r**2 + L**2*j2**2*(D12*k2**2 + i2**2*(D12 + 4*D66))))))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L))/(L**3*r**3*(i2 - k2)*(i2 + k2))


    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0_cyl(double r2, double L, np.ndarray[cDOUBLE, ndim=2] F,
            int m1, int m2, int n2):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    # sparse parameters
    k22_cond_1 = 9
    k22_cond_2 = 0
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k22_num

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    A11 = F[0,0]
    A12 = F[0,1]
    A16 = F[0,2]
    A22 = F[1,1]
    A26 = F[1,2]
    A66 = F[2,2]
    A44 = F[6,6]
    A45 = F[6,7]
    A55 = F[7,7]

    B11 = F[0,3]
    B12 = F[0,4]
    B16 = F[0,5]
    B22 = F[1,4]
    B26 = F[1,5]
    B66 = F[2,5]

    D11 = F[3,3]
    D12 = F[3,4]
    D16 = F[3,5]
    D22 = F[4,4]
    D26 = F[4,5]
    D66 = F[5,5]

    c = -1
    r = r2

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k2==i2 and l2==j2:
                        if i2!=0:
                            # k0_22 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.5*pi**3*A11*i2**2*r/L + 0.5*pi*A66*L*j2**2/r
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += -0.5*pi**2*i2*j2*(A12 + A66)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += -0.5*pi**2*i2*(A12*L**2*r + pi**2*B11*i2**2*r**2 + L**2*j2**2*(B12 + 2*B66))/(L**2*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += -0.5*pi**2*i2*j2*(A12 + A66)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.5*pi*A22*L*j2**2/r + 0.5*pi**3*A66*i2**2*r/L
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi*j2*(B22*L**2*j2**2 + r*(A22*L**2 + pi**2*i2**2*r*(B12 + 2*B66)))/(L*r**2)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += -0.5*pi**2*i2*(A12*L**2*r + pi**2*B11*i2**2*r**2 + L**2*j2**2*(B12 + 2*B66))/(L**2*r)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += 0.5*pi*j2*(B22*L**2*j2**2 + r*(A22*L**2 + pi**2*i2**2*r*(B12 + 2*B66)))/(L*r**2)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi*(D22*L**4*j2**4 + r*(2*B22*L**4*j2**2 + r*(A22*L**4 + pi**2*i2**2*(2*B12*L**2*r + pi**2*D11*i2**2*r**2 + L**2*j2**2*(2*D12 + 4*D66)))))/(L**3*r**3)

                        else:
                            # k0_22 cond_5
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += pi*A66*L*j2**2/r


    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0edges(int m1, int m2, int n2, double r1, double r2, double L,
             double kuBot, double kuTop,
             double kphixBot, double kphixTop):
    cdef int i1, k1, i2, j2, k2, l2, row, col, c
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv

    k22_cond_1 = 2
    k22_cond_2 = 2
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k22_num

    k0edgesr = np.zeros((fdim,), dtype=INT)
    k0edgesc = np.zeros((fdim,), dtype=INT)
    k0edgesv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k2==i2 and l2==j2:
                        # k0edges_22 cond_1
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += pi*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += pi**3*i2**2*(kphixBot*r1 + kphixTop*r2)/L**2

                    elif k2!=i2 and l2==j2:
                        # k0edges_22 cond_2
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += pi**3*i2*k2*((-1)**(i2 + k2)*kphixBot*r1 + kphixTop*r2)/L**2


    size = num0 + num1*m1 + num2*m2*n2

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fc, double P, double T, double r2, double alpharad, double L,
        int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double sina, cosa, xa, xb, r
    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    # sparse parameters
    k22_cond_1 = 1
    k22_cond_2 = 1
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k22_num

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    for section in range(s):
        c = -1

        xa = L*float(section)/s
        xb = L*float(section+1)/s

        r = r2 + sina*((xa+xb)/2.)

        for i2 in range(i0, m2+i0):
            for j2 in range(j0, n2+j0):
                row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                for k2 in range(i0, m2+i0):
                    for l2 in range(j0, n2+j0):
                        col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                        #NOTE symmetry
                        if row > col:
                            continue

                        if k2==i2 and l2==j2:
                            if i2!=0:
                                # kG0_22 cond_1
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+2
                                kG0v[c] += 0.125*(L*(2*L**2*P*j2**2 + pi*i2**2*(Fc - pi*P*r**2))*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) - 2*pi*i2*(xa - xb)*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2)))/(L**2*cosa*i2)

                        elif k2!=i2 and l2==j2:
                            # kG0_22 cond_2
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += 0.5*(i2*(-2*L**2*P*j2**2 + pi*k2**2*(Fc - pi*P*r**2))*sin(pi*k2*xb/L)*cos(pi*i2*xb/L) + i2*(2*L**2*P*j2**2 + pi*k2**2*(-Fc + pi*P*r**2))*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*(-2*L**2*P*j2**2 + pi*i2**2*(Fc - pi*P*r**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + k2*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L))/(L*cosa*(i2 - k2)*(i2 + k2))


    size = num0 + num1*m1 + num2*m2*n2

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkG0_cyl(double Fc, double P, double T, double r2, double L,
            int m1, int m2, int n2):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col
    cdef double r=r2
    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    # sparse parameters
    k22_cond_1 = 1
    k22_cond_2 = 0
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k22_num

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k2==i2 and l2==j2:
                        if i2!=0:
                            # kG0_22 cond_1
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += 0.25*pi*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2))/L


    size = num0 + num1*m1 + num2*m2*n2

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
