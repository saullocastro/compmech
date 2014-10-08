#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

cdef int i0 = 0
cdef int j0 = 1
cdef int num0 = 3
cdef int num1 = 3
cdef int num2 = 6
cdef double pi = 3.141592653589793

def fk0(double alpharad, double r2, double L, np.ndarray[cDOUBLE, ndim=2] F,
        int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r, sina, cosa, xa, xb
    cdef double sini1xa, cosi1xa, sini1xb, cosi1xb
    cdef double sini1xa_xb, sini1xaxb, cosi1xaxb
    cdef double sink1xa, sink1xb, cosk1xa, cosk1xb, sini2xa, sini2xb
    cdef double sin2i2xa, sin2i2xb, sini2xa_xb, sini2xaxb, cosi2xaxb
    cdef double cosi2xa, cosi2xb, cos2i2xa, cos2i2xb
    cdef double cosk2xa, cosk2xb, sink2xa, sink2xb

    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    sina = sin(alpharad)
    cosa = cos(alpharad)

    # sparse parameters
    k11_cond_1 = 9
    k11_cond_2 = 9
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 30
    k22_cond_2 = 36
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 5 + 2*6*m1 + k11_num + k22_num

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    A11 = F[0,0]
    A12 = F[0,1]
    A16 = F[0,2]
    A22 = F[1,1]
    A26 = F[1,2]
    A66 = F[2,2]
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

    with nogil:
        for section in range(s):
            c = -1

            xa = L*float(section)/s
            xb = L*float(section+1)/s

            r = r2 + sina*((xa+xb)/2.)

            # k0_00
            c += 1
            k0r[c] = 0
            k0c[c] = 0
            k0v[c] += -0.666666666666667*pi*(xa - xb)*(3*A11*(r*r) + sina*(3*A12*r*(-2*L + xa + xb) + A22*sina*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb))))/((L*L)*(cosa*cosa)*r)
            c += 1
            k0r[c] = 0
            k0c[c] = 1
            k0v[c] += 0.333333333333333*pi*r2*(xa - xb)*(3*A16*(r*r)*(-2*r + sina*(-2*L + xa + xb)) + 3*B16*cosa*r*(-2*r + sina*(-2*L + xa + xb)) + sina*(A26*r + B26*cosa)*(6*(L*L)*sina + 6*L*(r - sina*(xa + xb)) - 3*r*(xa + xb) + 2*sina*((xa*xa) + xa*xb + (xb*xb))))/((L*L)*cosa*(r*r))
            c += 1
            k0r[c] = 1
            k0c[c] = 0
            k0v[c] += 0.333333333333333*pi*r2*(xa - xb)*(3*A16*(r*r)*(-2*r + sina*(-2*L + xa + xb)) + 3*B16*cosa*r*(-2*r + sina*(-2*L + xa + xb)) + sina*(A26*r + B26*cosa)*(6*(L*L)*sina + 6*L*(r - sina*(xa + xb)) - 3*r*(xa + xb) + 2*sina*((xa*xa) + xa*xb + (xb*xb))))/((L*L)*cosa*(r*r))
            c += 1
            k0r[c] = 1
            k0c[c] = 1
            k0v[c] += -0.666666666666667*pi*(r2*r2)*(xa - xb)*(3*(r*r) + 3*r*sina*(2*L - xa - xb) + (sina*sina)*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb)))*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa))/((L*L)*(r*r*r))
            c += 1
            k0r[c] = 2
            k0c[c] = 2
            k0v[c] += -0.333333333333333*pi*(xa - xb)*(3*A11*(r*r) + A66*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb)) + sina*(3*A12*r*(-2*L + xa + xb) + A22*sina*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb))))/((L*L)*(cosa*cosa)*r)

            for i1 in range(i0, m1+i0):
                cosi1xa = cos(pi*i1*xa/L)
                cosi1xb = cos(pi*i1*xb/L)
                sini1xa = sin(pi*i1*xa/L)
                sini1xb = sin(pi*i1*xb/L)
                cosi1xa = cos(pi*i1*xa/L)
                cosi1xaxb = cos(pi*i1*(xa + xb)/L)
                sini1xa = sin(pi*i1*xa/L)
                sini1xa_xb = sin(pi*i1*(xa - xb)/L)
                sini1xaxb = sin(pi*i1*(xa + xb)/L)

                col = (i1-i0)*num1 + num0
                row = col
                if i1!=0:
                    # k0_01 cond_1
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+0
                    k0v[c] += (2*pi*A22*L*cosi1xa*i1*(sina*sina)*(L - xa) + 2*pi*A22*L*cosi1xb*i1*(sina*sina)*(-L + xb) + 2*sini1xa*((pi*pi)*A11*(i1*i1)*(r*r) + sina*((pi*pi)*A12*(i1*i1)*r*(-L + xa) + A22*(L*L)*sina)) - 2*sini1xb*((pi*pi)*A11*(i1*i1)*(r*r) + sina*((pi*pi)*A12*(i1*i1)*r*(-L + xb) + A22*(L*L)*sina)))/(pi*L*cosa*(i1*i1)*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+1
                    k0v[c] += (2*pi*L*cosi1xa*i1*sina*(B16*cosa*r + B26*cosa*(r + sina*(-L + xa)) + r*(A16*r + A26*(-L*sina + r + sina*xa))) + 2*pi*L*cosi1xb*i1*sina*(-B16*cosa*r - B26*cosa*(r + sina*(-L + xb)) - r*(A16*r + A26*(-L*sina + r + sina*xb))) + 2*sini1xa*((pi*pi)*A16*(i1*i1)*(r*r*r) + (pi*pi)*B16*cosa*(i1*i1)*(r*r) + sina*(A26*r + B26*cosa)*(-(L*L)*sina + (pi*pi)*(i1*i1)*r*(-L + xa))) + 2*sini1xb*(-(pi*pi)*A16*(i1*i1)*(r*r*r) - (pi*pi)*B16*cosa*(i1*i1)*(r*r) + sina*(A26*r + B26*cosa)*((L*L)*sina + (pi*pi)*(i1*i1)*r*(L - xb))))/(pi*L*cosa*(i1*i1)*(r*r))
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+2
                    k0v[c] += (2*L*sina*(sini1xa*(A22*(L*L)*cosa + (pi*pi)*B22*(i1*i1)*sina*(L - xa)) - sini1xb*(A22*(L*L)*cosa + (pi*pi)*B22*(i1*i1)*sina*(L - xb))) + 2*pi*cosi1xa*i1*(-A12*(L*L)*cosa*r - (pi*pi)*B11*(i1*i1)*(r*r) - sina*(A22*(L*L)*cosa*(-L + xa) + (pi*pi)*B12*(i1*i1)*r*(-L + xa) + B22*(L*L)*sina)) + 2*pi*cosi1xb*i1*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i1*i1)*(r*r) + sina*(A22*(L*L)*cosa*(-L + xb) + (pi*pi)*B12*(i1*i1)*r*(-L + xb) + B22*(L*L)*sina)))/(pi*(L*L)*cosa*(i1*i1)*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+0
                    k0v[c] += 2*r2*(-pi*L*cosi1xa*i1*sina*(B16*cosa*r + B26*cosa*(r + sina*(L - xa)) + r*(A16*r + A26*(L*sina + r - sina*xa))) + pi*L*cosi1xb*i1*sina*(B16*cosa*r + B26*cosa*(r + sina*(L - xb)) + r*(A16*r + A26*(L*sina + r - sina*xb))) + sini1xa*((pi*pi)*A16*(i1*i1)*(r*r)*(L*sina + r - sina*xa) + (pi*pi)*B16*cosa*(i1*i1)*r*(r + sina*(L - xa)) - (L*L)*(sina*sina)*(A26*r + B26*cosa)) + sini1xb*(-(pi*pi)*A16*(i1*i1)*(r*r)*(L*sina + r - sina*xb) - (pi*pi)*B16*cosa*(i1*i1)*r*(r + sina*(L - xb)) + (L*L)*(sina*sina)*(A26*r + B26*cosa)))/(pi*L*(i1*i1)*(r*r))
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+1
                    k0v[c] += r2*(2*A66*(r*r) + 4*B66*cosa*r + 2*D66*(cosa*cosa))*(pi*L*cosi1xa*i1*(sina*sina)*(L - xa) + pi*L*cosi1xb*i1*(sina*sina)*(-L + xb) + sini1xa*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*r*(L*sina + r - sina*xa)) - sini1xb*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*r*(L*sina + r - sina*xb)))/(pi*L*(i1*i1)*(r*r*r))
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+2
                    k0v[c] += -2*r2*(L*sina*(sini1xa*(A26*(L*L)*cosa*r + B26*((L*L)*(cosa*cosa) + (pi*pi)*(i1*i1)*r*(L*sina + r - sina*xa)) + (pi*pi)*(i1*i1)*(B16*(r*r) + cosa*(D16*r + D26*(L*sina + r - sina*xa)))) - sini1xb*(A26*(L*L)*cosa*r + B26*((L*L)*(cosa*cosa) + (pi*pi)*(i1*i1)*r*(L*sina + r - sina*xb)) + (pi*pi)*(i1*i1)*(B16*(r*r) + cosa*(D16*r + D26*(L*sina + r - sina*xb))))) + pi*cosi1xa*i1*(A26*(L*L)*cosa*r*(L*sina + r - sina*xa) + (pi*pi)*B16*L*(i1*i1)*(r*r)*sina + (pi*pi)*B16*(i1*i1)*(r*r*r) + B26*(L*L)*((cosa*cosa)*(L*sina + r - sina*xa) - r*(sina*sina)) + (pi*pi)*D16*L*cosa*(i1*i1)*r*sina + (pi*pi)*D16*cosa*(i1*i1)*(r*r) - D26*(L*L)*cosa*(sina*sina) - (pi*pi)*(i1*i1)*r*sina*xa*(B16*r + D16*cosa)) - pi*cosi1xb*i1*(A26*(L*L)*cosa*r*(L*sina + r - sina*xb) + (pi*pi)*B16*L*(i1*i1)*(r*r)*sina + (pi*pi)*B16*(i1*i1)*(r*r*r) + B26*(L*L)*((cosa*cosa)*(L*sina + r - sina*xb) - r*(sina*sina)) + (pi*pi)*D16*L*cosa*(i1*i1)*r*sina + (pi*pi)*D16*cosa*(i1*i1)*(r*r) - D26*(L*L)*cosa*(sina*sina) - (pi*pi)*(i1*i1)*r*sina*xb*(B16*r + D16*cosa)))/(pi*(L*L)*(i1*i1)*(r*r))

                    # k0_10 (using symmetry)
                    c += 1
                    k0r[c] = col+0
                    k0c[c] = 0
                    k0v[c] += (2*pi*A22*L*cosi1xa*i1*(sina*sina)*(L - xa) + 2*pi*A22*L*cosi1xb*i1*(sina*sina)*(-L + xb) + 2*sini1xa*((pi*pi)*A11*(i1*i1)*(r*r) + sina*((pi*pi)*A12*(i1*i1)*r*(-L + xa) + A22*(L*L)*sina)) - 2*sini1xb*((pi*pi)*A11*(i1*i1)*(r*r) + sina*((pi*pi)*A12*(i1*i1)*r*(-L + xb) + A22*(L*L)*sina)))/(pi*L*cosa*(i1*i1)*r)
                    c += 1
                    k0r[c] = col+1
                    k0c[c] = 0
                    k0v[c] += (2*pi*L*cosi1xa*i1*sina*(B16*cosa*r + B26*cosa*(r + sina*(-L + xa)) + r*(A16*r + A26*(-L*sina + r + sina*xa))) + 2*pi*L*cosi1xb*i1*sina*(-B16*cosa*r - B26*cosa*(r + sina*(-L + xb)) - r*(A16*r + A26*(-L*sina + r + sina*xb))) + 2*sini1xa*((pi*pi)*A16*(i1*i1)*(r*r*r) + (pi*pi)*B16*cosa*(i1*i1)*(r*r) + sina*(A26*r + B26*cosa)*(-(L*L)*sina + (pi*pi)*(i1*i1)*r*(-L + xa))) + 2*sini1xb*(-(pi*pi)*A16*(i1*i1)*(r*r*r) - (pi*pi)*B16*cosa*(i1*i1)*(r*r) + sina*(A26*r + B26*cosa)*((L*L)*sina + (pi*pi)*(i1*i1)*r*(L - xb))))/(pi*L*cosa*(i1*i1)*(r*r))
                    c += 1
                    k0r[c] = col+2
                    k0c[c] = 0
                    k0v[c] += (2*L*sina*(sini1xa*(A22*(L*L)*cosa + (pi*pi)*B22*(i1*i1)*sina*(L - xa)) - sini1xb*(A22*(L*L)*cosa + (pi*pi)*B22*(i1*i1)*sina*(L - xb))) + 2*pi*cosi1xa*i1*(-A12*(L*L)*cosa*r - (pi*pi)*B11*(i1*i1)*(r*r) - sina*(A22*(L*L)*cosa*(-L + xa) + (pi*pi)*B12*(i1*i1)*r*(-L + xa) + B22*(L*L)*sina)) + 2*pi*cosi1xb*i1*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i1*i1)*(r*r) + sina*(A22*(L*L)*cosa*(-L + xb) + (pi*pi)*B12*(i1*i1)*r*(-L + xb) + B22*(L*L)*sina)))/(pi*(L*L)*cosa*(i1*i1)*r)
                    c += 1
                    k0r[c] = col+0
                    k0c[c] = 1
                    k0v[c] += 2*r2*(-pi*L*cosi1xa*i1*sina*(B16*cosa*r + B26*cosa*(r + sina*(L - xa)) + r*(A16*r + A26*(L*sina + r - sina*xa))) + pi*L*cosi1xb*i1*sina*(B16*cosa*r + B26*cosa*(r + sina*(L - xb)) + r*(A16*r + A26*(L*sina + r - sina*xb))) + sini1xa*((pi*pi)*A16*(i1*i1)*(r*r)*(L*sina + r - sina*xa) + (pi*pi)*B16*cosa*(i1*i1)*r*(r + sina*(L - xa)) - (L*L)*(sina*sina)*(A26*r + B26*cosa)) + sini1xb*(-(pi*pi)*A16*(i1*i1)*(r*r)*(L*sina + r - sina*xb) - (pi*pi)*B16*cosa*(i1*i1)*r*(r + sina*(L - xb)) + (L*L)*(sina*sina)*(A26*r + B26*cosa)))/(pi*L*(i1*i1)*(r*r))
                    c += 1
                    k0r[c] = col+1
                    k0c[c] = 1
                    k0v[c] += r2*(2*A66*(r*r) + 4*B66*cosa*r + 2*D66*(cosa*cosa))*(pi*L*cosi1xa*i1*(sina*sina)*(L - xa) + pi*L*cosi1xb*i1*(sina*sina)*(-L + xb) + sini1xa*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*r*(L*sina + r - sina*xa)) - sini1xb*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*r*(L*sina + r - sina*xb)))/(pi*L*(i1*i1)*(r*r*r))
                    c += 1
                    k0r[c] = col+2
                    k0c[c] = 1
                    k0v[c] += -2*r2*(L*sina*(sini1xa*(A26*(L*L)*cosa*r + B26*((L*L)*(cosa*cosa) + (pi*pi)*(i1*i1)*r*(L*sina + r - sina*xa)) + (pi*pi)*(i1*i1)*(B16*(r*r) + cosa*(D16*r + D26*(L*sina + r - sina*xa)))) - sini1xb*(A26*(L*L)*cosa*r + B26*((L*L)*(cosa*cosa) + (pi*pi)*(i1*i1)*r*(L*sina + r - sina*xb)) + (pi*pi)*(i1*i1)*(B16*(r*r) + cosa*(D16*r + D26*(L*sina + r - sina*xb))))) + pi*cosi1xa*i1*(A26*(L*L)*cosa*r*(L*sina + r - sina*xa) + (pi*pi)*B16*L*(i1*i1)*(r*r)*sina + (pi*pi)*B16*(i1*i1)*(r*r*r) + B26*(L*L)*((cosa*cosa)*(L*sina + r - sina*xa) - r*(sina*sina)) + (pi*pi)*D16*L*cosa*(i1*i1)*r*sina + (pi*pi)*D16*cosa*(i1*i1)*(r*r) - D26*(L*L)*cosa*(sina*sina) - (pi*pi)*(i1*i1)*r*sina*xa*(B16*r + D16*cosa)) - pi*cosi1xb*i1*(A26*(L*L)*cosa*r*(L*sina + r - sina*xb) + (pi*pi)*B16*L*(i1*i1)*(r*r)*sina + (pi*pi)*B16*(i1*i1)*(r*r*r) + B26*(L*L)*((cosa*cosa)*(L*sina + r - sina*xb) - r*(sina*sina)) + (pi*pi)*D16*L*cosa*(i1*i1)*r*sina + (pi*pi)*D16*cosa*(i1*i1)*(r*r) - D26*(L*L)*cosa*(sina*sina) - (pi*pi)*(i1*i1)*r*sina*xb*(B16*r + D16*cosa)))/(pi*(L*L)*(i1*i1)*(r*r))

                for k1 in range(i0, m1+i0):
                    cosk1xa = cos(pi*k1*xa/L)
                    cosk1xb = cos(pi*k1*xb/L)
                    sink1xa = sin(pi*k1*xa/L)
                    sink1xb = sin(pi*k1*xb/L)
                    col = (k1-i0)*num1 + num0
                    if k1==i1:
                        if i1!=0:
                            # k0_11 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(-2*pi*A12*L*i1*r*sina*sini1xaxb + cosi1xaxb*(-(pi*pi)*A11*(i1*i1)*(r*r) + A22*(L*L)*(sina*sina))) - 2*pi*i1*(xa - xb)*((pi*pi)*A11*(i1*i1)*(r*r) + A22*(L*L)*(sina*sina)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(pi*L*i1*r*sina*sini1xaxb*(A16*r - A26*r + B16*cosa - B26*cosa) - cosi1xaxb*((L*L)*(sina*sina)*(A26*r + B26*cosa) + (pi*pi)*(i1*i1)*(r*r)*(A16*r + B16*cosa))) - 2*pi*i1*(xa - xb)*(-(L*L)*(sina*sina)*(A26*r + B26*cosa) + (pi*pi)*(i1*i1)*(r*r)*(A16*r + B16*cosa)))/((L*L)*i1*(r*r))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += (pi*A22*(L*L)*cosa*i1*sina*(-xa + xb) + sini1xa_xb*(L*cosi1xaxb*sina*(A22*(L*L)*cosa + 2*(pi*pi)*B12*(i1*i1)*r) - pi*i1*sini1xaxb*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i1*i1)*(r*r) - B22*(L*L)*(sina*sina))))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(pi*L*i1*r*sina*sini1xaxb*(A16*r - A26*r + B16*cosa - B26*cosa) - cosi1xaxb*((L*L)*(sina*sina)*(A26*r + B26*cosa) + (pi*pi)*(i1*i1)*(r*r)*(A16*r + B16*cosa))) - 2*pi*i1*(xa - xb)*(-(L*L)*(sina*sina)*(A26*r + B26*cosa) + (pi*pi)*(i1*i1)*(r*r)*(A16*r + B16*cosa)))/((L*L)*i1*(r*r))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += -0.5*(-2*L*sini1xa_xb*(2*pi*L*i1*r*sina*sini1xaxb + cosi1xaxb*((L*L)*(sina*sina) - (pi*pi)*(i1*i1)*(r*r))) + 2*pi*i1*(xa - xb)*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)))*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa))/((L*L)*i1*(r*r*r))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += 0.5*(2*pi*i1*sina*(xa - xb)*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i1*i1)*(D16 + D26)) + (pi*pi)*(i1*i1)*(r*r)*(B16 + B26)) + 2*sini1xa_xb*(-L*cosi1xaxb*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i1*i1)*(D16 - D26)) + (pi*pi)*(i1*i1)*(r*r)*(B16 - B26)) - pi*i1*sini1xaxb*(B26*(L*L)*r*((cosa*cosa) + (sina*sina)) + D26*(L*L)*cosa*(sina*sina) + (r*r)*(A26*(L*L)*cosa + (pi*pi)*(i1*i1)*(B16*r + D16*cosa)))))/((L*L)*i1*(r*r))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += (pi*A22*(L*L)*cosa*i1*sina*(-xa + xb) + sini1xa_xb*(L*cosi1xaxb*sina*(A22*(L*L)*cosa + 2*(pi*pi)*B12*(i1*i1)*r) - pi*i1*sini1xaxb*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i1*i1)*(r*r) - B22*(L*L)*(sina*sina))))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += 0.5*(2*pi*i1*sina*(xa - xb)*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i1*i1)*(D16 + D26)) + (pi*pi)*(i1*i1)*(r*r)*(B16 + B26)) + 2*sini1xa_xb*(-L*cosi1xaxb*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i1*i1)*(D16 - D26)) + (pi*pi)*(i1*i1)*(r*r)*(B16 - B26)) - pi*i1*sini1xaxb*(B26*(L*L)*r*((cosa*cosa) + (sina*sina)) + D26*(L*L)*cosa*(sina*sina) + (r*r)*(A26*(L*L)*cosa + (pi*pi)*(i1*i1)*(B16*r + D16*cosa)))))/((L*L)*i1*(r*r))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(2*pi*L*i1*sina*sini1xaxb*(B22*(L*L)*cosa + (pi*pi)*D12*(i1*i1)*r) + cosi1xaxb*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(i1*i1)*(2*B12*(L*L)*cosa*r + (pi*pi)*D11*(i1*i1)*(r*r) - D22*(L*L)*(sina*sina)))) - 2*pi*i1*(xa - xb)*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(i1*i1)*(2*B12*(L*L)*cosa*r + (pi*pi)*D11*(i1*i1)*(r*r) + D22*(L*L)*(sina*sina))))/((L*L*L*L)*i1*r)

                    else:
                        # k0_11 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += (-2*cosk1xa*k1*sini1xa*((pi*pi)*A11*(i1*i1)*(r*r) + A22*(L*L)*(sina*sina)) + 2*cosk1xb*k1*sini1xb*((pi*pi)*A11*(i1*i1)*(r*r) + A22*(L*L)*(sina*sina)) + 2*sink1xa*(pi*A12*L*r*sina*sini1xa*(-(i1*i1) + (k1*k1)) + cosi1xa*i1*((pi*pi)*A11*(k1*k1)*(r*r) + A22*(L*L)*(sina*sina))) - 2*sink1xb*(pi*A12*L*r*sina*sini1xb*(-(i1*i1) + (k1*k1)) + cosi1xb*i1*((pi*pi)*A11*(k1*k1)*(r*r) + A22*(L*L)*(sina*sina))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += (-2*pi*L*r*sina*sini1xb*sink1xb*(A16*(i1*i1)*r + B16*cosa*(i1*i1) + (k1*k1)*(A26*r + B26*cosa)) + 2*cosi1xa*i1*(pi*L*cosk1xa*k1*r*sina*(cosa*(B16 + B26) + r*(A16 + A26)) + sink1xa*(-(L*L)*(sina*sina)*(A26*r + B26*cosa) + (pi*pi)*(k1*k1)*(r*r)*(A16*r + B16*cosa))) + 2*cosi1xb*i1*(-pi*L*cosk1xb*k1*r*sina*(cosa*(B16 + B26) + r*(A16 + A26)) + sink1xb*((L*L)*(sina*sina)*(A26*r + B26*cosa) - (pi*pi)*(k1*k1)*(r*r)*(A16*r + B16*cosa))) + 2*cosk1xb*k1*sini1xb*(-(L*L)*(sina*sina)*(A26*r + B26*cosa) + (pi*pi)*(i1*i1)*(r*r)*(A16*r + B16*cosa)) + 2*sini1xa*(pi*L*r*sina*sink1xa*(A16*(i1*i1)*r + B16*cosa*(i1*i1) + (k1*k1)*(A26*r + B26*cosa)) + cosk1xa*k1*((L*L)*(sina*sina)*(A26*r + B26*cosa) - (pi*pi)*(i1*i1)*(r*r)*(A16*r + B16*cosa))))/(L*(r*r)*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += (2*L*cosk1xb*k1*sina*sini1xb*(A22*(L*L)*cosa + (pi*pi)*B12*r*(-(i1*i1) + (k1*k1))) + 2*cosi1xa*i1*(A22*(L*L*L)*cosa*sina*sink1xa - pi*cosk1xa*k1*(A12*(L*L)*cosa*r + (pi*pi)*B11*(k1*k1)*(r*r) + B22*(L*L)*(sina*sina))) + 2*cosi1xb*i1*(-A22*(L*L*L)*cosa*sina*sink1xb + pi*cosk1xb*k1*(A12*(L*L)*cosa*r + (pi*pi)*B11*(k1*k1)*(r*r) + B22*(L*L)*(sina*sina))) + 2*sini1xa*(-L*cosk1xa*k1*sina*(A22*(L*L)*cosa + (pi*pi)*B12*r*(-(i1*i1) + (k1*k1))) - pi*sink1xa*(B22*(L*L)*(k1*k1)*(sina*sina) + (i1*i1)*r*(A12*(L*L)*cosa + (pi*pi)*B11*(k1*k1)*r))) + 2*pi*sini1xb*sink1xb*(B22*(L*L)*(k1*k1)*(sina*sina) + (i1*i1)*r*(A12*(L*L)*cosa + (pi*pi)*B11*(k1*k1)*r)))/((L*L)*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += (2*pi*L*r*sina*sini1xb*sink1xb*(A26*(i1*i1)*r + B26*cosa*(i1*i1) + (k1*k1)*(A16*r + B16*cosa)) + 2*cosi1xa*i1*(-pi*L*cosk1xa*k1*r*sina*(cosa*(B16 + B26) + r*(A16 + A26)) + sink1xa*(-(L*L)*(sina*sina)*(A26*r + B26*cosa) + (pi*pi)*(k1*k1)*(r*r)*(A16*r + B16*cosa))) + 2*cosi1xb*i1*(pi*L*cosk1xb*k1*r*sina*(cosa*(B16 + B26) + r*(A16 + A26)) + sink1xb*((L*L)*(sina*sina)*(A26*r + B26*cosa) - (pi*pi)*(k1*k1)*(r*r)*(A16*r + B16*cosa))) + 2*cosk1xb*k1*sini1xb*(-(L*L)*(sina*sina)*(A26*r + B26*cosa) + (pi*pi)*(i1*i1)*(r*r)*(A16*r + B16*cosa)) + 2*sini1xa*(-pi*L*r*sina*sink1xa*(A26*(i1*i1)*r + B26*cosa*(i1*i1) + (k1*k1)*(A16*r + B16*cosa)) + cosk1xa*k1*((L*L)*(sina*sina)*(A26*r + B26*cosa) - (pi*pi)*(i1*i1)*(r*r)*(A16*r + B16*cosa))))/(L*(r*r)*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += (-2*A66*(r*r) - 4*B66*cosa*r - 2*D66*(cosa*cosa))*(cosk1xa*k1*sini1xa*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)) - cosk1xb*k1*sini1xb*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)) - sink1xa*(pi*L*r*sina*sini1xa*(i1 - k1)*(i1 + k1) + cosi1xa*i1*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r))) + sink1xb*(pi*L*r*sina*sini1xb*(i1 - k1)*(i1 + k1) + cosi1xb*i1*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r))))/(L*(r*r*r)*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += (-2*L*cosk1xb*k1*sina*sini1xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(k1*k1) + D26*(i1*i1))) + (pi*pi)*(r*r)*(B16*(k1*k1) + B26*(i1*i1))) + 2*cosi1xa*i1*(-L*sina*sink1xa*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(k1*k1)*(D16 + D26)) + (pi*pi)*(k1*k1)*(r*r)*(B16 + B26)) - pi*cosk1xa*k1*(-(L*L)*(sina*sina)*(B26*r + D26*cosa) + r*(B26*(L*L)*(cosa*cosa) + r*(A26*(L*L)*cosa + (pi*pi)*(k1*k1)*(B16*r + D16*cosa))))) + 2*cosi1xb*i1*(L*sina*sink1xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(k1*k1)*(D16 + D26)) + (pi*pi)*(k1*k1)*(r*r)*(B16 + B26)) + pi*cosk1xb*k1*(-(L*L)*(sina*sina)*(B26*r + D26*cosa) + r*(B26*(L*L)*(cosa*cosa) + r*(A26*(L*L)*cosa + (pi*pi)*(k1*k1)*(B16*r + D16*cosa))))) + 2*sini1xa*(L*cosk1xa*k1*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(k1*k1) + D26*(i1*i1))) + (pi*pi)*(r*r)*(B16*(k1*k1) + B26*(i1*i1))) - pi*sink1xa*(-(L*L)*(k1*k1)*(sina*sina)*(B26*r + D26*cosa) + (i1*i1)*r*(B26*(L*L)*(cosa*cosa) + r*(A26*(L*L)*cosa + (pi*pi)*(k1*k1)*(B16*r + D16*cosa))))) + 2*pi*sini1xb*sink1xb*(-(L*L)*(k1*k1)*(sina*sina)*(B26*r + D26*cosa) + (i1*i1)*r*(B26*(L*L)*(cosa*cosa) + r*(A26*(L*L)*cosa + (pi*pi)*(k1*k1)*(B16*r + D16*cosa)))))/((L*L)*(r*r)*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += (2*A22*(L*L*L)*cosa*cosk1xb*k1*sina*sini1xb + 2*cosi1xa*i1*(L*sina*sink1xa*(A22*(L*L)*cosa + (pi*pi)*B12*r*(i1 - k1)*(i1 + k1)) + pi*cosk1xa*k1*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i1*i1)*(r*r) + B22*(L*L)*(sina*sina))) + 2*cosi1xb*i1*(-L*sina*sink1xb*(A22*(L*L)*cosa + (pi*pi)*B12*r*(i1 - k1)*(i1 + k1)) - pi*cosk1xb*k1*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i1*i1)*(r*r) + B22*(L*L)*(sina*sina))) + 2*sini1xa*(-A22*(L*L*L)*cosa*cosk1xa*k1*sina + pi*sink1xa*(B22*(L*L)*(i1*i1)*(sina*sina) + (k1*k1)*r*(A12*(L*L)*cosa + (pi*pi)*B11*(i1*i1)*r))) - 2*pi*sini1xb*sink1xb*(B22*(L*L)*(i1*i1)*(sina*sina) + (k1*k1)*r*(A12*(L*L)*cosa + (pi*pi)*B11*(i1*i1)*r)))/((L*L)*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += (-2*L*cosk1xb*k1*sina*sini1xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i1*i1)*(D16 + D26)) + (pi*pi)*(i1*i1)*(r*r)*(B16 + B26)) + 2*cosi1xa*i1*(-L*sina*sink1xa*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(i1*i1) + D26*(k1*k1))) + (pi*pi)*(r*r)*(B16*(i1*i1) + B26*(k1*k1))) + pi*cosk1xa*k1*(-(L*L)*(sina*sina)*(B26*r + D26*cosa) + r*(B26*(L*L)*(cosa*cosa) + r*(A26*(L*L)*cosa + (pi*pi)*(i1*i1)*(B16*r + D16*cosa))))) + 2*cosi1xb*i1*(L*sina*sink1xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(i1*i1) + D26*(k1*k1))) + (pi*pi)*(r*r)*(B16*(i1*i1) + B26*(k1*k1))) - pi*cosk1xb*k1*(-(L*L)*(sina*sina)*(B26*r + D26*cosa) + r*(B26*(L*L)*(cosa*cosa) + r*(A26*(L*L)*cosa + (pi*pi)*(i1*i1)*(B16*r + D16*cosa))))) + 2*sini1xa*(L*cosk1xa*k1*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i1*i1)*(D16 + D26)) + (pi*pi)*(i1*i1)*(r*r)*(B16 + B26)) + pi*sink1xa*(-(L*L)*(i1*i1)*(sina*sina)*(B26*r + D26*cosa) + (k1*k1)*r*(B26*(L*L)*(cosa*cosa) + r*(A26*(L*L)*cosa + (pi*pi)*(i1*i1)*(B16*r + D16*cosa))))) - 2*pi*sini1xb*sink1xb*(-(L*L)*(i1*i1)*(sina*sina)*(B26*r + D26*cosa) + (k1*k1)*r*(B26*(L*L)*(cosa*cosa) + r*(A26*(L*L)*cosa + (pi*pi)*(i1*i1)*(B16*r + D16*cosa)))))/((L*L)*(r*r)*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += (2*pi*B22*(L*L*L)*cosa*sina*sini1xb*sink1xb*(-(i1*i1) + (k1*k1)) + 2*cosi1xa*((pi*pi*pi)*D12*L*cosk1xa*i1*k1*r*sina*(-(i1*i1) + (k1*k1)) + i1*sink1xa*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i1*i1) + (k1*k1)) + (k1*k1)*((pi*pi)*D11*(i1*i1)*(r*r) + D22*(L*L)*(sina*sina))))) + 2*cosi1xb*((pi*pi*pi)*D12*L*cosk1xb*i1*k1*r*sina*(i1 - k1)*(i1 + k1) - i1*sink1xb*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i1*i1) + (k1*k1)) + (k1*k1)*((pi*pi)*D11*(i1*i1)*(r*r) + D22*(L*L)*(sina*sina))))) + 2*cosk1xb*k1*sini1xb*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i1*i1) + (k1*k1)) + (i1*i1)*((pi*pi)*D11*(k1*k1)*(r*r) + D22*(L*L)*(sina*sina)))) + 2*sini1xa*(pi*B22*(L*L*L)*cosa*sina*sink1xa*(i1 - k1)*(i1 + k1) - cosk1xa*k1*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i1*i1) + (k1*k1)) + (i1*i1)*((pi*pi)*D11*(k1*k1)*(r*r) + D22*(L*L)*(sina*sina))))))/((L*L*L)*r*(i1 - k1)*(i1 + k1))

            for i2 in range(i0, m2+i0):
                cos2i2xa = cos(2*pi*i2*xa/L)
                cos2i2xb = cos(2*pi*i2*xb/L)
                cosi2xa = cos(pi*i2*xa/L)
                cosi2xaxb = cos(pi*i2*(xa + xb)/L)
                sin2i2xa = sin(2*pi*i2*xa/L)
                sin2i2xb = sin(2*pi*i2*xb/L)
                sini2xa = sin(pi*i2*xa/L)
                sini2xa_xb = sin(pi*i2*(xa - xb)/L)
                sini2xaxb = sin(pi*i2*(xa + xb)/L)
                cosi2xb = cos(pi*i2*xb/L)
                sini2xb = sin(pi*i2*xb/L)
                for k2 in range(i0, m2+i0):
                    cosk2xa = cos(pi*k2*xa/L)
                    cosk2xb = cos(pi*k2*xb/L)
                    sink2xa = sin(pi*k2*xa/L)
                    sink2xb = sin(pi*k2*xb/L)
                    for j2 in range(j0, n2+j0):
                        row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                        for l2 in range(j0, n2+j0):
                            col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                            if k2==i2 and l2==j2:
                                if i2!=0:
                                    # k0_22 cond_1
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*A12*L*i2*r*sina*sini2xaxb - cosi2xaxb*(-(pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) - 2*pi*i2*(xa - xb)*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+2
                                    k0v[c] += -0.25*pi*(2*pi*i2*r*sina*(xa - xb)*(cosa*(B16 + B26) + r*(A16 + A26)) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(-A16*r + A26*r - B16*cosa + B26*cosa) + sini2xaxb*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) - r*(A26*(L*L)*(-(j2*j2) + (sina*sina)) + (pi*pi)*(i2*i2)*r*(A16*r + B16*cosa))))/(pi*i2))/(L*(r*r))
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*pi*j2*(2*L*sini2xa_xb*(L*sina*sini2xaxb*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*cosi2xaxb*i2*r*(A12*r - A66*r + B12*cosa - B66*cosa))/(pi*i2) + 2*pi*i2*r*(-xa + xb)*(cosa*(B12 + B66) + r*(A12 + A66)))/(L*(r*r))
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(-L*sina*sini2xaxb*(A22*(L*L)*cosa*r + 2*(pi*pi)*B12*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(B22 + B66)) - pi*cosi2xaxb*i2*r*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) - B22*(L*L)*(sina*sina) - 2*B66*(L*L)*(j2*j2))) + 2*(pi*pi)*(i2*i2)*r*(xa - xb)*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2)))/((L*L*L)*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*j2*((pi*pi)*(i2*i2)*r*sina*(-2*B16 - 2*B26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(B16 - B26) + sini2xaxb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(-A26*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/(L*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*A12*L*i2*r*sina*sini2xaxb - cosi2xaxb*(-(pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) - 2*pi*i2*(xa - xb)*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*pi*j2*(2*L*sini2xa_xb*(-L*sina*sini2xaxb*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*cosi2xaxb*i2*r*(-A12*r + A66*r - B12*cosa + B66*cosa))/(pi*i2) + 2*pi*i2*r*(xa - xb)*(cosa*(B12 + B66) + r*(A12 + A66)))/(L*(r*r))
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+3
                                    k0v[c] += -0.25*pi*(2*pi*i2*r*sina*(xa - xb)*(cosa*(B16 + B26) + r*(A16 + A26)) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(-A16*r + A26*r - B16*cosa + B26*cosa) + sini2xaxb*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) - r*(A26*(L*L)*(-(j2*j2) + (sina*sina)) + (pi*pi)*(i2*i2)*r*(A16*r + B16*cosa))))/(pi*i2))/(L*(r*r))
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*j2*((pi*pi)*(i2*i2)*r*sina*(2*B16 + 2*B26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(-B16 + B26) + sini2xaxb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa - (pi*pi)*B16*(i2*i2)*r))))/(L*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(-L*sina*sini2xaxb*(A22*(L*L)*cosa*r + 2*(pi*pi)*B12*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(B22 + B66)) - pi*cosi2xaxb*i2*r*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) - B22*(L*L)*(sina*sina) - 2*B66*(L*L)*(j2*j2))) + 2*(pi*pi)*(i2*i2)*r*(xa - xb)*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2)))/((L*L*L)*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+0
                                    k0v[c] += -0.25*pi*(2*pi*i2*r*sina*(xa - xb)*(cosa*(B16 + B26) + r*(A16 + A26)) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(-A16*r + A26*r - B16*cosa + B26*cosa) + sini2xaxb*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) - r*(A26*(L*L)*(-(j2*j2) + (sina*sina)) + (pi*pi)*(i2*i2)*r*(A16*r + B16*cosa))))/(pi*i2))/(L*(r*r))
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*pi*j2*(2*L*sini2xa_xb*(-L*sina*sini2xaxb*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*cosi2xaxb*i2*r*(-A12*r + A66*r - B12*cosa + B66*cosa))/(pi*i2) + 2*pi*i2*r*(xa - xb)*(cosa*(B12 + B66) + r*(A12 + A66)))/(L*(r*r))
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+2
                                    k0v[c] += pi*(0.25*(L*L)*(-2*pi*cos2i2xa*i2*r*sina*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa)) + 2*pi*cos2i2xb*i2*r*sina*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa)) - (L*(-sin2i2xa + sin2i2xb) + 2*pi*i2*(xa - xb))*((j2*j2)*(A22*(r*r) + 2*B22*cosa*r + D22*(cosa*cosa)) + (sina*sina)*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa))))/(pi*i2) + 0.25*pi*i2*(r*r)*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa))*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(-xa + xb)))/((L*L)*(r*r*r))
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*(2*pi*i2*sina*(xa - xb)*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 + D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 + B26)) + 2*sini2xa_xb*(-L*cosi2xaxb*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 - D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 - B26)) - pi*i2*sini2xaxb*((pi*pi)*B16*(i2*i2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + B26*(L*L)*r*(-(j2*j2) + (sina*sina)) + cosa*(D26*(L*L)*(-(j2*j2) + (sina*sina)) + (r*r)*(A26*(L*L) + (pi*pi)*D16*(i2*i2))))))/((L*L)*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*j2*(2*L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(cosa*(D22 + 3*D66) + r*(B22 + 3*B66)) + cosi2xaxb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 - 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 - 2*B66)))) - 2*pi*i2*(xa - xb)*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 + 2*B66))))/((L*L)*i2*(r*r*r))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*pi*j2*(2*L*sini2xa_xb*(L*sina*sini2xaxb*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*cosi2xaxb*i2*r*(A12*r - A66*r + B12*cosa - B66*cosa))/(pi*i2) + 2*pi*i2*r*(-xa + xb)*(cosa*(B12 + B66) + r*(A12 + A66)))/(L*(r*r))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+1
                                    k0v[c] += -0.25*pi*(2*pi*i2*r*sina*(xa - xb)*(cosa*(B16 + B26) + r*(A16 + A26)) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(-A16*r + A26*r - B16*cosa + B26*cosa) + sini2xaxb*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) - r*(A26*(L*L)*(-(j2*j2) + (sina*sina)) + (pi*pi)*(i2*i2)*r*(A16*r + B16*cosa))))/(pi*i2))/(L*(r*r))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+3
                                    k0v[c] += pi*(0.25*(L*L)*(-2*pi*cos2i2xa*i2*r*sina*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa)) + 2*pi*cos2i2xb*i2*r*sina*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa)) - (L*(-sin2i2xa + sin2i2xb) + 2*pi*i2*(xa - xb))*((j2*j2)*(A22*(r*r) + 2*B22*cosa*r + D22*(cosa*cosa)) + (sina*sina)*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa))))/(pi*i2) + 0.25*pi*i2*(r*r)*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa))*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(-xa + xb)))/((L*L)*(r*r*r))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*j2*(2*L*sini2xa_xb*(-pi*L*i2*r*sina*sini2xaxb*(cosa*(D22 + 3*D66) + r*(B22 + 3*B66)) - cosi2xaxb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 - 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 - 2*B66)))) + 2*pi*i2*(xa - xb)*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 + 2*B66))))/((L*L)*i2*(r*r*r))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*(2*pi*i2*sina*(xa - xb)*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 + D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 + B26)) + 2*sini2xa_xb*(-L*cosi2xaxb*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 - D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 - B26)) - pi*i2*sini2xaxb*((pi*pi)*B16*(i2*i2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + B26*(L*L)*r*(-(j2*j2) + (sina*sina)) + cosa*(D26*(L*L)*(-(j2*j2) + (sina*sina)) + (r*r)*(A26*(L*L) + (pi*pi)*D16*(i2*i2))))))/((L*L)*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(-L*sina*sini2xaxb*(A22*(L*L)*cosa*r + 2*(pi*pi)*B12*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(B22 + B66)) - pi*cosi2xaxb*i2*r*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) - B22*(L*L)*(sina*sina) - 2*B66*(L*L)*(j2*j2))) + 2*(pi*pi)*(i2*i2)*r*(xa - xb)*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2)))/((L*L*L)*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*j2*((pi*pi)*(i2*i2)*r*sina*(2*B16 + 2*B26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(-B16 + B26) + sini2xaxb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa - (pi*pi)*B16*(i2*i2)*r))))/(L*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*(2*pi*i2*sina*(xa - xb)*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 + D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 + B26)) + 2*sini2xa_xb*(-L*cosi2xaxb*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 - D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 - B26)) - pi*i2*sini2xaxb*((pi*pi)*B16*(i2*i2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + B26*(L*L)*r*(-(j2*j2) + (sina*sina)) + cosa*(D26*(L*L)*(-(j2*j2) + (sina*sina)) + (r*r)*(A26*(L*L) + (pi*pi)*D16*(i2*i2))))))/((L*L)*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*j2*(2*L*sini2xa_xb*(-pi*L*i2*r*sina*sini2xaxb*(cosa*(D22 + 3*D66) + r*(B22 + 3*B66)) - cosi2xaxb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 - 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 - 2*B66)))) + 2*pi*i2*(xa - xb)*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 + 2*B66))))/((L*L)*i2*(r*r*r))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*L*i2*r*sina*sini2xaxb*(B22*(L*L)*cosa*r + (pi*pi)*D12*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(D22 + 2*D66)) + cosi2xaxb*(D22*((L*L*L*L)*(j2*j2*j2*j2) - (pi*pi)*(L*L)*(i2*i2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(i2*i2)*(2*B12*(L*L)*cosa*r + (pi*pi)*D11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(2*D12 - 4*D66)))))) - 2*pi*i2*(xa - xb)*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(i2*i2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(i2*i2)*(2*B12*(L*L)*cosa*r + (pi*pi)*D11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(2*D12 + 4*D66))))))/((L*L*L*L)*i2*(r*r*r))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*j2*((pi*pi)*(i2*i2)*r*sina*(-2*B16 - 2*B26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(B16 - B26) + sini2xaxb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(-A26*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/(L*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(-L*sina*sini2xaxb*(A22*(L*L)*cosa*r + 2*(pi*pi)*B12*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(B22 + B66)) - pi*cosi2xaxb*i2*r*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) - B22*(L*L)*(sina*sina) - 2*B66*(L*L)*(j2*j2))) + 2*(pi*pi)*(i2*i2)*r*(xa - xb)*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2)))/((L*L*L)*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*j2*(2*L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(cosa*(D22 + 3*D66) + r*(B22 + 3*B66)) + cosi2xaxb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 - 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 - 2*B66)))) - 2*pi*i2*(xa - xb)*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 + 2*B66))))/((L*L)*i2*(r*r*r))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*(2*pi*i2*sina*(xa - xb)*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 + D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 + B26)) + 2*sini2xa_xb*(-L*cosi2xaxb*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 - D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 - B26)) - pi*i2*sini2xaxb*((pi*pi)*B16*(i2*i2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + B26*(L*L)*r*(-(j2*j2) + (sina*sina)) + cosa*(D26*(L*L)*(-(j2*j2) + (sina*sina)) + (r*r)*(A26*(L*L) + (pi*pi)*D16*(i2*i2))))))/((L*L)*i2*(r*r))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*L*i2*r*sina*sini2xaxb*(B22*(L*L)*cosa*r + (pi*pi)*D12*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(D22 + 2*D66)) + cosi2xaxb*(D22*((L*L*L*L)*(j2*j2*j2*j2) - (pi*pi)*(L*L)*(i2*i2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(i2*i2)*(2*B12*(L*L)*cosa*r + (pi*pi)*D11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(2*D12 - 4*D66)))))) - 2*pi*i2*(xa - xb)*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(i2*i2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(i2*i2)*(2*B12*(L*L)*cosa*r + (pi*pi)*D11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(2*D12 + 4*D66))))))/((L*L*L*L)*i2*(r*r*r))

                                else:
                                    # k0_22 cond_5
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+0
                                    k0v[c] += pi*(-xa + xb)*(A22*(sina*sina) + A66*(j2*j2))/r
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+1
                                    k0v[c] += pi*(-xa + xb)*(A22*(sina*sina) + A66*(j2*j2))/r

                            elif k2!=i2 and l2==j2:
                                # k0_22 cond_2
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += (cosi2xa*(pi*A12*L*cosk2xa*r*sina*(-(i2*i2) + (k2*k2)) + k2*sink2xa*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) + cosi2xb*(pi*A12*L*cosk2xb*r*sina*(i2 - k2)*(i2 + k2) - k2*sink2xb*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) + i2*(-cosk2xa*sini2xa + cosk2xb*sini2xb)*((pi*pi)*A11*(k2*k2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+1
                                k0v[c] += pi*A16*j2*(cosi2xa*cosk2xa*((i2*i2) + (k2*k2)) - cosi2xb*cosk2xb*((i2*i2) + (k2*k2)) + 2*i2*k2*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += pi*(L*(A26*r + B26*cosa)*(cosi2xa*k2*(L*cosk2xa*(-(j2*j2) + (sina*sina)) + pi*k2*r*sina*sink2xa) - cosi2xb*k2*(L*cosk2xb*(-(j2*j2) + (sina*sina)) + pi*k2*r*sina*sink2xb) + i2*(sini2xa*(L*sink2xa*(-(j2*j2) + (sina*sina)) - pi*cosk2xa*k2*r*sina) + sini2xb*(L*sink2xb*(j2 - sina)*(j2 + sina) + pi*cosk2xb*k2*r*sina)))/pi - i2*r*(A16*r + B16*cosa)*(cosi2xa*i2*(-L*sina*sink2xa + pi*cosk2xa*k2*r) + cosi2xb*i2*(L*sina*sink2xb - pi*cosk2xb*k2*r) + k2*sini2xa*(L*cosk2xa*sina + pi*k2*r*sink2xa) - k2*sini2xb*(L*cosk2xb*sina + pi*k2*r*sink2xb)))/(L*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += j2*(cosi2xa*(L*cosk2xa*k2*sina*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*r*sink2xa*(A12*(i2*i2)*r + B12*cosa*(i2*i2) + (k2*k2)*(A66*r + B66*cosa))) + cosi2xb*(-L*cosk2xb*k2*sina*(cosa*(B22 + B66) + r*(A22 + A66)) - pi*r*sink2xb*(A12*(i2*i2)*r + B12*cosa*(i2*i2) + (k2*k2)*(A66*r + B66*cosa))) + i2*(-L*sina*sini2xb*sink2xb*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*cosk2xb*k2*r*sini2xb*(cosa*(B12 + B66) + r*(A12 + A66)) + sini2xa*(L*sina*sink2xa*(cosa*(B22 + B66) + r*(A22 + A66)) - pi*cosk2xa*k2*r*(cosa*(B12 + B66) + r*(A12 + A66)))))/((r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += (-cosi2xa*(L*cosk2xa*k2*sina*(A22*(L*L)*cosa*r + (pi*pi)*B12*(r*r)*(-(i2*i2) + (k2*k2)) + (L*L)*(j2*j2)*(B22 + B66)) + pi*r*sink2xa*(B12*(L*L)*(i2*i2)*(j2*j2) + B22*(L*L)*(k2*k2)*(sina*sina) + 2*B66*(L*L)*(j2*j2)*(k2*k2) + (i2*i2)*r*(A12*(L*L)*cosa + (pi*pi)*B11*(k2*k2)*r))) + cosi2xb*(L*cosk2xb*k2*sina*(A22*(L*L)*cosa*r + (pi*pi)*B12*(r*r)*(-(i2*i2) + (k2*k2)) + (L*L)*(j2*j2)*(B22 + B66)) + pi*r*sink2xb*(B12*(L*L)*(i2*i2)*(j2*j2) + B22*(L*L)*(k2*k2)*(sina*sina) + 2*B66*(L*L)*(j2*j2)*(k2*k2) + (i2*i2)*r*(A12*(L*L)*cosa + (pi*pi)*B11*(k2*k2)*r))) + i2*(-(L*L*L)*sina*sini2xa*sink2xa*(A22*cosa*r + (j2*j2)*(B22 + B66)) + pi*cosk2xa*k2*r*sini2xa*(A12*(L*L)*cosa*r + (pi*pi)*B11*(k2*k2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2)) + sini2xb*((L*L*L)*sina*sink2xb*(A22*cosa*r + (j2*j2)*(B22 + B66)) - pi*cosk2xb*k2*r*(A12*(L*L)*cosa*r + (pi*pi)*B11*(k2*k2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2)))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+5
                                k0v[c] += j2*(cosi2xa*(pi*L*r*sina*sink2xa*(B16*(i2*i2) + B26*(k2*k2)) - cosk2xa*k2*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + (pi*pi)*B16*r*(2*(i2*i2) + (k2*k2))))) + cosi2xb*(-pi*L*r*sina*sink2xb*(B16*(i2*i2) + B26*(k2*k2)) + cosk2xb*k2*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + (pi*pi)*B16*r*(2*(i2*i2) + (k2*k2))))) + i2*(pi*L*cosk2xb*k2*r*sina*sini2xb*(B16 + B26) + sini2xa*(-pi*L*cosk2xa*k2*r*sina*(B16 + B26) - sink2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + 3*(pi*pi)*B16*(k2*k2)*r))) + sini2xb*sink2xb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + 3*(pi*pi)*B16*(k2*k2)*r))))/(L*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+0
                                k0v[c] += -pi*A16*j2*(cosi2xa*cosk2xa*((i2*i2) + (k2*k2)) - cosi2xb*cosk2xb*((i2*i2) + (k2*k2)) + 2*i2*k2*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += (cosi2xa*(pi*A12*L*cosk2xa*r*sina*(-(i2*i2) + (k2*k2)) + k2*sink2xa*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) + cosi2xb*(pi*A12*L*cosk2xb*r*sina*(i2 - k2)*(i2 + k2) - k2*sink2xb*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) + i2*(-cosk2xa*sini2xa + cosk2xb*sini2xb)*((pi*pi)*A11*(k2*k2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += j2*(cosi2xa*(-L*cosk2xa*k2*sina*(cosa*(B22 + B66) + r*(A22 + A66)) - pi*r*sink2xa*(A12*(i2*i2)*r + B12*cosa*(i2*i2) + (k2*k2)*(A66*r + B66*cosa))) + cosi2xb*(L*cosk2xb*k2*sina*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*r*sink2xb*(A12*(i2*i2)*r + B12*cosa*(i2*i2) + (k2*k2)*(A66*r + B66*cosa))) + i2*(L*sina*sini2xb*sink2xb*(cosa*(B22 + B66) + r*(A22 + A66)) - pi*cosk2xb*k2*r*sini2xb*(cosa*(B12 + B66) + r*(A12 + A66)) + sini2xa*(-L*sina*sink2xa*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*cosk2xa*k2*r*(cosa*(B12 + B66) + r*(A12 + A66)))))/((r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+3
                                k0v[c] += pi*(L*(A26*r + B26*cosa)*(cosi2xa*k2*(L*cosk2xa*(-(j2*j2) + (sina*sina)) + pi*k2*r*sina*sink2xa) - cosi2xb*k2*(L*cosk2xb*(-(j2*j2) + (sina*sina)) + pi*k2*r*sina*sink2xb) + i2*(sini2xa*(L*sink2xa*(-(j2*j2) + (sina*sina)) - pi*cosk2xa*k2*r*sina) + sini2xb*(L*sink2xb*(j2 - sina)*(j2 + sina) + pi*cosk2xb*k2*r*sina)))/pi - i2*r*(A16*r + B16*cosa)*(cosi2xa*i2*(-L*sina*sink2xa + pi*cosk2xa*k2*r) + cosi2xb*i2*(L*sina*sink2xb - pi*cosk2xb*k2*r) + k2*sini2xa*(L*cosk2xa*sina + pi*k2*r*sink2xa) - k2*sini2xb*(L*cosk2xb*sina + pi*k2*r*sink2xb)))/(L*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+4
                                k0v[c] += j2*(cosi2xa*(-pi*L*r*sina*sink2xa*(B16*(i2*i2) + B26*(k2*k2)) + cosk2xa*k2*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + (pi*pi)*B16*r*(2*(i2*i2) + (k2*k2))))) + cosi2xb*(pi*L*r*sina*sink2xb*(B16*(i2*i2) + B26*(k2*k2)) - cosk2xb*k2*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + (pi*pi)*B16*r*(2*(i2*i2) + (k2*k2))))) + i2*(pi*L*cosk2xa*k2*r*sina*sini2xa*(B16 + B26) + sini2xa*sink2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + 3*(pi*pi)*B16*(k2*k2)*r)) - sini2xb*(pi*L*cosk2xb*k2*r*sina*(B16 + B26) + sink2xb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + 3*(pi*pi)*B16*(k2*k2)*r)))))/(L*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+5
                                k0v[c] += (-cosi2xa*(L*cosk2xa*k2*sina*(A22*(L*L)*cosa*r + (pi*pi)*B12*(r*r)*(-(i2*i2) + (k2*k2)) + (L*L)*(j2*j2)*(B22 + B66)) + pi*r*sink2xa*(B12*(L*L)*(i2*i2)*(j2*j2) + B22*(L*L)*(k2*k2)*(sina*sina) + 2*B66*(L*L)*(j2*j2)*(k2*k2) + (i2*i2)*r*(A12*(L*L)*cosa + (pi*pi)*B11*(k2*k2)*r))) + cosi2xb*(L*cosk2xb*k2*sina*(A22*(L*L)*cosa*r + (pi*pi)*B12*(r*r)*(-(i2*i2) + (k2*k2)) + (L*L)*(j2*j2)*(B22 + B66)) + pi*r*sink2xb*(B12*(L*L)*(i2*i2)*(j2*j2) + B22*(L*L)*(k2*k2)*(sina*sina) + 2*B66*(L*L)*(j2*j2)*(k2*k2) + (i2*i2)*r*(A12*(L*L)*cosa + (pi*pi)*B11*(k2*k2)*r))) + i2*(-(L*L*L)*sina*sini2xa*sink2xa*(A22*cosa*r + (j2*j2)*(B22 + B66)) + pi*cosk2xa*k2*r*sini2xa*(A12*(L*L)*cosa*r + (pi*pi)*B11*(k2*k2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2)) + sini2xb*((L*L*L)*sina*sink2xb*(A22*cosa*r + (j2*j2)*(B22 + B66)) - pi*cosk2xb*k2*r*(A12*(L*L)*cosa*r + (pi*pi)*B11*(k2*k2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2)))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += (pi*L*cosk2xb*r*sina*sini2xb*(A26*(i2*i2)*r + B26*cosa*(i2*i2) + (k2*k2)*(A16*r + B16*cosa)) + cosi2xa*i2*(pi*L*k2*r*sina*sink2xa*(cosa*(B16 + B26) + r*(A16 + A26)) + cosk2xa*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*(j2 - sina)*(j2 + sina) + (pi*pi)*(k2*k2)*r*(A16*r + B16*cosa)))) + cosi2xb*i2*(-pi*L*k2*r*sina*sink2xb*(cosa*(B16 + B26) + r*(A16 + A26)) - cosk2xb*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*(j2 - sina)*(j2 + sina) + (pi*pi)*(k2*k2)*r*(A16*r + B16*cosa)))) - k2*sini2xb*sink2xb*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*(j2 - sina)*(j2 + sina) + (pi*pi)*(i2*i2)*r*(A16*r + B16*cosa))) + sini2xa*(-pi*L*cosk2xa*r*sina*(A26*(i2*i2)*r + B26*cosa*(i2*i2) + (k2*k2)*(A16*r + B16*cosa)) + k2*sink2xa*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*(j2 - sina)*(j2 + sina) + (pi*pi)*(i2*i2)*r*(A16*r + B16*cosa)))))/(L*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += j2*(-L*k2*sina*sini2xb*sink2xb*(cosa*(B22 + B66) + r*(A22 + A66)) + cosi2xa*i2*(L*cosk2xa*sina*(cosa*(B22 + B66) + r*(A22 + A66)) - pi*k2*r*sink2xa*(cosa*(B12 + B66) + r*(A12 + A66))) + cosi2xb*i2*(-L*cosk2xb*sina*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*k2*r*sink2xb*(cosa*(B12 + B66) + r*(A12 + A66))) - pi*cosk2xb*r*sini2xb*(A66*(i2*i2)*r + B66*cosa*(i2*i2) + (k2*k2)*(A12*r + B12*cosa)) + sini2xa*(L*k2*sina*sink2xa*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*cosk2xa*r*(A66*(i2*i2)*r + B66*cosa*(i2*i2) + (k2*k2)*(A12*r + B12*cosa))))/((r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += (-cosk2xa*k2*sini2xa*((cosa*cosa)*((pi*pi)*D66*(i2*i2)*(r*r) + (L*L)*(D22*(j2*j2) + D66*(sina*sina))) + 2*cosa*r*((pi*pi)*B66*(i2*i2)*(r*r) + (L*L)*(B22*(j2*j2) + B66*(sina*sina))) + (r*r)*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A66*(sina*sina)))) + cosk2xb*k2*sini2xb*((cosa*cosa)*((pi*pi)*D66*(i2*i2)*(r*r) + (L*L)*(D22*(j2*j2) + D66*(sina*sina))) + 2*cosa*r*((pi*pi)*B66*(i2*i2)*(r*r) + (L*L)*(B22*(j2*j2) + B66*(sina*sina))) + (r*r)*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A66*(sina*sina)))) + sink2xa*(pi*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2)*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa)) + cosi2xa*i2*((cosa*cosa)*((pi*pi)*D66*(k2*k2)*(r*r) + (L*L)*(D22*(j2*j2) + D66*(sina*sina))) + 2*cosa*r*((pi*pi)*B66*(k2*k2)*(r*r) + (L*L)*(B22*(j2*j2) + B66*(sina*sina))) + (r*r)*((pi*pi)*A66*(k2*k2)*(r*r) + (L*L)*(A22*(j2*j2) + A66*(sina*sina))))) - sink2xb*(pi*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2)*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa)) + cosi2xb*i2*((cosa*cosa)*((pi*pi)*D66*(k2*k2)*(r*r) + (L*L)*(D22*(j2*j2) + D66*(sina*sina))) + 2*cosa*r*((pi*pi)*B66*(k2*k2)*(r*r) + (L*L)*(B22*(j2*j2) + B66*(sina*sina))) + (r*r)*((pi*pi)*A66*(k2*k2)*(r*r) + (L*L)*(A22*(j2*j2) + A66*(sina*sina))))))/(L*(r*r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+3
                                k0v[c] += pi*j2*(A26*(r*r) + 2*B26*cosa*r + D26*(cosa*cosa))*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((r*r)*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+4
                                k0v[c] += (-L*cosk2xb*k2*sina*sini2xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(k2*k2) + D26*(i2*i2))) + (pi*pi)*(r*r)*(B16*(k2*k2) + B26*(i2*i2))) + cosi2xa*i2*(-L*sina*sink2xa*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(k2*k2)*(D16 + D26)) + (pi*pi)*(k2*k2)*(r*r)*(B16 + B26)) - pi*cosk2xa*k2*(A26*(L*L)*cosa*(r*r) + (pi*pi)*B16*(k2*k2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + 3*B26*(L*L)*(j2*j2)*r + (pi*pi)*D16*cosa*(k2*k2)*(r*r) + 3*D26*(L*L)*cosa*(j2*j2) - (L*L)*(sina*sina)*(B26*r + D26*cosa))) + cosi2xb*i2*(L*sina*sink2xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(k2*k2)*(D16 + D26)) + (pi*pi)*(k2*k2)*(r*r)*(B16 + B26)) + pi*cosk2xb*k2*(A26*(L*L)*cosa*(r*r) + (pi*pi)*B16*(k2*k2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + 3*B26*(L*L)*(j2*j2)*r + (pi*pi)*D16*cosa*(k2*k2)*(r*r) + 3*D26*(L*L)*cosa*(j2*j2) - (L*L)*(sina*sina)*(B26*r + D26*cosa))) + sini2xa*(L*cosk2xa*k2*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(k2*k2) + D26*(i2*i2))) + (pi*pi)*(r*r)*(B16*(k2*k2) + B26*(i2*i2))) - pi*sink2xa*(B26*(L*L)*(cosa*cosa)*(i2*i2)*r + cosa*(D26*(L*L)*((j2*j2)*((i2*i2) + 2*(k2*k2)) - (k2*k2)*(sina*sina)) + (i2*i2)*(r*r)*(A26*(L*L) + (pi*pi)*D16*(k2*k2))) + r*((pi*pi)*B16*(i2*i2)*(k2*k2)*(r*r) + B26*(L*L)*((j2*j2)*((i2*i2) + 2*(k2*k2)) - (k2*k2)*(sina*sina))))) + pi*sini2xb*sink2xb*(B26*(L*L)*(cosa*cosa)*(i2*i2)*r + cosa*(D26*(L*L)*((j2*j2)*((i2*i2) + 2*(k2*k2)) - (k2*k2)*(sina*sina)) + (i2*i2)*(r*r)*(A26*(L*L) + (pi*pi)*D16*(k2*k2))) + r*((pi*pi)*B16*(i2*i2)*(k2*k2)*(r*r) + B26*(L*L)*((j2*j2)*((i2*i2) + 2*(k2*k2)) - (k2*k2)*(sina*sina)))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+5
                                k0v[c] += j2*(pi*L*r*sina*sini2xb*sink2xb*(-D66*cosa*(i2*i2) + cosa*(k2*k2)*(D22 + 2*D66) + r*(-B66*(i2*i2) + (k2*k2)*(B22 + 2*B66))) + cosi2xa*i2*(-pi*L*cosk2xa*k2*r*sina*(cosa*(D22 + D66) + r*(B22 + B66)) + sink2xa*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(k2*k2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(k2*k2)*(r*r)*(B12 + 2*B66)))) + cosi2xb*i2*(pi*L*cosk2xb*k2*r*sina*(cosa*(D22 + D66) + r*(B22 + B66)) - sink2xb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(k2*k2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(k2*k2)*(r*r)*(B12 + 2*B66)))) + cosk2xb*k2*sini2xb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(D12*(k2*k2) + 2*D66*(i2*i2)))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(r*r)*(B12*(k2*k2) + 2*B66*(i2*i2)))) + sini2xa*(pi*L*r*sina*sink2xa*(D66*cosa*(i2*i2) - cosa*(k2*k2)*(D22 + 2*D66) + r*(B66*(i2*i2) - (k2*k2)*(B22 + 2*B66))) - cosk2xa*k2*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(D12*(k2*k2) + 2*D66*(i2*i2)))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(r*r)*(B12*(k2*k2) + 2*B66*(i2*i2))))))/(L*(r*r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += j2*(L*k2*sina*sini2xb*sink2xb*(cosa*(B22 + B66) + r*(A22 + A66)) + cosi2xa*i2*(-L*cosk2xa*sina*(cosa*(B22 + B66) + r*(A22 + A66)) + pi*k2*r*sink2xa*(cosa*(B12 + B66) + r*(A12 + A66))) + cosi2xb*i2*(L*cosk2xb*sina*(cosa*(B22 + B66) + r*(A22 + A66)) - pi*k2*r*sink2xb*(cosa*(B12 + B66) + r*(A12 + A66))) + pi*cosk2xb*r*sini2xb*(A66*(i2*i2)*r + B66*cosa*(i2*i2) + (k2*k2)*(A12*r + B12*cosa)) + sini2xa*(-L*k2*sina*sink2xa*(cosa*(B22 + B66) + r*(A22 + A66)) - pi*cosk2xa*r*(A66*(i2*i2)*r + B66*cosa*(i2*i2) + (k2*k2)*(A12*r + B12*cosa))))/((r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+1
                                k0v[c] += (pi*L*cosk2xb*r*sina*sini2xb*(A26*(i2*i2)*r + B26*cosa*(i2*i2) + (k2*k2)*(A16*r + B16*cosa)) + cosi2xa*i2*(pi*L*k2*r*sina*sink2xa*(cosa*(B16 + B26) + r*(A16 + A26)) + cosk2xa*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*(j2 - sina)*(j2 + sina) + (pi*pi)*(k2*k2)*r*(A16*r + B16*cosa)))) + cosi2xb*i2*(-pi*L*k2*r*sina*sink2xb*(cosa*(B16 + B26) + r*(A16 + A26)) - cosk2xb*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*(j2 - sina)*(j2 + sina) + (pi*pi)*(k2*k2)*r*(A16*r + B16*cosa)))) - k2*sini2xb*sink2xb*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*(j2 - sina)*(j2 + sina) + (pi*pi)*(i2*i2)*r*(A16*r + B16*cosa))) + sini2xa*(-pi*L*cosk2xa*r*sina*(A26*(i2*i2)*r + B26*cosa*(i2*i2) + (k2*k2)*(A16*r + B16*cosa)) + k2*sink2xa*(B26*(L*L)*cosa*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*(j2 - sina)*(j2 + sina) + (pi*pi)*(i2*i2)*r*(A16*r + B16*cosa)))))/(L*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+2
                                k0v[c] += -pi*j2*(A26*(r*r) + 2*B26*cosa*r + D26*(cosa*cosa))*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((r*r)*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += (-cosk2xa*k2*sini2xa*((cosa*cosa)*((pi*pi)*D66*(i2*i2)*(r*r) + (L*L)*(D22*(j2*j2) + D66*(sina*sina))) + 2*cosa*r*((pi*pi)*B66*(i2*i2)*(r*r) + (L*L)*(B22*(j2*j2) + B66*(sina*sina))) + (r*r)*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A66*(sina*sina)))) + cosk2xb*k2*sini2xb*((cosa*cosa)*((pi*pi)*D66*(i2*i2)*(r*r) + (L*L)*(D22*(j2*j2) + D66*(sina*sina))) + 2*cosa*r*((pi*pi)*B66*(i2*i2)*(r*r) + (L*L)*(B22*(j2*j2) + B66*(sina*sina))) + (r*r)*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A66*(sina*sina)))) + sink2xa*(pi*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2)*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa)) + cosi2xa*i2*((cosa*cosa)*((pi*pi)*D66*(k2*k2)*(r*r) + (L*L)*(D22*(j2*j2) + D66*(sina*sina))) + 2*cosa*r*((pi*pi)*B66*(k2*k2)*(r*r) + (L*L)*(B22*(j2*j2) + B66*(sina*sina))) + (r*r)*((pi*pi)*A66*(k2*k2)*(r*r) + (L*L)*(A22*(j2*j2) + A66*(sina*sina))))) - sink2xb*(pi*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2)*(A66*(r*r) + 2*B66*cosa*r + D66*(cosa*cosa)) + cosi2xb*i2*((cosa*cosa)*((pi*pi)*D66*(k2*k2)*(r*r) + (L*L)*(D22*(j2*j2) + D66*(sina*sina))) + 2*cosa*r*((pi*pi)*B66*(k2*k2)*(r*r) + (L*L)*(B22*(j2*j2) + B66*(sina*sina))) + (r*r)*((pi*pi)*A66*(k2*k2)*(r*r) + (L*L)*(A22*(j2*j2) + A66*(sina*sina))))))/(L*(r*r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += j2*(pi*L*r*sina*sini2xb*sink2xb*(D66*cosa*(i2*i2) - cosa*(k2*k2)*(D22 + 2*D66) + r*(B66*(i2*i2) - (k2*k2)*(B22 + 2*B66))) + cosi2xa*i2*(pi*L*cosk2xa*k2*r*sina*(cosa*(D22 + D66) + r*(B22 + B66)) - sink2xa*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(k2*k2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(k2*k2)*(r*r)*(B12 + 2*B66)))) + cosi2xb*i2*(-pi*L*cosk2xb*k2*r*sina*(cosa*(D22 + D66) + r*(B22 + B66)) + sink2xb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(k2*k2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(k2*k2)*(r*r)*(B12 + 2*B66)))) - cosk2xb*k2*sini2xb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(D12*(k2*k2) + 2*D66*(i2*i2)))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(r*r)*(B12*(k2*k2) + 2*B66*(i2*i2)))) + sini2xa*(pi*L*r*sina*sink2xa*(-D66*cosa*(i2*i2) + cosa*(k2*k2)*(D22 + 2*D66) + r*(-B66*(i2*i2) + (k2*k2)*(B22 + 2*B66))) + cosk2xa*k2*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(D12*(k2*k2) + 2*D66*(i2*i2)))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(r*r)*(B12*(k2*k2) + 2*B66*(i2*i2))))))/(L*(r*r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+5
                                k0v[c] += (-L*cosk2xb*k2*sina*sini2xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(k2*k2) + D26*(i2*i2))) + (pi*pi)*(r*r)*(B16*(k2*k2) + B26*(i2*i2))) + cosi2xa*i2*(-L*sina*sink2xa*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(k2*k2)*(D16 + D26)) + (pi*pi)*(k2*k2)*(r*r)*(B16 + B26)) - pi*cosk2xa*k2*(A26*(L*L)*cosa*(r*r) + (pi*pi)*B16*(k2*k2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + 3*B26*(L*L)*(j2*j2)*r + (pi*pi)*D16*cosa*(k2*k2)*(r*r) + 3*D26*(L*L)*cosa*(j2*j2) - (L*L)*(sina*sina)*(B26*r + D26*cosa))) + cosi2xb*i2*(L*sina*sink2xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(k2*k2)*(D16 + D26)) + (pi*pi)*(k2*k2)*(r*r)*(B16 + B26)) + pi*cosk2xb*k2*(A26*(L*L)*cosa*(r*r) + (pi*pi)*B16*(k2*k2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + 3*B26*(L*L)*(j2*j2)*r + (pi*pi)*D16*cosa*(k2*k2)*(r*r) + 3*D26*(L*L)*cosa*(j2*j2) - (L*L)*(sina*sina)*(B26*r + D26*cosa))) + sini2xa*(L*cosk2xa*k2*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(k2*k2) + D26*(i2*i2))) + (pi*pi)*(r*r)*(B16*(k2*k2) + B26*(i2*i2))) - pi*sink2xa*(B26*(L*L)*(cosa*cosa)*(i2*i2)*r + cosa*(D26*(L*L)*((j2*j2)*((i2*i2) + 2*(k2*k2)) - (k2*k2)*(sina*sina)) + (i2*i2)*(r*r)*(A26*(L*L) + (pi*pi)*D16*(k2*k2))) + r*((pi*pi)*B16*(i2*i2)*(k2*k2)*(r*r) + B26*(L*L)*((j2*j2)*((i2*i2) + 2*(k2*k2)) - (k2*k2)*(sina*sina))))) + pi*sini2xb*sink2xb*(B26*(L*L)*(cosa*cosa)*(i2*i2)*r + cosa*(D26*(L*L)*((j2*j2)*((i2*i2) + 2*(k2*k2)) - (k2*k2)*(sina*sina)) + (i2*i2)*(r*r)*(A26*(L*L) + (pi*pi)*D16*(k2*k2))) + r*((pi*pi)*B16*(i2*i2)*(k2*k2)*(r*r) + B26*(L*L)*((j2*j2)*((i2*i2) + 2*(k2*k2)) - (k2*k2)*(sina*sina)))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += (-(L*L*L)*k2*sina*sini2xb*sink2xb*(A22*cosa*r + (j2*j2)*(B22 + B66)) + cosi2xa*i2*(L*cosk2xa*sina*(A22*(L*L)*cosa*r + (pi*pi)*B12*(r*r)*(i2 - k2)*(i2 + k2) + (L*L)*(j2*j2)*(B22 + B66)) - pi*k2*r*sink2xa*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2))) + cosi2xb*i2*(-L*cosk2xb*sina*(A22*(L*L)*cosa*r + (pi*pi)*B12*(r*r)*(i2 - k2)*(i2 + k2) + (L*L)*(j2*j2)*(B22 + B66)) + pi*k2*r*sink2xb*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2))) - pi*cosk2xb*r*sini2xb*(B22*(L*L)*(i2*i2)*(sina*sina) + 2*B66*(L*L)*(i2*i2)*(j2*j2) + (k2*k2)*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2))) + sini2xa*((L*L*L)*k2*sina*sink2xa*(A22*cosa*r + (j2*j2)*(B22 + B66)) + pi*cosk2xa*r*(B22*(L*L)*(i2*i2)*(sina*sina) + 2*B66*(L*L)*(i2*i2)*(j2*j2) + (k2*k2)*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2)))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+1
                                k0v[c] += j2*(-pi*L*cosk2xb*r*sina*sini2xb*(B16*(k2*k2) + B26*(i2*i2)) + cosi2xa*i2*(-pi*L*k2*r*sina*sink2xa*(B16 + B26) - cosk2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + (pi*pi)*B16*r*((i2*i2) + 2*(k2*k2))))) + cosi2xb*i2*(pi*L*k2*r*sina*sink2xb*(B16 + B26) + cosk2xb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + (pi*pi)*B16*r*((i2*i2) + 2*(k2*k2))))) + k2*sini2xb*sink2xb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + 3*(pi*pi)*B16*(i2*i2)*r)) + sini2xa*(pi*L*cosk2xa*r*sina*(B16*(k2*k2) + B26*(i2*i2)) - k2*sink2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + 3*(pi*pi)*B16*(i2*i2)*r))))/(L*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+2
                                k0v[c] += (-L*cosk2xb*k2*sina*sini2xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 + D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 + B26)) + cosi2xa*i2*(-L*sina*sink2xa*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(i2*i2) + D26*(k2*k2))) + (pi*pi)*(r*r)*(B16*(i2*i2) + B26*(k2*k2))) + pi*cosk2xa*k2*(A26*(L*L)*cosa*(r*r) + (pi*pi)*B16*(i2*i2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + 3*B26*(L*L)*(j2*j2)*r + (pi*pi)*D16*cosa*(i2*i2)*(r*r) + 3*D26*(L*L)*cosa*(j2*j2) - (L*L)*(sina*sina)*(B26*r + D26*cosa))) + cosi2xb*i2*(L*sina*sink2xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(i2*i2) + D26*(k2*k2))) + (pi*pi)*(r*r)*(B16*(i2*i2) + B26*(k2*k2))) - pi*cosk2xb*k2*(A26*(L*L)*cosa*(r*r) + (pi*pi)*B16*(i2*i2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + 3*B26*(L*L)*(j2*j2)*r + (pi*pi)*D16*cosa*(i2*i2)*(r*r) + 3*D26*(L*L)*cosa*(j2*j2) - (L*L)*(sina*sina)*(B26*r + D26*cosa))) + sini2xa*(L*cosk2xa*k2*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 + D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 + B26)) + pi*sink2xa*(B26*(L*L)*(cosa*cosa)*(k2*k2)*r + cosa*(D26*(L*L)*(-(i2*i2)*(sina*sina) + (j2*j2)*(2*(i2*i2) + (k2*k2))) + (k2*k2)*(r*r)*(A26*(L*L) + (pi*pi)*D16*(i2*i2))) + r*((pi*pi)*B16*(i2*i2)*(k2*k2)*(r*r) + B26*(L*L)*(-(i2*i2)*(sina*sina) + (j2*j2)*(2*(i2*i2) + (k2*k2)))))) - pi*sini2xb*sink2xb*(B26*(L*L)*(cosa*cosa)*(k2*k2)*r + cosa*(D26*(L*L)*(-(i2*i2)*(sina*sina) + (j2*j2)*(2*(i2*i2) + (k2*k2))) + (k2*k2)*(r*r)*(A26*(L*L) + (pi*pi)*D16*(i2*i2))) + r*((pi*pi)*B16*(i2*i2)*(k2*k2)*(r*r) + B26*(L*L)*(-(i2*i2)*(sina*sina) + (j2*j2)*(2*(i2*i2) + (k2*k2))))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += j2*(pi*L*r*sina*sini2xb*sink2xb*(-D66*cosa*(k2*k2) + cosa*(i2*i2)*(D22 + 2*D66) + r*(-B66*(k2*k2) + (i2*i2)*(B22 + 2*B66))) + cosi2xa*i2*(-pi*L*cosk2xa*k2*r*sina*(cosa*(D22 + D66) + r*(B22 + B66)) - sink2xa*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(D12*(i2*i2) + 2*D66*(k2*k2)))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(r*r)*(B12*(i2*i2) + 2*B66*(k2*k2))))) + cosi2xb*i2*(pi*L*cosk2xb*k2*r*sina*(cosa*(D22 + D66) + r*(B22 + B66)) + sink2xb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(D12*(i2*i2) + 2*D66*(k2*k2)))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(r*r)*(B12*(i2*i2) + 2*B66*(k2*k2))))) - cosk2xb*k2*sini2xb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 + 2*B66))) + sini2xa*(-pi*L*r*sina*sink2xa*(-D66*cosa*(k2*k2) + cosa*(i2*i2)*(D22 + 2*D66) + r*(-B66*(k2*k2) + (i2*i2)*(B22 + 2*B66))) + cosk2xa*k2*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 + 2*B66)))))/(L*(r*r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += (pi*(L*L*L)*r*sina*sini2xb*sink2xb*(-(i2*i2) + (k2*k2))*(B22*cosa*r + (j2*j2)*(D22 + 2*D66)) + cosi2xa*i2*((pi*pi*pi)*D12*L*cosk2xa*k2*(r*r*r)*sina*(-(i2*i2) + (k2*k2)) + sink2xa*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(k2*k2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i2*i2) + (k2*k2)) + (pi*pi)*D11*(i2*i2)*(k2*k2)*(r*r) + (L*L)*(j2*j2)*(D12*(i2*i2) + (k2*k2)*(D12 + 4*D66))))))) + cosi2xb*i2*((pi*pi*pi)*D12*L*cosk2xb*k2*(r*r*r)*sina*(i2 - k2)*(i2 + k2) - sink2xb*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(k2*k2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i2*i2) + (k2*k2)) + (pi*pi)*D11*(i2*i2)*(k2*k2)*(r*r) + (L*L)*(j2*j2)*(D12*(i2*i2) + (k2*k2)*(D12 + 4*D66))))))) + cosk2xb*k2*sini2xb*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(i2*i2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i2*i2) + (k2*k2)) + (pi*pi)*D11*(i2*i2)*(k2*k2)*(r*r) + (L*L)*(j2*j2)*(D12*(k2*k2) + (i2*i2)*(D12 + 4*D66)))))) + sini2xa*(pi*(L*L*L)*r*sina*sink2xa*(i2 - k2)*(i2 + k2)*(B22*cosa*r + (j2*j2)*(D22 + 2*D66)) - cosk2xa*k2*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(i2*i2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i2*i2) + (k2*k2)) + (pi*pi)*D11*(i2*i2)*(k2*k2)*(r*r) + (L*L)*(j2*j2)*(D12*(k2*k2) + (i2*i2)*(D12 + 4*D66))))))))/((L*L*L)*(r*r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+5
                                k0v[c] += pi*j2*(pi*D16*L*cosk2xb*k2*r*sina*sini2xb*(-(i2*i2) + (k2*k2)) + cosi2xa*i2*(pi*D16*L*r*sina*sink2xa*(-(i2*i2) + (k2*k2)) + 2*cosk2xa*k2*(D26*(L*L)*(2*(j2*j2) - (sina*sina)) + r*(2*B26*(L*L)*cosa + (pi*pi)*D16*r*((i2*i2) + (k2*k2))))) + cosi2xb*i2*(pi*D16*L*r*sina*sink2xb*(i2 - k2)*(i2 + k2) - 2*cosk2xb*k2*(D26*(L*L)*(2*(j2*j2) - (sina*sina)) + r*(2*B26*(L*L)*cosa + (pi*pi)*D16*r*((i2*i2) + (k2*k2))))) + sini2xa*(pi*D16*L*cosk2xa*k2*r*sina*(i2 - k2)*(i2 + k2) + sink2xa*(D26*(L*L)*((i2*i2) + (k2*k2))*(2*(j2*j2) - (sina*sina)) + 2*r*(B26*(L*L)*cosa*((i2*i2) + (k2*k2)) + 2*(pi*pi)*D16*(i2*i2)*(k2*k2)*r))) + sini2xb*sink2xb*(-2*B26*(L*L)*cosa*r*((i2*i2) + (k2*k2)) - 4*(pi*pi)*D16*(i2*i2)*(k2*k2)*(r*r) - D26*(L*L)*((i2*i2) + (k2*k2))*(2*(j2*j2) - (sina*sina))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+0
                                k0v[c] += j2*(pi*L*cosk2xb*r*sina*sini2xb*(B16*(k2*k2) + B26*(i2*i2)) + cosi2xa*i2*(pi*L*k2*r*sina*sink2xa*(B16 + B26) + cosk2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + (pi*pi)*B16*r*((i2*i2) + 2*(k2*k2))))) + cosi2xb*i2*(-pi*L*k2*r*sina*sink2xb*(B16 + B26) - cosk2xb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + (pi*pi)*B16*r*((i2*i2) + 2*(k2*k2))))) - k2*sini2xb*sink2xb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + 3*(pi*pi)*B16*(i2*i2)*r)) + sini2xa*(-pi*L*cosk2xa*r*sina*(B16*(k2*k2) + B26*(i2*i2)) + k2*sink2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(A26*(L*L)*cosa + 3*(pi*pi)*B16*(i2*i2)*r))))/(L*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+1
                                k0v[c] += (-(L*L*L)*k2*sina*sini2xb*sink2xb*(A22*cosa*r + (j2*j2)*(B22 + B66)) + cosi2xa*i2*(L*cosk2xa*sina*(A22*(L*L)*cosa*r + (pi*pi)*B12*(r*r)*(i2 - k2)*(i2 + k2) + (L*L)*(j2*j2)*(B22 + B66)) - pi*k2*r*sink2xa*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2))) + cosi2xb*i2*(-L*cosk2xb*sina*(A22*(L*L)*cosa*r + (pi*pi)*B12*(r*r)*(i2 - k2)*(i2 + k2) + (L*L)*(j2*j2)*(B22 + B66)) + pi*k2*r*sink2xb*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2) + B22*(L*L)*(sina*sina) + 2*B66*(L*L)*(j2*j2))) - pi*cosk2xb*r*sini2xb*(B22*(L*L)*(i2*i2)*(sina*sina) + 2*B66*(L*L)*(i2*i2)*(j2*j2) + (k2*k2)*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2))) + sini2xa*((L*L*L)*k2*sina*sink2xa*(A22*cosa*r + (j2*j2)*(B22 + B66)) + pi*cosk2xa*r*(B22*(L*L)*(i2*i2)*(sina*sina) + 2*B66*(L*L)*(i2*i2)*(j2*j2) + (k2*k2)*(A12*(L*L)*cosa*r + (pi*pi)*B11*(i2*i2)*(r*r) + B12*(L*L)*(j2*j2)))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+2
                                k0v[c] += j2*(-pi*L*r*sina*sini2xb*sink2xb*(-D66*cosa*(k2*k2) + cosa*(i2*i2)*(D22 + 2*D66) + r*(-B66*(k2*k2) + (i2*i2)*(B22 + 2*B66))) + cosi2xa*i2*(pi*L*cosk2xa*k2*r*sina*(cosa*(D22 + D66) + r*(B22 + B66)) + sink2xa*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(D12*(i2*i2) + 2*D66*(k2*k2)))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(r*r)*(B12*(i2*i2) + 2*B66*(k2*k2))))) + cosi2xb*i2*(-pi*L*cosk2xb*k2*r*sina*(cosa*(D22 + D66) + r*(B22 + B66)) - sink2xb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(D12*(i2*i2) + 2*D66*(k2*k2)))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(r*r)*(B12*(i2*i2) + 2*B66*(k2*k2))))) + cosk2xb*k2*sini2xb*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 + 2*B66))) + sini2xa*(pi*L*r*sina*sink2xa*(-D66*cosa*(k2*k2) + cosa*(i2*i2)*(D22 + 2*D66) + r*(-B66*(k2*k2) + (i2*i2)*(B22 + 2*B66))) - cosk2xa*k2*(B22*(L*L)*(cosa*cosa)*r + cosa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A22*(L*L) + (pi*pi)*(i2*i2)*(D12 + 2*D66))) + r*((L*L)*(B22*(j2*j2) + B66*(sina*sina)) + (pi*pi)*(i2*i2)*(r*r)*(B12 + 2*B66)))))/(L*(r*r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+3
                                k0v[c] += (-L*cosk2xb*k2*sina*sini2xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 + D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 + B26)) + cosi2xa*i2*(-L*sina*sink2xa*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(i2*i2) + D26*(k2*k2))) + (pi*pi)*(r*r)*(B16*(i2*i2) + B26*(k2*k2))) + pi*cosk2xa*k2*(A26*(L*L)*cosa*(r*r) + (pi*pi)*B16*(i2*i2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + 3*B26*(L*L)*(j2*j2)*r + (pi*pi)*D16*cosa*(i2*i2)*(r*r) + 3*D26*(L*L)*cosa*(j2*j2) - (L*L)*(sina*sina)*(B26*r + D26*cosa))) + cosi2xb*i2*(L*sina*sink2xb*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(D16*(i2*i2) + D26*(k2*k2))) + (pi*pi)*(r*r)*(B16*(i2*i2) + B26*(k2*k2))) - pi*cosk2xb*k2*(A26*(L*L)*cosa*(r*r) + (pi*pi)*B16*(i2*i2)*(r*r*r) + B26*(L*L)*(cosa*cosa)*r + 3*B26*(L*L)*(j2*j2)*r + (pi*pi)*D16*cosa*(i2*i2)*(r*r) + 3*D26*(L*L)*cosa*(j2*j2) - (L*L)*(sina*sina)*(B26*r + D26*cosa))) + sini2xa*(L*cosk2xa*k2*sina*(B26*(L*L)*(cosa*cosa) + cosa*r*(A26*(L*L) + (pi*pi)*(i2*i2)*(D16 + D26)) + (pi*pi)*(i2*i2)*(r*r)*(B16 + B26)) + pi*sink2xa*(B26*(L*L)*(cosa*cosa)*(k2*k2)*r + cosa*(D26*(L*L)*(-(i2*i2)*(sina*sina) + (j2*j2)*(2*(i2*i2) + (k2*k2))) + (k2*k2)*(r*r)*(A26*(L*L) + (pi*pi)*D16*(i2*i2))) + r*((pi*pi)*B16*(i2*i2)*(k2*k2)*(r*r) + B26*(L*L)*(-(i2*i2)*(sina*sina) + (j2*j2)*(2*(i2*i2) + (k2*k2)))))) - pi*sini2xb*sink2xb*(B26*(L*L)*(cosa*cosa)*(k2*k2)*r + cosa*(D26*(L*L)*(-(i2*i2)*(sina*sina) + (j2*j2)*(2*(i2*i2) + (k2*k2))) + (k2*k2)*(r*r)*(A26*(L*L) + (pi*pi)*D16*(i2*i2))) + r*((pi*pi)*B16*(i2*i2)*(k2*k2)*(r*r) + B26*(L*L)*(-(i2*i2)*(sina*sina) + (j2*j2)*(2*(i2*i2) + (k2*k2))))))/((L*L)*(r*r)*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+4
                                k0v[c] += pi*j2*(pi*D16*L*cosk2xb*k2*r*sina*sini2xb*(-(i2*i2) + (k2*k2)) + cosi2xa*i2*(pi*D16*L*r*sina*sink2xa*(-(i2*i2) + (k2*k2)) + 2*cosk2xa*k2*(D26*(L*L)*(2*(j2*j2) - (sina*sina)) + r*(2*B26*(L*L)*cosa + (pi*pi)*D16*r*((i2*i2) + (k2*k2))))) + cosi2xb*i2*(pi*D16*L*r*sina*sink2xb*(i2 - k2)*(i2 + k2) - 2*cosk2xb*k2*(D26*(L*L)*(2*(j2*j2) - (sina*sina)) + r*(2*B26*(L*L)*cosa + (pi*pi)*D16*r*((i2*i2) + (k2*k2))))) + sini2xa*(pi*D16*L*cosk2xa*k2*r*sina*(i2 - k2)*(i2 + k2) + sink2xa*(D26*(L*L)*((i2*i2) + (k2*k2))*(2*(j2*j2) - (sina*sina)) + 2*r*(B26*(L*L)*cosa*((i2*i2) + (k2*k2)) + 2*(pi*pi)*D16*(i2*i2)*(k2*k2)*r))) + sini2xb*sink2xb*(-2*B26*(L*L)*cosa*r*((i2*i2) + (k2*k2)) - 4*(pi*pi)*D16*(i2*i2)*(k2*k2)*(r*r) - D26*(L*L)*((i2*i2) + (k2*k2))*(2*(j2*j2) - (sina*sina))))/((L*L)*(r*r)*(-i2 + k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+5
                                k0v[c] += (pi*(L*L*L)*r*sina*sini2xb*sink2xb*(-(i2*i2) + (k2*k2))*(B22*cosa*r + (j2*j2)*(D22 + 2*D66)) + cosi2xa*i2*((pi*pi*pi)*D12*L*cosk2xa*k2*(r*r*r)*sina*(-(i2*i2) + (k2*k2)) + sink2xa*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(k2*k2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i2*i2) + (k2*k2)) + (pi*pi)*D11*(i2*i2)*(k2*k2)*(r*r) + (L*L)*(j2*j2)*(D12*(i2*i2) + (k2*k2)*(D12 + 4*D66))))))) + cosi2xb*i2*((pi*pi*pi)*D12*L*cosk2xb*k2*(r*r*r)*sina*(i2 - k2)*(i2 + k2) - sink2xb*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(k2*k2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i2*i2) + (k2*k2)) + (pi*pi)*D11*(i2*i2)*(k2*k2)*(r*r) + (L*L)*(j2*j2)*(D12*(i2*i2) + (k2*k2)*(D12 + 4*D66))))))) + cosk2xb*k2*sini2xb*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(i2*i2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i2*i2) + (k2*k2)) + (pi*pi)*D11*(i2*i2)*(k2*k2)*(r*r) + (L*L)*(j2*j2)*(D12*(k2*k2) + (i2*i2)*(D12 + 4*D66)))))) + sini2xa*(pi*(L*L*L)*r*sina*sink2xa*(i2 - k2)*(i2 + k2)*(B22*cosa*r + (j2*j2)*(D22 + 2*D66)) - cosk2xa*k2*(D22*((L*L*L*L)*(j2*j2*j2*j2) + (pi*pi)*(L*L)*(i2*i2)*(r*r)*(sina*sina)) + D66*(L*L*L*L)*(j2*j2)*(sina*sina) + r*(2*B22*(L*L*L*L)*cosa*(j2*j2) + r*(A22*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(B12*(L*L)*cosa*r*((i2*i2) + (k2*k2)) + (pi*pi)*D11*(i2*i2)*(k2*k2)*(r*r) + (L*L)*(j2*j2)*(D12*(k2*k2) + (i2*i2)*(D12 + 4*D66))))))))/((L*L*L)*(r*r*r)*(i2 - k2)*(i2 + k2))


    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0_cyl(double r2, double L, np.ndarray[cDOUBLE, ndim=2] F,
            int m1, int m2, int n2):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    # sparse parameters
    k11_cond_1 = 5
    k11_cond_2 = 4
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 18
    k22_cond_2 = 18
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 5 + 2*2*m1 + k11_num + k22_num

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    A11 = F[0,0]
    A12 = F[0,1]
    A16 = F[0,2]
    A22 = F[1,1]
    A26 = F[1,2]
    A66 = F[2,2]
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

    # k0_00
    c += 1
    k0r[c] = 0
    k0c[c] = 0
    k0v[c] += 2*pi*A11*r/L
    c += 1
    k0r[c] = 0
    k0c[c] = 1
    k0v[c] += 2*pi*r2*(A16*r + B16)/L
    c += 1
    k0r[c] = 1
    k0c[c] = 0
    k0v[c] += 2*pi*r2*(A16*r + B16)/L
    c += 1
    k0r[c] = 1
    k0c[c] = 1
    k0v[c] += 2*pi*(r2*r2)*(D66 + r*(A66*r + 2*B66))/(L*r)
    c += 1
    k0r[c] = 2
    k0c[c] = 2
    k0v[c] += pi*A11*r/L + 0.333333333333333*pi*A66*L/r

    for i1 in range(i0, m1+i0):
        col = (i1-i0)*num1 + num0
        row = col
        if i1!=0:
            # k0_01 cond_1
            c += 1
            k0r[c] = 0
            k0c[c] = col+2
            k0v[c] += (2*(-1)**i1 - 2)*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r)/((L*L)*i1)
            c += 1
            k0r[c] = 1
            k0c[c] = col+2
            k0v[c] += r2*(2*(-1)**i1 - 2)*(B26*(L*L) + r*(A26*(L*L) + (pi*pi)*(i1*i1)*(B16*r + D16)))/((L*L)*i1*r)

            # k0_10 cond_1 (using symmetry)
            c += 1
            k0r[c] = col+2
            k0c[c] = 0
            k0v[c] += (2*(-1)**i1 - 2)*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r)/((L*L)*i1)
            c += 1
            k0r[c] = col+2
            k0c[c] = 1
            k0v[c] += r2*(2*(-1)**i1 - 2)*(B26*(L*L) + r*(A26*(L*L) + (pi*pi)*(i1*i1)*(B16*r + D16)))/((L*L)*i1*r)

        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0
            if k1==i1:
                if i1!=0:
                    # k0_11 cond_1
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += (pi*pi*pi)*A11*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+1
                    k0v[c] += (pi*pi*pi)*(i1*i1)*(A16*r + B16)/L
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += (pi*pi*pi)*(i1*i1)*(A16*r + B16)/L
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += (pi*pi*pi)*(i1*i1)*(D66 + r*(A66*r + 2*B66))/(L*r)
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += pi*A22*L/r + 2*(pi*pi*pi)*B12*(i1*i1)/L + (pi*pi*pi*pi*pi)*D11*(i1*i1*i1*i1)*r/(L*L*L)

            else:
                # k0_11 cond_2
                c += 1
                k0r[c] = row+0
                k0c[c] = col+2
                k0v[c] += pi*i1*k1*(2*(-1)**(i1 + k1) - 2)*(A12*(L*L) + (pi*pi)*B11*(k1*k1)*r)/((L*L)*((i1*i1) - (k1*k1)))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+2
                k0v[c] += pi*i1*k1*(2*(-1)**(i1 + k1) - 2)*(B26*(L*L) + r*(A26*(L*L) + (pi*pi)*(k1*k1)*(B16*r + D16)))/((L*L)*r*((i1*i1) - (k1*k1)))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+0
                k0v[c] += pi*i1*k1*(-2*(-1)**(i1 + k1) + 2)*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r)/((L*L)*((i1*i1) - (k1*k1)))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+1
                k0v[c] += pi*i1*k1*(-2*(-1)**(i1 + k1) + 2)*(B26*(L*L) + r*(A26*(L*L) + (pi*pi)*(i1*i1)*(B16*r + D16)))/((L*L)*r*((i1*i1) - (k1*k1)))

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                    if k2==i2 and l2==j2:
                        if i2!=0:
                            # k0_22 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.5*(pi*pi*pi)*A11*(i2*i2)*r/L + 0.5*pi*A66*L*(j2*j2)/r
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += 0.5*(pi*pi)*i2*j2*(B12 + B66 + r*(A12 + A66))/r
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += -0.5*(pi*pi)*i2*(A12*(L*L)*r + (pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(B12 + 2*B66))/((L*L)*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.5*(pi*pi*pi)*A11*(i2*i2)*r/L + 0.5*pi*A66*L*(j2*j2)/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += -0.5*(pi*pi)*i2*j2*(B12 + B66 + r*(A12 + A66))/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+5
                            k0v[c] += -0.5*(pi*pi)*i2*(A12*(L*L)*r + (pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(B12 + 2*B66))/((L*L)*r)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += -0.5*(pi*pi)*i2*j2*(B12 + B66 + r*(A12 + A66))/r
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.25*pi*(2*(L*L)*(j2*j2)*(D22 + r*(A22*r + 2*B22)) + 2*(pi*pi)*(i2*i2)*(r*r)*(D66 + r*(A66*r + 2*B66)))/(L*(r*r*r))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+5
                            k0v[c] += 0.5*pi*j2*(D22*(L*L)*(j2*j2) + r*(B22*(L*L)*((j2*j2) + 1) + r*(A22*(L*L) + (pi*pi)*(i2*i2)*(B12*r + 2*B66*r + D12 + 2*D66))))/(L*(r*r*r))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+0
                            k0v[c] += 0.5*(pi*pi)*i2*j2*(B12 + B66 + r*(A12 + A66))/r
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += 0.25*pi*(2*(L*L)*(j2*j2)*(D22 + r*(A22*r + 2*B22)) + 2*(pi*pi)*(i2*i2)*(r*r)*(D66 + r*(A66*r + 2*B66)))/(L*(r*r*r))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+4
                            k0v[c] += -0.5*pi*j2*(D22*(L*L)*(j2*j2) + r*(B22*(L*L)*((j2*j2) + 1) + r*(A22*(L*L) + (pi*pi)*(i2*i2)*(B12*r + 2*B66*r + D12 + 2*D66))))/(L*(r*r*r))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+0
                            k0v[c] += -0.5*(pi*pi)*i2*(A12*(L*L)*r + (pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(B12 + 2*B66))/((L*L)*r)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+3
                            k0v[c] += -0.5*pi*j2*(D22*(L*L)*(j2*j2) + r*(B22*(L*L)*((j2*j2) + 1) + r*(A22*(L*L) + (pi*pi)*(i2*i2)*(B12*r + 2*B66*r + D12 + 2*D66))))/(L*(r*r*r))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+4
                            k0v[c] += 0.5*pi*(D22*(L*L*L*L)*(j2*j2*j2*j2) + r*(2*B22*(L*L*L*L)*(j2*j2) + r*(A22*(L*L*L*L) + (pi*pi)*(i2*i2)*(2*B12*(L*L)*r + (pi*pi)*D11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(2*D12 + 4*D66)))))/((L*L*L)*(r*r*r))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+1
                            k0v[c] += -0.5*(pi*pi)*i2*(A12*(L*L)*r + (pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(B12 + 2*B66))/((L*L)*r)
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi*j2*(D22*(L*L)*(j2*j2) + r*(B22*(L*L)*((j2*j2) + 1) + r*(A22*(L*L) + (pi*pi)*(i2*i2)*(B12*r + 2*B66*r + D12 + 2*D66))))/(L*(r*r*r))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+5
                            k0v[c] += 0.5*pi*(D22*(L*L*L*L)*(j2*j2*j2*j2) + r*(2*B22*(L*L*L*L)*(j2*j2) + r*(A22*(L*L*L*L) + (pi*pi)*(i2*i2)*(2*B12*(L*L)*r + (pi*pi)*D11*(i2*i2)*(r*r) + (L*L)*(j2*j2)*(2*D12 + 4*D66)))))/((L*L*L)*(r*r*r))

                        else:
                            # k0_22 cond_5
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += pi*A66*L*(j2*j2)/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += pi*A66*L*(j2*j2)/r

                    elif k2!=i2 and l2==j2:
                        # k0_22 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += -pi*A16*j2*((-1)**(i2 + k2) - 1)*((i2*i2) + (k2*k2))/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += -k2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(A26*(L*L)*(j2*j2) + (pi*pi)*(i2*i2)*r*(A16*r + B16)))/(L*(r*r)*(-(i2*i2) + (k2*k2)))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+5
                        k0v[c] += -j2*k2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(A26*(L*L) + (pi*pi)*B16*r*(2*(i2*i2) + (k2*k2))))/(L*(r*r)*(-(i2*i2) + (k2*k2)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += pi*A16*j2*((-1)**(i2 + k2) - 1)*((i2*i2) + (k2*k2))/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += -k2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(A26*(L*L)*(j2*j2) + (pi*pi)*(i2*i2)*r*(A16*r + B16)))/(L*(r*r)*(-(i2*i2) + (k2*k2)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += j2*k2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(A26*(L*L) + (pi*pi)*B16*r*(2*(i2*i2) + (k2*k2))))/(L*(r*r)*(-(i2*i2) + (k2*k2)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += -i2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(A26*(L*L)*(j2*j2) + (pi*pi)*(k2*k2)*r*(A16*r + B16)))/(L*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += pi*i2*j2*k2*(-2*(-1)**(i2 + k2) + 2)*(A26*(r*r) + 2*B26*r + D26)/((r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += pi*i2*k2*((-1)**(i2 + k2) - 1)*(3*D26*(L*L)*(j2*j2) + r*(B26*(L*L)*(3*(j2*j2) + 1) + r*(A26*(L*L) + (pi*pi)*(k2*k2)*(B16*r + D16))))/((L*L)*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += -i2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(A26*(L*L)*(j2*j2) + (pi*pi)*(k2*k2)*r*(A16*r + B16)))/(L*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += pi*i2*j2*k2*(2*(-1)**(i2 + k2) - 2)*(A26*(r*r) + 2*B26*r + D26)/((r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+5
                        k0v[c] += pi*i2*k2*((-1)**(i2 + k2) - 1)*(3*D26*(L*L)*(j2*j2) + r*(B26*(L*L)*(3*(j2*j2) + 1) + r*(A26*(L*L) + (pi*pi)*(k2*k2)*(B16*r + D16))))/((L*L)*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += i2*j2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(A26*(L*L) + (pi*pi)*B16*r*((i2*i2) + 2*(k2*k2))))/(L*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += -pi*i2*k2*((-1)**(i2 + k2) - 1)*(3*D26*(L*L)*(j2*j2) + r*(B26*(L*L)*(3*(j2*j2) + 1) + r*(A26*(L*L) + (pi*pi)*(i2*i2)*(B16*r + D16))))/((L*L)*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+5
                        k0v[c] += pi*i2*j2*k2*(-2*(-1)**(i2 + k2) + 2)*(2*D26*(L*L)*(j2*j2) + r*(2*B26*(L*L) + (pi*pi)*D16*r*((i2*i2) + (k2*k2))))/((L*L)*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+0
                        k0v[c] += -i2*j2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(A26*(L*L) + (pi*pi)*B16*r*((i2*i2) + 2*(k2*k2))))/(L*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+3
                        k0v[c] += -pi*i2*k2*((-1)**(i2 + k2) - 1)*(3*D26*(L*L)*(j2*j2) + r*(B26*(L*L)*(3*(j2*j2) + 1) + r*(A26*(L*L) + (pi*pi)*(i2*i2)*(B16*r + D16))))/((L*L)*(r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+4
                        k0v[c] += pi*i2*j2*k2*(2*(-1)**(i2 + k2) - 2)*(2*D26*(L*L)*(j2*j2) + r*(2*B26*(L*L) + (pi*pi)*D16*r*((i2*i2) + (k2*k2))))/((L*L)*(r*r)*((i2*i2) - (k2*k2)))


    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0

def fk0edges(int m1, int m2, int n2, double r1, double r2, double L,
             double kuBot, double kuTop,
             double kphixBot, double kphixTop):
    cdef int i1, k1, i2, j2, k2, l2, row, col, c
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv

    k11_cond_1 = 1
    k11_cond_2 = 1
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 4
    k22_cond_2 = 4
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k11_num + k22_num

    k0edgesr = np.zeros((fdim,), dtype=INT)
    k0edgesc = np.zeros((fdim,), dtype=INT)
    k0edgesv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0
        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0
            if k1==i1:
                # k0edges_11 cond_1
                c += 1
                k0edgesr[c] = row+2
                k0edgesc[c] = col+2
                k0edgesv[c] += 2*pi**3*i1**2*(kphixBot*r1 + kphixTop*r2)/L**2

            else:
                # k0edges_11 cond_2
                c += 1
                k0edgesr[c] = row+2
                k0edgesc[c] = col+2
                k0edgesv[c] += 2*pi**3*i1*k1*((-1)**(i1 + k1)*kphixBot*r1 + kphixTop*r2)/L**2

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                    if k2==i2 and l2==j2:
                        # k0edges_22 cond_1
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += pi*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += pi*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+4
                        k0edgesc[c] = col+4
                        k0edgesv[c] += pi**3*i2**2*(kphixBot*r1 + kphixTop*r2)/L**2
                        c += 1
                        k0edgesr[c] = row+5
                        k0edgesc[c] = col+5
                        k0edgesv[c] += pi**3*i2**2*(kphixBot*r1 + kphixTop*r2)/L**2

                    elif k2!=i2 and l2==j2:
                        # k0edges_22 cond_2
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+4
                        k0edgesc[c] = col+4
                        k0edgesv[c] += pi**3*i2*k2*((-1)**(i2 + k2)*kphixBot*r1 + kphixTop*r2)/L**2
                        c += 1
                        k0edgesr[c] = row+5
                        k0edgesc[c] = col+5
                        k0edgesv[c] += pi**3*i2*k2*((-1)**(i2 + k2)*kphixBot*r1 + kphixTop*r2)/L**2


    size = num0 + num1*m1 + num2*m2*n2

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fc, double P, double T, double r2, double alpharad, double L,
         int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double sina, cosa, r, xa, xb
    cdef double sini1xa, cosi1xa, sini1xb, cosi1xb
    cdef double sink1xa, sink1xb, cosk1xa, cosk1xb, sini2xa, sini2xb
    cdef double sin2i2xa, sin2i2xb
    cdef double cosi2xa, cosi2xb, cos2i2xa, cos2i2xb
    cdef double cosk2xa, cosk2xb, sink2xa, sink2xb
    cdef double sin2i1xa, cos2i1xa, sin2i1xb, cos2i1xb

    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    # sparse parameters
    k11_cond_1 = 4
    k11_cond_2 = 4
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 12
    k22_cond_2 = 14
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 1 + 2*2*m1 + k11_num + k22_num

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    with nogil:
        for section in range(s):
            c = -1

            xa = L*float(section)/s
            xb = L*float(section+1)/s

            r = r2 + sina*((xa+xb)/2.)

            # kG0_00
            c += 1
            kG0r[c] = 1
            kG0c[c] = 1
            kG0v[c] += -0.666666666666667*pi*P*cosa*(r2*r2)*(xa - xb)*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb))/(L*L)

            for i1 in range(i0, m1+i0):
                cosi1xa = cos(pi*i1*xa/L)
                cosi1xb = cos(pi*i1*xb/L)
                sini1xa = sin(pi*i1*xa/L)
                sini1xb = sin(pi*i1*xb/L)
                cos2i1xa = cos(2*pi*i1*xa/L)
                cos2i1xb = cos(2*pi*i1*xb/L)
                sin2i1xa = sin(2*pi*i1*xa/L)
                sin2i1xb = sin(2*pi*i1*xb/L)

                col = (i1-i0)*num1 + num0
                row = col
                if i1!=0:
                    # kG0_01 cond_1
                    c += 1
                    kG0r[c] = 1
                    kG0c[c] = col+1
                    kG0v[c] += 2*P*cosa*r2*(L*(sini1xa - sini1xb) + pi*cosi1xa*i1*(L - xa) + pi*cosi1xb*i1*(-L + xb))/(pi*(i1*i1))
                    c += 1
                    kG0r[c] = 1
                    kG0c[c] = col+2
                    kG0v[c] += T*cosa*r2*(L*cosi1xa - L*cosi1xb + pi*i1*(sini1xa*(-L + xa) + sini1xb*(L - xb)))/(pi*L*i1*(r*r))

                    # kG0_10 cond_1 (using symmetry)
                    c += 1
                    kG0r[c] = col+1
                    kG0c[c] = 1
                    kG0v[c] += 2*P*cosa*r2*(L*(sini1xa - sini1xb) + pi*cosi1xa*i1*(L - xa) + pi*cosi1xb*i1*(-L + xb))/(pi*(i1*i1))
                    c += 1
                    kG0r[c] = col+2
                    kG0c[c] = 1
                    kG0v[c] += T*cosa*r2*(L*cosi1xa - L*cosi1xb + pi*i1*(sini1xa*(-L + xa) + sini1xb*(L - xb)))/(pi*L*i1*(r*r))

                for k1 in range(i0, m1+i0):
                    cosk1xa = cos(pi*k1*xa/L)
                    cosk1xb = cos(pi*k1*xb/L)
                    sink1xa = sin(pi*k1*xa/L)
                    sink1xb = sin(pi*k1*xb/L)

                    col = (k1-i0)*num1 + num0
                    if k1==i1:
                        if i1!=0:
                            # kG0_11 cond_1
                            c += 1
                            kG0r[c] = row+1
                            kG0c[c] = col+1
                            kG0v[c] += 0.5*P*cosa*(L*sin2i1xa - L*sin2i1xb + 2*pi*i1*(-xa + xb))/i1
                            c += 1
                            kG0r[c] = row+1
                            kG0c[c] = col+2
                            kG0v[c] += 0.25*T*cosa*(cos2i1xa - cos2i1xb)/(r*r)
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+1
                            kG0v[c] += 0.25*T*cosa*(cos2i1xa - cos2i1xb)/(r*r)
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += 0.25*pi*i1*(Fc - pi*P*(r*r))*(L*sin2i1xa - L*sin2i1xb + 2*pi*i1*(xa - xb))/((L*L)*cosa)

                    else:
                        # kG0_11 cond_2
                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+1
                        kG0v[c] += 2*L*P*cosa*(cosi1xa*i1*sink1xa - cosi1xb*i1*sink1xb - cosk1xa*k1*sini1xa + cosk1xb*k1*sini1xb)/((i1*i1) - (k1*k1))
                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+2
                        kG0v[c] += T*cosa*k1*(cosi1xa*cosk1xa*i1 - cosi1xb*cosk1xb*i1 + k1*sini1xa*sink1xa - k1*sini1xb*sink1xb)/((r*r)*((i1*i1) - (k1*k1)))
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+1
                        kG0v[c] += T*cosa*i1*(-cosi1xa*cosk1xa*k1 + cosi1xb*cosk1xb*k1 - i1*sini1xa*sink1xa + i1*sini1xb*sink1xb)/((r*r)*((i1*i1) - (k1*k1)))
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += pi*i1*k1*(Fc - pi*P*(r*r))*(-cosi1xa*k1*sink1xa + cosi1xb*k1*sink1xb + cosk1xa*i1*sini1xa - cosk1xb*i1*sini1xb)/(L*cosa*(i1 - k1)*(i1 + k1))

            for i2 in range(i0, m2+i0):
                cos2i2xa = cos(2*pi*i2*xa/L)
                cos2i2xb = cos(2*pi*i2*xb/L)
                sin2i2xa = sin(2*pi*i2*xa/L)
                sin2i2xb = sin(2*pi*i2*xb/L)
                cosi2xa = cos(pi*i2*xa/L)
                cosi2xb = cos(pi*i2*xb/L)
                sini2xa = sin(pi*i2*xa/L)
                sini2xb = sin(pi*i2*xb/L)
                for k2 in range(i0, m2+i0):
                    cosk2xa = cos(pi*k2*xa/L)
                    cosk2xb = cos(pi*k2*xb/L)
                    sink2xa = sin(pi*k2*xa/L)
                    sink2xb = sin(pi*k2*xb/L)
                    for j2 in range(j0, n2+j0):
                        row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
                        for l2 in range(j0, n2+j0):
                            col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                            if k2==i2 and l2==j2:
                                if i2!=0:
                                    # kG0_22 cond_1
                                    c += 1
                                    kG0r[c] = row+2
                                    kG0c[c] = col+2
                                    kG0v[c] += 0.25*P*cosa*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/i2
                                    c += 1
                                    kG0r[c] = row+2
                                    kG0c[c] = col+4
                                    kG0v[c] += 0.125*T*cosa*(cos2i2xa - cos2i2xb)/(r*r)
                                    c += 1
                                    kG0r[c] = row+2
                                    kG0c[c] = col+5
                                    kG0v[c] += 0.25*P*j2*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/i2
                                    c += 1
                                    kG0r[c] = row+3
                                    kG0c[c] = col+3
                                    kG0v[c] += 0.25*P*cosa*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/i2
                                    c += 1
                                    kG0r[c] = row+3
                                    kG0c[c] = col+4
                                    kG0v[c] += 0.25*P*j2*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/i2
                                    c += 1
                                    kG0r[c] = row+3
                                    kG0c[c] = col+5
                                    kG0v[c] += 0.125*T*cosa*(cos2i2xa - cos2i2xb)/(r*r)
                                    c += 1
                                    kG0r[c] = row+4
                                    kG0c[c] = col+2
                                    kG0v[c] += 0.125*T*cosa*(cos2i2xa - cos2i2xb)/(r*r)
                                    c += 1
                                    kG0r[c] = row+4
                                    kG0c[c] = col+3
                                    kG0v[c] += 0.25*P*j2*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/i2
                                    c += 1
                                    kG0r[c] = row+4
                                    kG0c[c] = col+4
                                    kG0v[c] += 0.125*(L*(sin2i2xa - sin2i2xb)*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(Fc - pi*P*(r*r))) - 2*pi*i2*(xa - xb)*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r))))/((L*L)*cosa*i2)
                                    c += 1
                                    kG0r[c] = row+5
                                    kG0c[c] = col+2
                                    kG0v[c] += 0.25*P*j2*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/i2
                                    c += 1
                                    kG0r[c] = row+5
                                    kG0c[c] = col+3
                                    kG0v[c] += 0.125*T*cosa*(cos2i2xa - cos2i2xb)/(r*r)
                                    c += 1
                                    kG0r[c] = row+5
                                    kG0c[c] = col+5
                                    kG0v[c] += 0.125*(L*(sin2i2xa - sin2i2xb)*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(Fc - pi*P*(r*r))) - 2*pi*i2*(xa - xb)*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r))))/((L*L)*cosa*i2)

                            elif k2!=i2 and l2==j2:
                                # kG0_22 cond_2
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+2
                                kG0v[c] += L*P*cosa*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/((i2*i2) - (k2*k2))
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+4
                                kG0v[c] += T*cosa*k2*(cosi2xa*cosk2xa*i2 - cosi2xb*cosk2xb*i2 + k2*sini2xa*sink2xa - k2*sini2xb*sink2xb)/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+5
                                kG0v[c] += L*P*j2*(-cosi2xa*i2*sink2xa + cosi2xb*i2*sink2xb + cosk2xa*k2*sini2xa - cosk2xb*k2*sini2xb)/((i2*i2) - (k2*k2))
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+3
                                kG0v[c] += L*P*cosa*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/((i2*i2) - (k2*k2))
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+4
                                kG0v[c] += L*P*j2*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/((i2*i2) - (k2*k2))
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+5
                                kG0v[c] += T*cosa*k2*(cosi2xa*cosk2xa*i2 - cosi2xb*cosk2xb*i2 + k2*sini2xa*sink2xa - k2*sini2xb*sink2xb)/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+2
                                kG0v[c] += T*cosa*i2*(-cosi2xa*cosk2xa*k2 + cosi2xb*cosk2xb*k2 - i2*sini2xa*sink2xa + i2*sini2xb*sink2xb)/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+3
                                kG0v[c] += L*P*j2*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/((i2*i2) - (k2*k2))
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+4
                                kG0v[c] += 0.5*(cosi2xa*i2*sink2xa*(2*(L*L)*P*(j2*j2) + pi*(k2*k2)*(-Fc + pi*P*(r*r))) + cosi2xb*i2*sink2xb*(-2*(L*L)*P*(j2*j2) + pi*(k2*k2)*(Fc - pi*P*(r*r))) + cosk2xa*k2*sini2xa*(-2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(Fc - pi*P*(r*r))) + cosk2xb*k2*sini2xb*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r))))/(L*cosa*(i2 - k2)*(i2 + k2))
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+5
                                kG0v[c] += -T*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+2
                                kG0v[c] += L*P*j2*(-cosi2xa*i2*sink2xa + cosi2xb*i2*sink2xb + cosk2xa*k2*sini2xa - cosk2xb*k2*sini2xb)/((i2*i2) - (k2*k2))
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+3
                                kG0v[c] += T*cosa*i2*(-cosi2xa*cosk2xa*k2 + cosi2xb*cosk2xb*k2 - i2*sini2xa*sink2xa + i2*sini2xb*sink2xb)/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+4
                                kG0v[c] += T*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+5
                                kG0v[c] += 0.5*(cosi2xa*i2*sink2xa*(2*(L*L)*P*(j2*j2) + pi*(k2*k2)*(-Fc + pi*P*(r*r))) + cosi2xb*i2*sink2xb*(-2*(L*L)*P*(j2*j2) + pi*(k2*k2)*(Fc - pi*P*(r*r))) + cosk2xa*k2*sini2xa*(-2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(Fc - pi*P*(r*r))) + cosk2xb*k2*sini2xb*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r))))/(L*cosa*(i2 - k2)*(i2 + k2))


    size = num0 + num1*m1 + num2*m2*n2

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkG0_cyl(double Fc, double P, double T, double r2, double L,
            int m1, int m2, int n2):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col
    cdef double r
    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    # sparse parameters
    k11_cond_1 = 2
    k11_cond_2 = 2
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 8
    k22_cond_2 = 6
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 1 + 2*2*m1 + k11_num + k22_num

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1
    r = r2

    # kG0_00
    c += 1
    kG0r[c] = 1
    kG0c[c] = 1
    kG0v[c] += 0.666666666666667*pi*L*P*(r2*r2)

    for i1 in range(i0, m1+i0):
        col = (i1-i0)*num1 + num0
        row = col
        if i1!=0:
            # kG0_01 cond_1
            c += 1
            kG0r[c] = 1
            kG0c[c] = col+1
            kG0v[c] += 2*L*P*r2/i1
            c += 1
            kG0r[c] = 1
            kG0c[c] = col+2
            kG0v[c] += -T*r2*((-1)**i1 - 1)/(pi*i1*(r*r))

            # kG0_10 cond_1 (using symmetry)
            c += 1
            kG0r[c] = col+1
            kG0c[c] = 1
            kG0v[c] += 2*L*P*r2/i1
            c += 1
            kG0r[c] = col+2
            kG0c[c] = 1
            kG0v[c] += -T*r2*((-1)**i1 - 1)/(pi*i1*(r*r))

        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0
            if k1==i1:
                if i1!=0:
                    # kG0_11 cond_1
                    c += 1
                    kG0r[c] = row+1
                    kG0c[c] = col+1
                    kG0v[c] += pi*L*P
                    c += 1
                    kG0r[c] = row+2
                    kG0c[c] = col+2
                    kG0v[c] += 0.5*(pi*pi)*(i1*i1)*(-Fc + pi*P*(r*r))/L

            else:
                # kG0_11 cond_2
                c += 1
                kG0r[c] = row+1
                kG0c[c] = col+2
                kG0v[c] += -T*i1*k1*((-1)**(i1 + k1) - 1)/((r*r)*((i1*i1) - (k1*k1)))
                c += 1
                kG0r[c] = row+2
                kG0c[c] = col+1
                kG0v[c] += T*i1*k1*((-1)**(i1 + k1) - 1)/((r*r)*((i1*i1) - (k1*k1)))

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1
                    if k2==i2 and l2==j2:
                        if i2!=0:
                            # kG0_22 cond_1
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += 0.5*pi*L*P
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+5
                            kG0v[c] += -0.5*pi*L*P*j2
                            c += 1
                            kG0r[c] = row+3
                            kG0c[c] = col+3
                            kG0v[c] += 0.5*pi*L*P
                            c += 1
                            kG0r[c] = row+3
                            kG0c[c] = col+4
                            kG0v[c] += 0.5*pi*L*P*j2
                            c += 1
                            kG0r[c] = row+4
                            kG0c[c] = col+3
                            kG0v[c] += 0.5*pi*L*P*j2
                            c += 1
                            kG0r[c] = row+4
                            kG0c[c] = col+4
                            kG0v[c] += 0.25*pi*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r)))/L
                            c += 1
                            kG0r[c] = row+5
                            kG0c[c] = col+2
                            kG0v[c] += -0.5*pi*L*P*j2
                            c += 1
                            kG0r[c] = row+5
                            kG0c[c] = col+5
                            kG0v[c] += 0.25*pi*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r)))/L

                    elif k2!=i2 and l2==j2:
                        # kG0_22 cond_2
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+4
                        kG0v[c] += -T*i2*k2*((-1)**(i2 + k2) - 1)/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                        c += 1
                        kG0r[c] = row+3
                        kG0c[c] = col+5
                        kG0v[c] += -T*i2*k2*((-1)**(i2 + k2) - 1)/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                        c += 1
                        kG0r[c] = row+4
                        kG0c[c] = col+2
                        kG0v[c] += T*i2*k2*((-1)**(i2 + k2) - 1)/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                        c += 1
                        kG0r[c] = row+4
                        kG0c[c] = col+5
                        kG0v[c] += T*i2*j2*k2*((-1)**(i2 + k2) - 1)/((r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        kG0r[c] = row+5
                        kG0c[c] = col+3
                        kG0v[c] += T*i2*k2*((-1)**(i2 + k2) - 1)/((r*r)*(2.0*(i2*i2) - 2.0*(k2*k2)))
                        c += 1
                        kG0r[c] = row+5
                        kG0c[c] = col+4
                        kG0v[c] += -T*i2*j2*k2*((-1)**(i2 + k2) - 1)/((r*r)*((i2*i2) - (k2*k2)))


    size = num0 + num1*m1 + num2*m2*n2

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0

