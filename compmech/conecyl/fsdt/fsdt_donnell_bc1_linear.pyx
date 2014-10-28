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
cdef int num1 = 5
cdef int num2 = 10
cdef double pi = 3.141592653589793


def fk0(double alpharad, double r2, double L, np.ndarray[cDOUBLE, ndim=2] F,
        int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r, sina, cosa, xa, xb
    cdef double sini1xa, cosi1xa, sini1xb, cosi1xb
    cdef double sini1xa_xb, sini1xaxb, cosi1xaxb
    cdef double sink1xa, sink1xb, cosk1xa, cosk1xb, sini2xa, sini2xb
    cdef double sin2i2xa, sin2i2xb, sini2xa_xb, sini2xaxb, cosi2xaxb
    cdef double cosi2xa, cosi2xb, cos2i2xa, cos2i2xb
    cdef double cosk2xa, cosk2xb, sink2xa, sink2xb
    cdef double sin2i1xa, cos2i1xa, sin2i1xb, cos2i1xb

    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    sina = sin(alpharad)
    cosa = cos(alpharad)

    # sparse parameters
    k11_cond_1 = 25
    k11_cond_2 = 25
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 86
    k22_cond_2 = 100
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 5 + 10*m1 + k11_num + k22_num

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
            k0v[c] += 0.333333333333333*pi*r2*(xa - xb)*(3*A16*r*(-2*r + sina*(-2*L + xa + xb)) + A26*sina*(6*(L*L)*sina + 6*L*(r - sina*(xa + xb)) - 3*r*(xa + xb) + 2*sina*((xa*xa) + xa*xb + (xb*xb))))/((L*L)*cosa*r)
            c += 1
            k0r[c] = 1
            k0c[c] = 0
            k0v[c] += 0.333333333333333*pi*r2*(xa - xb)*(3*A16*r*(-2*r + sina*(-2*L + xa + xb)) + A26*sina*(6*(L*L)*sina + 6*L*(r - sina*(xa + xb)) - 3*r*(xa + xb) + 2*sina*((xa*xa) + xa*xb + (xb*xb))))/((L*L)*cosa*r)
            c += 1
            k0r[c] = 1
            k0c[c] = 1
            k0v[c] += -0.666666666666667*pi*(r2*r2)*(xa - xb)*(A44*(cosa*cosa)*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb)) + 3*A66*(r*r) + A66*sina*(3*(L*L)*sina + L*(6*r - 3*sina*(xa + xb)) - 3*r*(xa + xb) + sina*((xa*xa) + xa*xb + (xb*xb))))/((L*L)*r)
            c += 1
            k0r[c] = 2
            k0c[c] = 2
            k0v[c] += -0.333333333333333*pi*(xa - xb)*(3*A11*(r*r) + A66*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb)) + sina*(3*A12*r*(-2*L + xa + xb) + A22*sina*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb))))/((L*L)*(cosa*cosa)*r)

            for k1 in range(i0, m1+i0):
                cosk1xa = cos(pi*k1*xa/L)
                cosk1xb = cos(pi*k1*xb/L)
                sink1xa = sin(pi*k1*xa/L)
                sink1xb = sin(pi*k1*xb/L)

                col = (k1-i0)*num1 + num0

                if k1!=0:
                    # k0_01 cond_1
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+0
                    k0v[c] += (2*pi*A22*L*cosk1xa*k1*(sina*sina)*(L - xa) + 2*pi*A22*L*cosk1xb*k1*(sina*sina)*(-L + xb) + 2*sink1xa*((pi*pi)*A11*(k1*k1)*(r*r) + sina*((pi*pi)*A12*(k1*k1)*r*(-L + xa) + A22*(L*L)*sina)) - 2*sink1xb*((pi*pi)*A11*(k1*k1)*(r*r) + sina*((pi*pi)*A12*(k1*k1)*r*(-L + xb) + A22*(L*L)*sina)))/(pi*L*cosa*(k1*k1)*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+1
                    k0v[c] += (2*pi*L*cosk1xa*k1*sina*(A16*r + A26*(-L*sina + r + sina*xa)) + 2*pi*L*cosk1xb*k1*sina*(-A16*r - A26*(-L*sina + r + sina*xb)) + 2*sink1xa*((pi*pi)*A16*(k1*k1)*(r*r) + A26*sina*(-(L*L)*sina + (pi*pi)*(k1*k1)*r*(-L + xa))) + 2*sink1xb*(-(pi*pi)*A16*(k1*k1)*(r*r) + A26*sina*((L*L)*sina + (pi*pi)*(k1*k1)*r*(L - xb))))/(pi*L*cosa*(k1*k1)*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+2
                    k0v[c] += (2*A22*L*sina*(sink1xa - sink1xb) - 2*pi*cosk1xa*k1*(A12*r + A22*sina*(-L + xa)) + 2*pi*cosk1xb*k1*(A12*r + A22*sina*(-L + xb)))/(pi*(k1*k1)*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+3
                    k0v[c] += (2*pi*B22*L*k1*(sina*sina)*(sink1xa*(-L + xa) + sink1xb*(L - xb)) + 2*cosk1xa*((pi*pi)*B11*(k1*k1)*(r*r) + sina*((pi*pi)*B12*(k1*k1)*r*(-L + xa) + B22*(L*L)*sina)) - 2*cosk1xb*((pi*pi)*B11*(k1*k1)*(r*r) + sina*((pi*pi)*B12*(k1*k1)*r*(-L + xb) + B22*(L*L)*sina)))/(pi*L*cosa*(k1*k1)*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+4
                    k0v[c] += (2*pi*L*cosk1xa*k1*sina*(B16*r + B26*(-L*sina + r + sina*xa)) + 2*pi*L*cosk1xb*k1*sina*(-B16*r - B26*(-L*sina + r + sina*xb)) + 2*sink1xa*((pi*pi)*B16*(k1*k1)*(r*r) + B26*sina*(-(L*L)*sina + (pi*pi)*(k1*k1)*r*(-L + xa))) + 2*sink1xb*(-(pi*pi)*B16*(k1*k1)*(r*r) + B26*sina*((L*L)*sina + (pi*pi)*(k1*k1)*r*(L - xb))))/(pi*L*cosa*(k1*k1)*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+0
                    k0v[c] += 2*r2*(-pi*L*cosk1xa*k1*sina*(A16*r + A26*(L*sina + r - sina*xa)) + pi*L*cosk1xb*k1*sina*(A16*r + A26*(L*sina + r - sina*xb)) + sink1xa*((pi*pi)*A16*(k1*k1)*r*(L*sina + r - sina*xa) - A26*(L*L)*(sina*sina)) + sink1xb*(-(pi*pi)*A16*(k1*k1)*r*(L*sina + r - sina*xb) + A26*(L*L)*(sina*sina)))/(pi*L*(k1*k1)*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+1
                    k0v[c] += 2*r2*(pi*L*cosk1xa*k1*(L - xa)*(A44*(cosa*cosa) + A66*(sina*sina)) - pi*L*cosk1xb*k1*(L - xb)*(A44*(cosa*cosa) + A66*(sina*sina)) + sink1xa*(A44*(L*L)*(cosa*cosa) + A66*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*r*(L*sina + r - sina*xa))) - sink1xb*(A44*(L*L)*(cosa*cosa) + A66*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*r*(L*sina + r - sina*xb))))/(pi*L*(k1*k1)*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+2
                    k0v[c] += 2*cosa*r2*(-pi*L*cosk1xa*k1*(A26*(L*sina + r - sina*xa) + A45*r) + pi*L*cosk1xb*k1*(A26*(L*sina + r - sina*xb) + A45*r) + sink1xa*(-A26*(L*L)*sina + (pi*pi)*A45*(k1*k1)*r*(L - xa)) + sink1xb*(A26*(L*L)*sina + (pi*pi)*A45*(k1*k1)*r*(-L + xb)))/(pi*L*(k1*k1)*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+3
                    k0v[c] += 2*r2*(pi*L*k1*(sink1xa*(A45*cosa*r*(L - xa) + sina*(B16*r + B26*(L*sina + r - sina*xa))) - sink1xb*(A45*cosa*r*(L - xb) + sina*(B16*r + B26*(L*sina + r - sina*xb)))) - cosk1xa*(A45*(L*L)*cosa*r - (pi*pi)*B16*(k1*k1)*r*(L*sina + r - sina*xa) + B26*(L*L)*(sina*sina)) + cosk1xb*(A45*(L*L)*cosa*r - (pi*pi)*B16*(k1*k1)*r*(L*sina + r - sina*xb) + B26*(L*L)*(sina*sina)))/(pi*L*(k1*k1)*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+4
                    k0v[c] += 2*r2*(-pi*L*cosk1xa*k1*(L - xa)*(A44*cosa*r - B66*(sina*sina)) + pi*L*cosk1xb*k1*(L - xb)*(A44*cosa*r - B66*(sina*sina)) + sink1xa*(-A44*(L*L)*cosa*r + B66*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*r*(L*sina + r - sina*xa))) + sink1xb*(A44*(L*L)*cosa*r - B66*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*r*(L*sina + r - sina*xb))))/(pi*L*(k1*k1)*r)

                else:
                    # k0_01 cond_2
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+3
                    k0v[c] += pi*sina*(xa - xb)*(2*B12*r + B22*sina*(-2*L + xa + xb))/(L*cosa*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+3
                    k0v[c] += -pi*r2*(xa - xb)*(A45*cosa*r*(-2*L + xa + xb) + B26*sina*(-2*r + sina*(-2*L + xa + xb)))/(L*r)

            for i1 in range(i0, m1+i0):
                cosi1xa = cos(pi*i1*xa/L)
                cosi1xb = cos(pi*i1*xb/L)
                sini1xa = sin(pi*i1*xa/L)
                sini1xb = sin(pi*i1*xb/L)
                cos2i1xa = cos(2*pi*i1*xa/L)
                cos2i1xb = cos(2*pi*i1*xb/L)
                cosi1xaxb = cos(pi*i1*(xa + xb)/L)
                sin2i1xa = sin(2*pi*i1*xa/L)
                sin2i1xb = sin(2*pi*i1*xb/L)
                sini1xa_xb = sin(pi*i1*(xa - xb)/L)
                sini1xaxb = sin(pi*i1*(xa + xb)/L)

                row = (i1-i0)*num1 + num0
                for k1 in range(i0, m1+i0):
                    col = (k1-i0)*num1 + num0

                    #NOTE symmetry
                    if row > col:
                        continue

                    cosk1xa = cos(pi*k1*xa/L)
                    cosk1xb = cos(pi*k1*xb/L)
                    sink1xa = sin(pi*k1*xa/L)
                    sink1xb = sin(pi*k1*xb/L)
                    if k1==i1:
                        if k1!=0:
                            # k0_11 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(-2*pi*A12*L*i1*r*sina*sini1xaxb + cosi1xaxb*(-(pi*pi)*A11*(i1*i1)*(r*r) + A22*(L*L)*(sina*sina))) - 2*pi*i1*(xa - xb)*((pi*pi)*A11*(i1*i1)*(r*r) + A22*(L*L)*(sina*sina)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += (L*sini1xa_xb*(pi*L*i1*r*sina*sini1xaxb*(A16 - A26) - cosi1xaxb*((pi*pi)*A16*(i1*i1)*(r*r) + A26*(L*L)*(sina*sina))) - pi*i1*(xa - xb)*((pi*pi)*A16*(i1*i1)*(r*r) - A26*(L*L)*(sina*sina)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += 0.5*cosa*(pi*A12*(cos2i1xa - cos2i1xb) + A22*sina*(L*sin2i1xa - L*sin2i1xb + 2*pi*i1*(-xa + xb))/(i1*r))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += sini1xa_xb*(-2*pi*B12*L*cosi1xaxb*i1*r*sina + sini1xaxb*((pi*pi)*B11*(i1*i1)*(r*r) - B22*(L*L)*(sina*sina)))/(L*i1*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += (L*sini1xa_xb*(pi*L*i1*r*sina*sini1xaxb*(B16 - B26) - cosi1xaxb*((pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))) - pi*i1*(xa - xb)*((pi*pi)*B16*(i1*i1)*(r*r) - B26*(L*L)*(sina*sina)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += (L*sini1xa_xb*(pi*L*i1*r*sina*sini1xaxb*(A16 - A26) - cosi1xaxb*((pi*pi)*A16*(i1*i1)*(r*r) + A26*(L*L)*(sina*sina))) - pi*i1*(xa - xb)*((pi*pi)*A16*(i1*i1)*(r*r) - A26*(L*L)*(sina*sina)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(2*pi*A66*L*i1*r*sina*sini1xaxb + cosi1xaxb*(-(pi*pi)*A66*(i1*i1)*(r*r) + (L*L)*(A44*(cosa*cosa) + A66*(sina*sina)))) - 2*pi*i1*(xa - xb)*((pi*pi)*A66*(i1*i1)*(r*r) + (L*L)*(A44*(cosa*cosa) + A66*(sina*sina))))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += (pi*A26*cosa*i1*sina*(xa - xb) - cosa*sini1xa_xb*(A26*L*cosi1xaxb*sina + pi*i1*r*sini1xaxb*(A26 - A45)))/(i1*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += (-(pi*pi)*(i1*i1)*r*sina*(B16 + B26)*(xa - xb) + sini1xa_xb*(pi*L*cosi1xaxb*i1*r*sina*(B16 - B26) + sini1xaxb*(A45*(L*L)*cosa*r + (pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))))/(L*i1*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(2*pi*B66*L*i1*r*sina*sini1xaxb - cosi1xaxb*(A44*(L*L)*cosa*r + B66*(-L*sina + pi*i1*r)*(L*sina + pi*i1*r))) - 2*pi*i1*(xa - xb)*(-A44*(L*L)*cosa*r + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i1*i1)*(r*r)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += 0.5*cosa*(pi*A12*(cos2i1xa - cos2i1xb) + A22*sina*(L*sin2i1xa - L*sin2i1xb + 2*pi*i1*(-xa + xb))/(i1*r))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += (pi*A26*cosa*i1*sina*(xa - xb) - cosa*sini1xa_xb*(A26*L*cosi1xaxb*sina + pi*i1*r*sini1xaxb*(A26 - A45)))/(i1*r)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.5*(L*(sin2i1xa - sin2i1xb)*(A22*(L*L)*(cosa*cosa) - (pi*pi)*A55*(i1*i1)*(r*r)) - 2*pi*i1*(xa - xb)*(A22*(L*L)*(cosa*cosa) + (pi*pi)*A55*(i1*i1)*(r*r)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+3
                            k0v[c] += 0.5*(B22*(L*L)*cos2i1xa*cosa*sina - B22*(L*L)*cos2i1xb*cosa*sina + pi*i1*r*(L*(-sin2i1xa + sin2i1xb)*(A55*r + B12*cosa) + 2*pi*i1*(xa - xb)*(-A55*r + B12*cosa)))/(L*i1*r)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+4
                            k0v[c] += (pi*B26*cosa*i1*sina*(xa - xb) - sini1xa_xb*(B26*L*cosa*cosi1xaxb*sina + pi*i1*r*sini1xaxb*(A45*r + B26*cosa)))/(i1*r)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+0
                            k0v[c] += sini1xa_xb*(-2*pi*B12*L*cosi1xaxb*i1*r*sina + sini1xaxb*((pi*pi)*B11*(i1*i1)*(r*r) - B22*(L*L)*(sina*sina)))/(L*i1*r)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+1
                            k0v[c] += (-(pi*pi)*(i1*i1)*r*sina*(B16 + B26)*(xa - xb) + sini1xa_xb*(pi*L*cosi1xaxb*i1*r*sina*(B16 - B26) + sini1xaxb*(A45*(L*L)*cosa*r + (pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))))/(L*i1*r)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+2
                            k0v[c] += 0.5*(B22*(L*L)*cos2i1xa*cosa*sina - B22*(L*L)*cos2i1xb*cosa*sina + pi*i1*r*(L*(-sin2i1xa + sin2i1xb)*(A55*r + B12*cosa) + 2*pi*i1*(xa - xb)*(-A55*r + B12*cosa)))/(L*i1*r)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(2*pi*D12*L*i1*r*sina*sini1xaxb - cosi1xaxb*(D22*(L*L)*(sina*sina) + (r*r)*(A55*(L*L) - (pi*pi)*D11*(i1*i1)))) - 2*pi*i1*(xa - xb)*(D22*(L*L)*(sina*sina) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i1*i1))))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+4
                            k0v[c] += (-(pi*pi)*(i1*i1)*r*sina*(D16 + D26)*(xa - xb) + sini1xa_xb*(pi*L*cosi1xaxb*i1*r*sina*(D16 - D26) + sini1xaxb*(-A45*(L*L)*(r*r) + (pi*pi)*D16*(i1*i1)*(r*r) + D26*(L*L)*(sina*sina))))/(L*i1*r)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+0
                            k0v[c] += (L*sini1xa_xb*(pi*L*i1*r*sina*sini1xaxb*(B16 - B26) - cosi1xaxb*((pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))) - pi*i1*(xa - xb)*((pi*pi)*B16*(i1*i1)*(r*r) - B26*(L*L)*(sina*sina)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+1
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(2*pi*B66*L*i1*r*sina*sini1xaxb - cosi1xaxb*(A44*(L*L)*cosa*r + B66*(-L*sina + pi*i1*r)*(L*sina + pi*i1*r))) - 2*pi*i1*(xa - xb)*(-A44*(L*L)*cosa*r + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i1*i1)*(r*r)))/((L*L)*i1*r)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+2
                            k0v[c] += (pi*B26*cosa*i1*sina*(xa - xb) - sini1xa_xb*(B26*L*cosa*cosi1xaxb*sina + pi*i1*r*sini1xaxb*(A45*r + B26*cosa)))/(i1*r)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+3
                            k0v[c] += (-(pi*pi)*(i1*i1)*r*sina*(D16 + D26)*(xa - xb) + sini1xa_xb*(pi*L*cosi1xaxb*i1*r*sina*(D16 - D26) + sini1xaxb*(-A45*(L*L)*(r*r) + (pi*pi)*D16*(i1*i1)*(r*r) + D26*(L*L)*(sina*sina))))/(L*i1*r)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+4
                            k0v[c] += 0.5*(2*L*sini1xa_xb*(2*pi*D66*L*i1*r*sina*sini1xaxb + cosi1xaxb*(D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) - (pi*pi)*D66*(i1*i1)))) - 2*pi*i1*(xa - xb)*(D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i1*i1))))/((L*L)*i1*r)
                        else:
                            # k0_11 cond_3
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += 2*pi*r*(A55 + D22*(sina*sina)/(r*r))*(-xa + xb)

                    else:
                        # k0_11 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += (-2*cosk1xa*k1*sini1xa*((pi*pi)*A11*(i1*i1)*(r*r) + A22*(L*L)*(sina*sina)) + 2*cosk1xb*k1*sini1xb*((pi*pi)*A11*(i1*i1)*(r*r) + A22*(L*L)*(sina*sina)) + 2*sink1xa*(pi*A12*L*r*sina*sini1xa*(-(i1*i1) + (k1*k1)) + cosi1xa*i1*((pi*pi)*A11*(k1*k1)*(r*r) + A22*(L*L)*(sina*sina))) - 2*sink1xb*(pi*A12*L*r*sina*sini1xb*(-(i1*i1) + (k1*k1)) + cosi1xb*i1*((pi*pi)*A11*(k1*k1)*(r*r) + A22*(L*L)*(sina*sina))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += (-2*pi*L*r*sina*sini1xb*sink1xb*(A16*(i1*i1) + A26*(k1*k1)) + 2*cosi1xa*i1*(pi*L*cosk1xa*k1*r*sina*(A16 + A26) + sink1xa*((pi*pi)*A16*(k1*k1)*(r*r) - A26*(L*L)*(sina*sina))) + 2*cosi1xb*i1*(-pi*L*cosk1xb*k1*r*sina*(A16 + A26) + sink1xb*(-(pi*pi)*A16*(k1*k1)*(r*r) + A26*(L*L)*(sina*sina))) + 2*cosk1xb*k1*sini1xb*((pi*pi)*A16*(i1*i1)*(r*r) - A26*(L*L)*(sina*sina)) + 2*sini1xa*(pi*L*r*sina*sink1xa*(A16*(i1*i1) + A26*(k1*k1)) + cosk1xa*k1*(-(pi*pi)*A16*(i1*i1)*(r*r) + A26*(L*L)*(sina*sina))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 2*cosa*(cosi1xa*i1*(-pi*A12*cosk1xa*k1*r + A22*L*sina*sink1xa) + cosk1xb*k1*(pi*A12*cosi1xb*i1*r + A22*L*sina*sini1xb) + i1*sink1xb*(pi*A12*i1*r*sini1xb - A22*L*cosi1xb*sina) - sini1xa*(pi*A12*(i1*i1)*r*sink1xa + A22*L*cosk1xa*k1*sina))/(r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += (2*pi*B12*L*cosk1xb*r*sina*sini1xb*(i1 - k1)*(i1 + k1) + 2*cosi1xa*cosk1xa*i1*((pi*pi)*B11*(k1*k1)*(r*r) + B22*(L*L)*(sina*sina)) - 2*cosi1xb*cosk1xb*i1*((pi*pi)*B11*(k1*k1)*(r*r) + B22*(L*L)*(sina*sina)) - 2*k1*sini1xb*sink1xb*((pi*pi)*B11*(i1*i1)*(r*r) + B22*(L*L)*(sina*sina)) + 2*sini1xa*(pi*B12*L*cosk1xa*r*sina*(-(i1*i1) + (k1*k1)) + k1*sink1xa*((pi*pi)*B11*(i1*i1)*(r*r) + B22*(L*L)*(sina*sina))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += (-2*pi*L*r*sina*sini1xb*sink1xb*(B16*(i1*i1) + B26*(k1*k1)) + 2*cosi1xa*i1*(pi*L*cosk1xa*k1*r*sina*(B16 + B26) + sink1xa*((pi*pi)*B16*(k1*k1)*(r*r) - B26*(L*L)*(sina*sina))) + 2*cosi1xb*i1*(-pi*L*cosk1xb*k1*r*sina*(B16 + B26) + sink1xb*(-(pi*pi)*B16*(k1*k1)*(r*r) + B26*(L*L)*(sina*sina))) + 2*cosk1xb*k1*sini1xb*((pi*pi)*B16*(i1*i1)*(r*r) - B26*(L*L)*(sina*sina)) + 2*sini1xa*(pi*L*r*sina*sink1xa*(B16*(i1*i1) + B26*(k1*k1)) + cosk1xa*k1*(-(pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += (2*pi*L*r*sina*sini1xb*sink1xb*(A16*(k1*k1) + A26*(i1*i1)) + 2*cosi1xa*i1*(-pi*L*cosk1xa*k1*r*sina*(A16 + A26) + sink1xa*((pi*pi)*A16*(k1*k1)*(r*r) - A26*(L*L)*(sina*sina))) + 2*cosi1xb*i1*(pi*L*cosk1xb*k1*r*sina*(A16 + A26) + sink1xb*(-(pi*pi)*A16*(k1*k1)*(r*r) + A26*(L*L)*(sina*sina))) + 2*cosk1xb*k1*sini1xb*((pi*pi)*A16*(i1*i1)*(r*r) - A26*(L*L)*(sina*sina)) + 2*sini1xa*(-pi*L*r*sina*sink1xa*(A16*(k1*k1) + A26*(i1*i1)) + cosk1xa*k1*(-(pi*pi)*A16*(i1*i1)*(r*r) + A26*(L*L)*(sina*sina))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += (-2*cosk1xa*k1*sini1xa*((pi*pi)*A66*(i1*i1)*(r*r) + (L*L)*(A44*(cosa*cosa) + A66*(sina*sina))) + 2*cosk1xb*k1*sini1xb*((pi*pi)*A66*(i1*i1)*(r*r) + (L*L)*(A44*(cosa*cosa) + A66*(sina*sina))) + 2*sink1xa*(pi*A66*L*r*sina*sini1xa*(i1 - k1)*(i1 + k1) + cosi1xa*i1*((pi*pi)*A66*(k1*k1)*(r*r) + (L*L)*(A44*(cosa*cosa) + A66*(sina*sina)))) - 2*sink1xb*(pi*A66*L*r*sina*sini1xb*(i1 - k1)*(i1 + k1) + cosi1xb*i1*((pi*pi)*A66*(k1*k1)*(r*r) + (L*L)*(A44*(cosa*cosa) + A66*(sina*sina)))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += 2*cosa*(-A26*L*cosk1xb*k1*sina*sini1xb - cosi1xa*i1*(A26*L*sina*sink1xa + pi*cosk1xa*k1*r*(A26 + A45)) + cosi1xb*i1*(A26*L*sina*sink1xb + pi*cosk1xb*k1*r*(A26 + A45)) + pi*r*sini1xb*sink1xb*(A26*(i1*i1) + A45*(k1*k1)) + sini1xa*(A26*L*cosk1xa*k1*sina - pi*r*sink1xa*(A26*(i1*i1) + A45*(k1*k1))))/(r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += (2*pi*L*cosk1xb*r*sina*sini1xb*(B16*(k1*k1) + B26*(i1*i1)) + 2*cosi1xa*i1*(pi*L*k1*r*sina*sink1xa*(B16 + B26) - cosk1xa*(A45*(L*L)*cosa*r - (pi*pi)*B16*(k1*k1)*(r*r) + B26*(L*L)*(sina*sina))) + 2*cosi1xb*i1*(-pi*L*k1*r*sina*sink1xb*(B16 + B26) + cosk1xb*(A45*(L*L)*cosa*r - (pi*pi)*B16*(k1*k1)*(r*r) + B26*(L*L)*(sina*sina))) + 2*k1*sini1xb*sink1xb*(A45*(L*L)*cosa*r - (pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina)) + 2*sini1xa*(-pi*L*cosk1xa*r*sina*(B16*(k1*k1) + B26*(i1*i1)) - k1*sink1xa*(A45*(L*L)*cosa*r - (pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += (2*cosk1xa*k1*sini1xa*(A44*(L*L)*cosa*r - B66*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r))) + 2*cosk1xb*k1*sini1xb*(-A44*(L*L)*cosa*r + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i1*i1)*(r*r)) + 2*sink1xa*(pi*B66*L*r*sina*sini1xa*(i1 - k1)*(i1 + k1) + cosi1xa*i1*(-A44*(L*L)*cosa*r + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k1*k1)*(r*r))) + 2*sink1xb*(pi*B66*L*r*sina*sini1xb*(-(i1*i1) + (k1*k1)) + cosi1xb*i1*(A44*(L*L)*cosa*r - B66*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r)))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 2*cosa*(cosi1xa*i1*(pi*A12*cosk1xa*k1*r + A22*L*sina*sink1xa) - cosi1xb*i1*(pi*A12*cosk1xb*k1*r + A22*L*sina*sink1xb) + k1*sini1xa*(pi*A12*k1*r*sink1xa - A22*L*cosk1xa*sina) + k1*sini1xb*(-pi*A12*k1*r*sink1xb + A22*L*cosk1xb*sina))/(r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += 2*cosa*(cosi1xa*i1*(-A26*L*sina*sink1xa + pi*cosk1xa*k1*r*(A26 + A45)) + cosi1xb*i1*(A26*L*sina*sink1xb - pi*cosk1xb*k1*r*(A26 + A45)) + sini1xa*(A26*L*cosk1xa*k1*sina + pi*r*sink1xa*(A26*(k1*k1) + A45*(i1*i1))) - sini1xb*(A26*L*cosk1xb*k1*sina + pi*r*sink1xb*(A26*(k1*k1) + A45*(i1*i1))))/(r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += (2*cosi1xa*i1*sink1xa*(A22*(L*L)*(cosa*cosa) + (pi*pi)*A55*(k1*k1)*(r*r)) - 2*cosi1xb*i1*sink1xb*(A22*(L*L)*(cosa*cosa) + (pi*pi)*A55*(k1*k1)*(r*r)) - 2*cosk1xa*k1*sini1xa*(A22*(L*L)*(cosa*cosa) + (pi*pi)*A55*(i1*i1)*(r*r)) + 2*cosk1xb*k1*sini1xb*(A22*(L*L)*(cosa*cosa) + (pi*pi)*A55*(i1*i1)*(r*r)))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += (-2*B22*L*cosa*k1*sina*sini1xb*sink1xb + 2*cosi1xa*i1*(B22*L*cosa*cosk1xa*sina + pi*k1*r*sink1xa*(A55*r - B12*cosa)) + 2*cosi1xb*i1*(-B22*L*cosa*cosk1xb*sina + pi*k1*r*sink1xb*(-A55*r + B12*cosa)) + 2*pi*cosk1xb*r*sini1xb*(A55*(i1*i1)*r - B12*cosa*(k1*k1)) + 2*sini1xa*(B22*L*cosa*k1*sina*sink1xa + pi*cosk1xa*r*(-A55*(i1*i1)*r + B12*cosa*(k1*k1))))/(r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += (-2*B26*L*cosa*cosk1xb*k1*sina*sini1xb + 2*cosi1xa*i1*(-B26*L*cosa*sina*sink1xa + pi*cosk1xa*k1*r*(-A45*r + B26*cosa)) + 2*cosi1xb*i1*(B26*L*cosa*sina*sink1xb + pi*cosk1xb*k1*r*(A45*r - B26*cosa)) + 2*pi*r*sini1xb*sink1xb*(A45*(i1*i1)*r - B26*cosa*(k1*k1)) + 2*sini1xa*(B26*L*cosa*cosk1xa*k1*sina + pi*r*sink1xa*(-A45*(i1*i1)*r + B26*cosa*(k1*k1))))/(r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += (-2*cosi1xa*(pi*B12*L*r*sina*sink1xa*(i1 - k1)*(i1 + k1) + cosk1xa*k1*((pi*pi)*B11*(i1*i1)*(r*r) + B22*(L*L)*(sina*sina))) + 2*cosi1xb*(pi*B12*L*r*sina*sink1xb*(i1 - k1)*(i1 + k1) + cosk1xb*k1*((pi*pi)*B11*(i1*i1)*(r*r) + B22*(L*L)*(sina*sina))) - 2*i1*(sini1xa*sink1xa - sini1xb*sink1xb)*((pi*pi)*B11*(k1*k1)*(r*r) + B22*(L*L)*(sina*sina)))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += (2*cosi1xa*(pi*L*r*sina*sink1xa*(B16*(i1*i1) + B26*(k1*k1)) + cosk1xa*k1*(A45*(L*L)*cosa*r - (pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))) - 2*cosi1xb*(pi*L*r*sina*sink1xb*(B16*(i1*i1) + B26*(k1*k1)) + cosk1xb*k1*(A45*(L*L)*cosa*r - (pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))) - 2*i1*(pi*L*cosk1xa*k1*r*sina*sini1xa*(B16 + B26) - sini1xa*sink1xa*(A45*(L*L)*cosa*r - (pi*pi)*B16*(k1*k1)*(r*r) + B26*(L*L)*(sina*sina)) + sini1xb*(-pi*L*cosk1xb*k1*r*sina*(B16 + B26) + sink1xb*(A45*(L*L)*cosa*r - (pi*pi)*B16*(k1*k1)*(r*r) + B26*(L*L)*(sina*sina)))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += (-2*cosi1xa*(B22*L*cosa*cosk1xa*k1*sina + pi*r*sink1xa*(-A55*(k1*k1)*r + B12*cosa*(i1*i1))) + 2*cosi1xb*(B22*L*cosa*cosk1xb*k1*sina + pi*r*sink1xb*(-A55*(k1*k1)*r + B12*cosa*(i1*i1))) + 2*i1*(-B22*L*cosa*sina*sini1xa*sink1xa + pi*cosk1xa*k1*r*sini1xa*(-A55*r + B12*cosa) + sini1xb*(B22*L*cosa*sina*sink1xb + pi*cosk1xb*k1*r*(A55*r - B12*cosa))))/(r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += (2*cosi1xa*(pi*D12*L*cosk1xa*r*sina*(-(i1*i1) + (k1*k1)) + k1*sink1xa*(D22*(L*L)*(sina*sina) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i1*i1)))) + 2*cosi1xb*(pi*D12*L*cosk1xb*r*sina*(i1 - k1)*(i1 + k1) - k1*sink1xb*(D22*(L*L)*(sina*sina) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i1*i1)))) + 2*i1*(-cosk1xa*sini1xa + cosk1xb*sini1xb)*(D22*(L*L)*(sina*sina) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(k1*k1))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += (2*cosi1xa*(pi*L*r*sina*sink1xa*(D16*(i1*i1) + D26*(k1*k1)) - cosk1xa*k1*(-D26*(L*L)*(sina*sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i1*i1)))) + 2*cosi1xb*(-pi*L*r*sina*sink1xb*(D16*(i1*i1) + D26*(k1*k1)) + cosk1xb*k1*(-D26*(L*L)*(sina*sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i1*i1)))) + 2*i1*(pi*L*cosk1xb*k1*r*sina*sini1xb*(D16 + D26) + sini1xa*(-pi*L*cosk1xa*k1*r*sina*(D16 + D26) - sink1xa*(-D26*(L*L)*(sina*sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k1*k1)))) + sini1xb*sink1xb*(-D26*(L*L)*(sina*sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k1*k1)))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += (2*pi*L*r*sina*sini1xb*sink1xb*(B16*(k1*k1) + B26*(i1*i1)) + 2*cosi1xa*i1*(-pi*L*cosk1xa*k1*r*sina*(B16 + B26) + sink1xa*((pi*pi)*B16*(k1*k1)*(r*r) - B26*(L*L)*(sina*sina))) + 2*cosi1xb*i1*(pi*L*cosk1xb*k1*r*sina*(B16 + B26) + sink1xb*(-(pi*pi)*B16*(k1*k1)*(r*r) + B26*(L*L)*(sina*sina))) + 2*cosk1xb*k1*sini1xb*((pi*pi)*B16*(i1*i1)*(r*r) - B26*(L*L)*(sina*sina)) + 2*sini1xa*(-pi*L*r*sina*sink1xa*(B16*(k1*k1) + B26*(i1*i1)) + cosk1xa*k1*(-(pi*pi)*B16*(i1*i1)*(r*r) + B26*(L*L)*(sina*sina))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += (2*cosk1xa*k1*sini1xa*(A44*(L*L)*cosa*r - B66*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r))) + 2*cosk1xb*k1*sini1xb*(-A44*(L*L)*cosa*r + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i1*i1)*(r*r)) + 2*sink1xa*(pi*B66*L*r*sina*sini1xa*(i1 - k1)*(i1 + k1) + cosi1xa*i1*(-A44*(L*L)*cosa*r + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k1*k1)*(r*r))) + 2*sink1xb*(pi*B66*L*r*sina*sini1xb*(-(i1*i1) + (k1*k1)) + cosi1xb*i1*(A44*(L*L)*cosa*r - B66*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r)))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += (-2*B26*L*cosa*cosk1xb*k1*sina*sini1xb - 2*cosi1xa*i1*(B26*L*cosa*sina*sink1xa + pi*cosk1xa*k1*r*(-A45*r + B26*cosa)) + 2*cosi1xb*i1*(B26*L*cosa*sina*sink1xb + pi*cosk1xb*k1*r*(-A45*r + B26*cosa)) + 2*pi*r*sini1xb*sink1xb*(-A45*(k1*k1)*r + B26*cosa*(i1*i1)) + 2*sini1xa*(B26*L*cosa*cosk1xa*k1*sina + pi*r*sink1xa*(A45*(k1*k1)*r - B26*cosa*(i1*i1))))/(r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += (2*pi*L*cosk1xb*r*sina*sini1xb*(D16*(k1*k1) + D26*(i1*i1)) + 2*cosi1xa*i1*(pi*L*k1*r*sina*sink1xa*(D16 + D26) + cosk1xa*(-D26*(L*L)*(sina*sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k1*k1)))) + 2*cosi1xb*i1*(-pi*L*k1*r*sina*sink1xb*(D16 + D26) - cosk1xb*(-D26*(L*L)*(sina*sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k1*k1)))) - 2*k1*sini1xb*sink1xb*(-D26*(L*L)*(sina*sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i1*i1))) + 2*sini1xa*(-pi*L*cosk1xa*r*sina*(D16*(k1*k1) + D26*(i1*i1)) + k1*sink1xa*(-D26*(L*L)*(sina*sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i1*i1)))))/(L*r*(i1 - k1)*(i1 + k1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += (-2*cosk1xa*k1*sini1xa*(D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i1*i1))) + 2*cosk1xb*k1*sini1xb*(D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i1*i1))) + 2*sink1xa*(pi*D66*L*r*sina*sini1xa*(i1 - k1)*(i1 + k1) + cosi1xa*i1*(D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(k1*k1)))) - 2*sink1xb*(pi*D66*L*r*sina*sini1xb*(i1 - k1)*(i1 + k1) + cosi1xb*i1*(D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(k1*k1)))))/(L*r*(i1 - k1)*(i1 + k1))

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

                            #NOTE symmetry
                            if row > col:
                                continue

                            if k2==i2 and l2==j2:
                                if k2!=0:
                                    # k0_22 cond_1
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(-2*pi*A12*L*i2*r*sina*sini2xaxb + cosi2xaxb*(-(pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) - 2*pi*i2*(xa - xb)*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+2
                                    k0v[c] += 0.5*(L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(A16 - A26) + cosi2xaxb*(-(pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) - pi*i2*(xa - xb)*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*j2*(pi*cos2i2xa*i2*r*(-A12 + A66) + pi*cos2i2xb*i2*r*(A12 - A66) + sina*(A22 + A66)*(L*(-sin2i2xa + sin2i2xb) + 2*pi*i2*(xa - xb)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*cosa*(pi*A12*(cos2i2xa - cos2i2xb) + A22*sina*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/(i2*r))
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*A26*cosa*j2*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+6
                                    k0v[c] += -0.5*sini2xa_xb*(2*pi*B12*L*cosi2xaxb*i2*r*sina + sini2xaxb*(-(pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+7
                                    k0v[c] += (pi*pi)*B16*i2*j2*(xa - xb)/L
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+8
                                    k0v[c] += 0.5*(L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(B16 - B26) + cosi2xaxb*(-(pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) - pi*i2*(xa - xb)*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+9
                                    k0v[c] += 0.25*j2*(pi*cos2i2xa*i2*r*(-B12 + B66) + pi*cos2i2xb*i2*r*(B12 - B66) + sina*(B22 + B66)*(L*(-sin2i2xa + sin2i2xb) + 2*pi*i2*(xa - xb)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(-2*pi*A12*L*i2*r*sina*sini2xaxb + cosi2xaxb*(-(pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) - 2*pi*i2*(xa - xb)*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*j2*(pi*cos2i2xa*i2*r*(A12 - A66) + pi*cos2i2xb*i2*r*(-A12 + A66) + sina*(A22 + A66)*(L*(sin2i2xa - sin2i2xb) + 2*pi*i2*(-xa + xb)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+3
                                    k0v[c] += 0.5*(L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(A16 - A26) + cosi2xaxb*(-(pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) - pi*i2*(xa - xb)*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*A26*cosa*j2*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*cosa*(pi*A12*(cos2i2xa - cos2i2xb) + A22*sina*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/(i2*r))
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+6
                                    k0v[c] += (pi*pi)*B16*i2*j2*(-xa + xb)/L
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+7
                                    k0v[c] += -0.5*sini2xa_xb*(2*pi*B12*L*cosi2xaxb*i2*r*sina + sini2xaxb*(-(pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+8
                                    k0v[c] += 0.25*j2*(pi*cos2i2xa*i2*r*(B12 - B66) + pi*cos2i2xb*i2*r*(-B12 + B66) + sina*(B22 + B66)*(L*(sin2i2xa - sin2i2xb) + 2*pi*i2*(-xa + xb)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+9
                                    k0v[c] += 0.5*(L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(B16 - B26) + cosi2xaxb*(-(pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) - pi*i2*(xa - xb)*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+0
                                    k0v[c] += 0.5*(L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(A16 - A26) + cosi2xaxb*(-(pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) - pi*i2*(xa - xb)*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*j2*(pi*cos2i2xa*i2*r*(A12 - A66) + pi*cos2i2xb*i2*r*(-A12 + A66) + sina*(A22 + A66)*(L*(sin2i2xa - sin2i2xb) + 2*pi*i2*(-xa + xb)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*A66*L*i2*r*sina*sini2xaxb + cosi2xaxb*(-(pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina)))) - 2*pi*i2*(xa - xb)*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*cosa*(2*pi*A26*i2*sina*(xa - xb) - 2*sini2xa_xb*(A26*L*cosi2xaxb*sina + pi*i2*r*sini2xaxb*(A26 - A45)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*cosa*j2*(A22 + A44)*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+6
                                    k0v[c] += 0.25*((pi*pi)*(i2*i2)*r*sina*(-2*B16 - 2*B26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(B16 - B26) + sini2xaxb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+7
                                    k0v[c] += 0.25*j2*((L*L)*cos2i2xa*sina*(B22 + B66) - (L*L)*cos2i2xb*sina*(B22 + B66) + pi*i2*r*(L*(B12 - B66)*(-sin2i2xa + sin2i2xb) + pi*i2*(2*B12 + 2*B66)*(xa - xb)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+8
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*B66*L*i2*r*sina*sini2xaxb + cosi2xaxb*(B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) - r*(A44*(L*L)*cosa + (pi*pi)*B66*(i2*i2)*r))) - 2*pi*i2*(xa - xb)*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*j2*(pi*cos2i2xa*i2*r*(-A12 + A66) + pi*cos2i2xb*i2*r*(A12 - A66) + sina*(A22 + A66)*(L*(-sin2i2xa + sin2i2xb) + 2*pi*i2*(xa - xb)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+1
                                    k0v[c] += 0.5*(L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(A16 - A26) + cosi2xaxb*(-(pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) - pi*i2*(xa - xb)*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*A66*L*i2*r*sina*sini2xaxb + cosi2xaxb*(-(pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina)))) - 2*pi*i2*(xa - xb)*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*cosa*j2*(A22 + A44)*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*cosa*(2*pi*A26*i2*sina*(xa - xb) - 2*sini2xa_xb*(A26*L*cosi2xaxb*sina + pi*i2*r*sini2xaxb*(A26 - A45)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+6
                                    k0v[c] += 0.25*j2*(-(L*L)*cos2i2xa*sina*(B22 + B66) + (L*L)*cos2i2xb*sina*(B22 + B66) + pi*i2*r*(L*(B12 - B66)*(sin2i2xa - sin2i2xb) + pi*i2*(-2*B12 - 2*B66)*(xa - xb)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+7
                                    k0v[c] += 0.25*((pi*pi)*(i2*i2)*r*sina*(-2*B16 - 2*B26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(B16 - B26) + sini2xaxb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+9
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*B66*L*i2*r*sina*sini2xaxb + cosi2xaxb*(B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) - r*(A44*(L*L)*cosa + (pi*pi)*B66*(i2*i2)*r))) - 2*pi*i2*(xa - xb)*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*cosa*(pi*A12*(cos2i2xa - cos2i2xb) + A22*sina*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/(i2*r))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*A26*cosa*j2*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*cosa*(2*pi*A26*i2*sina*(xa - xb) - 2*sini2xa_xb*(A26*L*cosi2xaxb*sina + pi*i2*r*sini2xaxb*(A26 - A45)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*cosa*j2*(A22 + A44)*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*(L*(sin2i2xa - sin2i2xb)*(-(pi*pi)*A55*(i2*i2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))) - 2*pi*i2*(xa - xb)*((pi*pi)*A55*(i2*i2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+6
                                    k0v[c] += 0.25*(B22*(L*L)*cos2i2xa*cosa*sina - B22*(L*L)*cos2i2xb*cosa*sina + pi*i2*r*(L*(-sin2i2xa + sin2i2xb)*(A55*r + B12*cosa) + 2*pi*i2*(xa - xb)*(-A55*r + B12*cosa)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+7
                                    k0v[c] += 0.25*L*j2*(cos2i2xa - cos2i2xb)*(A45*r - B26*cosa)/(i2*r)
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+8
                                    k0v[c] += 0.5*(pi*B26*cosa*i2*sina*(xa - xb) - sini2xa_xb*(B26*L*cosa*cosi2xaxb*sina + pi*i2*r*sini2xaxb*(A45*r + B26*cosa)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+9
                                    k0v[c] += 0.25*j2*(-A44*r + B22*cosa)*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*A26*cosa*j2*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*cosa*(pi*A12*(cos2i2xa - cos2i2xb) + A22*sina*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/(i2*r))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*cosa*j2*(A22 + A44)*(L*sin2i2xa - L*sin2i2xb + 2*pi*i2*(-xa + xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*cosa*(2*pi*A26*i2*sina*(xa - xb) - 2*sini2xa_xb*(A26*L*cosi2xaxb*sina + pi*i2*r*sini2xaxb*(A26 - A45)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*(L*(sin2i2xa - sin2i2xb)*(-(pi*pi)*A55*(i2*i2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))) - 2*pi*i2*(xa - xb)*((pi*pi)*A55*(i2*i2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+6
                                    k0v[c] += 0.25*L*j2*(cos2i2xa - cos2i2xb)*(-A45*r + B26*cosa)/(i2*r)
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+7
                                    k0v[c] += 0.25*(B22*(L*L)*cos2i2xa*cosa*sina - B22*(L*L)*cos2i2xb*cosa*sina + pi*i2*r*(L*(-sin2i2xa + sin2i2xb)*(A55*r + B12*cosa) + 2*pi*i2*(xa - xb)*(-A55*r + B12*cosa)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+8
                                    k0v[c] += 0.25*j2*(A44*r - B22*cosa)*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+9
                                    k0v[c] += 0.5*(pi*B26*cosa*i2*sina*(xa - xb) - sini2xa_xb*(B26*L*cosa*cosi2xaxb*sina + pi*i2*r*sini2xaxb*(A45*r + B26*cosa)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+0
                                    k0v[c] += -0.5*sini2xa_xb*(2*pi*B12*L*cosi2xaxb*i2*r*sina + sini2xaxb*(-(pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+1
                                    k0v[c] += (pi*pi)*B16*i2*j2*(-xa + xb)/L
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*((pi*pi)*(i2*i2)*r*sina*(-2*B16 - 2*B26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(B16 - B26) + sini2xaxb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*j2*(-(L*L)*cos2i2xa*sina*(B22 + B66) + (L*L)*cos2i2xb*sina*(B22 + B66) + pi*i2*r*(L*(B12 - B66)*(sin2i2xa - sin2i2xb) + pi*i2*(-2*B12 - 2*B66)*(xa - xb)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*(B22*(L*L)*cos2i2xa*cosa*sina - B22*(L*L)*cos2i2xb*cosa*sina + pi*i2*r*(L*(-sin2i2xa + sin2i2xb)*(A55*r + B12*cosa) + 2*pi*i2*(xa - xb)*(-A55*r + B12*cosa)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*L*j2*(cos2i2xa - cos2i2xb)*(-A45*r + B26*cosa)/(i2*r)
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+6
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*D12*L*i2*r*sina*sini2xaxb - cosi2xaxb*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) - (pi*pi)*D11*(i2*i2)))) - 2*pi*i2*(xa - xb)*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i2*i2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+8
                                    k0v[c] += 0.25*((pi*pi)*(i2*i2)*r*sina*(-2*D16 - 2*D26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(D16 - D26) + sini2xaxb*(D26*(L*L)*(-(j2*j2) + (sina*sina)) + (r*r)*(-A45*(L*L) + (pi*pi)*D16*(i2*i2)))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+9
                                    k0v[c] += 0.25*j2*(-(L*L)*cos2i2xa*sina*(D22 + D66) + (L*L)*cos2i2xb*sina*(D22 + D66) + pi*i2*r*(L*(D12 - D66)*(sin2i2xa - sin2i2xb) + pi*i2*(-2*D12 - 2*D66)*(xa - xb)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+0
                                    k0v[c] += (pi*pi)*B16*i2*j2*(xa - xb)/L
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+1
                                    k0v[c] += -0.5*sini2xa_xb*(2*pi*B12*L*cosi2xaxb*i2*r*sina + sini2xaxb*(-(pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*j2*((L*L)*cos2i2xa*sina*(B22 + B66) - (L*L)*cos2i2xb*sina*(B22 + B66) + pi*i2*r*(L*(B12 - B66)*(-sin2i2xa + sin2i2xb) + pi*i2*(2*B12 + 2*B66)*(xa - xb)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*((pi*pi)*(i2*i2)*r*sina*(-2*B16 - 2*B26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(B16 - B26) + sini2xaxb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*L*j2*(cos2i2xa - cos2i2xb)*(A45*r - B26*cosa)/(i2*r)
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*(B22*(L*L)*cos2i2xa*cosa*sina - B22*(L*L)*cos2i2xb*cosa*sina + pi*i2*r*(L*(-sin2i2xa + sin2i2xb)*(A55*r + B12*cosa) + 2*pi*i2*(xa - xb)*(-A55*r + B12*cosa)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+7
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*D12*L*i2*r*sina*sini2xaxb - cosi2xaxb*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) - (pi*pi)*D11*(i2*i2)))) - 2*pi*i2*(xa - xb)*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i2*i2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+8
                                    k0v[c] += 0.25*j2*((L*L)*cos2i2xa*sina*(D22 + D66) - (L*L)*cos2i2xb*sina*(D22 + D66) + pi*i2*r*(L*(D12 - D66)*(-sin2i2xa + sin2i2xb) + pi*i2*(2*D12 + 2*D66)*(xa - xb)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+9
                                    k0v[c] += 0.25*((pi*pi)*(i2*i2)*r*sina*(-2*D16 - 2*D26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(D16 - D26) + sini2xaxb*(D26*(L*L)*(-(j2*j2) + (sina*sina)) + (r*r)*(-A45*(L*L) + (pi*pi)*D16*(i2*i2)))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+8
                                    k0c[c] = col+0
                                    k0v[c] += 0.5*(L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(B16 - B26) + cosi2xaxb*(-(pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) - pi*i2*(xa - xb)*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+8
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*j2*(pi*cos2i2xa*i2*r*(B12 - B66) + pi*cos2i2xb*i2*r*(-B12 + B66) + sina*(B22 + B66)*(L*(sin2i2xa - sin2i2xb) + 2*pi*i2*(-xa + xb)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+8
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*B66*L*i2*r*sina*sini2xaxb + cosi2xaxb*(B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) - r*(A44*(L*L)*cosa + (pi*pi)*B66*(i2*i2)*r))) - 2*pi*i2*(xa - xb)*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+8
                                    k0c[c] = col+4
                                    k0v[c] += 0.5*(pi*B26*cosa*i2*sina*(xa - xb) - sini2xa_xb*(B26*L*cosa*cosi2xaxb*sina + pi*i2*r*sini2xaxb*(A45*r + B26*cosa)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+8
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*j2*(A44*r - B22*cosa)*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+8
                                    k0c[c] = col+6
                                    k0v[c] += 0.25*((pi*pi)*(i2*i2)*r*sina*(-2*D16 - 2*D26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(D16 - D26) + sini2xaxb*(D26*(L*L)*(-(j2*j2) + (sina*sina)) + (r*r)*(-A45*(L*L) + (pi*pi)*D16*(i2*i2)))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+8
                                    k0c[c] = col+7
                                    k0v[c] += 0.25*j2*((L*L)*cos2i2xa*sina*(D22 + D66) - (L*L)*cos2i2xb*sina*(D22 + D66) + pi*i2*r*(L*(D12 - D66)*(-sin2i2xa + sin2i2xb) + pi*i2*(2*D12 + 2*D66)*(xa - xb)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+8
                                    k0c[c] = col+8
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*D66*L*i2*r*sina*sini2xaxb + cosi2xaxb*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) - (pi*pi)*D66*(i2*i2)))) - 2*pi*i2*(xa - xb)*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i2*i2))))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+9
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*j2*(pi*cos2i2xa*i2*r*(-B12 + B66) + pi*cos2i2xb*i2*r*(B12 - B66) + sina*(B22 + B66)*(L*(-sin2i2xa + sin2i2xb) + 2*pi*i2*(xa - xb)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+9
                                    k0c[c] = col+1
                                    k0v[c] += 0.5*(L*sini2xa_xb*(pi*L*i2*r*sina*sini2xaxb*(B16 - B26) + cosi2xaxb*(-(pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) - pi*i2*(xa - xb)*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+9
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*B66*L*i2*r*sina*sini2xaxb + cosi2xaxb*(B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) - r*(A44*(L*L)*cosa + (pi*pi)*B66*(i2*i2)*r))) - 2*pi*i2*(xa - xb)*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)))/((L*L)*i2*r)
                                    c += 1
                                    k0r[c] = row+9
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*j2*(-A44*r + B22*cosa)*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb))/(i2*r)
                                    c += 1
                                    k0r[c] = row+9
                                    k0c[c] = col+5
                                    k0v[c] += 0.5*(pi*B26*cosa*i2*sina*(xa - xb) - sini2xa_xb*(B26*L*cosa*cosi2xaxb*sina + pi*i2*r*sini2xaxb*(A45*r + B26*cosa)))/(i2*r)
                                    c += 1
                                    k0r[c] = row+9
                                    k0c[c] = col+6
                                    k0v[c] += 0.25*j2*(-(L*L)*cos2i2xa*sina*(D22 + D66) + (L*L)*cos2i2xb*sina*(D22 + D66) + pi*i2*r*(L*(D12 - D66)*(sin2i2xa - sin2i2xb) + pi*i2*(-2*D12 - 2*D66)*(xa - xb)))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+9
                                    k0c[c] = col+7
                                    k0v[c] += 0.25*((pi*pi)*(i2*i2)*r*sina*(-2*D16 - 2*D26)*(xa - xb) + 2*sini2xa_xb*(pi*L*cosi2xaxb*i2*r*sina*(D16 - D26) + sini2xaxb*(D26*(L*L)*(-(j2*j2) + (sina*sina)) + (r*r)*(-A45*(L*L) + (pi*pi)*D16*(i2*i2)))))/(L*i2*r)
                                    c += 1
                                    k0r[c] = row+9
                                    k0c[c] = col+9
                                    k0v[c] += 0.25*(2*L*sini2xa_xb*(2*pi*D66*L*i2*r*sina*sini2xaxb + cosi2xaxb*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) - (pi*pi)*D66*(i2*i2)))) - 2*pi*i2*(xa - xb)*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i2*i2))))/((L*L)*i2*r)

                                else:
                                    # k0_22 cond_5
                                    c += 1
                                    k0r[c] = row+6
                                    k0c[c] = col+6
                                    k0v[c] += pi*(-xa + xb)*(A55*(r*r) + D22*(sina*sina) + D66*(j2*j2))/r
                                    c += 1
                                    k0r[c] = row+7
                                    k0c[c] = col+7
                                    k0v[c] += pi*(-xa + xb)*(A55*(r*r) + D22*(sina*sina) + D66*(j2*j2))/r

                            elif k2!=i2 and l2==j2:
                                # k0_22 cond_2
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += (-cosk2xa*k2*sini2xa*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))) + cosk2xb*k2*sini2xb*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))) + sink2xa*(pi*A12*L*r*sina*sini2xa*(-(i2*i2) + (k2*k2)) + cosi2xa*i2*((pi*pi)*A11*(k2*k2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) - sink2xb*(pi*A12*L*r*sina*sini2xb*(-(i2*i2) + (k2*k2)) + cosi2xb*i2*((pi*pi)*A11*(k2*k2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+1
                                k0v[c] += pi*A16*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += (-pi*L*r*sina*sini2xb*sink2xb*(A16*(i2*i2) + A26*(k2*k2)) + cosi2xa*i2*(pi*L*cosk2xa*k2*r*sina*(A16 + A26) + sink2xa*((pi*pi)*A16*(k2*k2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) + cosi2xb*i2*(-pi*L*cosk2xb*k2*r*sina*(A16 + A26) - sink2xb*((pi*pi)*A16*(k2*k2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) + cosk2xb*k2*sini2xb*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina)) + sini2xa*(pi*L*r*sina*sink2xa*(A16*(i2*i2) + A26*(k2*k2)) - cosk2xa*k2*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += j2*(-L*cosk2xb*k2*sina*sini2xb*(A22 + A66) + cosi2xa*i2*(-L*sina*sink2xa*(A22 + A66) + pi*cosk2xa*k2*r*(A12 + A66)) + cosi2xb*i2*(L*sina*sink2xb*(A22 + A66) - pi*cosk2xb*k2*r*(A12 + A66)) - pi*r*sini2xb*sink2xb*(A12*(i2*i2) + A66*(k2*k2)) + sini2xa*(L*cosk2xa*k2*sina*(A22 + A66) + pi*r*sink2xa*(A12*(i2*i2) + A66*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += cosa*(cosi2xa*i2*(-pi*A12*cosk2xa*k2*r + A22*L*sina*sink2xa) + cosk2xb*k2*(pi*A12*cosi2xb*i2*r + A22*L*sina*sini2xb) + i2*sink2xb*(pi*A12*i2*r*sini2xb - A22*L*cosi2xb*sina) - sini2xa*(pi*A12*(i2*i2)*r*sink2xa + A22*L*cosk2xa*k2*sina))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+5
                                k0v[c] += A26*L*cosa*j2*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+6
                                k0v[c] += (pi*B12*L*cosk2xb*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xa*cosk2xa*i2*((pi*pi)*B11*(k2*k2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))) - cosi2xb*cosk2xb*i2*((pi*pi)*B11*(k2*k2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))) - k2*sini2xb*sink2xb*((pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))) + sini2xa*(pi*B12*L*cosk2xa*r*sina*(-(i2*i2) + (k2*k2)) + k2*sink2xa*((pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+7
                                k0v[c] += pi*B16*j2*(-2*cosi2xa*i2*k2*sink2xa + 2*cosi2xb*i2*k2*sink2xb + cosk2xa*sini2xa*((i2*i2) + (k2*k2)) - cosk2xb*sini2xb*((i2*i2) + (k2*k2)))/((i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+8
                                k0v[c] += (-pi*L*r*sina*sini2xb*sink2xb*(B16*(i2*i2) + B26*(k2*k2)) + cosi2xa*i2*(pi*L*cosk2xa*k2*r*sina*(B16 + B26) + sink2xa*((pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) + cosi2xb*i2*(-pi*L*cosk2xb*k2*r*sina*(B16 + B26) - sink2xb*((pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) + cosk2xb*k2*sini2xb*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina)) + sini2xa*(pi*L*r*sina*sink2xa*(B16*(i2*i2) + B26*(k2*k2)) - cosk2xa*k2*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+9
                                k0v[c] += j2*(-L*cosk2xb*k2*sina*sini2xb*(B22 + B66) + cosi2xa*i2*(-L*sina*sink2xa*(B22 + B66) + pi*cosk2xa*k2*r*(B12 + B66)) + cosi2xb*i2*(L*sina*sink2xb*(B22 + B66) - pi*cosk2xb*k2*r*(B12 + B66)) - pi*r*sini2xb*sink2xb*(B12*(i2*i2) + B66*(k2*k2)) + sini2xa*(L*cosk2xa*k2*sina*(B22 + B66) + pi*r*sink2xa*(B12*(i2*i2) + B66*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+0
                                k0v[c] += -pi*A16*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += (-cosk2xa*k2*sini2xa*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))) + cosk2xb*k2*sini2xb*((pi*pi)*A11*(i2*i2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2))) + sink2xa*(pi*A12*L*r*sina*sini2xa*(-(i2*i2) + (k2*k2)) + cosi2xa*i2*((pi*pi)*A11*(k2*k2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))) - sink2xb*(pi*A12*L*r*sina*sini2xb*(-(i2*i2) + (k2*k2)) + cosi2xb*i2*((pi*pi)*A11*(k2*k2)*(r*r) + (L*L)*(A22*(sina*sina) + A66*(j2*j2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += j2*(L*cosk2xb*k2*sina*sini2xb*(A22 + A66) + cosi2xa*i2*(L*sina*sink2xa*(A22 + A66) - pi*cosk2xa*k2*r*(A12 + A66)) + cosi2xb*i2*(-L*sina*sink2xb*(A22 + A66) + pi*cosk2xb*k2*r*(A12 + A66)) + pi*r*sini2xb*sink2xb*(A12*(i2*i2) + A66*(k2*k2)) + sini2xa*(-L*cosk2xa*k2*sina*(A22 + A66) - pi*r*sink2xa*(A12*(i2*i2) + A66*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+3
                                k0v[c] += (-pi*L*r*sina*sini2xb*sink2xb*(A16*(i2*i2) + A26*(k2*k2)) + cosi2xa*i2*(pi*L*cosk2xa*k2*r*sina*(A16 + A26) + sink2xa*((pi*pi)*A16*(k2*k2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) + cosi2xb*i2*(-pi*L*cosk2xb*k2*r*sina*(A16 + A26) - sink2xb*((pi*pi)*A16*(k2*k2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) + cosk2xb*k2*sini2xb*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina)) + sini2xa*(pi*L*r*sina*sink2xa*(A16*(i2*i2) + A26*(k2*k2)) - cosk2xa*k2*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+4
                                k0v[c] += A26*L*cosa*j2*(-cosi2xa*i2*sink2xa + cosi2xb*i2*sink2xb + cosk2xa*k2*sini2xa - cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+5
                                k0v[c] += cosa*(cosi2xa*i2*(-pi*A12*cosk2xa*k2*r + A22*L*sina*sink2xa) + cosk2xb*k2*(pi*A12*cosi2xb*i2*r + A22*L*sina*sini2xb) + i2*sink2xb*(pi*A12*i2*r*sini2xb - A22*L*cosi2xb*sina) - sini2xa*(pi*A12*(i2*i2)*r*sink2xa + A22*L*cosk2xa*k2*sina))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+6
                                k0v[c] += pi*B16*j2*(2*cosi2xa*i2*k2*sink2xa - 2*cosi2xb*i2*k2*sink2xb - cosk2xa*sini2xa*((i2*i2) + (k2*k2)) + cosk2xb*sini2xb*((i2*i2) + (k2*k2)))/((i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+7
                                k0v[c] += (pi*B12*L*cosk2xb*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xa*cosk2xa*i2*((pi*pi)*B11*(k2*k2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))) - cosi2xb*cosk2xb*i2*((pi*pi)*B11*(k2*k2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))) - k2*sini2xb*sink2xb*((pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))) + sini2xa*(pi*B12*L*cosk2xa*r*sina*(-(i2*i2) + (k2*k2)) + k2*sink2xa*((pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+8
                                k0v[c] += j2*(L*cosk2xb*k2*sina*sini2xb*(B22 + B66) + cosi2xa*i2*(L*sina*sink2xa*(B22 + B66) - pi*cosk2xa*k2*r*(B12 + B66)) + cosi2xb*i2*(-L*sina*sink2xb*(B22 + B66) + pi*cosk2xb*k2*r*(B12 + B66)) + pi*r*sini2xb*sink2xb*(B12*(i2*i2) + B66*(k2*k2)) + sini2xa*(-L*cosk2xa*k2*sina*(B22 + B66) - pi*r*sink2xa*(B12*(i2*i2) + B66*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+9
                                k0v[c] += (-pi*L*r*sina*sini2xb*sink2xb*(B16*(i2*i2) + B26*(k2*k2)) + cosi2xa*i2*(pi*L*cosk2xa*k2*r*sina*(B16 + B26) + sink2xa*((pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) + cosi2xb*i2*(-pi*L*cosk2xb*k2*r*sina*(B16 + B26) - sink2xb*((pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) + cosk2xb*k2*sini2xb*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina)) + sini2xa*(pi*L*r*sina*sink2xa*(B16*(i2*i2) + B26*(k2*k2)) - cosk2xa*k2*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += (pi*L*r*sina*sini2xb*sink2xb*(A16*(k2*k2) + A26*(i2*i2)) + cosi2xa*i2*(-pi*L*cosk2xa*k2*r*sina*(A16 + A26) + sink2xa*((pi*pi)*A16*(k2*k2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) + cosi2xb*i2*(pi*L*cosk2xb*k2*r*sina*(A16 + A26) - sink2xb*((pi*pi)*A16*(k2*k2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) + cosk2xb*k2*sini2xb*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina)) + sini2xa*(-pi*L*r*sina*sink2xa*(A16*(k2*k2) + A26*(i2*i2)) - cosk2xa*k2*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += j2*(L*cosk2xb*k2*sina*sini2xb*(A22 + A66) + cosi2xa*i2*(L*sina*sink2xa*(A22 + A66) + pi*cosk2xa*k2*r*(A12 + A66)) + cosi2xb*i2*(-L*sina*sink2xb*(A22 + A66) - pi*cosk2xb*k2*r*(A12 + A66)) - pi*r*sini2xb*sink2xb*(A12*(k2*k2) + A66*(i2*i2)) + sini2xa*(-L*cosk2xa*k2*sina*(A22 + A66) + pi*r*sink2xa*(A12*(k2*k2) + A66*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += (-cosk2xa*k2*sini2xa*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina))) + cosk2xb*k2*sini2xb*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina))) + sink2xa*(pi*A66*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2) + cosi2xa*i2*((pi*pi)*A66*(k2*k2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina)))) - sink2xb*(pi*A66*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xb*i2*((pi*pi)*A66*(k2*k2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+3
                                k0v[c] += pi*A26*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+4
                                k0v[c] += cosa*(-A26*L*cosk2xb*k2*sina*sini2xb - cosi2xa*i2*(A26*L*sina*sink2xa + pi*cosk2xa*k2*r*(A26 + A45)) + cosi2xb*i2*(A26*L*sina*sink2xb + pi*cosk2xb*k2*r*(A26 + A45)) + pi*r*sini2xb*sink2xb*(A26*(i2*i2) + A45*(k2*k2)) + sini2xa*(A26*L*cosk2xa*k2*sina - pi*r*sink2xa*(A26*(i2*i2) + A45*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+5
                                k0v[c] += L*cosa*j2*(A22 + A44)*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+6
                                k0v[c] += (pi*L*cosk2xb*r*sina*sini2xb*(B16*(k2*k2) + B26*(i2*i2)) + cosi2xa*i2*(pi*L*k2*r*sina*sink2xa*(B16 + B26) + cosk2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(-A45*(L*L)*cosa + (pi*pi)*B16*(k2*k2)*r))) + cosi2xb*i2*(-pi*L*k2*r*sina*sink2xb*(B16 + B26) + cosk2xb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa - (pi*pi)*B16*(k2*k2)*r))) + k2*sini2xb*sink2xb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa - (pi*pi)*B16*(i2*i2)*r)) + sini2xa*(-pi*L*cosk2xa*r*sina*(B16*(k2*k2) + B26*(i2*i2)) + k2*sink2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(-A45*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+7
                                k0v[c] += j2*(-L*k2*sina*sini2xb*sink2xb*(B22 + B66) + cosi2xa*i2*(L*cosk2xa*sina*(B22 + B66) - pi*k2*r*sink2xa*(B12 + B66)) + cosi2xb*i2*(-L*cosk2xb*sina*(B22 + B66) + pi*k2*r*sink2xb*(B12 + B66)) - pi*cosk2xb*r*sini2xb*(B12*(k2*k2) + B66*(i2*i2)) + sini2xa*(L*k2*sina*sink2xa*(B22 + B66) + pi*cosk2xa*r*(B12*(k2*k2) + B66*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+8
                                k0v[c] += (-cosk2xa*k2*sini2xa*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)) + cosk2xb*k2*sini2xb*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)) + sink2xa*(pi*B66*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2) + cosi2xa*i2*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k2*k2)*(r*r))) - sink2xb*(pi*B66*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xb*i2*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k2*k2)*(r*r))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+9
                                k0v[c] += pi*B26*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += j2*(-L*cosk2xb*k2*sina*sini2xb*(A22 + A66) + cosi2xa*i2*(-L*sina*sink2xa*(A22 + A66) - pi*cosk2xa*k2*r*(A12 + A66)) + cosi2xb*i2*(L*sina*sink2xb*(A22 + A66) + pi*cosk2xb*k2*r*(A12 + A66)) + pi*r*sini2xb*sink2xb*(A12*(k2*k2) + A66*(i2*i2)) + sini2xa*(L*cosk2xa*k2*sina*(A22 + A66) - pi*r*sink2xa*(A12*(k2*k2) + A66*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+1
                                k0v[c] += (pi*L*r*sina*sini2xb*sink2xb*(A16*(k2*k2) + A26*(i2*i2)) + cosi2xa*i2*(-pi*L*cosk2xa*k2*r*sina*(A16 + A26) + sink2xa*((pi*pi)*A16*(k2*k2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) + cosi2xb*i2*(pi*L*cosk2xb*k2*r*sina*(A16 + A26) - sink2xb*((pi*pi)*A16*(k2*k2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))) + cosk2xb*k2*sini2xb*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina)) + sini2xa*(-pi*L*r*sina*sink2xa*(A16*(k2*k2) + A26*(i2*i2)) - cosk2xa*k2*((pi*pi)*A16*(i2*i2)*(r*r) + A26*(L*L)*(j2 - sina)*(j2 + sina))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+2
                                k0v[c] += -pi*A26*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += (-cosk2xa*k2*sini2xa*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina))) + cosk2xb*k2*sini2xb*((pi*pi)*A66*(i2*i2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina))) + sink2xa*(pi*A66*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2) + cosi2xa*i2*((pi*pi)*A66*(k2*k2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina)))) - sink2xb*(pi*A66*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xb*i2*((pi*pi)*A66*(k2*k2)*(r*r) + (L*L)*(A22*(j2*j2) + A44*(cosa*cosa) + A66*(sina*sina)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += L*cosa*j2*(A22 + A44)*(-cosi2xa*i2*sink2xa + cosi2xb*i2*sink2xb + cosk2xa*k2*sini2xa - cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+5
                                k0v[c] += cosa*(-A26*L*cosk2xb*k2*sina*sini2xb - cosi2xa*i2*(A26*L*sina*sink2xa + pi*cosk2xa*k2*r*(A26 + A45)) + cosi2xb*i2*(A26*L*sina*sink2xb + pi*cosk2xb*k2*r*(A26 + A45)) + pi*r*sini2xb*sink2xb*(A26*(i2*i2) + A45*(k2*k2)) + sini2xa*(A26*L*cosk2xa*k2*sina - pi*r*sink2xa*(A26*(i2*i2) + A45*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+6
                                k0v[c] += j2*(L*k2*sina*sini2xb*sink2xb*(B22 + B66) + cosi2xa*i2*(-L*cosk2xa*sina*(B22 + B66) + pi*k2*r*sink2xa*(B12 + B66)) + cosi2xb*i2*(L*cosk2xb*sina*(B22 + B66) - pi*k2*r*sink2xb*(B12 + B66)) + pi*cosk2xb*r*sini2xb*(B12*(k2*k2) + B66*(i2*i2)) + sini2xa*(-L*k2*sina*sink2xa*(B22 + B66) - pi*cosk2xa*r*(B12*(k2*k2) + B66*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+7
                                k0v[c] += (pi*L*cosk2xb*r*sina*sini2xb*(B16*(k2*k2) + B26*(i2*i2)) + cosi2xa*i2*(pi*L*k2*r*sina*sink2xa*(B16 + B26) + cosk2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(-A45*(L*L)*cosa + (pi*pi)*B16*(k2*k2)*r))) + cosi2xb*i2*(-pi*L*k2*r*sina*sink2xb*(B16 + B26) + cosk2xb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa - (pi*pi)*B16*(k2*k2)*r))) + k2*sini2xb*sink2xb*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa - (pi*pi)*B16*(i2*i2)*r)) + sini2xa*(-pi*L*cosk2xa*r*sina*(B16*(k2*k2) + B26*(i2*i2)) + k2*sink2xa*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(-A45*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+8
                                k0v[c] += -pi*B26*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+9
                                k0v[c] += (-cosk2xa*k2*sini2xa*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)) + cosk2xb*k2*sini2xb*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)) + sink2xa*(pi*B66*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2) + cosi2xa*i2*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k2*k2)*(r*r))) - sink2xb*(pi*B66*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xb*i2*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k2*k2)*(r*r))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += cosa*(cosi2xa*i2*(pi*A12*cosk2xa*k2*r + A22*L*sina*sink2xa) - cosi2xb*i2*(pi*A12*cosk2xb*k2*r + A22*L*sina*sink2xb) + k2*sini2xa*(pi*A12*k2*r*sink2xa - A22*L*cosk2xa*sina) + k2*sini2xb*(-pi*A12*k2*r*sink2xb + A22*L*cosk2xb*sina))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+1
                                k0v[c] += A26*L*cosa*j2*(-cosi2xa*i2*sink2xa + cosi2xb*i2*sink2xb + cosk2xa*k2*sini2xa - cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+2
                                k0v[c] += cosa*(cosi2xa*i2*(-A26*L*sina*sink2xa + pi*cosk2xa*k2*r*(A26 + A45)) + cosi2xb*i2*(A26*L*sina*sink2xb - pi*cosk2xb*k2*r*(A26 + A45)) + sini2xa*(A26*L*cosk2xa*k2*sina + pi*r*sink2xa*(A26*(k2*k2) + A45*(i2*i2))) - sini2xb*(A26*L*cosk2xb*k2*sina + pi*r*sink2xb*(A26*(k2*k2) + A45*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += L*cosa*j2*(A22 + A44)*(-cosi2xa*i2*sink2xa + cosi2xb*i2*sink2xb + cosk2xa*k2*sini2xa - cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += (cosi2xa*i2*sink2xa*((pi*pi)*A55*(k2*k2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))) - cosi2xb*i2*sink2xb*((pi*pi)*A55*(k2*k2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))) - cosk2xa*k2*sini2xa*((pi*pi)*A55*(i2*i2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))) + cosk2xb*k2*sini2xb*((pi*pi)*A55*(i2*i2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+5
                                k0v[c] += pi*A45*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+6
                                k0v[c] += (-B22*L*cosa*k2*sina*sini2xb*sink2xb + cosi2xa*i2*(B22*L*cosa*cosk2xa*sina + pi*k2*r*sink2xa*(A55*r - B12*cosa)) + cosi2xb*i2*(-B22*L*cosa*cosk2xb*sina + pi*k2*r*sink2xb*(-A55*r + B12*cosa)) + pi*cosk2xb*r*sini2xb*(A55*(i2*i2)*r - B12*cosa*(k2*k2)) + sini2xa*(B22*L*cosa*k2*sina*sink2xa + pi*cosk2xa*r*(-A55*(i2*i2)*r + B12*cosa*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+7
                                k0v[c] += L*j2*(A45*r - B26*cosa)*(cosi2xa*cosk2xa*i2 - cosi2xb*cosk2xb*i2 + k2*sini2xa*sink2xa - k2*sini2xb*sink2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+8
                                k0v[c] += (-B26*L*cosa*cosk2xb*k2*sina*sini2xb + cosi2xa*i2*(-B26*L*cosa*sina*sink2xa + pi*cosk2xa*k2*r*(-A45*r + B26*cosa)) + cosi2xb*i2*(B26*L*cosa*sina*sink2xb + pi*cosk2xb*k2*r*(A45*r - B26*cosa)) + pi*r*sini2xb*sink2xb*(A45*(i2*i2)*r - B26*cosa*(k2*k2)) + sini2xa*(B26*L*cosa*cosk2xa*k2*sina + pi*r*sink2xa*(-A45*(i2*i2)*r + B26*cosa*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+9
                                k0v[c] += L*j2*(A44*r - B22*cosa)*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+0
                                k0v[c] += A26*L*cosa*j2*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+1
                                k0v[c] += cosa*(cosi2xa*i2*(pi*A12*cosk2xa*k2*r + A22*L*sina*sink2xa) - cosi2xb*i2*(pi*A12*cosk2xb*k2*r + A22*L*sina*sink2xb) + k2*sini2xa*(pi*A12*k2*r*sink2xa - A22*L*cosk2xa*sina) + k2*sini2xb*(-pi*A12*k2*r*sink2xb + A22*L*cosk2xb*sina))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+2
                                k0v[c] += L*cosa*j2*(A22 + A44)*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+3
                                k0v[c] += cosa*(cosi2xa*i2*(-A26*L*sina*sink2xa + pi*cosk2xa*k2*r*(A26 + A45)) + cosi2xb*i2*(A26*L*sina*sink2xb - pi*cosk2xb*k2*r*(A26 + A45)) + sini2xa*(A26*L*cosk2xa*k2*sina + pi*r*sink2xa*(A26*(k2*k2) + A45*(i2*i2))) - sini2xb*(A26*L*cosk2xb*k2*sina + pi*r*sink2xb*(A26*(k2*k2) + A45*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+4
                                k0v[c] += -pi*A45*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+5
                                k0v[c] += (cosi2xa*i2*sink2xa*((pi*pi)*A55*(k2*k2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))) - cosi2xb*i2*sink2xb*((pi*pi)*A55*(k2*k2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))) - cosk2xa*k2*sini2xa*((pi*pi)*A55*(i2*i2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))) + cosk2xb*k2*sini2xb*((pi*pi)*A55*(i2*i2)*(r*r) + (L*L)*(A22*(cosa*cosa) + A44*(j2*j2))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+6
                                k0v[c] += L*j2*(-A45*r + B26*cosa)*(cosi2xa*cosk2xa*i2 - cosi2xb*cosk2xb*i2 + k2*sini2xa*sink2xa - k2*sini2xb*sink2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+7
                                k0v[c] += (-B22*L*cosa*k2*sina*sini2xb*sink2xb + cosi2xa*i2*(B22*L*cosa*cosk2xa*sina + pi*k2*r*sink2xa*(A55*r - B12*cosa)) + cosi2xb*i2*(-B22*L*cosa*cosk2xb*sina + pi*k2*r*sink2xb*(-A55*r + B12*cosa)) + pi*cosk2xb*r*sini2xb*(A55*(i2*i2)*r - B12*cosa*(k2*k2)) + sini2xa*(B22*L*cosa*k2*sina*sink2xa + pi*cosk2xa*r*(-A55*(i2*i2)*r + B12*cosa*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+8
                                k0v[c] += L*j2*(-A44*r + B22*cosa)*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+9
                                k0v[c] += (-B26*L*cosa*cosk2xb*k2*sina*sini2xb + cosi2xa*i2*(-B26*L*cosa*sina*sink2xa + pi*cosk2xa*k2*r*(-A45*r + B26*cosa)) + cosi2xb*i2*(B26*L*cosa*sina*sink2xb + pi*cosk2xb*k2*r*(A45*r - B26*cosa)) + pi*r*sini2xb*sink2xb*(A45*(i2*i2)*r - B26*cosa*(k2*k2)) + sini2xa*(B26*L*cosa*cosk2xa*k2*sina + pi*r*sink2xa*(-A45*(i2*i2)*r + B26*cosa*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+0
                                k0v[c] += (cosi2xa*(pi*B12*L*r*sina*sink2xa*(-(i2*i2) + (k2*k2)) - cosk2xa*k2*((pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2)))) + cosi2xb*(pi*B12*L*r*sina*sink2xb*(i2 - k2)*(i2 + k2) + cosk2xb*k2*((pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2)))) + i2*(-sini2xa*sink2xa + sini2xb*sink2xb)*((pi*pi)*B11*(k2*k2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+1
                                k0v[c] += pi*B16*j2*(cosi2xa*sink2xa*((i2*i2) + (k2*k2)) - cosi2xb*sink2xb*((i2*i2) + (k2*k2)) - 2*cosk2xa*i2*k2*sini2xa + 2*cosk2xb*i2*k2*sini2xb)/((i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+2
                                k0v[c] += (cosi2xa*(pi*L*r*sina*sink2xa*(B16*(i2*i2) + B26*(k2*k2)) + cosk2xa*k2*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa - (pi*pi)*B16*(i2*i2)*r))) + cosi2xb*(-pi*L*r*sina*sink2xb*(B16*(i2*i2) + B26*(k2*k2)) + cosk2xb*k2*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(-A45*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))) + i2*(pi*L*cosk2xb*k2*r*sina*sini2xb*(B16 + B26) + sini2xa*(-pi*L*cosk2xa*k2*r*sina*(B16 + B26) + sink2xa*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa - (pi*pi)*B16*(k2*k2)*r))) + sini2xb*sink2xb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(-A45*(L*L)*cosa + (pi*pi)*B16*(k2*k2)*r))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+3
                                k0v[c] += j2*(cosi2xa*(L*cosk2xa*k2*sina*(B22 + B66) + pi*r*sink2xa*(B12*(i2*i2) + B66*(k2*k2))) - cosi2xb*(L*cosk2xb*k2*sina*(B22 + B66) + pi*r*sink2xb*(B12*(i2*i2) + B66*(k2*k2))) + i2*(L*sina*sini2xa*sink2xa*(B22 + B66) - pi*cosk2xa*k2*r*sini2xa*(B12 + B66) + sini2xb*(-L*sina*sink2xb*(B22 + B66) + pi*cosk2xb*k2*r*(B12 + B66))))/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+4
                                k0v[c] += (-cosi2xa*(B22*L*cosa*cosk2xa*k2*sina + pi*r*sink2xa*(-A55*(k2*k2)*r + B12*cosa*(i2*i2))) + cosi2xb*(B22*L*cosa*cosk2xb*k2*sina + pi*r*sink2xb*(-A55*(k2*k2)*r + B12*cosa*(i2*i2))) + i2*(-B22*L*cosa*sina*sini2xa*sink2xa + pi*cosk2xa*k2*r*sini2xa*(-A55*r + B12*cosa) + sini2xb*(B22*L*cosa*sina*sink2xb + pi*cosk2xb*k2*r*(A55*r - B12*cosa))))/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+5
                                k0v[c] += L*j2*(A45*r - B26*cosa)*(cosi2xa*cosk2xa*k2 - cosi2xb*cosk2xb*k2 + i2*sini2xa*sink2xa - i2*sini2xb*sink2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+6
                                k0v[c] += (cosi2xa*(pi*D12*L*cosk2xa*r*sina*(-(i2*i2) + (k2*k2)) + k2*sink2xa*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i2*i2)))) + cosi2xb*(pi*D12*L*cosk2xb*r*sina*(i2 - k2)*(i2 + k2) - k2*sink2xb*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i2*i2)))) - i2*(cosk2xa*sini2xa - cosk2xb*sini2xb)*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(k2*k2))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+7
                                k0v[c] += pi*D16*j2*(cosi2xa*cosk2xa*((i2*i2) + (k2*k2)) - cosi2xb*cosk2xb*((i2*i2) + (k2*k2)) + 2*i2*k2*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+8
                                k0v[c] += (cosi2xa*(pi*L*r*sina*sink2xa*(D16*(i2*i2) + D26*(k2*k2)) - cosk2xa*k2*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2)))) + cosi2xb*(-pi*L*r*sina*sink2xb*(D16*(i2*i2) + D26*(k2*k2)) + cosk2xb*k2*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2)))) + i2*(pi*L*cosk2xb*k2*r*sina*sini2xb*(D16 + D26) + sini2xa*(-pi*L*cosk2xa*k2*r*sina*(D16 + D26) - sink2xa*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))) + sini2xb*sink2xb*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+9
                                k0v[c] += j2*(cosi2xa*(L*cosk2xa*k2*sina*(D22 + D66) + pi*r*sink2xa*(D12*(i2*i2) + D66*(k2*k2))) - cosi2xb*(L*cosk2xb*k2*sina*(D22 + D66) + pi*r*sink2xb*(D12*(i2*i2) + D66*(k2*k2))) + i2*(L*sina*sini2xa*sink2xa*(D22 + D66) - pi*cosk2xa*k2*r*sini2xa*(D12 + D66) + sini2xb*(-L*sina*sink2xb*(D22 + D66) + pi*cosk2xb*k2*r*(D12 + D66))))/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+0
                                k0v[c] += pi*B16*j2*(-cosi2xa*sink2xa*((i2*i2) + (k2*k2)) + cosi2xb*sink2xb*((i2*i2) + (k2*k2)) + 2*cosk2xa*i2*k2*sini2xa - 2*cosk2xb*i2*k2*sini2xb)/((i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+1
                                k0v[c] += (cosi2xa*(pi*B12*L*r*sina*sink2xa*(-(i2*i2) + (k2*k2)) - cosk2xa*k2*((pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2)))) + cosi2xb*(pi*B12*L*r*sina*sink2xb*(i2 - k2)*(i2 + k2) + cosk2xb*k2*((pi*pi)*B11*(i2*i2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2)))) + i2*(-sini2xa*sink2xa + sini2xb*sink2xb)*((pi*pi)*B11*(k2*k2)*(r*r) + (L*L)*(B22*(sina*sina) + B66*(j2*j2))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+2
                                k0v[c] += j2*(cosi2xa*(-L*cosk2xa*k2*sina*(B22 + B66) - pi*r*sink2xa*(B12*(i2*i2) + B66*(k2*k2))) + cosk2xb*k2*(L*cosi2xb*sina*(B22 + B66) - pi*i2*r*sini2xb*(B12 + B66)) + i2*sini2xa*(-L*sina*sink2xa*(B22 + B66) + pi*cosk2xa*k2*r*(B12 + B66)) + sink2xb*(L*i2*sina*sini2xb*(B22 + B66) + pi*cosi2xb*r*(B12*(i2*i2) + B66*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+3
                                k0v[c] += (cosi2xa*(pi*L*r*sina*sink2xa*(B16*(i2*i2) + B26*(k2*k2)) + cosk2xa*k2*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa - (pi*pi)*B16*(i2*i2)*r))) + cosi2xb*(-pi*L*r*sina*sink2xb*(B16*(i2*i2) + B26*(k2*k2)) + cosk2xb*k2*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(-A45*(L*L)*cosa + (pi*pi)*B16*(i2*i2)*r))) + i2*(pi*L*cosk2xb*k2*r*sina*sini2xb*(B16 + B26) + sini2xa*(-pi*L*cosk2xa*k2*r*sina*(B16 + B26) + sink2xa*(B26*(L*L)*(-(j2*j2) + (sina*sina)) + r*(A45*(L*L)*cosa - (pi*pi)*B16*(k2*k2)*r))) + sini2xb*sink2xb*(B26*(L*L)*(j2 - sina)*(j2 + sina) + r*(-A45*(L*L)*cosa + (pi*pi)*B16*(k2*k2)*r))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+4
                                k0v[c] += L*j2*(-A45*r + B26*cosa)*(cosi2xa*cosk2xa*k2 - cosi2xb*cosk2xb*k2 + i2*sini2xa*sink2xa - i2*sini2xb*sink2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+5
                                k0v[c] += (-cosi2xa*(B22*L*cosa*cosk2xa*k2*sina + pi*r*sink2xa*(-A55*(k2*k2)*r + B12*cosa*(i2*i2))) + cosi2xb*(B22*L*cosa*cosk2xb*k2*sina + pi*r*sink2xb*(-A55*(k2*k2)*r + B12*cosa*(i2*i2))) + i2*(-B22*L*cosa*sina*sini2xa*sink2xa + pi*cosk2xa*k2*r*sini2xa*(-A55*r + B12*cosa) + sini2xb*(B22*L*cosa*sina*sink2xb + pi*cosk2xb*k2*r*(A55*r - B12*cosa))))/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+6
                                k0v[c] += -pi*D16*j2*(cosi2xa*cosk2xa*((i2*i2) + (k2*k2)) - cosi2xb*cosk2xb*((i2*i2) + (k2*k2)) + 2*i2*k2*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+7
                                k0v[c] += (cosi2xa*(pi*D12*L*cosk2xa*r*sina*(-(i2*i2) + (k2*k2)) + k2*sink2xa*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i2*i2)))) + cosi2xb*(pi*D12*L*cosk2xb*r*sina*(i2 - k2)*(i2 + k2) - k2*sink2xb*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(i2*i2)))) - i2*(cosk2xa*sini2xa - cosk2xb*sini2xb)*(D22*(L*L)*(sina*sina) + D66*(L*L)*(j2*j2) + (r*r)*(A55*(L*L) + (pi*pi)*D11*(k2*k2))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+8
                                k0v[c] += j2*(cosi2xa*(-L*cosk2xa*k2*sina*(D22 + D66) - pi*r*sink2xa*(D12*(i2*i2) + D66*(k2*k2))) + cosk2xb*k2*(L*cosi2xb*sina*(D22 + D66) - pi*i2*r*sini2xb*(D12 + D66)) + i2*sini2xa*(-L*sina*sink2xa*(D22 + D66) + pi*cosk2xa*k2*r*(D12 + D66)) + sink2xb*(L*i2*sina*sini2xb*(D22 + D66) + pi*cosi2xb*r*(D12*(i2*i2) + D66*(k2*k2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+9
                                k0v[c] += (cosi2xa*(pi*L*r*sina*sink2xa*(D16*(i2*i2) + D26*(k2*k2)) - cosk2xa*k2*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2)))) + cosi2xb*(-pi*L*r*sina*sink2xb*(D16*(i2*i2) + D26*(k2*k2)) + cosk2xb*k2*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2)))) + i2*(pi*L*cosk2xb*k2*r*sina*sini2xb*(D16 + D26) + sini2xa*(-pi*L*cosk2xa*k2*r*sina*(D16 + D26) - sink2xa*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))) + sini2xb*sink2xb*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+0
                                k0v[c] += (pi*L*r*sina*sini2xb*sink2xb*(B16*(k2*k2) + B26*(i2*i2)) + cosi2xa*i2*(-pi*L*cosk2xa*k2*r*sina*(B16 + B26) + sink2xa*((pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) + cosi2xb*i2*(pi*L*cosk2xb*k2*r*sina*(B16 + B26) - sink2xb*((pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) + cosk2xb*k2*sini2xb*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina)) + sini2xa*(-pi*L*r*sina*sink2xa*(B16*(k2*k2) + B26*(i2*i2)) - cosk2xa*k2*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+1
                                k0v[c] += j2*(L*cosk2xb*k2*sina*sini2xb*(B22 + B66) + cosi2xa*i2*(L*sina*sink2xa*(B22 + B66) + pi*cosk2xa*k2*r*(B12 + B66)) + cosi2xb*i2*(-L*sina*sink2xb*(B22 + B66) - pi*cosk2xb*k2*r*(B12 + B66)) - pi*r*sini2xb*sink2xb*(B12*(k2*k2) + B66*(i2*i2)) + sini2xa*(-L*cosk2xa*k2*sina*(B22 + B66) + pi*r*sink2xa*(B12*(k2*k2) + B66*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+2
                                k0v[c] += (-cosk2xa*k2*sini2xa*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)) + cosk2xb*k2*sini2xb*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)) + sink2xa*(pi*B66*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2) + cosi2xa*i2*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k2*k2)*(r*r))) - sink2xb*(pi*B66*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xb*i2*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k2*k2)*(r*r))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+3
                                k0v[c] += pi*B26*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+4
                                k0v[c] += (-B26*L*cosa*cosk2xb*k2*sina*sini2xb - cosi2xa*i2*(B26*L*cosa*sina*sink2xa + pi*cosk2xa*k2*r*(-A45*r + B26*cosa)) + cosi2xb*i2*(B26*L*cosa*sina*sink2xb + pi*cosk2xb*k2*r*(-A45*r + B26*cosa)) + pi*r*sini2xb*sink2xb*(-A45*(k2*k2)*r + B26*cosa*(i2*i2)) + sini2xa*(B26*L*cosa*cosk2xa*k2*sina + pi*r*sink2xa*(A45*(k2*k2)*r - B26*cosa*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+5
                                k0v[c] += L*j2*(-A44*r + B22*cosa)*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+6
                                k0v[c] += (pi*L*cosk2xb*r*sina*sini2xb*(D16*(k2*k2) + D26*(i2*i2)) + cosi2xa*i2*(pi*L*k2*r*sina*sink2xa*(D16 + D26) + cosk2xa*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))) + cosi2xb*i2*(-pi*L*k2*r*sina*sink2xb*(D16 + D26) - cosk2xb*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))) - k2*sini2xb*sink2xb*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2))) + sini2xa*(-pi*L*cosk2xa*r*sina*(D16*(k2*k2) + D26*(i2*i2)) + k2*sink2xa*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+7
                                k0v[c] += j2*(-L*k2*sina*sini2xb*sink2xb*(D22 + D66) + cosi2xa*i2*(L*cosk2xa*sina*(D22 + D66) - pi*k2*r*sink2xa*(D12 + D66)) + cosi2xb*i2*(-L*cosk2xb*sina*(D22 + D66) + pi*k2*r*sink2xb*(D12 + D66)) - pi*cosk2xb*r*sini2xb*(D12*(k2*k2) + D66*(i2*i2)) + sini2xa*(L*k2*sina*sink2xa*(D22 + D66) + pi*cosk2xa*r*(D12*(k2*k2) + D66*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+8
                                k0v[c] += (-cosk2xa*k2*sini2xa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i2*i2))) + cosk2xb*k2*sini2xb*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i2*i2))) + sink2xa*(pi*D66*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2) + cosi2xa*i2*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(k2*k2)))) - sink2xb*(pi*D66*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xb*i2*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(k2*k2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+9
                                k0v[c] += pi*D26*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+0
                                k0v[c] += j2*(-L*cosk2xb*k2*sina*sini2xb*(B22 + B66) + cosi2xa*i2*(-L*sina*sink2xa*(B22 + B66) - pi*cosk2xa*k2*r*(B12 + B66)) + cosi2xb*i2*(L*sina*sink2xb*(B22 + B66) + pi*cosk2xb*k2*r*(B12 + B66)) + pi*r*sini2xb*sink2xb*(B12*(k2*k2) + B66*(i2*i2)) + sini2xa*(L*cosk2xa*k2*sina*(B22 + B66) - pi*r*sink2xa*(B12*(k2*k2) + B66*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+1
                                k0v[c] += (pi*L*r*sina*sini2xb*sink2xb*(B16*(k2*k2) + B26*(i2*i2)) + cosi2xa*i2*(-pi*L*cosk2xa*k2*r*sina*(B16 + B26) + sink2xa*((pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) + cosi2xb*i2*(pi*L*cosk2xb*k2*r*sina*(B16 + B26) - sink2xb*((pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))) + cosk2xb*k2*sini2xb*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina)) + sini2xa*(-pi*L*r*sina*sink2xa*(B16*(k2*k2) + B26*(i2*i2)) - cosk2xa*k2*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(j2 - sina)*(j2 + sina))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+2
                                k0v[c] += -pi*B26*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+3
                                k0v[c] += (-cosk2xa*k2*sini2xa*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)) + cosk2xb*k2*sini2xb*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r)) + sink2xa*(pi*B66*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2) + cosi2xa*i2*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k2*k2)*(r*r))) - sink2xb*(pi*B66*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xb*i2*(-A44*(L*L)*cosa*r + B22*(L*L)*(j2*j2) + B66*(L*L)*(sina*sina) + (pi*pi)*B66*(k2*k2)*(r*r))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+4
                                k0v[c] += L*j2*(A44*r - B22*cosa)*(cosi2xa*i2*sink2xa - cosi2xb*i2*sink2xb - cosk2xa*k2*sini2xa + cosk2xb*k2*sini2xb)/(r*((i2*i2) - (k2*k2)))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+5
                                k0v[c] += (-B26*L*cosa*cosk2xb*k2*sina*sini2xb - cosi2xa*i2*(B26*L*cosa*sina*sink2xa + pi*cosk2xa*k2*r*(-A45*r + B26*cosa)) + cosi2xb*i2*(B26*L*cosa*sina*sink2xb + pi*cosk2xb*k2*r*(-A45*r + B26*cosa)) + pi*r*sini2xb*sink2xb*(-A45*(k2*k2)*r + B26*cosa*(i2*i2)) + sini2xa*(B26*L*cosa*cosk2xa*k2*sina + pi*r*sink2xa*(A45*(k2*k2)*r - B26*cosa*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+6
                                k0v[c] += j2*(L*k2*sina*sini2xb*sink2xb*(D22 + D66) + cosi2xa*i2*(-L*cosk2xa*sina*(D22 + D66) + pi*k2*r*sink2xa*(D12 + D66)) + cosi2xb*i2*(L*cosk2xb*sina*(D22 + D66) - pi*k2*r*sink2xb*(D12 + D66)) + pi*cosk2xb*r*sini2xb*(D12*(k2*k2) + D66*(i2*i2)) + sini2xa*(-L*k2*sina*sink2xa*(D22 + D66) - pi*cosk2xa*r*(D12*(k2*k2) + D66*(i2*i2))))/(r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+7
                                k0v[c] += (pi*L*cosk2xb*r*sina*sini2xb*(D16*(k2*k2) + D26*(i2*i2)) + cosi2xa*i2*(pi*L*k2*r*sina*sink2xa*(D16 + D26) + cosk2xa*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))) + cosi2xb*i2*(-pi*L*k2*r*sina*sink2xb*(D16 + D26) - cosk2xb*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))) - k2*sini2xb*sink2xb*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2))) + sini2xa*(-pi*L*cosk2xa*r*sina*(D16*(k2*k2) + D26*(i2*i2)) + k2*sink2xa*(D26*(L*L)*(j2 - sina)*(j2 + sina) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2)))))/(L*r*(i2 - k2)*(i2 + k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+8
                                k0v[c] += -pi*D26*j2*(2*cosi2xa*cosk2xa*i2*k2 - 2*cosi2xb*cosk2xb*i2*k2 + ((i2*i2) + (k2*k2))*(sini2xa*sink2xa - sini2xb*sink2xb))/((i2*i2) - (k2*k2))
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+9
                                k0v[c] += (-cosk2xa*k2*sini2xa*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i2*i2))) + cosk2xb*k2*sini2xb*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(i2*i2))) + sink2xa*(pi*D66*L*r*sina*sini2xa*(i2 - k2)*(i2 + k2) + cosi2xa*i2*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(k2*k2)))) - sink2xb*(pi*D66*L*r*sina*sini2xb*(i2 - k2)*(i2 + k2) + cosi2xb*i2*(D22*(L*L)*(j2*j2) + D66*(L*L)*(sina*sina) + (r*r)*(A44*(L*L) + (pi*pi)*D66*(k2*k2)))))/(L*r*(i2 - k2)*(i2 + k2))

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
    k11_cond_1 = 13
    k11_cond_2 = 12
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 50
    k22_cond_2 = 50
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 5 + 6*m1 + k11_num + k22_num

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

    # k0_00
    c += 1
    k0r[c] = 0
    k0c[c] = 0
    k0v[c] += 2*pi*A11*r/L
    c += 1
    k0r[c] = 0
    k0c[c] = 1
    k0v[c] += 2*pi*A16*r*r2/L
    c += 1
    k0r[c] = 1
    k0c[c] = 0
    k0v[c] += 2*pi*A16*r*r2/L
    c += 1
    k0r[c] = 1
    k0c[c] = 1
    k0v[c] += 0.666666666666667*pi*A44*L*(r2*r2)/r + 2*pi*A66*r*(r2*r2)/L
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
            k0v[c] += A12*(2*(-1)**i1 - 2)/i1
            c += 1
            k0r[c] = 0
            k0c[c] = col+3
            k0v[c] += pi*B11*r*(-2*(-1)**i1 + 2)/L
            c += 1
            k0r[c] = 1
            k0c[c] = col+1
            k0v[c] += 2*A44*L*r2/(i1*r)
            c += 1
            k0r[c] = 1
            k0c[c] = col+2
            k0v[c] += r2*(2*(-1)**i1 - 2)*(A26 + A45)/i1
            c += 1
            k0r[c] = 1
            k0c[c] = col+3
            k0v[c] += r2*(2*(-1)**i1 - 2)*(A45*(L*L) - (pi*pi)*B16*(i1*i1)*r)/(pi*L*(i1*i1))
            c += 1
            k0r[c] = 1
            k0c[c] = col+4
            k0v[c] += -2*A44*L*r2/i1

        else:
            # k0_01 cond_5
            c += 1
            k0r[c] = 1
            k0c[c] = col+3
            k0v[c] += -pi*A45*L*r2

        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0

            #NOTE symmetry
            if row > col:
                continue

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
                    k0v[c] += (pi*pi*pi)*A16*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+4
                    k0v[c] += (pi*pi*pi)*B16*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += (pi*pi*pi)*A16*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += pi*A44*L/r + (pi*pi*pi)*A66*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+4
                    k0v[c] += -pi*A44*L + (pi*pi*pi)*B66*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += pi*A22*L/r + (pi*pi*pi)*A55*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+3
                    k0v[c] += (pi*pi)*i1*(A55*r - B12)
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+2
                    k0v[c] += (pi*pi)*i1*(A55*r - B12)
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+3
                    k0v[c] += pi*A55*L*r + (pi*pi*pi)*D11*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+0
                    k0v[c] += (pi*pi*pi)*B16*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+1
                    k0v[c] += -pi*A44*L + (pi*pi*pi)*B66*(i1*i1)*r/L
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+4
                    k0v[c] += pi*A44*L*r + (pi*pi*pi)*D66*(i1*i1)*r/L

                else:
                    # k0_11 cond_5
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+3
                    k0v[c] += 2*pi*A55*L*r

            else:
                # k0_11 cond_2
                c += 1
                k0r[c] = row+0
                k0c[c] = col+2
                k0v[c] += pi*A12*i1*k1*(2*(-1)**(i1 + k1) - 2)/((i1*i1) - (k1*k1))
                c += 1
                k0r[c] = row+0
                k0c[c] = col+3
                k0v[c] += (pi*pi)*B11*i1*(k1*k1)*r*(-2*(-1)**(i1 + k1) + 2)/(L*((i1*i1) - (k1*k1)))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+2
                k0v[c] += pi*i1*k1*(2*(-1)**(i1 + k1) - 2)*(A26 + A45)/((i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+3
                k0v[c] += i1*(2*(-1)**(i1 + k1) - 2)*(A45*(L*L) - (pi*pi)*B16*(k1*k1)*r)/(L*(i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+0
                k0v[c] += pi*A12*i1*k1*(-2*(-1)**(i1 + k1) + 2)/((i1*i1) - (k1*k1))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+1
                k0v[c] += pi*i1*k1*(-2*(-1)**(i1 + k1) + 2)*(A26 + A45)/((i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+4
                k0v[c] += pi*i1*k1*(-2*(-1)**(i1 + k1) + 2)*(-A45*r + B26)/((i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+0
                k0v[c] += (pi*pi)*B11*(i1*i1)*k1*r*(2*(-1)**(i1 + k1) - 2)/(L*((i1*i1) - (k1*k1)))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+1
                k0v[c] += k1*(2*(-1)**(i1 + k1) - 2)*(-A45*(L*L) + (pi*pi)*B16*(i1*i1)*r)/(L*(i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+4
                k0v[c] += k1*r*(2*(-1)**(i1 + k1) - 2)*(A45*(L*L) + (pi*pi)*D16*(i1*i1))/(L*(i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+2
                k0v[c] += pi*i1*k1*(2*(-1)**(i1 + k1) - 2)*(-A45*r + B26)/((i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+3
                k0v[c] += i1*r*(-2*(-1)**(i1 + k1) + 2)*(A45*(L*L) + (pi*pi)*D16*(k1*k1))/(L*(i1 - k1)*(i1 + k1))

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
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi**3*A16*i2**2*r/L + 0.5*pi*A26*L*j2**2/r
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+5
                            k0v[c] += 0.5*pi*A26*L*j2/r
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+7
                            k0v[c] += -pi**2*B16*i2*j2
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+8
                            k0v[c] += 0.5*pi**3*B16*i2**2*r/L + 0.5*pi*B26*L*j2**2/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.5*pi**3*A11*i2**2*r/L + 0.5*pi*A66*L*j2**2/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += 0.5*pi**3*A16*i2**2*r/L + 0.5*pi*A26*L*j2**2/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += -0.5*pi*A26*L*j2/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+6
                            k0v[c] += pi**2*B16*i2*j2
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+9
                            k0v[c] += 0.5*pi**3*B16*i2**2*r/L + 0.5*pi*B26*L*j2**2/r
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += 0.5*pi**3*A16*i2**2*r/L + 0.5*pi*A26*L*j2**2/r
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi**3*A66*i2**2*r/L + 0.5*pi*L*(A22*j2**2 + A44)/r
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+5
                            k0v[c] += 0.5*pi*L*j2*(A22 + A44)/r
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+7
                            k0v[c] += -0.5*pi**2*i2*j2*(B12 + B66)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+8
                            k0v[c] += 0.5*pi*(-A44*L**2 + B22*L**2*j2**2/r + pi**2*B66*i2**2*r)/L
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+1
                            k0v[c] += 0.5*pi**3*A16*i2**2*r/L + 0.5*pi*A26*L*j2**2/r
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += 0.5*pi**3*A66*i2**2*r/L + 0.5*pi*L*(A22*j2**2 + A44)/r
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+4
                            k0v[c] += -0.5*pi*L*j2*(A22 + A44)/r
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+6
                            k0v[c] += 0.5*pi**2*i2*j2*(B12 + B66)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+9
                            k0v[c] += 0.5*pi*(-A44*L**2 + B22*L**2*j2**2/r + pi**2*B66*i2**2*r)/L
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+1
                            k0v[c] += -0.5*pi*A26*L*j2/r
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+3
                            k0v[c] += -0.5*pi*L*j2*(A22 + A44)/r
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+4
                            k0v[c] += 0.5*pi**3*A55*i2**2*r/L + 0.5*pi*L*(A22 + A44*j2**2)/r
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+6
                            k0v[c] += -0.5*pi**2*i2*(-A55*r + B12)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+9
                            k0v[c] += 0.5*pi*L*j2*(A44*r - B22)/r
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+0
                            k0v[c] += 0.5*pi*A26*L*j2/r
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi*L*j2*(A22 + A44)/r
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+5
                            k0v[c] += 0.5*pi**3*A55*i2**2*r/L + 0.5*pi*L*(A22 + A44*j2**2)/r
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+7
                            k0v[c] += -0.5*pi**2*i2*(-A55*r + B12)
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+8
                            k0v[c] += 0.5*pi*L*j2*(-A44*r + B22)/r
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+1
                            k0v[c] += pi**2*B16*i2*j2
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+3
                            k0v[c] += 0.5*pi**2*i2*j2*(B12 + B66)
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+4
                            k0v[c] += -0.5*pi**2*i2*(-A55*r + B12)
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+6
                            k0v[c] += 0.5*pi*D66*L*j2**2/r + 0.5*pi*r*(A55*L**2 + pi**2*D11*i2**2)/L
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+9
                            k0v[c] += 0.5*pi**2*i2*j2*(D12 + D66)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+0
                            k0v[c] += -pi**2*B16*i2*j2
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+2
                            k0v[c] += -0.5*pi**2*i2*j2*(B12 + B66)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+5
                            k0v[c] += -0.5*pi**2*i2*(-A55*r + B12)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+7
                            k0v[c] += 0.5*pi*D66*L*j2**2/r + 0.5*pi*r*(A55*L**2 + pi**2*D11*i2**2)/L
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+8
                            k0v[c] += -0.5*pi**2*i2*j2*(D12 + D66)
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+0
                            k0v[c] += 0.5*pi**3*B16*i2**2*r/L + 0.5*pi*B26*L*j2**2/r
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi*(-A44*L**2 + B22*L**2*j2**2/r + pi**2*B66*i2**2*r)/L
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+5
                            k0v[c] += 0.5*pi*L*j2*(-A44*r + B22)/r
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+7
                            k0v[c] += -0.5*pi**2*i2*j2*(D12 + D66)
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+8
                            k0v[c] += 0.5*pi*D22*L*j2**2/r + 0.5*pi*r*(A44*L**2 + pi**2*D66*i2**2)/L
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+1
                            k0v[c] += 0.5*pi**3*B16*i2**2*r/L + 0.5*pi*B26*L*j2**2/r
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+3
                            k0v[c] += 0.5*pi*(-A44*L**2 + B22*L**2*j2**2/r + pi**2*B66*i2**2*r)/L
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+4
                            k0v[c] += 0.5*pi*L*j2*(A44*r - B22)/r
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+6
                            k0v[c] += 0.5*pi**2*i2*j2*(D12 + D66)
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+9
                            k0v[c] += 0.5*pi*D22*L*j2**2/r + 0.5*pi*r*(A44*L**2 + pi**2*D66*i2**2)/L

                        else:
                            # k0_22 cond_5
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+6
                            k0v[c] += L*(pi*A55*r + pi*D66*(j2*j2)/r)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+7
                            k0v[c] += L*(pi*A55*r + pi*D66*(j2*j2)/r)

                    elif k2!=i2 and l2==j2:
                        # k0_22 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += pi*A16*i2*j2*k2*(-2*(-1)**(i2 + k2) + 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += -pi*i2*j2*k2*((-1)**(i2 + k2) - 1)*(A12 + A66)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += pi*A12*i2*k2*((-1)**(i2 + k2) - 1)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+6
                        k0v[c] += -i2*((-1)**(i2 + k2) - 1)*((pi*pi)*B11*(k2*k2)*(r*r) + B66*(L*L)*(j2*j2))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+9
                        k0v[c] += -pi*i2*j2*k2*((-1)**(i2 + k2) - 1)*(B12 + B66)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += pi*A16*i2*j2*k2*(2*(-1)**(i2 + k2) - 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += pi*i2*j2*k2*((-1)**(i2 + k2) - 1)*(A12 + A66)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+5
                        k0v[c] += pi*A12*i2*k2*((-1)**(i2 + k2) - 1)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+7
                        k0v[c] += -i2*((-1)**(i2 + k2) - 1)*((pi*pi)*B11*(k2*k2)*(r*r) + B66*(L*L)*(j2*j2))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+8
                        k0v[c] += pi*i2*j2*k2*((-1)**(i2 + k2) - 1)*(B12 + B66)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -pi*i2*j2*k2*((-1)**(i2 + k2) - 1)*(A12 + A66)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += pi*A26*i2*j2*k2*(-2*(-1)**(i2 + k2) + 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += pi*i2*k2*((-1)**(i2 + k2) - 1)*(A26 + A45)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+6
                        k0v[c] += -i2*((-1)**(i2 + k2) - 1)*(-A45*(L*L)*r + (pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2*j2))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+9
                        k0v[c] += pi*B26*i2*j2*k2*(-2*(-1)**(i2 + k2) + 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += pi*i2*j2*k2*((-1)**(i2 + k2) - 1)*(A12 + A66)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += pi*A26*i2*j2*k2*(2*(-1)**(i2 + k2) - 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+5
                        k0v[c] += pi*i2*k2*((-1)**(i2 + k2) - 1)*(A26 + A45)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+7
                        k0v[c] += -i2*((-1)**(i2 + k2) - 1)*(-A45*(L*L)*r + (pi*pi)*B16*(k2*k2)*(r*r) + B26*(L*L)*(j2*j2))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+8
                        k0v[c] += pi*B26*i2*j2*k2*(2*(-1)**(i2 + k2) - 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += -pi*A12*i2*k2*((-1)**(i2 + k2) - 1)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += -pi*i2*k2*((-1)**(i2 + k2) - 1)*(A26 + A45)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+5
                        k0v[c] += pi*A45*i2*j2*k2*(-2*(-1)**(i2 + k2) + 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+7
                        k0v[c] += L*i2*j2*((-1)**(i2 + k2) - 1)*(-A45*r + B26)/(r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+8
                        k0v[c] += -pi*i2*k2*((-1)**(i2 + k2) - 1)*(-A45*r + B26)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+1
                        k0v[c] += -pi*A12*i2*k2*((-1)**(i2 + k2) - 1)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+3
                        k0v[c] += -pi*i2*k2*((-1)**(i2 + k2) - 1)*(A26 + A45)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+4
                        k0v[c] += pi*A45*i2*j2*k2*(2*(-1)**(i2 + k2) - 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+6
                        k0v[c] += L*i2*j2*((-1)**(i2 + k2) - 1)*(A45*r - B26)/(r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+9
                        k0v[c] += -pi*i2*k2*((-1)**(i2 + k2) - 1)*(-A45*r + B26)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+0
                        k0v[c] += k2*((-1)**(i2 + k2) - 1)*((pi*pi)*B11*(i2*i2)*(r*r) + B66*(L*L)*(j2*j2))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+2
                        k0v[c] += k2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(-A45*(L*L) + (pi*pi)*B16*(i2*i2)*r))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+5
                        k0v[c] += L*j2*k2*((-1)**(i2 + k2) - 1)*(A45*r - B26)/(r*(-(i2*i2) + (k2*k2)))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+7
                        k0v[c] += -pi*D16*j2*((-1)**(i2 + k2) - 1)*((i2*i2) + (k2*k2))/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+8
                        k0v[c] += k2*((-1)**(i2 + k2) - 1)*(D26*(L*L)*(j2*j2) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2)))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+1
                        k0v[c] += k2*((-1)**(i2 + k2) - 1)*((pi*pi)*B11*(i2*i2)*(r*r) + B66*(L*L)*(j2*j2))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+3
                        k0v[c] += k2*((-1)**(i2 + k2) - 1)*(B26*(L*L)*(j2*j2) + r*(-A45*(L*L) + (pi*pi)*B16*(i2*i2)*r))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+4
                        k0v[c] += L*j2*k2*((-1)**(i2 + k2) - 1)*(-A45*r + B26)/(r*(-(i2*i2) + (k2*k2)))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+6
                        k0v[c] += pi*D16*j2*((-1)**(i2 + k2) - 1)*((i2*i2) + (k2*k2))/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+9
                        k0v[c] += k2*((-1)**(i2 + k2) - 1)*(D26*(L*L)*(j2*j2) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(i2*i2)))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+1
                        k0v[c] += -pi*i2*j2*k2*((-1)**(i2 + k2) - 1)*(B12 + B66)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+3
                        k0v[c] += pi*B26*i2*j2*k2*(-2*(-1)**(i2 + k2) + 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+4
                        k0v[c] += pi*i2*k2*((-1)**(i2 + k2) - 1)*(-A45*r + B26)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+6
                        k0v[c] += -i2*((-1)**(i2 + k2) - 1)*(D26*(L*L)*(j2*j2) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+9
                        k0v[c] += pi*D26*i2*j2*k2*(-2*(-1)**(i2 + k2) + 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+0
                        k0v[c] += pi*i2*j2*k2*((-1)**(i2 + k2) - 1)*(B12 + B66)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+2
                        k0v[c] += pi*B26*i2*j2*k2*(2*(-1)**(i2 + k2) - 2)/((i2*i2) - (k2*k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+5
                        k0v[c] += pi*i2*k2*((-1)**(i2 + k2) - 1)*(-A45*r + B26)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+7
                        k0v[c] += -i2*((-1)**(i2 + k2) - 1)*(D26*(L*L)*(j2*j2) + (r*r)*(A45*(L*L) + (pi*pi)*D16*(k2*k2)))/(L*r*((i2*i2) - (k2*k2)))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+8
                        k0v[c] += pi*D26*i2*j2*k2*(2*(-1)**(i2 + k2) - 2)/((i2*i2) - (k2*k2))

    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0edges(int m1, int m2, int n2, double r1, double r2,
             double kphixBot, double kphixTop):
    cdef int i1, k1, i2, j2, k2, l2, row, col, c
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv

    k11_cond_1 = 1
    k11_cond_2 = 1
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 2
    k22_cond_2 = 2
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

            #NOTE symmetry
            if row > col:
                continue

            if k1==i1:
                # k0edges_11 cond_1
                c += 1
                k0edgesr[c] = row+3
                k0edgesc[c] = col+3
                k0edgesv[c] += 2*pi*(kphixBot*r1 + kphixTop*r2)

            else:
                # k0edges_11 cond_2
                c += 1
                k0edgesr[c] = row+3
                k0edgesc[c] = col+3
                k0edgesv[c] += 2*pi*((-1)**(i1 + k1)*kphixBot*r1 + kphixTop*r2)

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
                        k0edgesr[c] = row+6
                        k0edgesc[c] = col+6
                        k0edgesv[c] += pi*(kphixBot*r1 + kphixTop*r2)
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+7
                        k0edgesv[c] += pi*(kphixBot*r1 + kphixTop*r2)

                    elif k2!=i2 and l2==j2:
                        # k0edges_22 cond_2
                        c += 1
                        k0edgesr[c] = row+6
                        k0edgesc[c] = col+6
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kphixBot*r1 + kphixTop*r2)
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+7
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kphixBot*r1 + kphixTop*r2)

    size = num0 + num1*m1 + num2*m2*n2

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fc, double P, double T, double r2, double alpharad, double L,
         int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double sina, cosa, xa, xb, r
    cdef double sini1xa, cosi1xa, sini1xb, cosi1xb
    cdef double sink1xa, sink1xb, cosk1xa, cosk1xb, sini2xa, sini2xb
    cdef double sin2i2xa, sin2i2xb
    cdef double cosi2xa, cosi2xb
    cdef double cosk2xa, cosk2xb, sink2xa, sink2xb
    cdef double sin2i1xa, sin2i1xb

    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    # sparse parameters
    k11_cond_1 = 1
    k11_cond_2 = 1
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 2
    k22_cond_2 = 4
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k11_num + k22_num

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

            for i1 in range(i0, m1+i0):
                sin2i1xa = sin(2*pi*i1*xa/L)
                sin2i1xb = sin(2*pi*i1*xb/L)
                cosi1xa = cos(pi*i1*xa/L)
                cosi1xb = cos(pi*i1*xb/L)
                sini1xa = sin(pi*i1*xa/L)
                sini1xb = sin(pi*i1*xb/L)

                row = (i1-i0)*num1 + num0
                for k1 in range(i0, m1+i0):
                    col = (k1-i0)*num1 + num0

                    #NOTE symmetry
                    if row > col:
                        continue

                    cosk1xa = cos(pi*k1*xa/L)
                    cosk1xb = cos(pi*k1*xb/L)
                    sink1xa = sin(pi*k1*xa/L)
                    sink1xb = sin(pi*k1*xb/L)

                    if k1==i1:
                        if i1!=0:
                            # kG0_11 cond_1
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += 0.25*pi*i1*(Fc - pi*P*(r*r))*(L*sin2i1xa - L*sin2i1xb + 2*pi*i1*(xa - xb))/((L*L)*cosa)
                    else:
                        # kG0_11 cond_2
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += pi*i1*k1*(Fc - pi*P*(r*r))*(-cosi1xa*k1*sink1xa + cosi1xb*k1*sink1xb + cosk1xa*i1*sini1xa - cosk1xb*i1*sini1xb)/(L*cosa*(i1 - k1)*(i1 + k1))

            for i2 in range(i0, m2+i0):
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

                            #NOTE symmetry
                            if row > col:
                                continue

                            if k2==i2 and l2==j2:
                                if i2!=0:
                                    # kG0_22 cond_1
                                    c += 1
                                    kG0r[c] = row+4
                                    kG0c[c] = col+4
                                    kG0v[c] += 0.125*(L*(sin2i2xa - sin2i2xb)*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(Fc - pi*P*(r*r))) - 2*pi*i2*(xa - xb)*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r))))/((L*L)*cosa*i2)
                                    c += 1
                                    kG0r[c] = row+5
                                    kG0c[c] = col+5
                                    kG0v[c] += 0.125*(L*(sin2i2xa - sin2i2xb)*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(Fc - pi*P*(r*r))) - 2*pi*i2*(xa - xb)*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r))))/((L*L)*cosa*i2)

                            elif k2!=i2 and l2==j2:
                                # kG0_22 cond_2
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
    cdef double r=r2
    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    # sparse parameters
    k11_cond_1 = 1
    k11_cond_2 = 0
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 2
    k22_cond_2 = 2
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = k11_num + k22_num

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    for i1 in range(i0, m1+i0):
        row = (i1-i0)*num1 + num0
        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0

            #NOTE symmetry
            if row > col:
                continue

            if k1==i1:
                if i1!=0:
                    # kG0_11 cond_1
                    c += 1
                    kG0r[c] = row+2
                    kG0c[c] = col+2
                    kG0v[c] += 0.5*(pi*pi)*(i1*i1)*(-Fc + pi*P*(r*r))/L

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
                            kG0r[c] = row+4
                            kG0c[c] = col+4
                            kG0v[c] += 0.25*pi*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r)))/L
                            c += 1
                            kG0r[c] = row+5
                            kG0c[c] = col+5
                            kG0v[c] += 0.25*pi*(2*(L*L)*P*(j2*j2) + pi*(i2*i2)*(-Fc + pi*P*(r*r)))/L

                    elif k2!=i2 and l2==j2:
                        # kG0_22 cond_2
                        c += 1
                        kG0r[c] = row+4
                        kG0c[c] = col+5
                        kG0v[c] += T*i2*j2*k2*((-1)**(i2 + k2) - 1)/((r*r)*((i2*i2) - (k2*k2)))
                        c += 1
                        kG0r[c] = row+5
                        kG0c[c] = col+4
                        kG0v[c] += -T*i2*j2*k2*((-1)**(i2 + k2) - 1)/((r*r)*((i2*i2) - (k2*k2)))

    size = num0 + num1*m1 + num2*m2*n2

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
