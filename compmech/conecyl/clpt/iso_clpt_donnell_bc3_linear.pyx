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


def fk0(double alpharad, double r2, double L, double E11, double nu, double h,
        int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double r, sina, cosa, xa, xb
    cdef double sini1xa, cosi1xa, sini1xb, cosi1xb
    cdef double sini1xa_xb, sini1xaxb, cosi1xaxb
    cdef double sink1xa, sink1xb, cosk1xa, cosk1xb, sini2xa, sini2xb
    cdef double sin2i2xa, sin2i2xb, sini2xa_xb, sini2xaxb, cosi2xaxb
    cdef double cosi2xa, cosi2xb, cos2i2xa, cos2i2xb
    cdef double cosk2xa, cosk2xb, sink2xa, sink2xb
    cdef double sin2i1xa, sin2i1xb, cos2i1xa, cos2i1xb

    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    sina = sin(alpharad)
    cosa = cos(alpharad)

    # sparse parameters
    k11_cond_1 = 5
    k11_cond_2 = 5
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 18
    k22_cond_2 = 18
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 3 + 3*m1 + k11_num + k22_num

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

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
            k0v[c] += 0.666666666666667*pi*E11*h*(xa - xb)*(3*nu*r*sina*(-2*L + xa + xb) + 3*(r*r) + (sina*sina)*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb)))/((L*L)*(cosa*cosa)*r*((nu*nu) - 1))
            c += 1
            k0r[c] = 1
            k0c[c] = 1
            k0v[c] += -0.333333333333333*pi*E11*h*(r2*r2)*(xa - xb)*(3*(r*r) + 3*r*sina*(2*L - xa - xb) + (sina*sina)*(3*(L*L) - 3*L*(xa + xb) + (xa*xa) + xa*xb + (xb*xb)))/((L*L)*r*(nu + 1))
            c += 1
            k0r[c] = 2
            k0c[c] = 2
            k0v[c] += -0.166666666666667*pi*E11*h*(xa - xb)*(3*(L*L)*(nu - 2*(sina*sina) - 1) + 3*L*(nu*(4*r*sina - xa - xb) + 2*(sina*sina)*(xa + xb) + xa + xb) - 6*nu*r*sina*(xa + xb) - 6*(r*r) + (nu - 2*(sina*sina) - 1)*((xa*xa) + xa*xb + (xb*xb)))/((L*L)*(cosa*cosa)*r*((nu*nu) - 1))

            for i1 in range(i0, m1+i0):
                cosi1xa = cos(pi*i1*xa/L)
                cosi1xb = cos(pi*i1*xb/L)
                sini1xa = sin(pi*i1*xa/L)
                sini1xb = sin(pi*i1*xb/L)
                cosi1xaxb = cos(pi*i1*(xa + xb)/L)
                sini1xa_xb = sin(pi*i1*(xa - xb)/L)
                sini1xaxb = sin(pi*i1*(xa + xb)/L)
                sin2i1xa = sin(2*pi*i1*xa/L)
                sin2i1xb = sin(2*pi*i1*xb/L)
                cos2i1xa = cos(2*pi*i1*xa/L)
                cos2i1xb = cos(2*pi*i1*xb/L)

                col = (i1-i0)*num1 + num0
                row = col

                if i1 != 0:
                    # k0_01 cond_1
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+0
                    k0v[c] += 2*E11*h*(pi*L*cosk1xa*k1*(sina*sina)*(-L + xa) + pi*L*cosk1xb*k1*(sina*sina)*(L - xb) - sink1xa*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*r*(nu*sina*(-L + xa) + r)) + sink1xb*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*r*(nu*sina*(-L + xb) + r)))/(pi*L*cosa*(k1*k1)*r*((nu*nu) - 1))
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+2
                    k0v[c] += 2*E11*h*(L*sina*(-sink1xa + sink1xb) + pi*cosk1xa*k1*(nu*r + sina*(-L + xa)) - pi*cosk1xb*k1*(-L*sina + nu*r + sina*xb))/(pi*(k1*k1)*r*((nu*nu) - 1))
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+1
                    k0v[c] += E11*h*r2*(pi*L*cosk1xa*k1*(sina*sina)*(L - xa) + pi*L*cosk1xb*k1*(sina*sina)*(-L + xb) + sink1xa*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*r*(L*sina + r - sina*xa)) - sink1xb*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*r*(L*sina + r - sina*xb)))/(pi*L*(k1*k1)*r*(nu + 1))

                for k1 in range(i0, m1+i0):
                    col = (k1-i0)*num1 + num0

                    #NOTE symmetry
                    if row > col:
                        continue

                    cosk1xa = cos(pi*k1*xa/L)
                    cosk1xb = cos(pi*k1*xb/L)
                    sink1xa = sin(pi*k1*xa/L)
                    sink1xb = sin(pi*k1*xb/L)
                    if k1 == i1:
                        if i1 != 0:
                            # k0_11 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.5*E11*h*(2*L*sini1xa_xb*(2*pi*L*i1*nu*r*sina*sini1xaxb + cosi1xaxb*(-L*sina + pi*i1*r)*(L*sina + pi*i1*r)) + 2*pi*i1*(xa - xb)*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)))/((L*L)*i1*r*((nu*nu) - 1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += 0.5*E11*cosa*h*(pi*i1*nu*r*(-cos2i1xa + cos2i1xb) + sina*(-L*sin2i1xa + L*sin2i1xb + 2*pi*i1*(xa - xb)))/(i1*r*((nu*nu) - 1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.25*E11*h*(2*L*sini1xa_xb*(2*pi*L*i1*r*sina*sini1xaxb + cosi1xaxb*((L*L)*(sina*sina) - (pi*pi)*(i1*i1)*(r*r))) - 2*pi*i1*(xa - xb)*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)))/((L*L)*i1*r*(nu + 1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += 0.5*E11*cosa*h*(pi*i1*nu*r*(-cos2i1xa + cos2i1xb) + sina*(-L*sin2i1xa + L*sin2i1xb + 2*pi*i1*(xa - xb)))/(i1*r*((nu*nu) - 1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.0416666666666667*E11*h*(2*L*sini1xa_xb*(-2*(pi*pi*pi)*L*(h*h)*(i1*i1*i1)*nu*r*sina*sini1xaxb + cosi1xaxb*(-12*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(h*h)*(i1*i1)*((L*L)*(sina*sina) - (pi*pi)*(i1*i1)*(r*r)))) + 2*pi*i1*(xa - xb)*(12*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(h*h)*(i1*i1)*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r))))/((L*L*L*L)*i1*r*((nu*nu) - 1))

                    else:
                        # k0_11 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += 2*E11*h*(cosk1xa*k1*sini1xa*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)) - cosk1xb*k1*sini1xb*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)) - sink1xa*(pi*L*nu*r*sina*sini1xa*(-(i1*i1) + (k1*k1)) + cosi1xa*i1*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r))) + sink1xb*(pi*L*nu*r*sina*sini1xb*(-(i1*i1) + (k1*k1)) + cosi1xb*i1*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r))))/(L*r*(i1 - k1)*(i1 + k1)*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 2*E11*cosa*h*(cosi1xa*i1*(-L*sina*sink1xa + pi*cosk1xa*k1*nu*r) + cosi1xb*i1*(L*sina*sink1xb - pi*cosk1xb*k1*nu*r) + sini1xa*(L*cosk1xa*k1*sina + pi*(i1*i1)*nu*r*sink1xa) - sini1xb*(L*cosk1xb*k1*sina + pi*(i1*i1)*nu*r*sink1xb))/(r*(i1 - k1)*(i1 + k1)*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += E11*h*(-cosk1xa*k1*sini1xa*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)) + cosk1xb*k1*sini1xb*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)) + sink1xa*(pi*L*r*sina*sini1xa*(i1 - k1)*(i1 + k1) + cosi1xa*i1*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r))) - sink1xb*(pi*L*r*sina*sini1xb*(i1 - k1)*(i1 + k1) + cosi1xb*i1*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r))))/(L*r*(i1 - k1)*(i1 + k1)*(nu + 1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 2*E11*cosa*h*(-cosi1xa*i1*(L*sina*sink1xa + pi*cosk1xa*k1*nu*r) + cosi1xb*i1*(L*sina*sink1xb + pi*cosk1xb*k1*nu*r) + k1*sini1xa*(L*cosk1xa*sina - pi*k1*nu*r*sink1xa) + k1*sini1xb*(-L*cosk1xb*sina + pi*k1*nu*r*sink1xb))/(r*(i1 - k1)*(i1 + k1)*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += E11*h*(-cosi1xa*i1*((pi*pi*pi)*L*cosk1xa*(h*h)*k1*nu*r*sina*(-(i1*i1) + (k1*k1)) + sink1xa*(12*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(h*h)*(k1*k1)*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)))) + cosi1xb*i1*((pi*pi*pi)*L*cosk1xb*(h*h)*k1*nu*r*sina*(-(i1*i1) + (k1*k1)) + sink1xb*(12*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(h*h)*(k1*k1)*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r)))) + k1*(12*(L*L*L*L)*(cosa*cosa) + (pi*pi)*(h*h)*(i1*i1)*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r)))*(cosk1xa*sini1xa - cosk1xb*sini1xb))/((L*L*L)*r*(6.0*(i1*i1) - 6.0*(k1*k1))*((nu*nu) - 1))

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

                            if k2 == i2 and l2 == j2:
                                if i2 != 0:
                                    # k0_22 cond_1
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+0
                                    k0v[c] += 0.125*E11*h*(2*L*sini2xa_xb*(4*pi*L*i2*nu*r*sina*sini2xaxb + cosi2xaxb*((L*L)*((j2*j2)*(nu - 1) - 2*(sina*sina)) + 2*(pi*pi)*(i2*i2)*(r*r))) + 2*pi*i2*(xa - xb)*((L*L)*(-(j2*j2)*(nu - 1) + 2*(sina*sina)) + 2*(pi*pi)*(i2*i2)*(r*r)))/((L*L)*i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+3
                                    k0v[c] += 0.125*E11*h*j2*(-(L*L)*cos2i2xa*sina*(nu - 3) + (L*L)*cos2i2xb*sina*(nu - 3) + pi*i2*r*(L*(3*nu - 1)*(-sin2i2xa + sin2i2xb) - 2*pi*i2*(nu + 1)*(xa - xb)))/(L*i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+0
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*E11*cosa*h*(pi*i2*nu*r*(-cos2i2xa + cos2i2xb) + sina*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb)))/(i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+1
                                    k0v[c] += 0.125*E11*h*(2*L*sini2xa_xb*(4*pi*L*i2*nu*r*sina*sini2xaxb + cosi2xaxb*((L*L)*((j2*j2)*(nu - 1) - 2*(sina*sina)) + 2*(pi*pi)*(i2*i2)*(r*r))) + 2*pi*i2*(xa - xb)*((L*L)*(-(j2*j2)*(nu - 1) + 2*(sina*sina)) + 2*(pi*pi)*(i2*i2)*(r*r)))/((L*L)*i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+2
                                    k0v[c] += 0.125*E11*h*j2*((L*L)*cos2i2xa*sina*(nu - 3) - (L*L)*cos2i2xb*sina*(nu - 3) + pi*i2*r*(L*(3*nu - 1)*(sin2i2xa - sin2i2xb) + 2*pi*i2*(nu + 1)*(xa - xb)))/(L*i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+1
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*E11*cosa*h*(pi*i2*nu*r*(-cos2i2xa + cos2i2xb) + sina*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb)))/(i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+1
                                    k0v[c] += 0.125*E11*h*j2*((L*L)*cos2i2xa*sina*(nu - 3) - (L*L)*cos2i2xb*sina*(nu - 3) + pi*i2*r*(L*(3*nu - 1)*(sin2i2xa - sin2i2xb) + 2*pi*i2*(nu + 1)*(xa - xb)))/(L*i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+2
                                    k0v[c] += 0.125*pi*E11*h*(2*L*sini2xa_xb*(-2*pi*L*i2*r*sina*sini2xaxb*(nu - 1) + cosi2xaxb*(2*(L*L)*(j2*j2) + (nu - 1)*(-L*sina + pi*i2*r)*(L*sina + pi*i2*r)))/(pi*i2) + (xa - xb)*(4*(L*L)*(j2*j2) - 2*(nu - 1)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r))))/((L*L)*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+5
                                    k0v[c] += 0.25*E11*L*cosa*h*j2*(-cos2i2xa + cos2i2xb)/(i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+0
                                    k0v[c] += 0.125*E11*h*j2*(-(L*L)*cos2i2xa*sina*(nu - 3) + (L*L)*cos2i2xb*sina*(nu - 3) + pi*i2*r*(L*(3*nu - 1)*(-sin2i2xa + sin2i2xb) - 2*pi*i2*(nu + 1)*(xa - xb)))/(L*i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+3
                                    k0v[c] += 0.125*pi*E11*h*(2*L*sini2xa_xb*(-2*pi*L*i2*r*sina*sini2xaxb*(nu - 1) + cosi2xaxb*(2*(L*L)*(j2*j2) + (nu - 1)*(-L*sina + pi*i2*r)*(L*sina + pi*i2*r)))/(pi*i2) + (xa - xb)*(4*(L*L)*(j2*j2) - 2*(nu - 1)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r))))/((L*L)*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+4
                                    k0v[c] += 0.25*E11*L*cosa*h*j2*(cos2i2xa - cos2i2xb)/(i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+0
                                    k0v[c] += 0.25*E11*cosa*h*(pi*i2*nu*r*(-cos2i2xa + cos2i2xb) + sina*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb)))/(i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+3
                                    k0v[c] += 0.25*E11*L*cosa*h*j2*(cos2i2xa - cos2i2xb)/(i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+4
                                    k0c[c] = col+4
                                    k0v[c] += 0.0104166666666667*E11*h*(2*L*sini2xa_xb*(4*pi*L*(h*h)*i2*r*sina*sini2xaxb*((L*L)*(j2*j2)*(nu - 2) - (pi*pi)*(i2*i2)*nu*(r*r)) - cosi2xaxb*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + 4*(pi*pi)*(L*L)*(i2*i2)*(j2*j2)*(r*r)*(2*nu - 1) + (L*L)*(sina*sina)*(-(L*L)*(j2*j2)*(nu - 1) - 2*(pi*pi)*(i2*i2)*(r*r)) + 2*(pi*pi*pi*pi)*(i2*i2*i2*i2)*(r*r*r*r)))) + 2*pi*i2*(xa - xb)*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*((L*L)*(sina*sina)*(-(L*L)*(j2*j2)*(nu - 1) + 2*(pi*pi)*(i2*i2)*(r*r)) + 2*((L*L)*(j2*j2) + (pi*pi)*(i2*i2)*(r*r))**2)))/((L*L*L*L)*i2*(r*r*r)*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+1
                                    k0v[c] += 0.25*E11*cosa*h*(pi*i2*nu*r*(-cos2i2xa + cos2i2xb) + sina*(-L*sin2i2xa + L*sin2i2xb + 2*pi*i2*(xa - xb)))/(i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+2
                                    k0v[c] += 0.25*E11*L*cosa*h*j2*(-cos2i2xa + cos2i2xb)/(i2*r*((nu*nu) - 1))
                                    c += 1
                                    k0r[c] = row+5
                                    k0c[c] = col+5
                                    k0v[c] += 0.0104166666666667*E11*h*(2*L*sini2xa_xb*(4*pi*L*(h*h)*i2*r*sina*sini2xaxb*((L*L)*(j2*j2)*(nu - 2) - (pi*pi)*(i2*i2)*nu*(r*r)) - cosi2xaxb*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + 4*(pi*pi)*(L*L)*(i2*i2)*(j2*j2)*(r*r)*(2*nu - 1) + (L*L)*(sina*sina)*(-(L*L)*(j2*j2)*(nu - 1) - 2*(pi*pi)*(i2*i2)*(r*r)) + 2*(pi*pi*pi*pi)*(i2*i2*i2*i2)*(r*r*r*r)))) + 2*pi*i2*(xa - xb)*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*((L*L)*(sina*sina)*(-(L*L)*(j2*j2)*(nu - 1) + 2*(pi*pi)*(i2*i2)*(r*r)) + 2*((L*L)*(j2*j2) + (pi*pi)*(i2*i2)*(r*r))**2)))/((L*L*L*L)*i2*(r*r*r)*((nu*nu) - 1))

                                else:
                                    # k0_22 cond_5
                                    c += 1
                                    k0r[c] = row+2
                                    k0c[c] = col+2
                                    k0v[c] += pi*E11*h*(2*(j2*j2) - (sina*sina)*(nu - 1))*(xa - xb)/(r*(2.0*(nu*nu) - 2.0))
                                    c += 1
                                    k0r[c] = row+3
                                    k0c[c] = col+3
                                    k0v[c] += pi*E11*h*(2*(j2*j2) - (sina*sina)*(nu - 1))*(xa - xb)/(r*(2.0*(nu*nu) - 2.0))

                            elif k2 != i2 and l2 == j2:
                                # k0_22 cond_2
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += E11*h*(cosk2xa*k2*sini2xa*((L*L)*(-(j2*j2)*(nu - 1) + 2*(sina*sina)) + 2*(pi*pi)*(i2*i2)*(r*r)) + cosk2xb*k2*sini2xb*((L*L)*((j2*j2)*(nu - 1) - 2*(sina*sina)) - 2*(pi*pi)*(i2*i2)*(r*r)) + sink2xa*(pi*L*nu*r*sina*sini2xa*(i2 + k2)*(2*i2 - 2*k2) + cosi2xa*i2*((L*L)*((j2*j2)*(nu - 1) - 2*(sina*sina)) - 2*(pi*pi)*(k2*k2)*(r*r))) + sink2xb*(pi*L*nu*r*sina*sini2xb*(-2*(i2*i2) + 2*(k2*k2)) + cosi2xb*i2*((L*L)*(-(j2*j2)*nu + (j2*j2) + 2*(sina*sina)) + 2*(pi*pi)*(k2*k2)*(r*r))))/(L*r*(i2 + k2)*(2.0*i2 - 2.0*k2)*(nu - 1)*(nu + 1))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += E11*h*j2*(L*k2*sina*sini2xb*sink2xb*(nu - 3) + cosi2xa*i2*(-L*cosk2xa*sina*(nu - 3) + pi*k2*r*sink2xa*(nu + 1)) + cosi2xb*i2*(L*cosk2xb*sina*(nu - 3) - pi*k2*r*sink2xb*(nu + 1)) + pi*cosk2xb*r*sini2xb*(2*(i2*i2)*nu - (k2*k2)*nu + (k2*k2)) + sini2xa*(-L*k2*sina*sink2xa*(nu - 3) + pi*cosk2xa*r*(-2*(i2*i2)*nu + (k2*k2)*(nu - 1))))/(r*(i2 + k2)*(2.0*i2 - 2.0*k2)*(nu - 1)*(nu + 1))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += E11*cosa*h*(cosi2xa*i2*(-L*sina*sink2xa + pi*cosk2xa*k2*nu*r) + cosi2xb*i2*(L*sina*sink2xb - pi*cosk2xb*k2*nu*r) + sini2xa*(L*cosk2xa*k2*sina + pi*(i2*i2)*nu*r*sink2xa) - sini2xb*(L*cosk2xb*k2*sina + pi*(i2*i2)*nu*r*sink2xb))/(r*(i2 - k2)*(i2 + k2)*((nu*nu) - 1))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += E11*h*(cosk2xa*k2*sini2xa*((L*L)*(-(j2*j2)*(nu - 1) + 2*(sina*sina)) + 2*(pi*pi)*(i2*i2)*(r*r)) + cosk2xb*k2*sini2xb*((L*L)*((j2*j2)*(nu - 1) - 2*(sina*sina)) - 2*(pi*pi)*(i2*i2)*(r*r)) + sink2xa*(pi*L*nu*r*sina*sini2xa*(i2 + k2)*(2*i2 - 2*k2) + cosi2xa*i2*((L*L)*((j2*j2)*(nu - 1) - 2*(sina*sina)) - 2*(pi*pi)*(k2*k2)*(r*r))) + sink2xb*(pi*L*nu*r*sina*sini2xb*(-2*(i2*i2) + 2*(k2*k2)) + cosi2xb*i2*((L*L)*(-(j2*j2)*nu + (j2*j2) + 2*(sina*sina)) + 2*(pi*pi)*(k2*k2)*(r*r))))/(L*r*(i2 + k2)*(2.0*i2 - 2.0*k2)*(nu - 1)*(nu + 1))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += E11*h*j2*(-L*k2*sina*sini2xb*sink2xb*(nu - 3) + cosi2xa*i2*(L*cosk2xa*sina*(nu - 3) - pi*k2*r*sink2xa*(nu + 1)) + cosi2xb*i2*(-L*cosk2xb*sina*(nu - 3) + pi*k2*r*sink2xb*(nu + 1)) + pi*cosk2xb*r*sini2xb*(-2*(i2*i2)*nu + (k2*k2)*(nu - 1)) + sini2xa*(L*k2*sina*sink2xa*(nu - 3) + pi*cosk2xa*r*(2*(i2*i2)*nu - (k2*k2)*nu + (k2*k2))))/(r*(i2 + k2)*(2.0*i2 - 2.0*k2)*(nu - 1)*(nu + 1))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+5
                                k0v[c] += E11*cosa*h*(cosi2xa*i2*(-L*sina*sink2xa + pi*cosk2xa*k2*nu*r) + cosi2xb*i2*(L*sina*sink2xb - pi*cosk2xb*k2*nu*r) + sini2xa*(L*cosk2xa*k2*sina + pi*(i2*i2)*nu*r*sink2xa) - sini2xb*(L*cosk2xb*k2*sina + pi*(i2*i2)*nu*r*sink2xb))/(r*(i2 - k2)*(i2 + k2)*((nu*nu) - 1))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += E11*h*j2*(cosi2xa*(-L*cosk2xa*k2*sina*(nu - 3) + pi*r*sink2xa*((i2*i2)*(nu - 1) - 2*(k2*k2)*nu)) + cosi2xb*(L*cosk2xb*k2*sina*(nu - 3) + pi*r*sink2xb*(-(i2*i2)*nu + (i2*i2) + 2*(k2*k2)*nu)) + i2*(L*sina*sini2xb*sink2xb*(nu - 3) - pi*cosk2xb*k2*r*sini2xb*(nu + 1) + sini2xa*(-L*sina*sink2xa*(nu - 3) + pi*cosk2xa*k2*r*(nu + 1))))/(r*(i2 + k2)*(2.0*i2 - 2.0*k2)*(nu - 1)*(nu + 1))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += E11*h*(cosi2xa*(pi*L*cosk2xa*r*sina*(i2 - k2)*(i2 + k2)*(nu - 1) + k2*sink2xa*(-2*(L*L)*(j2*j2) + (nu - 1)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r)))) + cosi2xb*(-pi*L*cosk2xb*r*sina*(i2 - k2)*(i2 + k2)*(nu - 1) + k2*sink2xb*(2*(L*L)*(j2*j2) - (nu - 1)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r)))) + i2*(2*(L*L)*(j2*j2) - (nu - 1)*((L*L)*(sina*sina) + (pi*pi)*(k2*k2)*(r*r)))*(cosk2xa*sini2xa - cosk2xb*sini2xb))/(L*r*(i2 + k2)*(2.0*i2 - 2.0*k2)*(nu - 1)*(nu + 1))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+5
                                k0v[c] += E11*L*cosa*h*j2*(-cosi2xa*cosk2xa*k2 + cosi2xb*cosk2xb*k2 - i2*sini2xa*sink2xa + i2*sini2xb*sink2xb)/(((i2*i2) - (k2*k2))*(-(nu*nu)*r + r))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += E11*h*j2*(cosi2xa*(L*cosk2xa*k2*sina*(nu - 3) + pi*r*sink2xa*(-(i2*i2)*nu + (i2*i2) + 2*(k2*k2)*nu)) + cosi2xb*(-L*cosk2xb*k2*sina*(nu - 3) + pi*r*sink2xb*((i2*i2)*(nu - 1) - 2*(k2*k2)*nu)) + i2*(-L*sina*sini2xb*sink2xb*(nu - 3) + pi*cosk2xb*k2*r*sini2xb*(nu + 1) + sini2xa*(L*sina*sink2xa*(nu - 3) - pi*cosk2xa*k2*r*(nu + 1))))/(r*(i2 + k2)*(2.0*i2 - 2.0*k2)*(nu - 1)*(nu + 1))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += E11*h*(cosi2xa*(pi*L*cosk2xa*r*sina*(i2 - k2)*(i2 + k2)*(nu - 1) + k2*sink2xa*(-2*(L*L)*(j2*j2) + (nu - 1)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r)))) + cosi2xb*(-pi*L*cosk2xb*r*sina*(i2 - k2)*(i2 + k2)*(nu - 1) + k2*sink2xb*(2*(L*L)*(j2*j2) - (nu - 1)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r)))) + i2*(2*(L*L)*(j2*j2) - (nu - 1)*((L*L)*(sina*sina) + (pi*pi)*(k2*k2)*(r*r)))*(cosk2xa*sini2xa - cosk2xb*sini2xb))/(L*r*(i2 + k2)*(2.0*i2 - 2.0*k2)*(nu - 1)*(nu + 1))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += E11*L*cosa*h*j2*(-cosi2xa*cosk2xa*k2 + cosi2xb*cosk2xb*k2 - i2*sini2xa*sink2xa + i2*sini2xb*sink2xb)/(r*((i2*i2) - (k2*k2))*((nu*nu) - 1))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += E11*cosa*h*(-cosi2xa*i2*(L*sina*sink2xa + pi*cosk2xa*k2*nu*r) + cosi2xb*i2*(L*sina*sink2xb + pi*cosk2xb*k2*nu*r) + k2*sini2xa*(L*cosk2xa*sina - pi*k2*nu*r*sink2xa) + k2*sini2xb*(-L*cosk2xb*sina + pi*k2*nu*r*sink2xb))/(r*(i2 - k2)*(i2 + k2)*((nu*nu) - 1))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += E11*L*cosa*h*j2*(cosi2xa*cosk2xa*i2 - cosi2xb*cosk2xb*i2 + k2*sini2xa*sink2xa - k2*sini2xb*sink2xb)/(r*((i2*i2) - (k2*k2))*((nu*nu) - 1))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += E11*h*(2*pi*(L*L*L)*(h*h)*(j2*j2)*r*sina*sini2xb*sink2xb*(-(i2*i2) + (k2*k2))*(nu - 2) + cosi2xa*(2*(pi*pi*pi)*L*cosk2xa*(h*h)*i2*k2*nu*(r*r*r)*sina*(i2 - k2)*(i2 + k2) - i2*sink2xa*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + (L*L)*(j2*j2)*(-(L*L)*(sina*sina)*(nu - 1) + (pi*pi)*(r*r)*(2*(i2*i2)*nu - 2*(k2*k2)*(nu - 2))) + 2*(pi*pi)*(k2*k2)*(r*r)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r))))) + cosi2xb*(2*(pi*pi*pi)*L*cosk2xb*(h*h)*i2*k2*nu*(r*r*r)*sina*(-(i2*i2) + (k2*k2)) + i2*sink2xb*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + (L*L)*(j2*j2)*(-(L*L)*(sina*sina)*(nu - 1) + (pi*pi)*(r*r)*(2*(i2*i2)*nu - 2*(k2*k2)*(nu - 2))) + 2*(pi*pi)*(k2*k2)*(r*r)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r))))) - cosk2xb*k2*sini2xb*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + (L*L)*(j2*j2)*(-(L*L)*(sina*sina)*(nu - 1) + (pi*pi)*(r*r)*(-2*(i2*i2)*(nu - 2) + 2*(k2*k2)*nu)) + 2*(pi*pi)*(i2*i2)*(r*r)*((L*L)*(sina*sina) + (pi*pi)*(k2*k2)*(r*r)))) + sini2xa*(2*pi*(L*L*L)*(h*h)*(j2*j2)*r*sina*sink2xa*(i2 - k2)*(i2 + k2)*(nu - 2) + cosk2xa*k2*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + (L*L)*(j2*j2)*(-(L*L)*(sina*sina)*(nu - 1) + (pi*pi)*(r*r)*(-2*(i2*i2)*(nu - 2) + 2*(k2*k2)*nu)) + 2*(pi*pi)*(i2*i2)*(r*r)*((L*L)*(sina*sina) + (pi*pi)*(k2*k2)*(r*r))))))/((L*L*L)*(r*r*r)*(i2 + k2)*(24.0*i2 - 24.0*k2)*((nu*nu) - 1))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+1
                                k0v[c] += E11*cosa*h*(-cosi2xa*i2*(L*sina*sink2xa + pi*cosk2xa*k2*nu*r) + cosi2xb*i2*(L*sina*sink2xb + pi*cosk2xb*k2*nu*r) + k2*sini2xa*(L*cosk2xa*sina - pi*k2*nu*r*sink2xa) + k2*sini2xb*(-L*cosk2xb*sina + pi*k2*nu*r*sink2xb))/(r*(i2 - k2)*(i2 + k2)*((nu*nu) - 1))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+2
                                k0v[c] += E11*L*cosa*h*j2*(cosi2xa*cosk2xa*i2 - cosi2xb*cosk2xb*i2 + k2*sini2xa*sink2xa - k2*sini2xb*sink2xb)/(((i2*i2) - (k2*k2))*(-(nu*nu)*r + r))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+5
                                k0v[c] += E11*h*(2*pi*(L*L*L)*(h*h)*(j2*j2)*r*sina*sini2xb*sink2xb*(-(i2*i2) + (k2*k2))*(nu - 2) + cosi2xa*(2*(pi*pi*pi)*L*cosk2xa*(h*h)*i2*k2*nu*(r*r*r)*sina*(i2 - k2)*(i2 + k2) - i2*sink2xa*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + (L*L)*(j2*j2)*(-(L*L)*(sina*sina)*(nu - 1) + (pi*pi)*(r*r)*(2*(i2*i2)*nu - 2*(k2*k2)*(nu - 2))) + 2*(pi*pi)*(k2*k2)*(r*r)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r))))) + cosi2xb*(2*(pi*pi*pi)*L*cosk2xb*(h*h)*i2*k2*nu*(r*r*r)*sina*(-(i2*i2) + (k2*k2)) + i2*sink2xb*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + (L*L)*(j2*j2)*(-(L*L)*(sina*sina)*(nu - 1) + (pi*pi)*(r*r)*(2*(i2*i2)*nu - 2*(k2*k2)*(nu - 2))) + 2*(pi*pi)*(k2*k2)*(r*r)*((L*L)*(sina*sina) + (pi*pi)*(i2*i2)*(r*r))))) - cosk2xb*k2*sini2xb*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + (L*L)*(j2*j2)*(-(L*L)*(sina*sina)*(nu - 1) + (pi*pi)*(r*r)*(-2*(i2*i2)*(nu - 2) + 2*(k2*k2)*nu)) + 2*(pi*pi)*(i2*i2)*(r*r)*((L*L)*(sina*sina) + (pi*pi)*(k2*k2)*(r*r)))) + sini2xa*(2*pi*(L*L*L)*(h*h)*(j2*j2)*r*sina*sink2xa*(i2 - k2)*(i2 + k2)*(nu - 2) + cosk2xa*k2*(24*(L*L*L*L)*(cosa*cosa)*(r*r) + (h*h)*(2*(L*L*L*L)*(j2*j2*j2*j2) + (L*L)*(j2*j2)*(-(L*L)*(sina*sina)*(nu - 1) + (pi*pi)*(r*r)*(-2*(i2*i2)*(nu - 2) + 2*(k2*k2)*nu)) + 2*(pi*pi)*(i2*i2)*(r*r)*((L*L)*(sina*sina) + (pi*pi)*(k2*k2)*(r*r))))))/((L*L*L)*(r*r*r)*(i2 + k2)*(24.0*i2 - 24.0*k2)*((nu*nu) - 1))

    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0_cyl(double r2, double L, double E11, double nu, double h,
            int m1, int m2, int n2):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    # sparse parameters
    k11_cond_1 = 3
    k11_cond_2 = 2
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 10
    k22_cond_2 = 8
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 3 + 1*m1 + k11_num + k22_num

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1
    r = r2

    # k0_00
    c += 1
    k0r[c] = 0
    k0c[c] = 0
    k0v[c] += 2*pi*E11*h*r/(-L*(nu*nu) + L)
    c += 1
    k0r[c] = 1
    k0c[c] = 1
    k0v[c] += pi*E11*h*r*(r2*r2)/(L*nu + L)
    c += 1
    k0r[c] = 2
    k0c[c] = 2
    k0v[c] += 0.166666666666667*pi*E11*h*((L*L)*(nu - 1) - 6*(r*r))/(L*r*((nu*nu) - 1))

    for i1 in range(i0, m1+i0):
        col = (i1-i0)*num1 + num0
        row = col
        if i1 != 0:
            # k0_01 cond_1
            c += 1
            k0r[c] = 0
            k0c[c] = col+2
            k0v[c] += E11*h*nu*(-2*(-1)**k1 + 2)/(k1*((nu*nu) - 1))

        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0

            #NOTE symmetry
            if row > col:
                continue

            if k1 == i1:
                if i1 != 0:
                    # k0_11 cond_1
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += (pi*pi*pi)*E11*h*(i1*i1)*r/(-L*(nu*nu) + L)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += (pi*pi*pi)*E11*h*(i1*i1)*r/(2*L*nu + 2*L)
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += -0.0833333333333333*pi*E11*h*(12*(L*L*L*L) + (pi*pi*pi*pi)*(h*h)*(i1*i1*i1*i1)*(r*r))/((L*L*L)*r*((nu*nu) - 1))

            else:
                # k0_11 cond_2
                c += 1
                k0r[c] = row+0
                k0c[c] = col+2
                k0v[c] += pi*E11*h*i1*k1*nu*(-2*(-1)**(i1 + k1) + 2)/(((i1*i1) - (k1*k1))*((nu*nu) - 1))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+0
                k0v[c] += pi*E11*h*i1*k1*nu*(2*(-1)**(i1 + k1) - 2)/(((i1*i1) - (k1*k1))*((nu*nu) - 1))

    for i2 in range(i0, m2+i0):
        for j2 in range(j0, n2+j0):
            row = (i2-i0)*num2 + (j2-j0)*num2*m2 + num0 + num1*m1
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, n2+j0):
                    col = (k2-i0)*num2 + (l2-j0)*num2*m2 + num0 + num1*m1

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k2 == i2 and l2 == j2:
                        if i2 != 0:
                            # k0_22 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += pi*E11*h*(L*(j2*j2)/(4*nu*r + 4*r) + (pi*pi)*(i2*i2)*r/(-2*L*(nu*nu) + 2*L))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += (pi*pi)*E11*h*i2*j2/(4.0*nu - 4.0)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += pi*E11*h*(L*(j2*j2)/(4*nu*r + 4*r) + (pi*pi)*(i2*i2)*r/(-2*L*(nu*nu) + 2*L))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += (pi*pi)*E11*h*i2*j2/(-4*nu + 4)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += (pi*pi)*E11*h*i2*j2/(-4*nu + 4)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.25*pi*E11*h*(2*(L*L)*(j2*j2)/(-(nu*nu)*r + r) + (pi*pi)*(i2*i2)*r/(nu + 1))/L
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+0
                            k0v[c] += (pi*pi)*E11*h*i2*j2/(4.0*nu - 4.0)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += 0.25*pi*E11*h*(2*(L*L)*(j2*j2)/(-(nu*nu)*r + r) + (pi*pi)*(i2*i2)*r/(nu + 1))/L
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+4
                            k0v[c] += -0.0416666666666667*pi*E11*h*(12*(L*L*L*L)*(r*r) + (h*h)*((L*L)*(j2*j2) + (pi*pi)*(i2*i2)*(r*r))**2)/((L*L*L)*(r*r*r)*((nu*nu) - 1))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+5
                            k0v[c] += -0.0416666666666667*pi*E11*h*(12*(L*L*L*L)*(r*r) + (h*h)*((L*L)*(j2*j2) + (pi*pi)*(i2*i2)*(r*r))**2)/((L*L*L)*(r*r*r)*((nu*nu) - 1))

                        else:
                            # k0_22_ cond_5
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += pi*E11*L*h*(j2*j2)/(-(nu*nu)*r + r)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += pi*E11*L*h*(j2*j2)/(-(nu*nu)*r + r)

                    elif k2 != i2 and l2 == j2:
                        # k0_22 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += -pi*E11*h*i2*k2*nu*((-1)**(i2 + k2) - 1)/(((i2*i2) - (k2*k2))*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+5
                        k0v[c] += -pi*E11*h*i2*k2*nu*((-1)**(i2 + k2) - 1)/(((i2*i2) - (k2*k2))*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+5
                        k0v[c] += E11*L*h*j2*k2*((-1)**(i2 + k2) - 1)/(r*(-(i2*i2) + (k2*k2))*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += -E11*L*h*j2*k2*((-1)**(i2 + k2) - 1)/(r*(-(i2*i2) + (k2*k2))*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += pi*E11*h*i2*k2*nu*((-1)**(i2 + k2) - 1)/(((i2*i2) - (k2*k2))*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += -E11*L*h*i2*j2*((-1)**(i2 + k2) - 1)/(r*((i2*i2) - (k2*k2))*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+1
                        k0v[c] += pi*E11*h*i2*k2*nu*((-1)**(i2 + k2) - 1)/(((i2*i2) - (k2*k2))*((nu*nu) - 1))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+2
                        k0v[c] += E11*L*h*i2*j2*((-1)**(i2 + k2) - 1)/(r*((i2*i2) - (k2*k2))*((nu*nu) - 1))

    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0
