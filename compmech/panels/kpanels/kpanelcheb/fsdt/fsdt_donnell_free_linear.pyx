#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

from scipy.sparse import csr_matrix
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

cdef int num0 = 1
cdef int num1 = 5
cdef int num2 = 5
cdef int num3 = 5
cdef int num4 = 5
cdef double pi = 3.141592653589793

def fk0(double r1, double L, double tmin, double tmax,
        np.ndarray[cDOUBLE, ndim=2] F,
        int m2, int n3, int m4, int n4, double alpharad, int s):
    cdef int i2, k2, j3, l3, i4, j4, k4, l4, c, row, col, section
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r, sina, cosa, xa, xb

    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    sina = sin(alpharad)
    cosa = cos(alpharad)

    fdim = (4 + 8 + 8*m2 + 8*n3 + 20*m4*n4 +
            4 + 4*m2 + 4*n3 + 10*m4*n4 +
            4*m2*m2 + 4*m2*n3 + 10*m2*m4*n4 +
            4*n3*n3 + 10*n3*m4*n4 +
            25*m4*n4*m4*n4)

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

            xa = -L/2. + L*float(section)/s
            xb = -L/2. + L*float(section+1)/s

            r = r1 - sina*((xa+xb)/2. + L/2.)

            # k0_00
            c += 1
            k0r[c] = 0
            k0c[c] = 0
            k0v[c] += -0.0833333333333333*(tmax - tmin)*(xa - xb)*(12*A11*(r*r) + sina*(12*A12*r*(L + xa + xb) + A22*sina*(3*(L*L) + 6*L*(xa + xb) + 4*(xa*xa) + 4*xa*xb + 4*(xb*xb))))/((L*L)*r)
            c += 1
            k0r[c] = 0
            k0c[c] = 1
            k0v[c] += -0.25*(xa - xb)*(4*A16*r + sina*(2*A26*(L + xa + xb) + (tmax - tmin)*(2*A12*r + A22*sina*(L + xa + xb))))/(L*r)
            c += 1
            k0r[c] = 0
            k0c[c] = 2
            k0v[c] += 0.0833333333333333*(tmax - tmin)*(xa - xb)*(6*A16*r*(-2*r + sina*(L + xa + xb)) + A26*sina*(3*(L*L)*sina + L*(-6*r + 6*sina*(xa + xb)) - 6*r*(xa + xb) + 4*sina*((xa*xa) + xa*xb + (xb*xb))))/((L*L)*r)
            c += 1
            k0r[c] = 0
            k0c[c] = 3
            k0v[c] += -0.25*(xa - xb)*(4*A12*r + sina*(2*A22*(L + xa + xb) - (tmax - tmin)*(2*A16*r + A26*sina*(L + xa + xb))))/(L*r)
            c += 1
            k0r[c] = 1
            k0c[c] = 0
            k0v[c] += -0.25*(xa - xb)*(4*A16*r + sina*(2*A26*(L + xa + xb) + (tmax - tmin)*(2*A12*r + A22*sina*(L + xa + xb))))/(L*r)
            c += 1
            k0r[c] = 1
            k0c[c] = 1
            k0v[c] += -0.333333333333333*(3*A66 + sina*(tmax - tmin)*(A22*sina*(tmax - tmin) + 3*A26))*(xa - xb)/(r*(tmax - tmin))
            c += 1
            k0r[c] = 1
            k0c[c] = 2
            k0v[c] += 0.25*(-2*r + sina*(L + xa + xb))*(xa - xb)*(A26*sina*(tmax - tmin) + 2*A66)/(L*r)
            c += 1
            k0r[c] = 1
            k0c[c] = 3
            k0v[c] += 0.166666666666667*(-xa + xb)*(3*A22*sina + 2*A26*((sina*sina)*(-tmax + tmin) + 3/(tmax - tmin)) - 3*A66*sina)/r
            c += 1
            k0r[c] = 2
            k0c[c] = 0
            k0v[c] += 0.0833333333333333*(tmax - tmin)*(xa - xb)*(6*A16*r*(-2*r + sina*(L + xa + xb)) + A26*sina*(3*(L*L)*sina + L*(-6*r + 6*sina*(xa + xb)) - 6*r*(xa + xb) + 4*sina*((xa*xa) + xa*xb + (xb*xb))))/((L*L)*r)
            c += 1
            k0r[c] = 2
            k0c[c] = 1
            k0v[c] += 0.25*(-2*r + sina*(L + xa + xb))*(xa - xb)*(A26*sina*(tmax - tmin) + 2*A66)/(L*r)
            c += 1
            k0r[c] = 2
            k0c[c] = 2
            k0v[c] += -0.0833333333333333*(tmax - tmin)*(xa - xb)*(A44*(cosa*cosa)*(3*(L*L) + 6*L*(xa + xb) + 4*(xa*xa) + 4*xa*xb + 4*(xb*xb)) + A66*(12*(r*r) - 12*r*sina*(L + xa + xb) + (sina*sina)*(3*(L*L) + 6*L*(xa + xb) + 4*(xa*xa) + 4*xa*xb + 4*(xb*xb))))/((L*L)*r)
            c += 1
            k0r[c] = 2
            k0c[c] = 3
            k0v[c] += 0.25*(xa - xb)*(2*A26*(-2*r + sina*(L + xa + xb)) - (tmax - tmin)*(A44*(cosa*cosa)*(L + xa + xb) + A66*sina*(-2*r + sina*(L + xa + xb))))/(L*r)
            c += 1
            k0r[c] = 3
            k0c[c] = 0
            k0v[c] += -0.25*(xa - xb)*(4*A12*r + sina*(2*A22*(L + xa + xb) - (tmax - tmin)*(2*A16*r + A26*sina*(L + xa + xb))))/(L*r)
            c += 1
            k0r[c] = 3
            k0c[c] = 1
            k0v[c] += 0.166666666666667*(-xa + xb)*(6*A26/(tmax - tmin) - 3*A66*sina + sina*(3*A22 + 2*A26*sina*(-tmax + tmin)))/r
            c += 1
            k0r[c] = 3
            k0c[c] = 2
            k0v[c] += 0.25*(xa - xb)*(2*A26*(-2*r + sina*(L + xa + xb)) - (tmax - tmin)*(A44*(cosa*cosa)*(L + xa + xb) + A66*sina*(-2*r + sina*(L + xa + xb))))/(L*r)
            c += 1
            k0r[c] = 3
            k0c[c] = 3
            k0v[c] += -0.333333333333333*(3*A22 + (tmax - tmin)*(-3*A26*sina + (tmax - tmin)*(A44*(cosa*cosa) + A66*(sina*sina))))*(xa - xb)/(r*(tmax - tmin))

            # k0_01
            col = num0
            c += 1
            k0r[c] = 0
            k0c[c] = col+0
            k0v[c] += -0.5*sina*(tmax - tmin)*(xa - xb)*(2*B12*r + B22*sina*(L + xa + xb))/(L*r)
            c += 1
            k0r[c] = 0
            k0c[c] = col+1
            k0v[c] += 0.5*sina*(tmax - tmin)*(xa - xb)*(2*B16*r + B26*sina*(L + xa + xb))/(L*r)
            c += 1
            k0r[c] = 1
            k0c[c] = col+0
            k0v[c] += -0.5*sina*(xa - xb)*(B22*sina*(tmax - tmin) + 2*B26)/r
            c += 1
            k0r[c] = 1
            k0c[c] = col+1
            k0v[c] += 0.5*sina*(xa - xb)*(B26*sina*(tmax - tmin) + 2*B66)/r
            c += 1
            k0r[c] = 2
            k0c[c] = col+0
            k0v[c] += 0.5*(tmax - tmin)*(xa - xb)*(A45*cosa*r*(L + xa + xb) + B26*sina*(-2*r + sina*(L + xa + xb)))/(L*r)
            c += 1
            k0r[c] = 2
            k0c[c] = col+1
            k0v[c] += 0.5*(tmax - tmin)*(xa - xb)*(A44*cosa*r*(L + xa + xb) - B66*sina*(-2*r + sina*(L + xa + xb)))/(L*r)
            c += 1
            k0r[c] = 3
            k0c[c] = col+0
            k0v[c] += 0.5*(xa - xb)*(-2*B22*sina + (tmax - tmin)*(A45*cosa*r + B26*(sina*sina)))/r
            c += 1
            k0r[c] = 3
            k0c[c] = col+1
            k0v[c] += 0.5*(xa - xb)*(2*B26*sina + (tmax - tmin)*(A44*cosa*r - B66*(sina*sina)))/r

            # k0_02
            for k2 in range(1, m2+1):
                col = num0 + num1 + num2*(k2-1)
                c += 1
                k0r[c] = 0
                k0c[c] = col+0
                k0v[c] += 0.5*(tmax - tmin)*(pi*B22*L*k2*(sina*sina)*(-(L + 2*xa)*sin(0.5*pi*k2*(L + 2*xa)/L) + (L + 2*xb)*sin(0.5*pi*k2*(L + 2*xb)/L)) - (2*(pi*pi)*B11*(k2*k2)*(r*r) + sina*((pi*pi)*B12*(k2*k2)*r*(L + 2*xa) + 2*B22*(L*L)*sina))*cos(0.5*pi*k2*(L + 2*xa)/L) + (2*(pi*pi)*B11*(k2*k2)*(r*r) + sina*((pi*pi)*B12*(k2*k2)*r*(L + 2*xb) + 2*B22*(L*L)*sina))*cos(0.5*pi*k2*(L + 2*xb)/L))/((pi*pi)*L*(k2*k2)*r)
                c += 1
                k0r[c] = 0
                k0c[c] = col+1
                k0v[c] += -0.5*(tmax - tmin)*(pi*L*k2*sina*(-(2*B16*r + B26*(2*r + sina*(L + 2*xa)))*sin(0.5*pi*k2*(L + 2*xa)/L) + (2*B16*r + B26*(2*r + sina*(L + 2*xb)))*sin(0.5*pi*k2*(L + 2*xb)/L)) + (2*(pi*pi)*B16*(k2*k2)*(r*r) + B26*sina*(-2*(L*L)*sina + (pi*pi)*(k2*k2)*r*(L + 2*xa)))*cos(0.5*pi*k2*(L + 2*xa)/L) - (2*(pi*pi)*B16*(k2*k2)*(r*r) + B26*sina*(-2*(L*L)*sina + (pi*pi)*(k2*k2)*r*(L + 2*xb)))*cos(0.5*pi*k2*(L + 2*xb)/L))/((pi*pi)*L*(k2*k2)*r)
                c += 1
                k0r[c] = 1
                k0c[c] = col+0
                k0v[c] += 0.5*(-L*sina*(B22*sina*(tmax - tmin) + 2*B26)*(sin(0.5*pi*k2*(L + 2*xa)/L) - sin(0.5*pi*k2*(L + 2*xb)/L))/(pi*k2) - r*(B12*sina*(tmax - tmin) + 2*B16)*(cos(0.5*pi*k2*(L + 2*xa)/L) - cos(0.5*pi*k2*(L + 2*xb)/L)))/r
                c += 1
                k0r[c] = 1
                k0c[c] = col+1
                k0v[c] += 0.5*(B26*sina*(tmax - tmin) + 2*B66)*(L*sina*(sin(0.5*pi*k2*(L + 2*xa)/L) - sin(0.5*pi*k2*(L + 2*xb)/L)) - pi*k2*r*cos(0.5*pi*k2*(L + 2*xa)/L) + pi*k2*r*cos(0.5*pi*k2*(L + 2*xb)/L))/(pi*k2*r)
                c += 1
                k0r[c] = 2
                k0c[c] = col+0
                k0v[c] += 0.5*(tmax - tmin)*(pi*B16*k2*r*(2*L*sina*(-sin(0.5*pi*k2*(L + 2*xa)/L) + sin(0.5*pi*k2*(L + 2*xb)/L)) - pi*k2*(2*r - sina*(L + 2*xa))*cos(0.5*pi*k2*(L + 2*xa)/L) + pi*k2*(2*r - sina*(L + 2*xb))*cos(0.5*pi*k2*(L + 2*xb)/L)) - L*(-2*L*(A45*cosa*r + B26*(sina*sina))*cos(0.5*pi*k2*(L + 2*xa)/L) + 2*L*(A45*cosa*r + B26*(sina*sina))*cos(0.5*pi*k2*(L + 2*xb)/L) + pi*k2*(-(A45*cosa*r*(L + 2*xa) + B26*sina*(-2*r + sina*(L + 2*xa)))*sin(0.5*pi*k2*(L + 2*xa)/L) + (A45*cosa*r*(L + 2*xb) + B26*sina*(-2*r + sina*(L + 2*xb)))*sin(0.5*pi*k2*(L + 2*xb)/L))))/((pi*pi)*L*(k2*k2)*r)
                c += 1
                k0r[c] = 2
                k0c[c] = col+1
                k0v[c] += 0.5*(tmax - tmin)*(pi*L*k2*((L + 2*xa)*sin(0.5*pi*k2*(L + 2*xa)/L) - (L + 2*xb)*sin(0.5*pi*k2*(L + 2*xb)/L))*(A44*cosa*r - B66*(sina*sina)) + (2*A44*(L*L)*cosa*r + B66*(-2*(L*L)*(sina*sina) + (pi*pi)*(k2*k2)*r*(L*sina - 2*r + 2*sina*xa)))*cos(0.5*pi*k2*(L + 2*xa)/L) - (2*A44*(L*L)*cosa*r + B66*(-2*(L*L)*(sina*sina) + (pi*pi)*(k2*k2)*r*(L*sina - 2*r + 2*sina*xb)))*cos(0.5*pi*k2*(L + 2*xb)/L))/((pi*pi)*L*(k2*k2)*r)
                c += 1
                k0r[c] = 3
                k0c[c] = col+0
                k0v[c] += 0.5*(L*(-2*B22*sina + (tmax - tmin)*(A45*cosa*r + B26*(sina*sina)))*(sin(0.5*pi*k2*(L + 2*xa)/L) - sin(0.5*pi*k2*(L + 2*xb)/L))/(pi*k2) + r*(-2*B12 + B16*sina*(tmax - tmin))*(cos(0.5*pi*k2*(L + 2*xa)/L) - cos(0.5*pi*k2*(L + 2*xb)/L)))/r
                c += 1
                k0r[c] = 3
                k0c[c] = col+1
                k0v[c] += (L*(B26*sina*(tmax - tmin) + 0.5*(tmax - tmin)**2*(A44*cosa*r - B66*(sina*sina)))*(-sin(0.5*pi*k2*(L + 2*xa)/L) + sin(0.5*pi*k2*(L + 2*xb)/L))/(pi*k2) + 0.5*r*(2*B26 + B66*sina*(-tmax + tmin))*(tmax - tmin)*(cos(0.5*pi*k2*(L + 2*xa)/L) - cos(0.5*pi*k2*(L + 2*xb)/L)))/(r*(-tmax + tmin))

            # k0_03
            for l3 in range(1, n3+1):
                col = num0 + num1 + num2*m2 + num3*(l3-1)
                c += 1
                k0r[c] = 0
                k0c[c] = col+0
                k0v[c] += -0.5*((-1)**l3 - 1)*(xa - xb)*(2*B16*r + B26*sina*(L + xa + xb))/(L*r)
                c += 1
                k0r[c] = 0
                k0c[c] = col+1
                k0v[c] += -0.5*((-1)**l3 - 1)*(xa - xb)*(2*B12*r + B22*sina*(L + xa + xb))/(L*r)
                c += 1
                k0r[c] = 1
                k0c[c] = col+0
                k0v[c] += -(xa - xb)*((pi*pi)*B66*(l3*l3)*((-1)**l3 - 1) + sina*(tmax - tmin)*((-1)**l3*(pi*pi)*B26*(l3*l3) + B22*sina*((-1)**l3 - 1)*(tmax - tmin)))/((pi*pi)*(l3*l3)*r*(tmax - tmin))
                c += 1
                k0r[c] = 1
                k0c[c] = col+1
                k0v[c] += -(xa - xb)*((-1)**l3*(pi*pi)*B22*(l3*l3)*sina*(tmax - tmin) + B26*((-1)**l3 - 1)*((pi*pi)*(l3*l3) - (sina*sina)*(tmax - tmin)**2))/((pi*pi)*(l3*l3)*r*(tmax - tmin))
                c += 1
                k0r[c] = 2
                k0c[c] = col+0
                k0v[c] += -(-1)**(l3 - 1)*B66*(-2*r + sina*(L + xa + xb))*(xa - xb)/(L*r)
                c += 1
                k0r[c] = 2
                k0c[c] = col+1
                k0v[c] += -(-1)**(l3 - 1)*B26*(-2*r + sina*(L + xa + xb))*(xa - xb)/(L*r)
                c += 1
                k0r[c] = 3
                k0c[c] = col+0
                k0v[c] += -(xa - xb)*(-(pi*pi)*(l3*l3)*(-(-1)**l3*(B26 + B66*sina*(-tmax + tmin)) + B26) - ((-1)**l3 - 1)*(tmax - tmin)**2*(A45*cosa*r + B26*(sina*sina)))/((pi*pi)*(l3*l3)*r*(tmax - tmin))
                c += 1
                k0r[c] = 3
                k0c[c] = col+1
                k0v[c] += (xa - xb)*((pi*pi)*(l3*l3)*(-(-1)**l3*(B22 + B26*sina*(-tmax + tmin)) + B22) + ((-1)**l3 - 1)*(tmax - tmin)**2*(A44*cosa*r - B66*(sina*sina)))/((pi*pi)*(l3*l3)*r*(tmax - tmin))

            # k0_04
            for l4 in range(1, n4+1):
                for k4 in range(1, m4+1):
                    col = (num0 + num1 + num2*m2 + num3*n3 +
                            num4*((l4-1)*m4 + (k4-1)))
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+0
                    k0v[c] += 0.5*((-1)**l4 - 1)*(tmax - tmin)*(-sina*(2*A22*L*sina*(-sin(0.5*pi*k4*(L + 2*xa)/L) + sin(0.5*pi*k4*(L + 2*xb)/L)) + pi*k4*(2*A12*r + A22*sina*(L + 2*xa))*cos(0.5*pi*k4*(L + 2*xa)/L) - pi*k4*(2*A12*r + A22*sina*(L + 2*xb))*cos(0.5*pi*k4*(L + 2*xb)/L)) + pi*k4*r*(2*A12*L*sina*cos(0.5*pi*k4*(L + 2*xa)/L) - 2*A12*L*sina*cos(0.5*pi*k4*(L + 2*xb)/L) + pi*k4*((2*A11*r + A12*sina*(L + 2*xa))*sin(0.5*pi*k4*(L + 2*xa)/L) - (2*A11*r + A12*sina*(L + 2*xb))*sin(0.5*pi*k4*(L + 2*xb)/L)))/L)/((pi*pi*pi)*(k4*k4)*l4*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+1
                    k0v[c] += 0.5*((-1)**l4 - 1)*(tmax - tmin)*(2*(pi*pi)*A16*(k4*k4)*(r*r)*sin(0.5*pi*k4*(L + 2*xa)/L) - 2*(pi*pi)*A16*(k4*k4)*(r*r)*sin(0.5*pi*k4*(L + 2*xb)/L) - 2*A26*(L*L)*(sina*sina)*sin(0.5*pi*k4*(L + 2*xa)/L) + 2*A26*(L*L)*(sina*sina)*sin(0.5*pi*k4*(L + 2*xb)/L) + (pi*pi)*A26*L*(k4*k4)*r*sina*sin(0.5*pi*k4*(L + 2*xa)/L) - (pi*pi)*A26*L*(k4*k4)*r*sina*sin(0.5*pi*k4*(L + 2*xb)/L) + 2*(pi*pi)*A26*(k4*k4)*r*sina*xa*sin(0.5*pi*k4*(L + 2*xa)/L) - 2*(pi*pi)*A26*(k4*k4)*r*sina*xb*sin(0.5*pi*k4*(L + 2*xb)/L) + pi*L*k4*sina*(2*A16*r + A26*(L*sina + 2*r + 2*sina*xa))*cos(0.5*pi*k4*(L + 2*xa)/L) - pi*L*k4*sina*(2*A16*r + A26*L*sina + 2*A26*r + 2*A26*sina*xb)*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi*pi)*L*(k4*k4)*l4*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+2
                    k0v[c] += -0.5*cosa*((-1)**l4 - 1)*(tmax - tmin)*(2*A22*L*sina*(-sin(0.5*pi*k4*(L + 2*xa)/L) + sin(0.5*pi*k4*(L + 2*xb)/L)) + pi*k4*(2*A12*r + A22*sina*(L + 2*xa))*cos(0.5*pi*k4*(L + 2*xa)/L) - pi*k4*(2*A12*r + A22*sina*(L + 2*xb))*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi*pi)*(k4*k4)*l4*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+3
                    k0v[c] += -0.5*((-1)**l4 - 1)*(2*B26*L*sina*cos(0.5*pi*k4*(L + 2*xa)/L) - 2*B26*L*sina*cos(0.5*pi*k4*(L + 2*xb)/L) + pi*k4*((2*B16*r + B26*sina*(L + 2*xa))*sin(0.5*pi*k4*(L + 2*xa)/L) - (2*B16*r + B26*sina*(L + 2*xb))*sin(0.5*pi*k4*(L + 2*xb)/L)))/((pi*pi)*(k4*k4)*r)
                    c += 1
                    k0r[c] = 0
                    k0c[c] = col+4
                    k0v[c] += -0.5*((-1)**l4 - 1)*(2*B22*L*sina*cos(0.5*pi*k4*(L + 2*xa)/L) - 2*B22*L*sina*cos(0.5*pi*k4*(L + 2*xb)/L) + pi*k4*((2*B12*r + B22*sina*(L + 2*xa))*sin(0.5*pi*k4*(L + 2*xa)/L) - (2*B12*r + B22*sina*(L + 2*xb))*sin(0.5*pi*k4*(L + 2*xb)/L)))/((pi*pi)*(k4*k4)*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+0
                    k0v[c] += 2*((-1)**l4*A22*L*(sina*sina)*(tmax - tmin)*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) - pi*k4*r*((-1)**l4*A12*sina*(tmax - tmin) + A16*((-1)**l4 - 1))*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+1
                    k0v[c] += (A22*L*sina*((-1)**l4 - 1)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L)) + (-2*(-1)**l4*(A26*sina*(tmax - tmin) + A66) + 2*A66)*(L*sina*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + pi*k4*r*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+2
                    k0v[c] += L*cosa*(-(-1)**l4*(A22*sina*(tmax - tmin) + A26) + A26)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+3
                    k0v[c] += (-pi*B12*r*sina*((-1)**l4 - 1)*(tmax - tmin)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L)) - B22*L*(sina*sina)*((-1)**l4 - 1)*(tmax - tmin)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/k4 + (pi*pi)*L*(l4*l4)*(-(-1)**l4*(B26*sina*(tmax - tmin) + B66) + B66)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/(k4*(tmax - tmin)))/((pi*pi*pi)*(l4*l4)*r)
                    c += 1
                    k0r[c] = 1
                    k0c[c] = col+4
                    k0v[c] += (B26*sina*(2*(-1)**l4 - 2)*(tmax - tmin)**2*(-L*sina*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + pi*k4*r*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L) + (pi*pi)*L*(l4*l4)*(-(-1)**l4*(B22*sina*(tmax - tmin) + B26) + B26)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)))/((pi*pi*pi)*k4*(l4*l4)*r*(tmax - tmin))
                    c += 1
                    k0r[c] = 2
                    k0c[c] = col+0
                    k0v[c] += 0.5*((-1)**l4 - 1)*(tmax - tmin)*(-(pi*pi)*A16*L*(k4*k4)*r*sina*sin(0.5*pi*k4*(L + 2*xa)/L) + (pi*pi)*A16*L*(k4*k4)*r*sina*sin(0.5*pi*k4*(L + 2*xb)/L) + 2*(pi*pi)*A16*(k4*k4)*(r*r)*sin(0.5*pi*k4*(L + 2*xa)/L) - 2*(pi*pi)*A16*(k4*k4)*(r*r)*sin(0.5*pi*k4*(L + 2*xb)/L) - 2*(pi*pi)*A16*(k4*k4)*r*sina*xa*sin(0.5*pi*k4*(L + 2*xa)/L) + 2*(pi*pi)*A16*(k4*k4)*r*sina*xb*sin(0.5*pi*k4*(L + 2*xb)/L) - 2*A26*(L*L)*(sina*sina)*sin(0.5*pi*k4*(L + 2*xa)/L) + 2*A26*(L*L)*(sina*sina)*sin(0.5*pi*k4*(L + 2*xb)/L) + pi*L*k4*sina*(-2*A16*r + A26*(-2*r + sina*(L + 2*xa)))*cos(0.5*pi*k4*(L + 2*xa)/L) + pi*L*k4*sina*(2*A16*r + A26*(2*r - sina*(L + 2*xb)))*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi*pi)*L*(k4*k4)*l4*r)
                    c += 1
                    k0r[c] = 2
                    k0c[c] = col+1
                    k0v[c] += 0.5*((-1)**l4 - 1)*(tmax - tmin)*(2*A44*(L*L)*(cosa*cosa)*sin(0.5*pi*k4*(L + 2*xa)/L) - 2*A44*(L*L)*(cosa*cosa)*sin(0.5*pi*k4*(L + 2*xb)/L) + 2*A66*(L*L)*(sina*sina)*sin(0.5*pi*k4*(L + 2*xa)/L) - 2*A66*(L*L)*(sina*sina)*sin(0.5*pi*k4*(L + 2*xb)/L) - (pi*pi)*A66*L*(k4*k4)*r*sina*sin(0.5*pi*k4*(L + 2*xa)/L) + (pi*pi)*A66*L*(k4*k4)*r*sina*sin(0.5*pi*k4*(L + 2*xb)/L) + 2*(pi*pi)*A66*(k4*k4)*(r*r)*sin(0.5*pi*k4*(L + 2*xa)/L) - 2*(pi*pi)*A66*(k4*k4)*(r*r)*sin(0.5*pi*k4*(L + 2*xb)/L) - 2*(pi*pi)*A66*(k4*k4)*r*sina*xa*sin(0.5*pi*k4*(L + 2*xa)/L) + 2*(pi*pi)*A66*(k4*k4)*r*sina*xb*sin(0.5*pi*k4*(L + 2*xb)/L) - pi*L*k4*(L + 2*xa)*(A44*(cosa*cosa) + A66*(sina*sina))*cos(0.5*pi*k4*(L + 2*xa)/L) + pi*L*k4*(L + 2*xb)*(A44*(cosa*cosa) + A66*(sina*sina))*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi*pi)*L*(k4*k4)*l4*r)
                    c += 1
                    k0r[c] = 2
                    k0c[c] = col+2
                    k0v[c] += 0.5*cosa*((-1)**l4 - 1)*(tmax - tmin)*(-A26*(2*L*sina*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) + pi*k4*(2*r - sina*(L + 2*xa))*cos(0.5*pi*k4*(L + 2*xa)/L) - pi*k4*(2*r - sina*(L + 2*xb))*cos(0.5*pi*k4*(L + 2*xb)/L)) - pi*A45*k4*r*(2*L*cos(0.5*pi*k4*(L + 2*xa)/L) - 2*L*cos(0.5*pi*k4*(L + 2*xb)/L) + pi*k4*((L + 2*xa)*sin(0.5*pi*k4*(L + 2*xa)/L) - (L + 2*xb)*sin(0.5*pi*k4*(L + 2*xb)/L)))/L)/((pi*pi*pi)*(k4*k4)*l4*r)
                    c += 1
                    k0r[c] = 2
                    k0c[c] = col+3
                    k0v[c] += 0.5*B66*((-1)**l4 - 1)*(2*L*sina*cos(0.5*pi*k4*(L + 2*xa)/L) - 2*L*sina*cos(0.5*pi*k4*(L + 2*xb)/L) + pi*k4*((-2*r + sina*(L + 2*xa))*sin(0.5*pi*k4*(L + 2*xa)/L) + (2*r - sina*(L + 2*xb))*sin(0.5*pi*k4*(L + 2*xb)/L)))/((pi*pi)*(k4*k4)*r)
                    c += 1
                    k0r[c] = 2
                    k0c[c] = col+4
                    k0v[c] += 0.5*B26*((-1)**l4 - 1)*(2*L*sina*cos(0.5*pi*k4*(L + 2*xa)/L) - 2*L*sina*cos(0.5*pi*k4*(L + 2*xb)/L) + pi*k4*((-2*r + sina*(L + 2*xa))*sin(0.5*pi*k4*(L + 2*xa)/L) + (2*r - sina*(L + 2*xb))*sin(0.5*pi*k4*(L + 2*xb)/L)))/((pi*pi)*(k4*k4)*r)
                    c += 1
                    k0r[c] = 3
                    k0c[c] = col+0
                    k0v[c] += (-A66*L*sina*((-1)**l4 - 1)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L))/k4 + L*sina*(-(-1)**l4*(A22 + A26*sina*(-tmax + tmin)) + A22)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L))/k4 + pi*r*((-1)**l4*(A12 + A16*sina*(-tmax + tmin)) - A12)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)))/((pi*pi)*l4*r)
                    c += 1
                    k0r[c] = 3
                    k0c[c] = col+1
                    k0v[c] += -((-1)**l4*A44*L*(cosa*cosa)*(tmax - tmin)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L)) + A26*L*sina*((-1)**l4 - 1)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L)) - (-2*(-1)**l4*(A26 + A66*sina*(-tmax + tmin)) + 2*A26)*(L*sina*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + pi*k4*r*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = 3
                    k0c[c] = col+2
                    k0v[c] += 2*cosa*((-1)**l4*pi*A45*k4*r*(tmax - tmin)*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + L*(-(-1)**l4*A26*sina*(tmax - tmin) + A22*((-1)**l4 - 1) + A44*((-1)**l4 - 1))*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = 3
                    k0c[c] = col+3
                    k0v[c] += (A45*L*cosa*r*((-1)**l4 - 1)*(tmax - tmin)**2*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/k4 + pi*B16*r*sina*((-1)**l4 - 1)*(tmax - tmin)**2*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L)) + B26*L*(sina*sina)*((-1)**l4 - 1)*(tmax - tmin)**2*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/k4 + (pi*pi)*L*(l4*l4)*(-(-1)**l4*(B26 + B66*sina*(-tmax + tmin)) + B26)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/k4)/((pi*pi*pi)*(l4*l4)*r*(tmax - tmin))
                    c += 1
                    k0r[c] = 3
                    k0c[c] = col+4
                    k0v[c] += (A44*L*cosa*r*((-1)**l4 - 1)*(tmax - tmin)**2*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) + B66*sina*(2*(-1)**l4 - 2)*(tmax - tmin)**2*(L*sina*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) - pi*k4*r*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L) + (pi*pi)*L*(l4*l4)*(-(-1)**l4*(B22 + B26*sina*(-tmax + tmin)) + B22)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)))/((pi*pi*pi)*k4*(l4*l4)*r*(tmax - tmin))

            # k0_11
            row = num0
            col = num0
            c += 1
            k0r[c] = row+0
            k0c[c] = col+0
            k0v[c] += r*(A55 + D22*(sina*sina)/(r*r))*(tmax - tmin)*(-xa + xb)
            c += 1
            k0r[c] = row+0
            k0c[c] = col+1
            k0v[c] += r*(A45 - D26*(sina*sina)/(r*r))*(tmax - tmin)*(-xa + xb)
            c += 1
            k0r[c] = row+1
            k0c[c] = col+0
            k0v[c] += r*(A45 - D26*(sina*sina)/(r*r))*(tmax - tmin)*(-xa + xb)
            c += 1
            k0r[c] = row+1
            k0c[c] = col+1
            k0v[c] += r*(A44 + D66*(sina*sina)/(r*r))*(tmax - tmin)*(-xa + xb)

            # k0_12
            row = num0
            for k2 in range(1, m2+1):
                col = num0 + num1 + num2*(k2-1)
                c += 1
                k0r[c] = row+0
                k0c[c] = col+0
                k0v[c] += (tmax - tmin)*(-pi*D12*k2*r*sina*cos(0.5*pi*k2*(L + 2*xa)/L) + pi*D12*k2*r*sina*cos(0.5*pi*k2*(L + 2*xb)/L) - L*(A55*(r*r) + D22*(sina*sina))*(sin(0.5*pi*k2*(L + 2*xa)/L) - sin(0.5*pi*k2*(L + 2*xb)/L)))/(pi*k2*r)
                c += 1
                k0r[c] = row+0
                k0c[c] = col+1
                k0v[c] += (tmax - tmin)*(-pi*D26*k2*r*sina*cos(0.5*pi*k2*(L + 2*xa)/L) + pi*D26*k2*r*sina*cos(0.5*pi*k2*(L + 2*xb)/L) - L*(A45*(r*r) - D26*(sina*sina))*(sin(0.5*pi*k2*(L + 2*xa)/L) - sin(0.5*pi*k2*(L + 2*xb)/L)))/(pi*k2*r)
                c += 1
                k0r[c] = row+1
                k0c[c] = col+0
                k0v[c] += (tmax - tmin)*(pi*D16*k2*r*sina*cos(0.5*pi*k2*(L + 2*xa)/L) - pi*D16*k2*r*sina*cos(0.5*pi*k2*(L + 2*xb)/L) - L*(A45*(r*r) - D26*(sina*sina))*(sin(0.5*pi*k2*(L + 2*xa)/L) - sin(0.5*pi*k2*(L + 2*xb)/L)))/(pi*k2*r)
                c += 1
                k0r[c] = row+1
                k0c[c] = col+1
                k0v[c] += (tmax - tmin)*(pi*D66*k2*r*sina*cos(0.5*pi*k2*(L + 2*xa)/L) - pi*D66*k2*r*sina*cos(0.5*pi*k2*(L + 2*xb)/L) - L*(A44*(r*r) + D66*(sina*sina))*(sin(0.5*pi*k2*(L + 2*xa)/L) - sin(0.5*pi*k2*(L + 2*xb)/L)))/(pi*k2*r)

            # k0_13
            row = num0
            for l3 in range(1, n3+1):
                col = num0 + num1 + num2*m2 + num3*(l3-1)
                c += 1
                k0r[c] = row+0
                k0c[c] = col+0
                k0v[c] += -D26*sina*((-1)**l3 - 1)*(xa - xb)/r
                c += 1
                k0r[c] = row+0
                k0c[c] = col+1
                k0v[c] += -D22*sina*((-1)**l3 - 1)*(xa - xb)/r
                c += 1
                k0r[c] = row+1
                k0c[c] = col+0
                k0v[c] += D66*sina*((-1)**l3 - 1)*(xa - xb)/r
                c += 1
                k0r[c] = row+1
                k0c[c] = col+1
                k0v[c] += D26*sina*((-1)**l3 - 1)*(xa - xb)/r

            # k0_14
            row = num0
            for k4 in range(1, m4+1):
                for l4 in range(1, n4+1):
                    col = (num0 + num1 + num2*m2 + num3*n3 +
                            num4*((l4-1)*m4 + (k4-1)))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += sina*((-1)**l4 - 1)*(tmax - tmin)*(pi*B12*k4*r*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) - B22*L*sina*cos(0.5*pi*k4*(L + 2*xa)/L) + B22*L*sina*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+1
                    k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(pi*B26*k4*r*sina*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) + L*(A45*cosa*r + B26*(sina*sina))*cos(0.5*pi*k4*(L + 2*xa)/L) - L*(A45*cosa*r + B26*(sina*sina))*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+2
                    k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(pi*A55*k4*(r*r)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) - B22*L*cosa*sina*cos(0.5*pi*k4*(L + 2*xa)/L) + B22*L*cosa*sina*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+3
                    k0v[c] += -D26*L*sina*((-1)**l4 - 1)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/(pi*k4*r)
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+4
                    k0v[c] += -D22*L*sina*((-1)**l4 - 1)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/(pi*k4*r)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += sina*((-1)**l4 - 1)*(tmax - tmin)*(pi*B16*k4*r*(-sin(0.5*pi*k4*(L + 2*xa)/L) + sin(0.5*pi*k4*(L + 2*xb)/L)) + B26*L*sina*cos(0.5*pi*k4*(L + 2*xa)/L) - B26*L*sina*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(pi*B66*k4*r*sina*(-sin(0.5*pi*k4*(L + 2*xa)/L) + sin(0.5*pi*k4*(L + 2*xb)/L)) + L*(A44*cosa*r - B66*(sina*sina))*cos(0.5*pi*k4*(L + 2*xa)/L) + (-A44*L*cosa*r + B66*L*(sina*sina))*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+2
                    k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(pi*A45*k4*(r*r)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) + B26*L*cosa*sina*cos(0.5*pi*k4*(L + 2*xa)/L) - B26*L*cosa*sina*cos(0.5*pi*k4*(L + 2*xb)/L))/((pi*pi)*k4*l4*r)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+3
                    k0v[c] += D66*L*sina*((-1)**l4 - 1)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/(pi*k4*r)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+4
                    k0v[c] += D26*L*sina*((-1)**l4 - 1)*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))/(pi*k4*r)

            # k0_22
            for i2 in range(1, m2+1):
                row = num0 + num1 + num2*(i2-1)
                for k2 in range(1, m2+1):
                    col = num0 + num1 + num2*(k2-1)
                    if k2 == i2:
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += 0.25*(tmax - tmin)*(A55*L*r*(-sin(pi*i2*(L + 2*xa)/L) + sin(pi*i2*(L + 2*xb)/L))/(pi*i2) - 2*A55*r*xa + 2*A55*r*xb + pi*D11*i2*r*(sin(pi*i2*(L + 2*xa)/L) - sin(pi*i2*(L + 2*xb)/L))/L + 2*(pi*pi)*D11*(i2*i2)*r*(-xa + xb)/(L*L) - 2*D12*sina*(cos(pi*i2*(L + 2*xa)/L) - cos(pi*i2*(L + 2*xb)/L)) + D22*L*(sina*sina)*(-sin(pi*i2*(L + 2*xa)/L) + sin(pi*i2*(L + 2*xb)/L))/(pi*i2*r) - 2*D22*(sina*sina)*xa/r + 2*D22*(sina*sina)*xb/r)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += 0.25*(tmax - tmin)*(A45*L*r*(-sin(pi*i2*(L + 2*xa)/L) + sin(pi*i2*(L + 2*xb)/L))/(pi*i2) - 2*A45*r*xa + 2*A45*r*xb + D16*sina*(cos(pi*i2*(L + 2*xa)/L) - cos(pi*i2*(L + 2*xb)/L)) + pi*D16*i2*r*(sin(pi*i2*(L + 2*xa)/L) - sin(pi*i2*(L + 2*xb)/L))/L + 2*(pi*pi)*D16*(i2*i2)*r*(-xa + xb)/(L*L) + D26*L*(sina*sina)*(sin(pi*i2*(L + 2*xa)/L) - sin(pi*i2*(L + 2*xb)/L))/(pi*i2*r) - D26*sina*(cos(pi*i2*(L + 2*xa)/L) - cos(pi*i2*(L + 2*xb)/L)) + 2*D26*(sina*sina)*xa/r - 2*D26*(sina*sina)*xb/r)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += 0.25*(tmax - tmin)*(A45*L*r*(-sin(pi*i2*(L + 2*xa)/L) + sin(pi*i2*(L + 2*xb)/L))/(pi*i2) - 2*A45*r*xa + 2*A45*r*xb + D16*sina*(cos(pi*i2*(L + 2*xa)/L) - cos(pi*i2*(L + 2*xb)/L)) + pi*D16*i2*r*(sin(pi*i2*(L + 2*xa)/L) - sin(pi*i2*(L + 2*xb)/L))/L + 2*(pi*pi)*D16*(i2*i2)*r*(-xa + xb)/(L*L) + D26*L*(sina*sina)*(sin(pi*i2*(L + 2*xa)/L) - sin(pi*i2*(L + 2*xb)/L))/(pi*i2*r) - D26*sina*(cos(pi*i2*(L + 2*xa)/L) - cos(pi*i2*(L + 2*xb)/L)) + 2*D26*(sina*sina)*xa/r - 2*D26*(sina*sina)*xb/r)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += 0.25*(tmax - tmin)*(A44*r*(-L*sin(pi*i2*(L + 2*xa)/L) + L*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(-xa + xb))/(pi*i2) + 2*D66*sina*(cos(pi*i2*(L + 2*xa)/L) - cos(pi*i2*(L + 2*xb)/L)) + D66*(sina*sina)*(-L*sin(pi*i2*(L + 2*xa)/L) + L*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(-xa + xb))/(pi*i2*r) + pi*D66*i2*r*(L*sin(pi*i2*(L + 2*xa)/L) - L*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(-xa + xb))/(L*L))
                    else:
                        # k0_22 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += -(tmax - tmin)*(A55*L*r*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) + (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/pi - pi*D11*i2*k2*r*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) - (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/L + D12*i2*sina*(-i2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) - k2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + (i2 - k2)*cos(0.5*pi*(L + 2*xa)*(i2 + k2)/L) + (i2 + k2)*cos(0.5*pi*(L + 2*xa)*(i2 - k2)/L)) + D12*k2*sina*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + (i2 - k2)*cos(0.5*pi*(L + 2*xa)*(i2 + k2)/L) - (i2 + k2)*cos(0.5*pi*(L + 2*xa)*(i2 - k2)/L)) + D22*L*(sina*sina)*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) + (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/(pi*r))/((i2 + k2)*(2.0*i2 - 2.0*k2))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += (tmax - tmin)*(-A45*L*r*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) + (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/pi + D16*i2*sina*(-i2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) - k2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + (i2 - k2)*cos(0.5*pi*(L + 2*xa)*(i2 + k2)/L) + (i2 + k2)*cos(0.5*pi*(L + 2*xa)*(i2 - k2)/L)) + pi*D16*i2*k2*r*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) - (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/L + D26*L*(sina*sina)*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) + (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/(pi*r) - D26*k2*sina*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + (i2 - k2)*cos(0.5*pi*(L + 2*xa)*(i2 + k2)/L) - (i2 + k2)*cos(0.5*pi*(L + 2*xa)*(i2 - k2)/L)))/((i2 + k2)*(2.0*i2 - 2.0*k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += (tmax - tmin)*(-A45*L*r*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) + (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/pi + D16*k2*sina*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + (i2 - k2)*cos(0.5*pi*(L + 2*xa)*(i2 + k2)/L) - (i2 + k2)*cos(0.5*pi*(L + 2*xa)*(i2 - k2)/L)) + pi*D16*i2*k2*r*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) - (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/L + D26*L*(sina*sina)*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) + (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/(pi*r) - D26*i2*sina*(-i2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) - k2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + (i2 - k2)*cos(0.5*pi*(L + 2*xa)*(i2 + k2)/L) + (i2 + k2)*cos(0.5*pi*(L + 2*xa)*(i2 - k2)/L)))/((i2 + k2)*(2.0*i2 - 2.0*k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += (tmax - tmin)*(-A44*L*r*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) + (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/pi - D66*L*(sina*sina)*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) + (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/(pi*r) + D66*i2*sina*(-i2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) - k2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + (i2 - k2)*cos(0.5*pi*(L + 2*xa)*(i2 + k2)/L) + (i2 + k2)*cos(0.5*pi*(L + 2*xa)*(i2 - k2)/L)) + D66*k2*sina*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 - k2)/L) + k2*cos(0.5*pi*(L + 2*xb)*(i2 + k2)/L) + (i2 - k2)*cos(0.5*pi*(L + 2*xa)*(i2 + k2)/L) - (i2 + k2)*cos(0.5*pi*(L + 2*xa)*(i2 - k2)/L)) + pi*D66*i2*k2*r*((-i2 + k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xb*(2*i2 + 2*k2))/L) + (i2 - k2)*cos(0.5*pi*(L*(i2 + k2 - 1) + xa*(2*i2 + 2*k2))/L) - (i2 + k2)*(sin(0.5*pi*(L + 2*xa)*(i2 - k2)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k2)/L)))/L)/((i2 + k2)*(2.0*i2 - 2.0*k2))

            # k0_23
            for i2 in range(1, m2+1):
                row = num0 + num1 + num2*(i2-1)
                for l3 in range(1, n3+1):
                    col = num0 + num1 + num2*m2 + num3*(l3-1)
                    # k0_23
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += -((-1)**l3 - 1)*(pi*D16*i2*r*(cos(0.5*pi*i2*(L + 2*xa)/L) - cos(0.5*pi*i2*(L + 2*xb)/L)) + D26*L*sina*(sin(0.5*pi*i2*(L + 2*xa)/L) - sin(0.5*pi*i2*(L + 2*xb)/L)))/(pi*i2*r)
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+1
                    k0v[c] += -((-1)**l3 - 1)*(pi*D12*i2*r*(cos(0.5*pi*i2*(L + 2*xa)/L) - cos(0.5*pi*i2*(L + 2*xb)/L)) + D22*L*sina*(sin(0.5*pi*i2*(L + 2*xa)/L) - sin(0.5*pi*i2*(L + 2*xb)/L)))/(pi*i2*r)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += D66*((-1)**l3 - 1)*(L*sina*(sin(0.5*pi*i2*(L + 2*xa)/L) - sin(0.5*pi*i2*(L + 2*xb)/L)) - pi*i2*r*cos(0.5*pi*i2*(L + 2*xa)/L) + pi*i2*r*cos(0.5*pi*i2*(L + 2*xb)/L))/(pi*i2*r)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += D26*((-1)**l3 - 1)*(L*sina*(sin(0.5*pi*i2*(L + 2*xa)/L) - sin(0.5*pi*i2*(L + 2*xb)/L)) - pi*i2*r*cos(0.5*pi*i2*(L + 2*xa)/L) + pi*i2*r*cos(0.5*pi*i2*(L + 2*xb)/L))/(pi*i2*r)

            # k0_24
            for i2 in range(1, m2+1):
                row = num0 + num1 + num2*(i2-1)
                for k4 in range(1, m4+1):
                    for l4 in range(1, n4+1):
                        col = (num0 + num1 + num2*m2 + num3*n3 +
                                num4*((l4-1)*m4 + (k4-1)))
                        if k4 == i2:
                            # k0_24 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.25*(tmax - tmin)*(2*pi*B12*L*i2*r*sina*(2*(-1)**(l4 - 1)*sin(pi*i2*(L + 2*xb)/L) + ((-1)**l4 - 1)*sin(pi*i2*(L + 2*xa)/L)) + ((-1)**l4 - 1)*((pi*pi)*B11*(i2*i2)*(r*r) - B22*(L*L)*(sina*sina))*cos(pi*i2*(L + 2*xa)/L) - ((-1)**l4 - 1)*((pi*pi)*B11*(i2*i2)*(r*r) - B22*(L*L)*(sina*sina))*cos(pi*i2*(L + 2*xb)/L))/((pi*pi)*L*i2*l4*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += 0.25*((-1)**l4 - 1)*(tmax - tmin)*(pi*i2*r*sina*(-L*(B16 - B26)*sin(pi*i2*(L + 2*xa)/L) + L*(B16 - B26)*sin(pi*i2*(L + 2*xb)/L) + pi*i2*(2*B16 + 2*B26)*(xa - xb)) + (A45*(L*L)*cosa*r + (pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(sina*sina))*cos(pi*i2*(L + 2*xa)/L) - (A45*(L*L)*cosa*r + (pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(sina*sina))*cos(pi*i2*(L + 2*xb)/L))/((pi*pi)*L*i2*l4*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += 0.25*(tmax - tmin)*(-B22*(L*L)*cosa*sina*((-1)**l4 - 1)*cos(pi*i2*(L + 2*xa)/L) + B22*(L*L)*cosa*sina*((-1)**l4 - 1)*cos(pi*i2*(L + 2*xb)/L) + pi*i2*r*(4*(-1)**(l4 - 1)*pi*i2*(A55*r*xb + B12*cosa*xa) + ((-1)**l4 - 1)*(L*(A55*r + B12*cosa)*sin(pi*i2*(L + 2*xa)/L) - L*(A55*r + B12*cosa)*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(A55*r*xa + B12*cosa*xb))))/((pi*pi)*L*i2*l4*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += -0.25*((-1)**l4 - 1)*(pi*D16*i2*r*cos(pi*i2*(L + 2*xa)/L) - pi*D16*i2*r*cos(pi*i2*(L + 2*xb)/L) + D26*sina*(L*sin(pi*i2*(L + 2*xa)/L) - L*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(xa - xb)))/(pi*i2*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += -0.25*((-1)**l4 - 1)*(pi*D12*i2*r*cos(pi*i2*(L + 2*xa)/L) - pi*D12*i2*r*cos(pi*i2*(L + 2*xb)/L) + D22*sina*(L*sin(pi*i2*(L + 2*xa)/L) - L*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(xa - xb)))/(pi*i2*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += 0.25*(tmax - tmin)*(pi*i2*r*sina*((-1)**(l4 - 1)*pi*i2*xa*(4*B16 + 4*B26) + ((-1)**l4 - 1)*(-L*(B16 - B26)*sin(pi*i2*(L + 2*xa)/L) + L*(B16 - B26)*sin(pi*i2*(L + 2*xb)/L) + pi*i2*xb*(2*B16 + 2*B26))) + ((-1)**l4 - 1)*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(sina*sina))*cos(pi*i2*(L + 2*xa)/L) - ((-1)**l4 - 1)*((pi*pi)*B16*(i2*i2)*(r*r) + B26*(L*L)*(sina*sina))*cos(pi*i2*(L + 2*xb)/L))/((pi*pi)*L*i2*l4*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.25*(tmax - tmin)*(2*pi*B66*L*i2*r*sina*(2*(-1)**(l4 - 1)*sin(pi*i2*(L + 2*xa)/L) + ((-1)**l4 - 1)*sin(pi*i2*(L + 2*xb)/L)) + ((-1)**l4 - 1)*(A44*(L*L)*cosa*r - B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r))*cos(pi*i2*(L + 2*xa)/L) - ((-1)**l4 - 1)*(A44*(L*L)*cosa*r - B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i2*i2)*(r*r))*cos(pi*i2*(L + 2*xb)/L))/((pi*pi)*L*i2*l4*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += 0.25*(tmax - tmin)*(B26*(L*L)*cosa*sina*((-1)**l4 - 1)*cos(pi*i2*(L + 2*xa)/L) - B26*(L*L)*cosa*sina*((-1)**l4 - 1)*cos(pi*i2*(L + 2*xb)/L) + pi*i2*r*(4*(-1)**(l4 - 1)*pi*i2*(A45*r*xb + B26*cosa*xa) + ((-1)**l4 - 1)*(L*(A45*r + B26*cosa)*sin(pi*i2*(L + 2*xa)/L) - L*(A45*r + B26*cosa)*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(A45*r*xa + B26*cosa*xb))))/((pi*pi)*L*i2*l4*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += 0.25*D66*((-1)**l4 - 1)*(-pi*i2*r*cos(pi*i2*(L + 2*xa)/L) + pi*i2*r*cos(pi*i2*(L + 2*xb)/L) + sina*(L*sin(pi*i2*(L + 2*xa)/L) - L*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(xa - xb)))/(pi*i2*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += 0.25*D26*((-1)**l4 - 1)*(-pi*i2*r*cos(pi*i2*(L + 2*xa)/L) + pi*i2*r*cos(pi*i2*(L + 2*xb)/L) + sina*(L*sin(pi*i2*(L + 2*xa)/L) - L*sin(pi*i2*(L + 2*xb)/L) + 2*pi*i2*(xa - xb)))/(pi*i2*r)
                        else:
                            # k0_24 cond_2
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(B11*i2*k4*r*(-i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/L + B12*i2*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) - (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/pi + B12*k4*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/pi - B22*L*(sina*sina)*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/((pi*pi)*r))/(l4*(i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(A45*L*cosa*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/(pi*pi) - B16*i2*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) - (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/pi + B16*i2*k4*r*(-i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/L + B26*L*(sina*sina)*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/((pi*pi)*r) + B26*k4*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/pi)/(l4*(i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(pi*A55*k4*r*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L))) + pi*B12*cosa*i2*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) - (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L))) - B22*L*cosa*sina*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/r)/((pi*pi)*l4*(i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += ((-1)**l4 - 1)*(D16*i2*((-i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L)) - D26*L*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/(pi*r))/((i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += ((-1)**l4 - 1)*(-D12*i2*(-i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L)) - D22*L*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/(pi*r))/((i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(-B16*k4*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/pi + B16*i2*k4*r*(-i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/L + B26*L*(sina*sina)*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/((pi*pi)*r) + B26*i2*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) - (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/pi)/(l4*(i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(-A44*L*cosa*((-i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L))/(pi*pi) - B66*L*(sina*sina)*((-i2 + k4)*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L))/((pi*pi)*r) - B66*i2*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) - (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/pi - B66*k4*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/pi - B66*i2*k4*r*((-i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L))/L)/(l4*(i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += ((-1)**l4 - 1)*(tmax - tmin)*(pi*A45*k4*r*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L))) + B26*L*cosa*sina*(i2*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) - i2*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L))/r + pi*B26*cosa*i2*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) - (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L))))/((pi*pi)*l4*(i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += D66*((-1)**l4 - 1)*(L*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/(pi*r) + i2*((-i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/((i2 + k4)*(2.0*i2 - 2.0*k4))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += D26*((-1)**l4 - 1)*(L*sina*((-i2 + k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xb*(2*i2 + 2*k4))/L) + (i2 - k4)*cos(0.5*pi*(L*(i2 + k4 - 1) + xa*(2*i2 + 2*k4))/L) + (i2 + k4)*(sin(0.5*pi*(L + 2*xa)*(i2 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/(pi*r) + i2*((-i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 + k4)/L) + (i2 - k4)*cos(0.5*pi*(L + 2*xb)*(i2 + k4)/L) - (i2 + k4)*cos(0.5*pi*(L + 2*xa)*(i2 - k4)/L) + (i2 + k4)*cos(0.5*pi*(L + 2*xb)*(i2 - k4)/L)))/((i2 + k4)*(2.0*i2 - 2.0*k4))

            # k0_33
            for j3 in range(1, n3+1):
                row = num0 + num1 + num2*m2 + num3*(j3-1)
                for l3 in range(1, n3+1):
                    col = num0 + num1 + num2*m2 + num3*(l3-1)
                    if l3 == j3:
                        # k0_33 cond_1
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += -0.5*(xa - xb)*((pi*pi)*D66*(j3*j3) + (tmax - tmin)**2*(A55*(r*r) + D22*(sina*sina)))/(r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += -0.5*(xa - xb)*(A45*(r*r)*(tmax - tmin)**2 + D26*((pi*pi)*(j3*j3) - (sina*sina)*(tmax - tmin)**2))/(r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += -0.5*(xa - xb)*(A45*(r*r)*(tmax - tmin)**2 + D26*((pi*pi)*(j3*j3) - (sina*sina)*(tmax - tmin)**2))/(r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += -0.5*(xa - xb)*((pi*pi)*D22*(j3*j3) + (tmax - tmin)**2*(A44*(r*r) + D66*(sina*sina)))/(r*(tmax - tmin))
                    else:
                        # k0_33 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += -D26*sina*((-1)**(j3 + l3) - 1)*(xa - xb)/r
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += sina*((-1)**(j3 + l3) - 1)*(xa - xb)*(D22*(l3*l3) + D66*(j3*j3))/(r*(j3 - l3)*(j3 + l3))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += -sina*((-1)**(j3 + l3) - 1)*(xa - xb)*(D22*(j3*j3) + D66*(l3*l3))/(r*(j3 - l3)*(j3 + l3))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += D26*sina*((-1)**(j3 + l3) - 1)*(xa - xb)/r

            # k0_34
            for j3 in range(1, n3+1):
                row = num0 + num1 + num2*m2 + num3*(j3-1)
                for k4 in range(1, m4+1):
                    for l4 in range(1, n4+1):
                        col = (num0 + num1 + num2*m2 + num3*n3 +
                                num4*((l4-1)*m4 + (k4-1)))
                        if l4 == j3:
                            # k0_34 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.5*pi*B16*j3*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += 0.5*j3*(pi*B66*k4*r*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) + L*sina*(B22 + B66)*cos(0.5*pi*k4*(L + 2*xa)/L) - L*sina*(B22 + B66)*cos(0.5*pi*k4*(L + 2*xb)/L))/(k4*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += 0.5*L*j3*(A45*r - B26*cosa)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L))/(k4*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += -(-pi*D12*k4*r*sina*(tmax - tmin)**2*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) - L*((pi*pi)*D66*(j3*j3) + (tmax - tmin)**2*(A55*(r*r) + D22*(sina*sina)))*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/(pi*k4*r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += -(-pi*D26*k4*r*sina*(tmax - tmin)**2*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) - L*(A45*(r*r)*(tmax - tmin)**2 + D26*((pi*pi)*(j3*j3) - (sina*sina)*(tmax - tmin)**2))*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/(pi*k4*r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += 0.5*j3*(pi*B12*k4*r*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) - L*sina*(B22 + B66)*cos(0.5*pi*k4*(L + 2*xa)/L) + L*sina*(B22 + B66)*cos(0.5*pi*k4*(L + 2*xb)/L))/(k4*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.5*pi*B26*j3*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += 0.5*L*j3*(A44*r - B22*cosa)*(cos(0.5*pi*k4*(L + 2*xa)/L) - cos(0.5*pi*k4*(L + 2*xb)/L))/(k4*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += -(pi*D16*k4*r*sina*(tmax - tmin)**2*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) - L*(A45*(r*r)*(tmax - tmin)**2 + D26*((pi*pi)*(j3*j3) - (sina*sina)*(tmax - tmin)**2))*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/(pi*k4*r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += -(pi*D66*k4*r*sina*(tmax - tmin)**2*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) - L*((pi*pi)*D22*(j3*j3) + (tmax - tmin)**2*(A44*(r*r) + D66*(sina*sina)))*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/(pi*k4*r*(tmax - tmin))
                        else:
                            # k0_34 cond_2
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += l4*(-2*(-1)**(j3 + l4) + 2)*(-pi*B12*k4*r*sina*(tmax - tmin)**2*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + L*(B22*(sina*sina)*(tmax - tmin)**2 + (pi*pi)*B66*(j3*j3))*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/((pi*pi)*k4*r*(j3 - l4)*(j3 + l4)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += l4*(-2*(-1)**(j3 + l4) + 2)*(-pi*B26*k4*r*sina*(tmax - tmin)**2*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + L*(-A45*cosa*r*(tmax - tmin)**2 + B26*((pi*pi)*(j3*j3) - (sina*sina)*(tmax - tmin)**2))*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/((pi*pi)*k4*r*(j3 - l4)*(j3 + l4)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += l4*(2*(-1)**(j3 + l4) - 2)*(tmax - tmin)*(pi*A55*k4*(r*r)*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) - B22*L*cosa*sina*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/((pi*pi)*k4*r*(j3 - l4)*(j3 + l4))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += -((-1)**(j3 + l4) - 1)*(pi*D16*(j3*j3)*k4*r*cos(0.5*pi*k4*(L + 2*xa)/L) - pi*D16*(j3*j3)*k4*r*cos(0.5*pi*k4*(L + 2*xb)/L) + D26*L*sina*((j3*j3) - (l4*l4))*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)))/(pi*k4*r*(j3 - l4)*(j3 + l4))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += -((-1)**(j3 + l4) - 1)*(pi*D66*(j3*j3)*k4*r*cos(0.5*pi*k4*(L + 2*xa)/L) - pi*D66*(j3*j3)*k4*r*cos(0.5*pi*k4*(L + 2*xb)/L) - L*sina*(D22*(l4*l4) + D66*(j3*j3))*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)))/(pi*k4*r*(j3 - l4)*(j3 + l4))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += l4*(-2*(-1)**(j3 + l4) + 2)*(pi*B16*k4*r*sina*(tmax - tmin)**2*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + B26*L*((pi*pi)*(j3*j3) - (sina*sina)*(tmax - tmin)**2)*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/((pi*pi)*k4*r*(j3 - l4)*(j3 + l4)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += l4*(-2*(-1)**(j3 + l4) + 2)*(pi*B66*k4*r*sina*(tmax - tmin)**2*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + L*((pi*pi)*B22*(j3*j3) - (tmax - tmin)**2*(A44*cosa*r - B66*(sina*sina)))*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/((pi*pi)*k4*r*(j3 - l4)*(j3 + l4)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += l4*(2*(-1)**(j3 + l4) - 2)*(tmax - tmin)*(pi*A45*k4*(r*r)*sin(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L) + B26*L*cosa*sina*cos(0.5*pi*(L*(k4 - 1) + k4*(xa + xb))/L))*sin(0.5*pi*k4*(xa - xb)/L)/((pi*pi)*k4*r*(j3 - l4)*(j3 + l4))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += -((-1)**(j3 + l4) - 1)*(pi*D12*(j3*j3)*k4*r*cos(0.5*pi*k4*(L + 2*xa)/L) - pi*D12*(j3*j3)*k4*r*cos(0.5*pi*k4*(L + 2*xb)/L) + L*sina*(D22*(j3*j3) + D66*(l4*l4))*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)))/(pi*k4*r*(j3 - l4)*(j3 + l4))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += -D26*((-1)**(j3 + l4) - 1)*(-L*sina*((j3*j3) - (l4*l4))*(sin(0.5*pi*k4*(L + 2*xa)/L) - sin(0.5*pi*k4*(L + 2*xb)/L)) + pi*(j3*j3)*k4*r*cos(0.5*pi*k4*(L + 2*xa)/L) - pi*(j3*j3)*k4*r*cos(0.5*pi*k4*(L + 2*xb)/L))/(pi*k4*r*(j3 - l4)*(j3 + l4))

            # k0_44
            for i4 in range(1, m4+1):
                for j4 in range(1, n4+1):
                    row = (num0 + num1 + num2*m2 + num3*n3 +
                            num4*((j4-1)*m4 + (i4-1)))
                    for k4 in range(1, m4+1):
                        for l4 in range(1, n4+1):
                            col = (num0 + num1 + num2*m2 + num3*n3 +
                                    num4*((l4-1)*m4 + (k4-1)))
                            if k4 == i4 and l4 == j4:
                                # k0_44 cond_1
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += -0.125*pi*A11*i4*r*(tmax - tmin)*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(L*L) + 0.25*A12*sina*(tmax - tmin)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) - 0.125*A22*(sina*sina)*(tmax - tmin)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r) + 0.125*pi*A66*(j4*j4)*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(i4*r*(tmax - tmin))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+1
                                k0v[c] += (-A16*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) - pi*A16*i4*r*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(L*L) + A26*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + pi*A26*(j4*j4)*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(i4*r) + A26*(sina*sina)*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r))/(8.0*tmax - 8.0*tmin)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += 0.125*cosa*(tmax - tmin)*(pi*A12*i4*r*cos(pi*i4*(L + 2*xa)/L) - pi*A12*i4*r*cos(pi*i4*(L + 2*xb)/L) + A22*sina*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb)))/(pi*i4*r)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += 0.5*(pi*pi)*B16*i4*j4*(xa - xb)/L
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += 0.125*j4*(-(L*L)*sina*(B22 + B66)*cos(pi*i4*(L + 2*xa)/L) + (L*L)*sina*(B22 + B66)*cos(pi*i4*(L + 2*xb)/L) + pi*i4*r*(L*(B12 - B66)*sin(pi*i4*(L + 2*xa)/L) - L*(B12 - B66)*sin(pi*i4*(L + 2*xb)/L) + pi*i4*(2*B12 + 2*B66)*(xa - xb)))/(L*i4*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+0
                                k0v[c] += (-A16*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) - pi*A16*i4*r*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(L*L) + A26*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + pi*A26*(j4*j4)*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(i4*r) + A26*(sina*sina)*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r))/(8.0*tmax - 8.0*tmin)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += 0.125*pi*A22*(j4*j4)*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(i4*r*(tmax - tmin)) - 0.125*A44*(cosa*cosa)*(tmax - tmin)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r) - 0.25*A66*sina*(tmax - tmin)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) - 0.125*A66*(sina*sina)*(tmax - tmin)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r) - 0.125*pi*A66*i4*r*(tmax - tmin)*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(L*L)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += 0.125*cosa*(tmax - tmin)*(A26*sina*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)) + pi*i4*r*(A26 - A45)*cos(pi*i4*(L + 2*xa)/L) - pi*i4*r*(A26 - A45)*cos(pi*i4*(L + 2*xb)/L))/(pi*i4*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+3
                                k0v[c] += 0.125*j4*((L*L)*sina*(B22 + B66)*cos(pi*i4*(L + 2*xa)/L) - (L*L)*sina*(B22 + B66)*cos(pi*i4*(L + 2*xb)/L) + pi*i4*r*(-L*(B12 - B66)*sin(pi*i4*(L + 2*xa)/L) + L*(B12 - B66)*sin(pi*i4*(L + 2*xb)/L) + pi*i4*(2*B12 + 2*B66)*(xa - xb)))/(L*i4*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+4
                                k0v[c] += 0.5*(pi*pi)*B26*i4*j4*(xa - xb)/L
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += 0.125*cosa*(tmax - tmin)*(pi*A12*i4*r*cos(pi*i4*(L + 2*xa)/L) - pi*A12*i4*r*cos(pi*i4*(L + 2*xb)/L) + A22*sina*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb)))/(pi*i4*r)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += 0.125*cosa*(tmax - tmin)*(A26*sina*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)) + pi*i4*r*(A26 - A45)*cos(pi*i4*(L + 2*xa)/L) - pi*i4*r*(A26 - A45)*cos(pi*i4*(L + 2*xb)/L))/(pi*i4*r)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += 0.125*(-A22*(cosa*cosa)*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)) + (pi*pi)*A44*(j4*j4)*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb)) + (pi*pi)*A55*(i4*i4)*(r*r)*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(L*L))/(pi*i4*r*(tmax - tmin))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+3
                                k0v[c] += 0.125*L*j4*(A45*r - B26*cosa)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/(i4*r)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+4
                                k0v[c] += 0.125*L*j4*(A44*r - B22*cosa)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/(i4*r)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += 0.5*(pi*pi)*B16*i4*j4*(xa - xb)/L
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+1
                                k0v[c] += 0.125*j4*((L*L)*sina*(B22 + B66)*cos(pi*i4*(L + 2*xa)/L) - (L*L)*sina*(B22 + B66)*cos(pi*i4*(L + 2*xb)/L) + pi*i4*r*(-L*(B12 - B66)*sin(pi*i4*(L + 2*xa)/L) + L*(B12 - B66)*sin(pi*i4*(L + 2*xb)/L) + pi*i4*(2*B12 + 2*B66)*(xa - xb)))/(L*i4*r)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+2
                                k0v[c] += 0.125*L*j4*(A45*r - B26*cosa)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/(i4*r)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += (A55*r*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(pi*i4) + pi*D11*i4*r*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(L*L) - 2*D12*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + D22*(sina*sina)*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(pi*i4*r) + pi*D66*(j4*j4)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(i4*r))/(8.0*tmax - 8.0*tmin)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += (A45*r*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(pi*i4) + D16*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + pi*D16*i4*r*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(L*L) - D26*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + pi*D26*(j4*j4)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(i4*r) + D26*(sina*sina)*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r))/(8.0*tmax - 8.0*tmin)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += 0.125*j4*(-(L*L)*sina*(B22 + B66)*cos(pi*i4*(L + 2*xa)/L) + (L*L)*sina*(B22 + B66)*cos(pi*i4*(L + 2*xb)/L) + pi*i4*r*(L*(B12 - B66)*sin(pi*i4*(L + 2*xa)/L) - L*(B12 - B66)*sin(pi*i4*(L + 2*xb)/L) + pi*i4*(2*B12 + 2*B66)*(xa - xb)))/(L*i4*r)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+1
                                k0v[c] += 0.5*(pi*pi)*B26*i4*j4*(xa - xb)/L
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+2
                                k0v[c] += 0.125*L*j4*(A44*r - B22*cosa)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/(i4*r)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += (A45*r*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(pi*i4) + D16*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + pi*D16*i4*r*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(L*L) - D26*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + pi*D26*(j4*j4)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(i4*r) + D26*(sina*sina)*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r))/(8.0*tmax - 8.0*tmin)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += (A44*r*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(pi*i4) + pi*D22*(j4*j4)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(i4*r) + 2*D66*sina*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + D66*(sina*sina)*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(pi*i4*r) + pi*D66*i4*r*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(-xa + xb))/(L*L))/(8.0*tmax - 8.0*tmin)

                            elif k4 != i4 and l4 == j4:
                                # k0_44 cond_2
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += (-pi*A11*i4*k4*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L + A12*i4*sina*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + A12*k4*sina*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + A22*L*(sina*sina)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + pi*A66*L*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r)/((i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+1
                                k0v[c] += (-A16*i4*sina*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - pi*A16*i4*k4*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L + pi*A26*L*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r - A26*L*(sina*sina)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + A26*k4*sina*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += cosa*(tmax - tmin)*(pi*A12*i4*r*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + A22*L*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))))/(pi*r*(i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += pi*B16*j4*((i4 - k4)**2*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 - k4)**2*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)**2*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += j4*(pi*B12*i4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) - B22*L*sina*((-i4 + k4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L))/r - B66*L*sina*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r + pi*B66*k4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+0
                                k0v[c] += (-A16*k4*sina*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - pi*A16*i4*k4*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L + pi*A26*L*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r - A26*L*(sina*sina)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + A26*i4*sina*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += (pi*A22*L*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r + A44*L*(cosa*cosa)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + A66*L*(sina*sina)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - A66*i4*sina*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - A66*k4*sina*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - pi*A66*i4*k4*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L)/((i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += cosa*(tmax - tmin)*(-A26*L*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + A26*i4*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - A45*k4*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+3
                                k0v[c] += j4*(pi*B12*k4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) + B22*L*sina*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r + B66*L*sina*((-i4 + k4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L))/r + pi*B66*i4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+4
                                k0v[c] += pi*B26*j4*((i4 - k4)**2*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 - k4)**2*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)**2*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += cosa*(tmax - tmin)*(A12*k4*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + A22*L*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += cosa*(tmax - tmin)*(-A26*L*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + A26*k4*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - A45*i4*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += (A22*(L*L)*(cosa*cosa)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) + (pi*pi)*A44*(L*L)*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) - (pi*pi)*A55*i4*k4*(r*r)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))))/(pi*L*r*(i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+3
                                k0v[c] += L*j4*(A45*r - B26*cosa)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(r*(i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+4
                                k0v[c] += L*j4*(A44*r - B22*cosa)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(r*(i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += pi*B16*j4*(-(i4 - k4)**2*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)**2*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)**2*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+1
                                k0v[c] += j4*(-pi*B12*i4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) + B22*L*sina*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r + B66*L*sina*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r + pi*B66*k4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+2
                                k0v[c] += L*j4*(-A45*r + B26*cosa)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (-i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(r*(i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += (-A55*L*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + pi*D11*i4*k4*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L - D12*i4*sina*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - D12*k4*sina*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - D22*L*(sina*sina)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - pi*D66*L*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r)/((i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += (-A45*L*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + D16*i4*sina*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + pi*D16*i4*k4*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L - pi*D26*L*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r + D26*L*(sina*sina)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - D26*k4*sina*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += j4*(pi*B12*k4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) + B22*L*sina*((-i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L))/r - B66*L*sina*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r - pi*B66*i4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+1
                                k0v[c] += pi*B26*j4*(-(i4 - k4)**2*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)**2*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)**2*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+2
                                k0v[c] += L*j4*(-A44*r + B22*cosa)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (-i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(r*(i4 + k4)*(4.0*i4 - 4.0*k4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += (-A45*L*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + D16*k4*sina*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + pi*D16*i4*k4*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L - pi*D26*L*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r + D26*L*(sina*sina)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - D26*i4*sina*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += (-A44*L*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi - pi*D22*L*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r - D66*L*(sina*sina)*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + D66*i4*sina*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + D66*k4*sina*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + pi*D66*i4*k4*r*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L)/((i4 + k4)*(4.0*i4 - 4.0*k4)*(tmax - tmin))

                            elif k4 != i4 and l4 != j4:
                                # k0_44 cond_3
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += A16*j4*l4*((-1)**(j4 + l4) - 1)*(-(i4*i4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + (i4*i4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - 2*i4*k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - 2*i4*k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - (k4*k4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + (k4*k4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - (i4 - k4)**2*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)**2*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+1
                                k0v[c] += j4*l4*((-1)**(j4 + l4) - 1)*(A66*L*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + A66*k4*r*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + r*(pi*A12*i4*((i4 - k4)*(-cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L)) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L)) + A22*L*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/r)/pi)/(r*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += A26*L*cosa*j4*l4*((-1)**(j4 + l4) - 1)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += j4*((-1)**(j4 + l4) - 1)*(B11*i4*k4*r*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/L + B12*i4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi - B12*k4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi - B22*L*(sina*sina)*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((pi*pi)*r) - B66*L*(l4*l4)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r)/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += j4*r*((-1)**(j4 + l4) - 1)*(-B16*i4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - B16*i4*k4*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (-i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/L - B26*L*(l4*l4)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(r*r) + B26*L*(sina*sina)*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((pi*pi)*(r*r)) - B26*k4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+0
                                k0v[c] += j4*l4*((-1)**(j4 + l4) - 1)*(A12*k4*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + A22*L*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + A66*L*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - A66*i4*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += A26*j4*l4*((-1)**(j4 + l4) - 1)*(-(i4*i4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + (i4*i4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - 2*i4*k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - 2*i4*k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - (k4*k4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + (k4*k4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - (i4 - k4)**2*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)**2*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += L*cosa*j4*l4*((-1)**(j4 + l4) - 1)*(A22 + A44)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+3
                                k0v[c] += j4*((-1)**(j4 + l4) - 1)*(A45*L*cosa*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(pi*pi) + B16*k4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + B16*i4*k4*r*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/L - B26*L*(l4*l4)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r + B26*L*(sina*sina)*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((pi*pi)*r) + B26*i4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi)/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+4
                                k0v[c] += j4*((-1)**(j4 + l4) - 1)*(A44*L*cosa*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(pi*pi) - B22*L*(l4*l4)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r - B66*L*(sina*sina)*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((pi*pi)*r) - B66*i4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + B66*k4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + B66*i4*k4*r*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/L)/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += -A26*L*cosa*j4*l4*((-1)**(j4 + l4) - 1)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += -L*cosa*j4*l4*((-1)**(j4 + l4) - 1)*(A22 + A44)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += A45*j4*l4*((-1)**(j4 + l4) - 1)*(-(i4*i4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + (i4*i4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - 2*i4*k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - 2*i4*k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - (k4*k4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + (k4*k4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - (i4 - k4)**2*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)**2*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+3
                                k0v[c] += j4*((-1)**(j4 + l4) - 1)*(tmax - tmin)*(pi*A55*i4*r*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) - pi*B12*cosa*k4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) - B22*L*cosa*sina*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r)/((pi*pi)*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+4
                                k0v[c] += j4*((-1)**(j4 + l4) - 1)*(tmax - tmin)*(pi*A45*i4*r*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) + B26*L*cosa*sina*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r - pi*B26*cosa*k4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))))/((pi*pi)*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += l4*((-1)**(j4 + l4) - 1)*(-B11*i4*k4*r*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/L - B12*i4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + B12*k4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + B22*L*(sina*sina)*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((pi*pi)*r) + B66*L*(j4*j4)*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r)/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+1
                                k0v[c] += l4*((-1)**(j4 + l4) - 1)*(A45*L*cosa*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (-i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(pi*pi) + B16*i4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi - B16*i4*k4*r*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/L + B26*L*(j4*j4)*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r + B26*L*(sina*sina)*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (-i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((pi*pi)*r) + B26*k4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi)/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+2
                                k0v[c] += l4*((-1)**(j4 + l4) - 1)*(tmax - tmin)*(-pi*A55*k4*r*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) - pi*B12*cosa*i4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) + B22*L*cosa*sina*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r)/((pi*pi)*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += ((-1)**(j4 + l4) - 1)*(D16*i4*(l4*l4)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - D16*(j4*j4)*k4*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - D26*L*(j4*j4)*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + D26*L*(l4*l4)*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += ((-1)**(j4 + l4) - 1)*(D12*i4*(l4*l4)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) + D22*L*(l4*l4)*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + D66*L*(j4*j4)*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - D66*(j4*j4)*k4*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += l4*((-1)**(j4 + l4) - 1)*(B16*k4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi - B16*i4*k4*r*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/L + B26*L*(j4*j4)*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r + B26*L*(sina*sina)*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (-i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((pi*pi)*r) - B26*i4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi)/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+1
                                k0v[c] += l4*((-1)**(j4 + l4) - 1)*(A44*L*cosa*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (-i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(pi*pi) + B22*L*(j4*j4)*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r + B66*L*(sina*sina)*(tmax - tmin)**2*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/((pi*pi)*r) + B66*i4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi + B66*k4*sina*(tmax - tmin)**2*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/pi - B66*i4*k4*r*(tmax - tmin)**2*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/L)/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+2
                                k0v[c] += l4*((-1)**(j4 + l4) - 1)*(tmax - tmin)*(-pi*A45*k4*r*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))) - B26*L*cosa*sina*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/r - pi*B26*cosa*i4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L))))/((pi*pi)*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += ((-1)**(j4 + l4) - 1)*(-D12*(j4*j4)*k4*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - D22*L*(j4*j4)*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - D66*L*(l4*l4)*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + D66*i4*(l4*l4)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += D26*((-1)**(j4 + l4) - 1)*(L*(j4*j4)*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) - L*(l4*l4)*sina*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(pi*r) + i4*(l4*l4)*(-i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)) - (j4*j4)*k4*(i4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - i4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) + (i4 - k4)*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) - (i4 + k4)*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L)))/((i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4))

                            elif k4 == i4 and l4 != j4:
                                # k0_44 cond_4
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+1
                                k0v[c] += 0.25*j4*l4*((-1)**(j4 + l4) - 1)*(-pi*i4*r*(A12 - A66)*cos(pi*i4*(L + 2*xa)/L) + pi*i4*r*(A12 - A66)*cos(pi*i4*(L + 2*xb)/L) + sina*(A22 + A66)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)))/(pi*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += -0.25*A26*cosa*j4*l4*((-1)**(j4 + l4) - 1)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += 0.5*j4*((-1)**(j4 + l4) - 1)*(2*pi*B12*L*i4*r*sina*(tmax - tmin)**2*cos(pi*i4*(L + xa + xb)/L) + ((pi*pi)*B66*(L*L)*(l4*l4) - (tmax - tmin)**2*((pi*pi)*B11*(i4*i4)*(r*r) - B22*(L*L)*(sina*sina)))*sin(pi*i4*(L + xa + xb)/L))*sin(pi*i4*(xa - xb)/L)/((pi*pi)*L*i4*r*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += 0.25*j4*((-1)**(j4 + l4) - 1)*(B16*i4*r*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) - B16*sina*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/pi - B26*(L*L)*(l4*l4)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/(i4*r) + B26*(L*L)*(sina*sina)*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/((pi*pi)*i4*r) - B26*sina*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/pi)/(L*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+0
                                k0v[c] += -0.25*j4*l4*((-1)**(j4 + l4) - 1)*(-pi*i4*r*(A12 - A66)*cos(pi*i4*(L + 2*xa)/L) + pi*i4*r*(A12 - A66)*cos(pi*i4*(L + 2*xb)/L) + sina*(A22 + A66)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)))/(pi*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += -0.25*cosa*j4*l4*((-1)**(j4 + l4) - 1)*(A22 + A44)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+3
                                k0v[c] += 0.25*j4*((-1)**(j4 + l4) - 1)*(A45*(L*L)*cosa*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/((pi*pi)*i4) + B16*i4*r*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + B16*sina*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/pi - B26*(L*L)*(l4*l4)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/(i4*r) + B26*(L*L)*(sina*sina)*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/((pi*pi)*i4*r) + B26*sina*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/pi)/(L*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+4
                                k0v[c] += -0.5*j4*((-1)**(j4 + l4) - 1)*(2*pi*B66*L*i4*r*sina*(tmax - tmin)**2*cos(pi*i4*(L + xa + xb)/L) + (-(pi*pi)*B22*(L*L)*(l4*l4) + (tmax - tmin)**2*(A44*(L*L)*cosa*r - B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i4*i4)*(r*r)))*sin(pi*i4*(L + xa + xb)/L))*sin(pi*i4*(xa - xb)/L)/((pi*pi)*L*i4*r*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += 0.25*A26*cosa*j4*l4*((-1)**(j4 + l4) - 1)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += 0.25*cosa*j4*l4*((-1)**(j4 + l4) - 1)*(A22 + A44)*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/(pi*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+3
                                k0v[c] += 0.25*j4*((-1)**(j4 + l4) - 1)*(tmax - tmin)*(-B22*(L*L)*cosa*sina*cos(pi*i4*(L + 2*xa)/L) + B22*(L*L)*cosa*sina*cos(pi*i4*(L + 2*xb)/L) + pi*i4*r*(L*(A55*r + B12*cosa)*sin(pi*i4*(L + 2*xa)/L) - L*(A55*r + B12*cosa)*sin(pi*i4*(L + 2*xb)/L) - 2*pi*i4*(xa - xb)*(-A55*r + B12*cosa)))/((pi*pi)*L*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+4
                                k0v[c] += 0.25*j4*((-1)**(j4 + l4) - 1)*(tmax - tmin)*(B26*(L*L)*cosa*sina*cos(pi*i4*(L + 2*xa)/L) - B26*(L*L)*cosa*sina*cos(pi*i4*(L + 2*xb)/L) + pi*i4*r*(L*(A45*r + B26*cosa)*sin(pi*i4*(L + 2*xa)/L) - L*(A45*r + B26*cosa)*sin(pi*i4*(L + 2*xb)/L) - 2*pi*i4*(xa - xb)*(-A45*r + B26*cosa)))/((pi*pi)*L*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += -0.5*l4*((-1)**(j4 + l4) - 1)*(2*pi*B12*L*i4*r*sina*(tmax - tmin)**2*cos(pi*i4*(L + xa + xb)/L) + ((pi*pi)*B66*(L*L)*(j4*j4) - (tmax - tmin)**2*((pi*pi)*B11*(i4*i4)*(r*r) - B22*(L*L)*(sina*sina)))*sin(pi*i4*(L + xa + xb)/L))*sin(pi*i4*(xa - xb)/L)/((pi*pi)*L*i4*r*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+1
                                k0v[c] += 0.25*l4*((-1)**(j4 + l4) - 1)*(-A45*(L*L)*cosa*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/((pi*pi)*i4) - B16*i4*r*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) - B16*sina*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/pi + B26*(L*L)*(j4*j4)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/(i4*r) - B26*(L*L)*(sina*sina)*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/((pi*pi)*i4*r) - B26*sina*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/pi)/(L*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+2
                                k0v[c] += 0.25*l4*((-1)**(j4 + l4) - 1)*(tmax - tmin)*(B22*(L*L)*cosa*sina*cos(pi*i4*(L + 2*xa)/L) - B22*(L*L)*cosa*sina*cos(pi*i4*(L + 2*xb)/L) + pi*i4*r*(-L*(A55*r + B12*cosa)*sin(pi*i4*(L + 2*xa)/L) + L*(A55*r + B12*cosa)*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)*(-A55*r + B12*cosa)))/((pi*pi)*L*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += -0.25*((-1)**(j4 + l4) - 1)*(pi*D16*i4*r*cos(pi*i4*(L + 2*xa)/L) - pi*D16*i4*r*cos(pi*i4*(L + 2*xb)/L) + D26*sina*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)))/(pi*i4*r)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += 0.25*((-1)**(j4 + l4) - 1)*(-pi*i4*r*(-D12*(l4*l4) + D66*(j4*j4))*cos(pi*i4*(L + 2*xa)/L) + pi*i4*r*(-D12*(l4*l4) + D66*(j4*j4))*cos(pi*i4*(L + 2*xb)/L) + sina*(D22*(l4*l4) + D66*(j4*j4))*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)))/(pi*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += 0.25*l4*((-1)**(j4 + l4) - 1)*(-B16*i4*r*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L)) + B16*sina*(tmax - tmin)**2*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/pi + B26*(L*L)*(j4*j4)*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/(i4*r) - B26*(L*L)*(sina*sina)*(tmax - tmin)**2*(cos(pi*i4*(L + 2*xa)/L) - cos(pi*i4*(L + 2*xb)/L))/((pi*pi)*i4*r) + B26*sina*(tmax - tmin)**2*(-L*sin(pi*i4*(L + 2*xa)/L) + L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb))/pi)/(L*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+1
                                k0v[c] += 0.5*l4*((-1)**(j4 + l4) - 1)*(2*pi*B66*L*i4*r*sina*(tmax - tmin)**2*cos(pi*i4*(L + xa + xb)/L) + (-(pi*pi)*B22*(L*L)*(j4*j4) + (tmax - tmin)**2*(A44*(L*L)*cosa*r - B66*(L*L)*(sina*sina) + (pi*pi)*B66*(i4*i4)*(r*r)))*sin(pi*i4*(L + xa + xb)/L))*sin(pi*i4*(xa - xb)/L)/((pi*pi)*L*i4*r*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+2
                                k0v[c] += 0.25*l4*((-1)**(j4 + l4) - 1)*(tmax - tmin)*(-B26*(L*L)*cosa*sina*cos(pi*i4*(L + 2*xa)/L) + B26*(L*L)*cosa*sina*cos(pi*i4*(L + 2*xb)/L) + pi*i4*r*(-L*(A45*r + B26*cosa)*sin(pi*i4*(L + 2*xa)/L) + L*(A45*r + B26*cosa)*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)*(-A45*r + B26*cosa)))/((pi*pi)*L*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += -0.25*((-1)**(j4 + l4) - 1)*(pi*i4*r*(D12*(j4*j4) - D66*(l4*l4))*cos(pi*i4*(L + 2*xa)/L) - pi*i4*r*(D12*(j4*j4) - D66*(l4*l4))*cos(pi*i4*(L + 2*xb)/L) + sina*(D22*(j4*j4) + D66*(l4*l4))*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)))/(pi*i4*r*(j4 - l4)*(j4 + l4))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += 0.25*D26*((-1)**(j4 + l4) - 1)*(-pi*i4*r*cos(pi*i4*(L + 2*xa)/L) + pi*i4*r*cos(pi*i4*(L + 2*xb)/L) + sina*(L*sin(pi*i4*(L + 2*xa)/L) - L*sin(pi*i4*(L + 2*xb)/L) + 2*pi*i4*(xa - xb)))/(pi*i4*r)

    size = num0 + num1 + num2*m2 + num3*n3 + num4*m4*n4

    k0 = csr_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0_cyl(double r1, double L, double tmin, double tmax,
        np.ndarray[cDOUBLE, ndim=2] F,
        int m2, int n3, int m4, int n4):
    cdef int i2, k2, j3, l3, i4, j4, k4, l4, c, row, col
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    fdim = (1 + 1 + 4*m2 + 4*n3 + 10 + 7*m2 + 7*n3 + 17*m2*m2 +
            17*m2*n3 + 17*m2*m4*n4 + 17*n3*n3 + 17*n3*m4*n4 + 17*m4*n4*m4*n4)

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
    r = r1

    # k0_00
    c += 1
    k0r[c] = 0
    k0c[c] = 0
    k0v[c] += A11*r*(tmax - tmin)/L

    col = num0
    # k0_01
    c += 1
    k0r[c] = 0
    k0c[c] = col+2
    k0v[c] += A12*(tmax - tmin)

    # k0_02
    for k2 in range(1, m2+1):
        col = num0 + num1 + num2*(k2-1)
        c += 1
        k0r[c] = 0
        k0c[c] = col+0
        k0v[c] += A11*r*((-1)**k2 - 1)*(tmax - tmin)/L
        c += 1
        k0r[c] = 0
        k0c[c] = col+1
        k0v[c] += A16*r*((-1)**k2 - 1)*(tmax - tmin)/L
        c += 1
        k0r[c] = 0
        k0c[c] = col+3
        k0v[c] += B11*r*((-1)**k2 - 1)*(tmax - tmin)/L
        c += 1
        k0r[c] = 0
        k0c[c] = col+4
        k0v[c] += B16*r*((-1)**k2 - 1)*(tmax - tmin)/L

    # k0_03
    for l3 in range(1, n3+1):
        col = num0 + num1 + num2*m2 + num3*(l3-1)
        c += 1
        k0r[c] = 0
        k0c[c] = col+0
        k0v[c] += A16*((-1)**l3 - 1)
        c += 1
        k0r[c] = 0
        k0c[c] = col+1
        k0v[c] += A12*((-1)**l3 - 1)
        c += 1
        k0r[c] = 0
        k0c[c] = col+3
        k0v[c] += B16*((-1)**l3 - 1)
        c += 1
        k0r[c] = 0
        k0c[c] = col+4
        k0v[c] += B12*((-1)**l3 - 1)

    # k0_04
    pass

    row = num0
    col = num0
    # k0_11
    c += 1
    k0r[c] = row+1
    k0c[c] = col+1
    k0v[c] += A44*L*(tmax - tmin)/r
    c += 1
    k0r[c] = row+1
    k0c[c] = col+3
    k0v[c] += -A45*L*(tmax - tmin)
    c += 1
    k0r[c] = row+1
    k0c[c] = col+4
    k0v[c] += -A44*L*(tmax - tmin)
    c += 1
    k0r[c] = row+2
    k0c[c] = col+2
    k0v[c] += A22*L*(tmax - tmin)/r
    c += 1
    k0r[c] = row+3
    k0c[c] = col+1
    k0v[c] += -A45*L*(tmax - tmin)
    c += 1
    k0r[c] = row+3
    k0c[c] = col+3
    k0v[c] += A55*L*r*(tmax - tmin)
    c += 1
    k0r[c] = row+3
    k0c[c] = col+4
    k0v[c] += A45*L*r*(tmax - tmin)
    c += 1
    k0r[c] = row+4
    k0c[c] = col+1
    k0v[c] += -A44*L*(tmax - tmin)
    c += 1
    k0r[c] = row+4
    k0c[c] = col+3
    k0v[c] += A45*L*r*(tmax - tmin)
    c += 1
    k0r[c] = row+4
    k0c[c] = col+4
    k0v[c] += A44*L*r*(tmax - tmin)

    # k0_12
    c += 1
    k0r[c] = row+1
    k0c[c] = col+2
    k0v[c] += A45*((-1)**k2 - 1)*(-tmax + tmin)
    c += 1
    k0r[c] = row+2
    k0c[c] = col+0
    k0v[c] += A12*((-1)**k2 - 1)*(tmax - tmin)
    c += 1
    k0r[c] = row+2
    k0c[c] = col+1
    k0v[c] += A26*((-1)**k2 - 1)*(tmax - tmin)
    c += 1
    k0r[c] = row+2
    k0c[c] = col+3
    k0v[c] += B12*((-1)**k2 - 1)*(tmax - tmin)
    c += 1
    k0r[c] = row+2
    k0c[c] = col+4
    k0v[c] += B26*((-1)**k2 - 1)*(tmax - tmin)
    c += 1
    k0r[c] = row+3
    k0c[c] = col+2
    k0v[c] += A55*r*((-1)**k2 - 1)*(tmax - tmin)
    c += 1
    k0r[c] = row+4
    k0c[c] = col+2
    k0v[c] += A45*r*((-1)**k2 - 1)*(tmax - tmin)

    # k0_13
    c += 1
    k0r[c] = row+1
    k0c[c] = col+2
    k0v[c] += -A44*L*((-1)**l3 - 1)/r
    c += 1
    k0r[c] = row+2
    k0c[c] = col+0
    k0v[c] += A26*L*((-1)**l3 - 1)/r
    c += 1
    k0r[c] = row+2
    k0c[c] = col+1
    k0v[c] += A22*L*((-1)**l3 - 1)/r
    c += 1
    k0r[c] = row+2
    k0c[c] = col+3
    k0v[c] += B26*L*((-1)**l3 - 1)/r
    c += 1
    k0r[c] = row+2
    k0c[c] = col+4
    k0v[c] += B22*L*((-1)**l3 - 1)/r
    c += 1
    k0r[c] = row+3
    k0c[c] = col+2
    k0v[c] += A45*L*((-1)**l3 - 1)
    c += 1
    k0r[c] = row+4
    k0c[c] = col+2
    k0v[c] += A44*L*((-1)**l3 - 1)

    # k0_14
    pass

    # k0_22
    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for k2 in range(1, m2+1):
            col = num0 + num1 + num2*(k2-1)
            if k2 == i2:
                # k0_22 cond_1
                c += 1
                k0r[c] = row+0
                k0c[c] = col+0
                k0v[c] += 0.5*(pi*pi)*A11*(i2*i2)*r*(tmax - tmin)/L
                c += 1
                k0r[c] = row+0
                k0c[c] = col+1
                k0v[c] += 0.5*(pi*pi)*A16*(i2*i2)*r*(tmax - tmin)/L
                c += 1
                k0r[c] = row+0
                k0c[c] = col+3
                k0v[c] += 0.5*(pi*pi)*B11*(i2*i2)*r*(tmax - tmin)/L
                c += 1
                k0r[c] = row+0
                k0c[c] = col+4
                k0v[c] += 0.5*(pi*pi)*B16*(i2*i2)*r*(tmax - tmin)/L
                c += 1
                k0r[c] = row+1
                k0c[c] = col+0
                k0v[c] += 0.5*(pi*pi)*A16*(i2*i2)*r*(tmax - tmin)/L
                c += 1
                k0r[c] = row+1
                k0c[c] = col+1
                k0v[c] += r*(tmax - tmin)*(0.5*A44*L/(r*r) + 0.5*(pi*pi)*A66*(i2*i2)/L)
                c += 1
                k0r[c] = row+1
                k0c[c] = col+3
                k0v[c] += 0.5*(tmax - tmin)*(-A45*(L*L) + (pi*pi)*B16*(i2*i2)*r)/L
                c += 1
                k0r[c] = row+1
                k0c[c] = col+4
                k0v[c] += 0.5*(tmax - tmin)*(-A44*(L*L) + (pi*pi)*B66*(i2*i2)*r)/L
                c += 1
                k0r[c] = row+2
                k0c[c] = col+2
                k0v[c] += r*(tmax - tmin)*(0.5*A22*L/(r*r) + 0.5*(pi*pi)*A55*(i2*i2)/L)
                c += 1
                k0r[c] = row+3
                k0c[c] = col+0
                k0v[c] += 0.5*(pi*pi)*B11*(i2*i2)*r*(tmax - tmin)/L
                c += 1
                k0r[c] = row+3
                k0c[c] = col+1
                k0v[c] += 0.5*(tmax - tmin)*(-A45*(L*L) + (pi*pi)*B16*(i2*i2)*r)/L
                c += 1
                k0r[c] = row+3
                k0c[c] = col+3
                k0v[c] += 0.5*r*(tmax - tmin)*(A55*(L*L) + (pi*pi)*D11*(i2*i2))/L
                c += 1
                k0r[c] = row+3
                k0c[c] = col+4
                k0v[c] += 0.5*r*(tmax - tmin)*(A45*(L*L) + (pi*pi)*D16*(i2*i2))/L
                c += 1
                k0r[c] = row+4
                k0c[c] = col+0
                k0v[c] += 0.5*(pi*pi)*B16*(i2*i2)*r*(tmax - tmin)/L
                c += 1
                k0r[c] = row+4
                k0c[c] = col+1
                k0v[c] += 0.5*(tmax - tmin)*(-A44*(L*L) + (pi*pi)*B66*(i2*i2)*r)/L
                c += 1
                k0r[c] = row+4
                k0c[c] = col+3
                k0v[c] += 0.5*r*(tmax - tmin)*(A45*(L*L) + (pi*pi)*D16*(i2*i2))/L
                c += 1
                k0r[c] = row+4
                k0c[c] = col+4
                k0v[c] += 0.5*r*(tmax - tmin)*(A44*(L*L) + (pi*pi)*D66*(i2*i2))/L

            else:
                # k0_22 cond_2
                c += 1
                k0r[c] = row+0
                k0c[c] = col+2
                k0v[c] += A12*(i2*i2)*((-1)**(i2 + k2) - 1)*(tmax - tmin)/((i2*i2) - (k2*k2))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+2
                k0v[c] += ((-1)**(i2 + k2) - 1)*(tmax - tmin)*(A26*(i2*i2) + A45*(k2*k2))/((i2 - k2)*(i2 + k2))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+0
                k0v[c] += A12*(k2*k2)*((-1)**(i2 + k2) - 1)*(tmax - tmin)/(-(i2*i2) + (k2*k2))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+1
                k0v[c] += -((-1)**(i2 + k2) - 1)*(tmax - tmin)*(A26*(k2*k2) + A45*(i2*i2))/((i2 - k2)*(i2 + k2))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+3
                k0v[c] += ((-1)**(i2 + k2) - 1)*(tmax - tmin)*(A55*(i2*i2)*r - B12*(k2*k2))/((i2 - k2)*(i2 + k2))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+4
                k0v[c] += ((-1)**(i2 + k2) - 1)*(tmax - tmin)*(A45*(i2*i2)*r - B26*(k2*k2))/((i2 - k2)*(i2 + k2))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+2
                k0v[c] += ((-1)**(i2 + k2) - 1)*(tmax - tmin)*(-A55*(k2*k2)*r + B12*(i2*i2))/((i2 - k2)*(i2 + k2))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+2
                k0v[c] += ((-1)**(i2 + k2) - 1)*(tmax - tmin)*(-A45*(k2*k2)*r + B26*(i2*i2))/((i2 - k2)*(i2 + k2))

    # k0_23
    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for l3 in range(1, n3+1):
            col = num0 + num1 + num2*m2 + num3*(l3-1)
            c += 1
            k0r[c] = row+0
            k0c[c] = col+0
            k0v[c] += A16*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+0
            k0c[c] = col+1
            k0v[c] += A12*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+0
            k0c[c] = col+3
            k0v[c] += B16*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+0
            k0c[c] = col+4
            k0v[c] += B12*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+1
            k0c[c] = col+0
            k0v[c] += A66*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+1
            k0c[c] = col+1
            k0v[c] += A26*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+1
            k0c[c] = col+3
            k0v[c] += B66*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+1
            k0c[c] = col+4
            k0v[c] += B26*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+2
            k0c[c] = col+2
            k0v[c] += A45*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+3
            k0c[c] = col+0
            k0v[c] += B16*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+3
            k0c[c] = col+1
            k0v[c] += B12*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+3
            k0c[c] = col+3
            k0v[c] += D16*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+3
            k0c[c] = col+4
            k0v[c] += D12*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+4
            k0c[c] = col+0
            k0v[c] += B66*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+4
            k0c[c] = col+1
            k0v[c] += B26*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+4
            k0c[c] = col+3
            k0v[c] += D66*((-1)**i2 - 1)*((-1)**l3 - 1)
            c += 1
            k0r[c] = row+4
            k0c[c] = col+4
            k0v[c] += D26*((-1)**i2 - 1)*((-1)**l3 - 1)

    # k0_24
    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for k4 in range(1, m4+1):
            for l4 in range(1, n4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 +
                        num4*((l4-1)*m4 + (k4-1)))
                if k4 == i2:
                    # k0_24 cond_1
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+2
                    k0v[c] += (-1)**(l4 - 1)*A44*L/r
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+0
                    k0v[c] += -(-1)**(l4 - 1)*A26*L/r
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+1
                    k0v[c] += -(-1)**(l4 - 1)*A22*L/r
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+3
                    k0v[c] += -(-1)**(l4 - 1)*B26*L/r
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+4
                    k0v[c] += -(-1)**(l4 - 1)*B22*L/r
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+2
                    k0v[c] += -(-1)**(l4 - 1)*A45*L
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+2
                    k0v[c] += -(-1)**(l4 - 1)*A44*L

                else:
                    # k0_24 cond_2
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += (-1)**(l4 - 1)*A16*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+1
                    k0v[c] += (-1)**(l4 - 1)*A12*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+3
                    k0v[c] += (-1)**(l4 - 1)*B16*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+4
                    k0v[c] += (-1)**(l4 - 1)*B12*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += (-1)**(l4 - 1)*A66*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += A26*(i2*i2)*((-1)**l4 - 1)*((-1)**(i2 + k4) - 1)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+3
                    k0v[c] += B66*(i2*i2)*((-1)**l4 - 1)*((-1)**(i2 + k4) - 1)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+4
                    k0v[c] += B26*(i2*i2)*((-1)**l4 - 1)*((-1)**(i2 + k4) - 1)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += A45*(i2*i2)*((-1)**l4 - 1)*((-1)**(i2 + k4) - 1)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+0
                    k0v[c] += (-1)**(l4 - 1)*B16*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+1
                    k0v[c] += B12*(i2*i2)*((-1)**l4 - 1)*((-1)**(i2 + k4) - 1)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+3
                    k0v[c] += D16*(i2*i2)*((-1)**l4 - 1)*((-1)**(i2 + k4) - 1)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+4
                    k0v[c] += D12*(i2*i2)*((-1)**l4 - 1)*((-1)**(i2 + k4) - 1)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+0
                    k0v[c] += (-1)**(l4 - 1)*B66*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+1
                    k0v[c] += B26*(i2*i2)*((-1)**l4 - 1)*((-1)**(i2 + k4) - 1)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+3
                    k0v[c] += (-1)**(l4 - 1)*D66*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+4
                    k0v[c] += (-1)**(l4 - 1)*D26*(i2*i2)*(-2*(-1)**(i2 + k4) + 2)/((i2*i2) - (k4*k4))

    # k0_33
    for j3 in range(1, n3+1):
        row = num0 + num1 + num2*m2 + num3*(j3-1)
        for l3 in range(1, n3+1):
            col = num0 + num1 + num2*m2 + num3*(l3-1)
            if l3 == j3:
                # k0_33 cond_1
                c += 1
                k0r[c] = row+0
                k0c[c] = col+0
                k0v[c] += (pi*pi)*A66*L*(j3*j3)/(2*r*tmax - 2*r*tmin)
                c += 1
                k0r[c] = row+0
                k0c[c] = col+1
                k0v[c] += (pi*pi)*A26*L*(j3*j3)/(2*r*tmax - 2*r*tmin)
                c += 1
                k0r[c] = row+0
                k0c[c] = col+3
                k0v[c] += (pi*pi)*B66*L*(j3*j3)/(2*r*tmax - 2*r*tmin)
                c += 1
                k0r[c] = row+0
                k0c[c] = col+4
                k0v[c] += (pi*pi)*B26*L*(j3*j3)/(2*r*tmax - 2*r*tmin)
                c += 1
                k0r[c] = row+1
                k0c[c] = col+0
                k0v[c] += (pi*pi)*A26*L*(j3*j3)/(2*r*tmax - 2*r*tmin)
                c += 1
                k0r[c] = row+1
                k0c[c] = col+1
                k0v[c] += 0.5*L*((pi*pi)*A22*(j3*j3) + A44*(tmax - tmin)**2)/(r*(tmax - tmin))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+3
                k0v[c] += L*(-0.5*A45*(tmax - tmin) + (pi*pi)*B26*(j3*j3)/(2*r*tmax - 2*r*tmin))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+4
                k0v[c] += L*(-0.5*A44*(tmax - tmin) + (pi*pi)*B22*(j3*j3)/(2*r*tmax - 2*r*tmin))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+2
                k0v[c] += 0.5*L*(A22*(tmax - tmin)**2 + (pi*pi)*A44*(j3*j3))/(r*(tmax - tmin))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+0
                k0v[c] += (pi*pi)*B66*L*(j3*j3)/(2*r*tmax - 2*r*tmin)
                c += 1
                k0r[c] = row+3
                k0c[c] = col+1
                k0v[c] += L*(-0.5*A45*(tmax - tmin) + (pi*pi)*B26*(j3*j3)/(2*r*tmax - 2*r*tmin))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+3
                k0v[c] += L*(0.5*A55*r*(tmax - tmin) + (pi*pi)*D66*(j3*j3)/(2*r*tmax - 2*r*tmin))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+4
                k0v[c] += L*(0.5*A45*r*(tmax - tmin) + (pi*pi)*D26*(j3*j3)/(2*r*tmax - 2*r*tmin))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+0
                k0v[c] += (pi*pi)*B26*L*(j3*j3)/(2*r*tmax - 2*r*tmin)
                c += 1
                k0r[c] = row+4
                k0c[c] = col+1
                k0v[c] += L*(-0.5*A44*(tmax - tmin) + (pi*pi)*B22*(j3*j3)/(2*r*tmax - 2*r*tmin))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+3
                k0v[c] += L*(0.5*A45*r*(tmax - tmin) + (pi*pi)*D26*(j3*j3)/(2*r*tmax - 2*r*tmin))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+4
                k0v[c] += L*(0.5*A44*r*(tmax - tmin) + (pi*pi)*D22*(j3*j3)/(2*r*tmax - 2*r*tmin))

            else:
                # k0_33 cond_2
                c += 1
                k0r[c] = row+0
                k0c[c] = col+2
                k0v[c] += A26*L*(j3*j3)*((-1)**(j3 + l3) - 1)/(r*(j3 - l3)*(j3 + l3))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+2
                k0v[c] += L*((-1)**(j3 + l3) - 1)*(A22*(j3*j3) + A44*(l3*l3))/(r*(j3 - l3)*(j3 + l3))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+0
                k0v[c] += -A26*L*(l3*l3)*((-1)**(j3 + l3) - 1)/(r*(j3 - l3)*(j3 + l3))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+1
                k0v[c] += -L*((-1)**(j3 + l3) - 1)*(A22*(l3*l3) + A44*(j3*j3))/(r*(j3 - l3)*(j3 + l3))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+3
                k0v[c] += L*((-1)**(j3 + l3) - 1)*(A45*(j3*j3)*r - B26*(l3*l3))/(r*(j3 - l3)*(j3 + l3))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+4
                k0v[c] += L*((-1)**(j3 + l3) - 1)*(A44*(j3*j3)*r - B22*(l3*l3))/(r*(j3 - l3)*(j3 + l3))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+2
                k0v[c] += L*((-1)**(j3 + l3) - 1)*(-A45*(l3*l3)*r + B26*(j3*j3))/(r*(j3 - l3)*(j3 + l3))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+2
                k0v[c] += L*((-1)**(j3 + l3) - 1)*(-A44*(l3*l3)*r + B22*(j3*j3))/(r*(j3 - l3)*(j3 + l3))

    # k0_34
    for j3 in range(1, n3+1):
        row = num0 + num1 + num2*m2 + num3*(j3-1)
        for k4 in range(1, m4+1):
            for l4 in range(1, n4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 +
                        num4*((l4-1)*m4 + (k4-1)))
                if l4 == j3:
                    # k0_34 cond_1
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+2
                    k0v[c] += -0.5*A45*((-1)**k4 - 1)*(tmax - tmin)
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+0
                    k0v[c] += (-1)**(k4 - 1)*A12*(-tmax + tmin)
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+1
                    k0v[c] += 0.5*A26*((-1)**k4 - 1)*(tmax - tmin)
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+3
                    k0v[c] += 0.5*B12*((-1)**k4 - 1)*(tmax - tmin)
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+4
                    k0v[c] += 0.5*B26*((-1)**k4 - 1)*(tmax - tmin)
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+2
                    k0v[c] += (-1)**(k4 - 1)*A55*r*(-tmax + tmin)
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+2
                    k0v[c] += 0.5*A45*r*((-1)**k4 - 1)*(tmax - tmin)

                else:
                    # k0_34 cond_2
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += A16*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+1
                    k0v[c] += A66*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+3
                    k0v[c] += B16*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+4
                    k0v[c] += B66*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += A12*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += A26*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+3
                    k0v[c] += B12*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+4
                    k0v[c] += B26*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += A45*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+0
                    k0v[c] += B16*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+1
                    k0v[c] += B66*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+3
                    k0v[c] += D16*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+4
                    k0v[c] += D66*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+0
                    k0v[c] += B12*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+1
                    k0v[c] += B26*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+3
                    k0v[c] += D12*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+4
                    k0v[c] += D26*(j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)/((j3*j3) - (l4*l4))

    # k0_44
    for i4 in range(1, m4+1):
        for j4 in range(1, n4+1):
            row = (num0 + num1 + num2*m2 + num3*n3 +
                    num4*((j4-1)*m4 + (i4-1)))
            for k4 in range(1, m4+1):
                for l4 in range(1, n4+1):
                    col = (num0 + num1 + num2*m2 + num3*n3 +
                            num4*((l4-1)*m4 + (k4-1)))
                    if k4 == i4 and l4 == j4:
                        # k0_44 cond_1
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += 0.25*(pi*pi)*(A11*(i4*i4)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j4*j4))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += 0.25*(pi*pi)*(A16*(i4*i4)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j4*j4))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += 0.25*(pi*pi)*(B11*(i4*i4)*(r*r)*(tmax - tmin)**2 + B66*(L*L)*(j4*j4))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += 0.25*(pi*pi)*(B16*(i4*i4)*(r*r)*(tmax - tmin)**2 + B26*(L*L)*(j4*j4))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += 0.25*(pi*pi)*(A16*(i4*i4)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j4*j4))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += 0.25*((pi*pi)*A22*(L*L)*(j4*j4) + (tmax - tmin)**2*(A44*(L*L) + (pi*pi)*A66*(i4*i4)*(r*r)))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += 0.25*((pi*pi)*B26*(L*L)*(j4*j4) + r*(tmax - tmin)**2*(-A45*(L*L) + (pi*pi)*B16*(i4*i4)*r))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += 0.25*((pi*pi)*B22*(L*L)*(j4*j4) + r*(tmax - tmin)**2*(-A44*(L*L) + (pi*pi)*B66*(i4*i4)*r))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += 0.25*((pi*pi)*A44*(L*L)*(j4*j4) + (tmax - tmin)**2*(A22*(L*L) + (pi*pi)*A55*(i4*i4)*(r*r)))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += 0.25*(pi*pi)*(B11*(i4*i4)*(r*r)*(tmax - tmin)**2 + B66*(L*L)*(j4*j4))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += 0.25*((pi*pi)*B26*(L*L)*(j4*j4) + r*(tmax - tmin)**2*(-A45*(L*L) + (pi*pi)*B16*(i4*i4)*r))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += 0.25*((pi*pi)*D66*(L*L)*(j4*j4) + (r*r)*(tmax - tmin)**2*(A55*(L*L) + (pi*pi)*D11*(i4*i4)))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += 0.25*((pi*pi)*D26*(L*L)*(j4*j4) + (r*r)*(tmax - tmin)**2*(A45*(L*L) + (pi*pi)*D16*(i4*i4)))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += 0.25*(pi*pi)*(B16*(i4*i4)*(r*r)*(tmax - tmin)**2 + B26*(L*L)*(j4*j4))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += 0.25*((pi*pi)*B22*(L*L)*(j4*j4) + r*(tmax - tmin)**2*(-A44*(L*L) + (pi*pi)*B66*(i4*i4)*r))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += 0.25*((pi*pi)*D26*(L*L)*(j4*j4) + (r*r)*(tmax - tmin)**2*(A45*(L*L) + (pi*pi)*D16*(i4*i4)))/(L*r*(tmax - tmin))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += 0.25*((pi*pi)*D22*(L*L)*(j4*j4) + (r*r)*(tmax - tmin)**2*(A44*(L*L) + (pi*pi)*D66*(i4*i4)))/(L*r*(tmax - tmin))

                    elif k4 != i4 and l4 == j4:
                        # k0_44 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += A12*(i4*i4)*((-1)**(i4 + k4) - 1)*(tmax - tmin)/(2.0*(i4*i4) - 2.0*(k4*k4))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += ((-1)**(i4 + k4) - 1)*(tmax - tmin)*(A26*(i4*i4) + A45*(k4*k4))/((i4 + k4)*(2.0*i4 - 2.0*k4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += A12*(k4*k4)*((-1)**(i4 + k4) - 1)*(tmax - tmin)/(-2.0*(i4*i4) + 2.0*(k4*k4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -((-1)**(i4 + k4) - 1)*(tmax - tmin)*(A26*(k4*k4) + A45*(i4*i4))/((i4 + k4)*(2.0*i4 - 2.0*k4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += ((-1)**(i4 + k4) - 1)*(tmax - tmin)*(A55*(i4*i4)*r - B12*(k4*k4))/((i4 + k4)*(2.0*i4 - 2.0*k4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += ((-1)**(i4 + k4) - 1)*(tmax - tmin)*(A45*(i4*i4)*r - B26*(k4*k4))/((i4 + k4)*(2.0*i4 - 2.0*k4))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += ((-1)**(i4 + k4) - 1)*(tmax - tmin)*(-A55*(k4*k4)*r + B12*(i4*i4))/((i4 + k4)*(2.0*i4 - 2.0*k4))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += ((-1)**(i4 + k4) - 1)*(tmax - tmin)*(-A45*(k4*k4)*r + B26*(i4*i4))/((i4 + k4)*(2.0*i4 - 2.0*k4))

                    elif k4 != i4 and l4 != j4:
                        # k0_44 cond_3
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += -A16*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*(A12*(i4*i4)*(l4*l4) + A66*(j4*j4)*(k4*k4))/((i4 - k4)*(i4 + k4)*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += -B16*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*(B12*(i4*i4)*(l4*l4) + B66*(j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*(A12*(j4*j4)*(k4*k4) + A66*(i4*i4)*(l4*l4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += -A26*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*(B12*(j4*j4)*(k4*k4) + B66*(i4*i4)*(l4*l4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += -B26*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += -A45*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += -B16*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*(B12*(i4*i4)*(l4*l4) + B66*(j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += -D16*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*(D12*(i4*i4)*(l4*l4) + D66*(j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*(B12*(j4*j4)*(k4*k4) + B66*(i4*i4)*(l4*l4))/((i4 - k4)*(i4 + k4)*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += -B26*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*(D12*(j4*j4)*(k4*k4) + D66*(i4*i4)*(l4*l4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += -D26*((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))/(((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4))

                    elif k4 == i4 and l4 != j4:
                        # k0_44 cond_4
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += A26*L*(j4*j4)*((-1)**(j4 + l4) - 1)/(r*(j4 + l4)*(2.0*j4 - 2.0*l4))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += L*((-1)**(j4 + l4) - 1)*(A22*(j4*j4) + A44*(l4*l4))/(r*(j4 + l4)*(2.0*j4 - 2.0*l4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += -A26*L*(l4*l4)*((-1)**(j4 + l4) - 1)/(r*(j4 + l4)*(2.0*j4 - 2.0*l4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -L*((-1)**(j4 + l4) - 1)*(A22*(l4*l4) + A44*(j4*j4))/(r*(j4 + l4)*(2.0*j4 - 2.0*l4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += L*((-1)**(j4 + l4) - 1)*(A45*(j4*j4)*r - B26*(l4*l4))/(r*(j4 + l4)*(2.0*j4 - 2.0*l4))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += L*((-1)**(j4 + l4) - 1)*(A44*(j4*j4)*r - B22*(l4*l4))/(r*(j4 + l4)*(2.0*j4 - 2.0*l4))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += L*((-1)**(j4 + l4) - 1)*(-A45*(l4*l4)*r + B26*(j4*j4))/(r*(j4 + l4)*(2.0*j4 - 2.0*l4))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += L*((-1)**(j4 + l4) - 1)*(-A44*(l4*l4)*r + B22*(j4*j4))/(r*(j4 + l4)*(2.0*j4 - 2.0*l4))

    size = num0 + num1 + num2*m2 + num3*n3 + num4*m4*n4

    k0 = csr_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0edges(int m2, int n3, int m4, int n4, double r1, double r2, double L,
             double tmin, double tmax,
             double kuBot, double kuTop,
             double kvBot, double kvTop,
             double kwBot, double kwTop,
             double kphixBot, double kphixTop,
             double kphitBot, double kphitTop,
             double kuLeft, double kuRight,
             double kvLeft, double kvRight,
             double kwLeft, double kwRight,
             double kphixLeft, double kphixRight,
             double kphitLeft, double kphitRight):
    cdef int i2, k2, j3, l3, i4, j4, k4, l4, row, col, c
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv

    fdim = (5 + 5*m2 + 5*m2*m2 + 5*n3*n3 + 5*n3*m4*n4 + 5*m4*n4*m4*n4
           + 5 + 5*n3 + 5*m2*m2 + 5*m2*m4*n4 + 5*n3*n3 + 4*m4*n4*m4*n4)

    k0edgesr = np.zeros((fdim,), dtype=INT)
    k0edgesc = np.zeros((fdim,), dtype=INT)
    k0edgesv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    row = num0
    col = num0
    # k0edgesBT_11
    c += 1
    k0edgesr[c] = row+0
    k0edgesc[c] = col+0
    k0edgesv[c] += (tmax - tmin)*(kuBot*r1 + kuTop*r2)
    c += 1
    k0edgesr[c] = row+1
    k0edgesc[c] = col+1
    k0edgesv[c] += (tmax - tmin)*(kvBot*r1 + kvTop*r2)
    c += 1
    k0edgesr[c] = row+2
    k0edgesc[c] = col+2
    k0edgesv[c] += (tmax - tmin)*(kwBot*r1 + kwTop*r2)
    c += 1
    k0edgesr[c] = row+3
    k0edgesc[c] = col+3
    k0edgesv[c] += (tmax - tmin)*(kphixBot*r1 + kphixTop*r2)
    c += 1
    k0edgesr[c] = row+4
    k0edgesc[c] = col+4
    k0edgesv[c] += (tmax - tmin)*(kphitBot*r1 + kphitTop*r2)

    row = num0
    for k2 in range(1, m2+1):
        col = num0 + num1 + num2*(k2-1)
        # k0edgesBT_12
        c += 1
        k0edgesr[c] = row+0
        k0edgesc[c] = col+0
        k0edgesv[c] += (tmax - tmin)*((-1)**k2*kuTop*r2 + kuBot*r1)
        c += 1
        k0edgesr[c] = row+1
        k0edgesc[c] = col+1
        k0edgesv[c] += (tmax - tmin)*((-1)**k2*kvTop*r2 + kvBot*r1)
        c += 1
        k0edgesr[c] = row+2
        k0edgesc[c] = col+2
        k0edgesv[c] += (tmax - tmin)*((-1)**k2*kwTop*r2 + kwBot*r1)
        c += 1
        k0edgesr[c] = row+3
        k0edgesc[c] = col+3
        k0edgesv[c] += (tmax - tmin)*((-1)**k2*kphixTop*r2 + kphixBot*r1)
        c += 1
        k0edgesr[c] = row+4
        k0edgesc[c] = col+4
        k0edgesv[c] += (tmax - tmin)*((-1)**k2*kphitTop*r2 + kphitBot*r1)

    # k0edgesBT_13
    pass

    # k0edgesBT_14
    pass

    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for k2 in range(1, m2+1):
            col = num0 + num1 + num2*(k2-1)
            if k2 == i2:
                # k0edgesBT_22 cond_1
                c += 1
                k0edgesr[c] = row+0
                k0edgesc[c] = col+0
                k0edgesv[c] += (tmax - tmin)*(kuBot*r1 + kuTop*r2)
                c += 1
                k0edgesr[c] = row+1
                k0edgesc[c] = col+1
                k0edgesv[c] += (tmax - tmin)*(kvBot*r1 + kvTop*r2)
                c += 1
                k0edgesr[c] = row+2
                k0edgesc[c] = col+2
                k0edgesv[c] += (tmax - tmin)*(kwBot*r1 + kwTop*r2)
                c += 1
                k0edgesr[c] = row+3
                k0edgesc[c] = col+3
                k0edgesv[c] += (tmax - tmin)*(kphixBot*r1 + kphixTop*r2)
                c += 1
                k0edgesr[c] = row+4
                k0edgesc[c] = col+4
                k0edgesv[c] += (tmax - tmin)*(kphitBot*r1 + kphitTop*r2)

            else:
                # k0edgesBT_22 cond_2
                c += 1
                k0edgesr[c] = row+0
                k0edgesc[c] = col+0
                k0edgesv[c] += (tmax - tmin)*((-1)**(i2 + k2)*kuTop*r2 + kuBot*r1)
                c += 1
                k0edgesr[c] = row+1
                k0edgesc[c] = col+1
                k0edgesv[c] += (tmax - tmin)*((-1)**(i2 + k2)*kvTop*r2 + kvBot*r1)
                c += 1
                k0edgesr[c] = row+2
                k0edgesc[c] = col+2
                k0edgesv[c] += (tmax - tmin)*((-1)**(i2 + k2)*kwTop*r2 + kwBot*r1)
                c += 1
                k0edgesr[c] = row+3
                k0edgesc[c] = col+3
                k0edgesv[c] += (tmax - tmin)*((-1)**(i2 + k2)*kphixTop*r2 + kphixBot*r1)
                c += 1
                k0edgesr[c] = row+4
                k0edgesc[c] = col+4
                k0edgesv[c] += (tmax - tmin)*((-1)**(i2 + k2)*kphitTop*r2 + kphitBot*r1)

    # k0edgesBT_23
    pass

    # k0edgesBT_24
    pass

    for j3 in range(1, n3+1):
        row = num0 + num1 + num2*m2 + num3*(j3-1)
        for l3 in range(1, n3+1):
            col = num0 + num1 + num2*m2 + num3*(l3-1)
            if l3 == j3:
                # k0edgesBT_33 cond_1
                c += 1
                k0edgesr[c] = row+0
                k0edgesc[c] = col+0
                k0edgesv[c] += 0.5*(tmax - tmin)*(kuBot*r1 + kuTop*r2)
                c += 1
                k0edgesr[c] = row+1
                k0edgesc[c] = col+1
                k0edgesv[c] += 0.5*(tmax - tmin)*(kvBot*r1 + kvTop*r2)
                c += 1
                k0edgesr[c] = row+2
                k0edgesc[c] = col+2
                k0edgesv[c] += 0.5*(tmax - tmin)*(kwBot*r1 + kwTop*r2)
                c += 1
                k0edgesr[c] = row+3
                k0edgesc[c] = col+3
                k0edgesv[c] += 0.5*(tmax - tmin)*(kphixBot*r1 + kphixTop*r2)
                c += 1
                k0edgesr[c] = row+4
                k0edgesc[c] = col+4
                k0edgesv[c] += 0.5*(tmax - tmin)*(kphitBot*r1 + kphitTop*r2)

            else:
                # k0edgesBT_33 cond_2
                pass

    for j3 in range(1, n3+1):
        row = num0 + num1 + num2*m2 + num3*(j3-1)
        for k4 in range(1, m4+1):
            for l4 in range(1, n4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 +
                        num4*((l4-1)*m4 + (k4-1)))
                if l4 == j3:
                    # k0edgesBT_34 cond_1
                    c += 1
                    k0edgesr[c] = row+0
                    k0edgesc[c] = col+0
                    k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**k4*kuTop*r2 + kuBot*r1)
                    c += 1
                    k0edgesr[c] = row+1
                    k0edgesc[c] = col+1
                    k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**k4*kvTop*r2 + kvBot*r1)
                    c += 1
                    k0edgesr[c] = row+2
                    k0edgesc[c] = col+2
                    k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**k4*kwTop*r2 + kwBot*r1)
                    c += 1
                    k0edgesr[c] = row+3
                    k0edgesc[c] = col+3
                    k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**k4*kphixTop*r2 + kphixBot*r1)
                    c += 1
                    k0edgesr[c] = row+4
                    k0edgesc[c] = col+4
                    k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**k4*kphitTop*r2 + kphitBot*r1)

                else:
                    # k0edgesBT_34 cond_2
                    pass

    for i4 in range(1, m4+1):
        for j4 in range(1, n4+1):
            row = (num0 + num1 + num2*m2 + num3*n3 +
                    num4*((j4-1)*m4 + (i4-1)))
            for k4 in range(1, m4+1):
                for l4 in range(1, n4+1):
                    col = (num0 + num1 + num2*m2 + num3*n3 +
                            num4*((l4-1)*m4 + (k4-1)))
                    if k4 == i4 and l4 == j4:
                        # k0edgesBT_44 cond_1
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kvBot*r1 + kvTop*r2)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kwBot*r1 + kwTop*r2)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kphixBot*r1 + kphixTop*r2)
                        c += 1
                        k0edgesr[c] = row+4
                        k0edgesc[c] = col+4
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kphitBot*r1 + kphitTop*r2)

                    elif k4 != i4 and l4 == j4:
                        # k0edgesBT_44 cond_2
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i4 + k4)*kuTop*r2 + kuBot*r1)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i4 + k4)*kvTop*r2 + kvBot*r1)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i4 + k4)*kwTop*r2 + kwBot*r1)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i4 + k4)*kphixTop*r2 + kphixBot*r1)
                        c += 1
                        k0edgesr[c] = row+4
                        k0edgesc[c] = col+4
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i4 + k4)*kphitTop*r2 + kphitBot*r1)

                    elif k4 != i4 and l4 != j4:
                        # k0edgesBT_44 cond_3
                        pass

                    elif k4 == i4 and l4 != j4:
                        # k0edgesBT_44 cond_4
                        pass

    row = num0
    col = num0
    # k0edgesLR_11
    c += 1
    k0edgesr[c] = row+0
    k0edgesc[c] = col+0
    k0edgesv[c] += L*(kuLeft + kuRight)
    c += 1
    k0edgesr[c] = row+1
    k0edgesc[c] = col+1
    k0edgesv[c] += L*(kvLeft + kvRight)
    c += 1
    k0edgesr[c] = row+2
    k0edgesc[c] = col+2
    k0edgesv[c] += L*(kwLeft + kwRight)
    c += 1
    k0edgesr[c] = row+3
    k0edgesc[c] = col+3
    k0edgesv[c] += L*(kphixLeft + kphixRight)
    c += 1
    k0edgesr[c] = row+4
    k0edgesc[c] = col+4
    k0edgesv[c] += L*(kphitLeft + kphitRight)

    # k0edgesLR_12
    pass

    row = num0
    for l3 in range(1, n3+1):
        col = num0 + num1 + num2*m2 + num3*(l3-1)
        # k0edgesLR_13
        c += 1
        k0edgesr[c] = row+0
        k0edgesc[c] = col+0
        k0edgesv[c] += L*((-1)**l3*kuLeft + kuRight)
        c += 1
        k0edgesr[c] = row+1
        k0edgesc[c] = col+1
        k0edgesv[c] += L*((-1)**l3*kvLeft + kvRight)
        c += 1
        k0edgesr[c] = row+2
        k0edgesc[c] = col+2
        k0edgesv[c] += L*((-1)**l3*kwLeft + kwRight)
        c += 1
        k0edgesr[c] = row+3
        k0edgesc[c] = col+3
        k0edgesv[c] += L*((-1)**l3*kphixLeft + kphixRight)
        c += 1
        k0edgesr[c] = row+4
        k0edgesc[c] = col+4
        k0edgesv[c] += L*((-1)**l3*kphitLeft + kphitRight)

    # k0edgesLR_14
    pass

    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for k2 in range(1, m2+1):
            col = num0 + num1 + num2*(k2-1)
            if k2 == i2:
                # k0edgesLR_22 cond_1
                c += 1
                k0edgesr[c] = row+0
                k0edgesc[c] = col+0
                k0edgesv[c] += 0.5*L*(kuLeft + kuRight)
                c += 1
                k0edgesr[c] = row+1
                k0edgesc[c] = col+1
                k0edgesv[c] += 0.5*L*(kvLeft + kvRight)
                c += 1
                k0edgesr[c] = row+2
                k0edgesc[c] = col+2
                k0edgesv[c] += 0.5*L*(kwLeft + kwRight)
                c += 1
                k0edgesr[c] = row+3
                k0edgesc[c] = col+3
                k0edgesv[c] += 0.5*L*(kphixLeft + kphixRight)
                c += 1
                k0edgesr[c] = row+4
                k0edgesc[c] = col+4
                k0edgesv[c] += 0.5*L*(kphitLeft + kphitRight)

            else:
                # k0edgesLR_22 cond_2
                pass

    # k0edgesLR_23
    pass

    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for k4 in range(1, m4+1):
            for l4 in range(1, n4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 +
                        num4*((l4-1)*m4 + (k4-1)))
                if k4 == i2:
                    # k0edgesLR_24 cond_1
                    c += 1
                    k0edgesr[c] = row+0
                    k0edgesc[c] = col+0
                    k0edgesv[c] += 0.5*L*((-1)**l4*kuLeft + kuRight)
                    c += 1
                    k0edgesr[c] = row+1
                    k0edgesc[c] = col+1
                    k0edgesv[c] += 0.5*L*((-1)**l4*kvLeft + kvRight)
                    c += 1
                    k0edgesr[c] = row+2
                    k0edgesc[c] = col+2
                    k0edgesv[c] += 0.5*L*((-1)**l4*kwLeft + kwRight)
                    c += 1
                    k0edgesr[c] = row+3
                    k0edgesc[c] = col+3
                    k0edgesv[c] += 0.5*L*((-1)**l4*kphixLeft + kphixRight)
                    c += 1
                    k0edgesr[c] = row+4
                    k0edgesc[c] = col+4
                    k0edgesv[c] += 0.5*L*((-1)**l4*kphitLeft + kphitRight)

                else:
                    # k0edgesLR_24 cond_2
                    pass

    for j3 in range(1, n3+1):
        row = num0 + num1 + num2*m2 + num3*(j3-1)
        for l3 in range(1, n3+1):
            col = num0 + num1 + num2*m2 + num3*(l3-1)
            if l3 == j3:
                # k0edgesLR_33 cond_1
                c += 1
                k0edgesr[c] = row+0
                k0edgesc[c] = col+0
                k0edgesv[c] += L*(kuLeft + kuRight)
                c += 1
                k0edgesr[c] = row+1
                k0edgesc[c] = col+1
                k0edgesv[c] += L*(kvLeft + kvRight)
                c += 1
                k0edgesr[c] = row+2
                k0edgesc[c] = col+2
                k0edgesv[c] += L*(kwLeft + kwRight)
                c += 1
                k0edgesr[c] = row+3
                k0edgesc[c] = col+3
                k0edgesv[c] += L*(kphixLeft + kphixRight)
                c += 1
                k0edgesr[c] = row+4
                k0edgesc[c] = col+4
                k0edgesv[c] += L*(kphitLeft + kphitRight)

            else:
                # k0edgesLR_33 cond_2
                c += 1
                k0edgesr[c] = row+0
                k0edgesc[c] = col+0
                k0edgesv[c] += L*((-1)**(j3 + l3)*kuLeft + kuRight)
                c += 1
                k0edgesr[c] = row+1
                k0edgesc[c] = col+1
                k0edgesv[c] += L*((-1)**(j3 + l3)*kvLeft + kvRight)
                c += 1
                k0edgesr[c] = row+2
                k0edgesc[c] = col+2
                k0edgesv[c] += L*((-1)**(j3 + l3)*kwLeft + kwRight)
                c += 1
                k0edgesr[c] = row+3
                k0edgesc[c] = col+3
                k0edgesv[c] += L*((-1)**(j3 + l3)*kphixLeft + kphixRight)
                c += 1
                k0edgesr[c] = row+4
                k0edgesc[c] = col+4
                k0edgesv[c] += L*((-1)**(j3 + l3)*kphitLeft + kphitRight)

    # k0edgesLR_34
    pass

    for i4 in range(1, m4+1):
        for j4 in range(1, n4+1):
            row = (num0 + num1 + num2*m2 + num3*n3 +
                    num4*((j4-1)*m4 + (i4-1)))
            for k4 in range(1, m4+1):
                for l4 in range(1, n4+1):
                    col = (num0 + num1 + num2*m2 + num3*n3 +
                            num4*((l4-1)*m4 + (k4-1)))
                    if k4 == i4 and l4 == j4:
                        # k0edgesLR_44 cond_1
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*L*(kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*L*(kvLeft + kvRight)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*L*(kwLeft + kwRight)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += 0.5*L*(kphixLeft + kphixRight)
                        c += 1
                        k0edgesr[c] = row+4
                        k0edgesc[c] = col+4
                        k0edgesv[c] += 0.5*L*(kphitLeft + kphitRight)

                    elif k4 != i4 and l4 == j4:
                        # k0edgesLR_44 cond_2
                        pass

                    elif k4 != i4 and l4 != j4:
                        # k0edgesLR_44 cond_3
                        pass

                    elif k4 == i4 and l4 != j4:
                        # k0edgesLR_44 cond_4
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*L*((-1)**(j4 + l4)*kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*L*((-1)**(j4 + l4)*kvLeft + kvRight)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*L*((-1)**(j4 + l4)*kwLeft + kwRight)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += 0.5*L*((-1)**(j4 + l4)*kphixLeft + kphixRight)
                        c += 1
                        k0edgesr[c] = row+4
                        k0edgesc[c] = col+4
                        k0edgesv[c] += 0.5*L*((-1)**(j4 + l4)*kphitLeft + kphitRight)

    size = num0 + num1 + num2*m2 + num3*n3 + num4*m4*n4

    k0edges = csr_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fx, double Ft, double Fxt, double Ftx, double r1, double L,
        double tmin, double tmax, int m2, int n3, int m4, int n4,
        double alpharad, int s):
    cdef int i2, k2, j3, l3, i4, j4, k4, l4, c, row, col, section
    cdef double xa, xb, r, sina

    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    fdim = 1*m4*n4*m4*n4

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)

    with nogil:
        for section in range(s):
            c = -1

            xa = -L/2. + L*float(section)/s
            xb = -L/2. + L*float(section+1)/s

            r = r1 - sina*((xa+xb)/2. + L/2.)

            # kG0_44
            for i4 in range(1, m4+1):
                for j4 in range(1, n4+1):
                    row = (num0 + num1 + num2*m2 + num3*n3 +
                            num4*((j4-1)*m4 + (i4-1)))
                    for k4 in range(1, m4+1):
                        for l4 in range(1, n4+1):
                            col = (num0 + num1 + num2*m2 + num3*n3 +
                                    num4*((l4-1)*m4 + (k4-1)))
                            if k4 == i4 and l4 == j4:
                                # kG0_44 cond_1
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+2
                                kG0v[c] += 0.125*pi*(L*(Ft*L*(j4*j4) + Fx*(i4*i4)*r*(-tmax + tmin))*sin(pi*i4*(L + 2*xa)/L) - L*(Ft*L*(j4*j4) + Fx*(i4*i4)*r*(-tmax + tmin))*sin(pi*i4*(L + 2*xb)/L) - 2*pi*i4*(xa - xb)*(Ft*L*(j4*j4) + Fx*(i4*i4)*r*(tmax - tmin)))/((L*L)*i4*r*(tmax - tmin))
                            elif k4 != i4 and l4 == j4:
                                # kG0_44 cond_2
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+2
                                kG0v[c] += pi*(Ft*(j4*j4)*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) - (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/(r*(tmax - tmin)) - Fx*i4*k4*((-i4 + k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xb*(2*i4 + 2*k4))/L) + (i4 - k4)*cos(0.5*pi*(L*(i4 + k4 - 1) + xa*(2*i4 + 2*k4))/L) + (i4 + k4)*(sin(0.5*pi*(L + 2*xa)*(i4 - k4)/L) - sin(0.5*pi*(L + 2*xb)*(i4 - k4)/L)))/L)/((i4 + k4)*(4.0*i4 - 4.0*k4))
                            elif k4 != i4 and l4 != j4:
                                # kG0_44 cond_3
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+2
                                kG0v[c] += j4*l4*((-1)**(j4 + l4) - 1)*(Ftx*r*(tmax - tmin) + Fxt*L)*(-(i4*i4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + (i4*i4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - 2*i4*k4*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) - 2*i4*k4*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - (k4*k4)*cos(0.5*pi*(L + 2*xb)*(i4 - k4)/L) + (k4*k4)*cos(0.5*pi*(L + 2*xb)*(i4 + k4)/L) - (i4 - k4)**2*cos(0.5*pi*(L + 2*xa)*(i4 + k4)/L) + (i4 + k4)**2*cos(0.5*pi*(L + 2*xa)*(i4 - k4)/L))/(L*r*(i4 + k4)*(2.0*i4 - 2.0*k4)*(j4 - l4)*(j4 + l4)*(tmax - tmin))
                            elif k4 == i4 and l4 != j4:
                                # kG0_44 cond_4
                                pass

    size = num0 + num1 + num2*m2 + num3*n3 + num4*m4*n4

    kG0 = csr_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkG0_cyl(double Fx, double Ft, double Fxt, double Ftx, double r1,
        double L, double tmin, double tmax, int m2, int n3, int m4, int n4):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col
    cdef double r=r1
    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    fdim = (2*m2*m2 + 1*m2*n3 + 1*m2*m4*n4 + 1*n3*n3 + 1*n3*m4*n4 +
            1*m4*n4*m4*n4)

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kG0_22
    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for k2 in range(1, m2+1):
            col = num0 + num1 + num2*(k2-1)
            if k2 == i2:
                # kG0_22 cond_1
                c += 1
                kG0r[c] = row+2
                kG0c[c] = col+2
                kG0v[c] += 0.5*(pi*pi)*Fx*(i2*i2)/L
            else:
                # kG0_22 cond_2
                pass
    # kG0_23
    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for l3 in range(1, n3+1):
            col = num0 + num1 + num2*m2 + num3*(l3-1)
            c += 1
            kG0r[c] = row+2
            kG0c[c] = col+2
            kG0v[c] += ((-1)**i2 - 1)*((-1)**l3 - 1)*(Ftx*r*(tmax - tmin) + Fxt*L)/(L*r*(tmax - tmin))

    # kG0_24
    for i2 in range(1, m2+1):
        row = num0 + num1 + num2*(i2-1)
        for k4 in range(1, m4+1):
            for l4 in range(1, n4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 +
                        num4*((l4-1)*m4 + (k4-1)))
                if k4 == i2:
                    # kG0_24 cond_1
                    pass

                else:
                    # kG0_24 cond_2
                    c += 1
                    kG0r[c] = row+2
                    kG0c[c] = col+2
                    kG0v[c] += 2*(-1)**(l4 - 1)*i2*(-(-1)**(i2 + k4)*i2 + i2)*(Ftx*r*(tmax - tmin) + Fxt*L)/(L*r*((i2*i2) - (k4*k4))*(tmax - tmin))

    # kG0_33
    for j3 in range(1, n3+1):
        row = num0 + num1 + num2*m2 + num3*(j3-1)
        for l3 in range(1, n3+1):
            col = num0 + num1 + num2*m2 + num3*(l3-1)
            if l3 == j3:
                # kG0_33 cond_1
                c += 1
                kG0r[c] = row+2
                kG0c[c] = col+2
                kG0v[c] += (pi*pi)*Ft*(j3*j3)/(2*r*tmax - 2*r*tmin)

            else:
                # kG0_33 cond_2
                pass

    # kG0_34
    for j3 in range(1, n3+1):
        row = num0 + num1 + num2*m2 + num3*(j3-1)
        for k4 in range(1, m4+1):
            for l4 in range(1, n4+1):
                col = (num0 + num1 + num2*m2 + num3*n3 +
                        num4*((l4-1)*m4 + (k4-1)))
                if l4 == j3:
                    # kG0_34 cond_1
                    pass

                else:
                    # kG0_34 cond_2
                    c += 1
                    kG0r[c] = row+2
                    kG0c[c] = col+2
                    kG0v[c] += (j3*j3)*((-1)**k4 - 1)*((-1)**(j3 + l4) - 1)*(Ftx*r*(tmax - tmin) + Fxt*L)/(L*r*((j3*j3) - (l4*l4))*(tmax - tmin))

    # kG0_44
    for i4 in range(1, m4+1):
        for j4 in range(1, n4+1):
            row = (num0 + num1 + num2*m2 + num3*n3 +
                    num4*((j4-1)*m4 + (i4-1)))
            for k4 in range(1, m4+1):
                for l4 in range(1, n4+1):
                    col = (num0 + num1 + num2*m2 + num3*n3 +
                            num4*((l4-1)*m4 + (k4-1)))
                    if k4 == i4 and l4 == j4:
                        # kG0_44 cond_1
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += 0.25*(pi*pi)*(Ft*L*(j4*j4) + Fx*(i4*i4)*r*(tmax - tmin))/(L*r*(tmax - tmin))

                    elif k4 != i4 and l4 == j4:
                        # kG0_44 cond_2
                        pass

                    elif k4 != i4 and l4 != j4:
                        # kG0_44 cond_3
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += -((-1)**(i4 + k4) - 1)*((-1)**(j4 + l4) - 1)*((i4*i4)*(l4*l4) + (j4*j4)*(k4*k4))*(Ftx*r*(tmax - tmin) + Fxt*L)/(L*r*((i4*i4) - (k4*k4))*(j4 - l4)*(j4 + l4)*(tmax - tmin))

                    elif k4 == i4 and l4 != j4:
                        # kG0_44 cond_4
                        pass

    size = num0 + num1 + num2*m2 + num3*n3 + num4*m4*n4

    kG0 = csr_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
