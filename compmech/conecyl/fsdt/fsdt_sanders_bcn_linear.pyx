#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np
cimport cython
from cpython cimport bool


from compmech import INT


DOUBLE = np.float64


cdef extern from "math.h":
    double cos(double t)
    double sin(double t)


cdef int i0 = 0
cdef int j0 = 1
cdef int num0 = 3
cdef int num1 = 5
cdef int num2 = 10
cdef double pi = 3.141592653589793


def fk0(double alpharad, double r2, double L, double [:, ::1] F,
           int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r, sina, cosa, xa, xb
    cdef long [:] k0r, k0c
    cdef double [:] k0v

    sina = sin(alpharad)
    cosa = cos(alpharad)

    # sparse parameters
    k11_cond_1 = 25
    k11_cond_2 = 25
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 82
    k22_cond_2 = 100
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 9 + 15*m1 + k11_num + k22_num

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

        # k0_00
        c += 1
        k0r[c] = 0
        k0c[c] = 0
        k0v[c] += -0.666666666666667*pi*(xa - xb)*(3*A11*r**2 + sina*(3*A12*r*(-2*L + xa + xb) + A22*sina*(3*L**2 - 3*L*(xa + xb) + xa**2 + xa*xb + xb**2)))/(L**2*cosa**2*r)
        c += 1
        k0r[c] = 0
        k0c[c] = 1
        k0v[c] += 0.333333333333333*pi*r2*(xa - xb)*(3*A16*r**2*(-2*r + sina*(-2*L + xa + xb)) + 3*B16*r*(-2*r + sina*(-2*L + xa + xb)) + sina*(A26*r + B26)*(6*L**2*sina + 6*L*(r - sina*(xa + xb)) - 3*r*(xa + xb) + 2*sina*(xa**2 + xa*xb + xb**2)))/(L**2*cosa*r**2)
        c += 1
        k0r[c] = 0
        k0c[c] = 2
        k0v[c] += -0.666666666666667*pi*(xa - xb)*(3*A11*r**2 + sina*(3*A12*r*(-2*L + xa + xb) + A22*sina*(3*L**2 - 3*L*(xa + xb) + xa**2 + xa*xb + xb**2)))/(L**2*cosa**2*r)
        c += 1
        k0r[c] = 1
        k0c[c] = 0
        k0v[c] += 0.333333333333333*pi*r2*(xa - xb)*(3*A16*r**2*(-2*r + sina*(-2*L + xa + xb)) + 3*B16*r*(-2*r + sina*(-2*L + xa + xb)) + sina*(A26*r + B26)*(6*L**2*sina + 6*L*(r - sina*(xa + xb)) - 3*r*(xa + xb) + 2*sina*(xa**2 + xa*xb + xb**2)))/(L**2*cosa*r**2)
        c += 1
        k0r[c] = 1
        k0c[c] = 1
        k0v[c] += 0.666666666666667*pi*r2**2*(xa - xb)*(-D66*(3*r**2 + 3*r*sina*(2*L - xa - xb) + sina**2*(3*L**2 - 3*L*(xa + xb) + xa**2 + xa*xb + xb**2)) + r*(-A44*cosa**2*r*(3*L**2 - 3*L*(xa + xb) + xa**2 + xa*xb + xb**2) - (A66*r + 2*B66)*(3*r**2 + 3*r*sina*(2*L - xa - xb) + sina**2*(3*L**2 - 3*L*(xa + xb) + xa**2 + xa*xb + xb**2))))/(L**2*r**3)
        c += 1
        k0r[c] = 1
        k0c[c] = 2
        k0v[c] += 0.333333333333333*pi*r2*(xa - xb)*(3*A16*r**2*(-2*r + sina*(-2*L + xa + xb)) + 3*B16*r*(-2*r + sina*(-2*L + xa + xb)) + sina*(A26*r + B26)*(6*L**2*sina + 6*L*(r - sina*(xa + xb)) - 3*r*(xa + xb) + 2*sina*(xa**2 + xa*xb + xb**2)))/(L**2*cosa*r**2)
        c += 1
        k0r[c] = 2
        k0c[c] = 0
        k0v[c] += -0.666666666666667*pi*(xa - xb)*(3*A11*r**2 + sina*(3*A12*r*(-2*L + xa + xb) + A22*sina*(3*L**2 - 3*L*(xa + xb) + xa**2 + xa*xb + xb**2)))/(L**2*cosa**2*r)
        c += 1
        k0r[c] = 2
        k0c[c] = 1
        k0v[c] += 0.333333333333333*pi*r2*(xa - xb)*(3*A16*r**2*(-2*r + sina*(-2*L + xa + xb)) + 3*B16*r*(-2*r + sina*(-2*L + xa + xb)) + sina*(A26*r + B26)*(6*L**2*sina + 6*L*(r - sina*(xa + xb)) - 3*r*(xa + xb) + 2*sina*(xa**2 + xa*xb + xb**2)))/(L**2*cosa*r**2)
        c += 1
        k0r[c] = 2
        k0c[c] = 2
        k0v[c] += -0.333333333333333*pi*(xa - xb)*(9*A11*r**2 + A66*(3*L**2 - 3*L*(xa + xb) + xa**2 + xa*xb + xb**2) + 3*sina*(3*A12*r*(-2*L + xa + xb) + A22*sina*(3*L**2 - 3*L*(xa + xb) + xa**2 + xa*xb + xb**2)))/(L**2*cosa**2*r)

        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            row = col

            if i1 != 0:
                # k0_01 cond_1
                c += 1
                k0r[c] = 0
                k0c[c] = col+0
                k0v[c] += (2*pi*A22*L*i1*sina**2*(-L + xb)*cos(pi*i1*xb/L) + 2*pi*A22*L*i1*sina**2*(L - xa)*cos(pi*i1*xa/L) + 2*(pi**2*A11*i1**2*r**2 + sina*(pi**2*A12*i1**2*r*(-L + xa) + A22*L**2*sina))*sin(pi*i1*xa/L) - 2*(pi**2*A11*i1**2*r**2 + sina*(pi**2*A12*i1**2*r*(-L + xb) + A22*L**2*sina))*sin(pi*i1*xb/L))/(pi*L*cosa*i1**2*r)
                c += 1
                k0r[c] = 0
                k0c[c] = col+1
                k0v[c] += (2*pi*L*i1*sina*(-B16*r - B26*(r + sina*(-L + xb)) - r*(A16*r + A26*(-L*sina + r + sina*xb)))*cos(pi*i1*xb/L) + 2*pi*L*i1*sina*(B16*r + B26*(r + sina*(-L + xa)) + r*(A16*r + A26*(-L*sina + r + sina*xa)))*cos(pi*i1*xa/L) + 2*(-pi**2*A16*i1**2*r**3 - pi**2*B16*i1**2*r**2 + sina*(A26*r + B26)*(L**2*sina + pi**2*i1**2*r*(L - xb)))*sin(pi*i1*xb/L) + 2*(pi**2*A16*i1**2*r**3 + pi**2*B16*i1**2*r**2 + sina*(A26*r + B26)*(-L**2*sina + pi**2*i1**2*r*(-L + xa)))*sin(pi*i1*xa/L))/(pi*L*cosa*i1**2*r**2)
                c += 1
                k0r[c] = 0
                k0c[c] = col+2
                k0v[c] += (2*A22*L*sina*(sin(pi*i1*xa/L) - sin(pi*i1*xb/L)) - 2*pi*i1*(A12*r + A22*sina*(-L + xa))*cos(pi*i1*xa/L) + 2*pi*i1*(A12*r + A22*sina*(-L + xb))*cos(pi*i1*xb/L))/(pi*i1**2*r)
                c += 1
                k0r[c] = 0
                k0c[c] = col+3
                k0v[c] += (2*pi*B22*L*i1*sina**2*((-L + xa)*sin(pi*i1*xa/L) + (L - xb)*sin(pi*i1*xb/L)) + 2*(pi**2*B11*i1**2*r**2 + sina*(pi**2*B12*i1**2*r*(-L + xa) + B22*L**2*sina))*cos(pi*i1*xa/L) - 2*(pi**2*B11*i1**2*r**2 + sina*(pi**2*B12*i1**2*r*(-L + xb) + B22*L**2*sina))*cos(pi*i1*xb/L))/(pi*L*cosa*i1**2*r)
                c += 1
                k0r[c] = 0
                k0c[c] = col+4
                k0v[c] += (2*pi*L*i1*sina*(-B16*r - B26*(-L*sina + r + sina*xb))*cos(pi*i1*xb/L) + 2*pi*L*i1*sina*(B16*r + B26*(-L*sina + r + sina*xa))*cos(pi*i1*xa/L) + 2*(-pi**2*B16*i1**2*r**2 + B26*sina*(L**2*sina + pi**2*i1**2*r*(L - xb)))*sin(pi*i1*xb/L) + 2*(pi**2*B16*i1**2*r**2 + B26*sina*(-L**2*sina + pi**2*i1**2*r*(-L + xa)))*sin(pi*i1*xa/L))/(pi*L*cosa*i1**2*r)
                c += 1
                k0r[c] = 1
                k0c[c] = col+0
                k0v[c] += 2*r2*(-pi*L*i1*sina*(B16*r + B26*(r + sina*(L - xa)) + r*(A16*r + A26*(L*sina + r - sina*xa)))*cos(pi*i1*xa/L) + pi*L*i1*sina*(B16*r + B26*(r + sina*(L - xb)) + r*(A16*r + A26*(L*sina + r - sina*xb)))*cos(pi*i1*xb/L) + (pi**2*A16*i1**2*r**2*(L*sina + r - sina*xa) + pi**2*B16*i1**2*r*(r + sina*(L - xa)) - L**2*sina**2*(A26*r + B26))*sin(pi*i1*xa/L) + (-pi**2*A16*i1**2*r**2*(L*sina + r - sina*xb) - pi**2*B16*i1**2*r*(r + sina*(L - xb)) + L**2*sina**2*(A26*r + B26))*sin(pi*i1*xb/L))/(pi*L*i1**2*r**2)
                c += 1
                k0r[c] = 1
                k0c[c] = col+1
                k0v[c] += -2*r2*(-L*(L*(A44*cosa**2*r**2 + sina**2*(A66*r**2 + 2*B66*r + D66))*(sin(pi*i1*xa/L) - sin(pi*i1*xb/L)) + pi*i1*(A44*cosa**2*r**2*(L - xa) + sina*(A66*r**2 + 2*B66*r + D66)*(L*sina + r - sina*xa))*cos(pi*i1*xa/L) - pi*i1*(A44*cosa**2*r**2*(L - xb) + sina*(A66*r**2 + 2*B66*r + D66)*(L*sina + r - sina*xb))*cos(pi*i1*xb/L)) + pi*i1*r*(D66 + r*(A66*r + 2*B66))*(L*sina*cos(pi*i1*xa/L) - L*sina*cos(pi*i1*xb/L) + pi*i1*(-(L*sina + r - sina*xa)*sin(pi*i1*xa/L) + (L*sina + r - sina*xb)*sin(pi*i1*xb/L))))/(pi*L*i1**2*r**3)
                c += 1
                k0r[c] = 1
                k0c[c] = col+2
                k0v[c] += -2*cosa*r2*(pi*A45*i1*r**2*(L*cos(pi*i1*xa/L) - L*cos(pi*i1*xb/L) + pi*i1*((-L + xa)*sin(pi*i1*xa/L) + (L - xb)*sin(pi*i1*xb/L))) + L*(A26*r + B26)*(L*sina*(sin(pi*i1*xa/L) - sin(pi*i1*xb/L)) + pi*i1*(r + sina*(L - xa))*cos(pi*i1*xa/L) - pi*i1*(L*sina + r - sina*xb)*cos(pi*i1*xb/L)))/(pi*L*i1**2*r**2)
                c += 1
                k0r[c] = 1
                k0c[c] = col+3
                k0v[c] += 2*r2*(-L*(L*(A45*cosa*r**2 + sina**2*(B26*r + D26))*cos(pi*i1*xa/L) - L*(A45*cosa*r**2 + sina**2*(B26*r + D26))*cos(pi*i1*xb/L) + pi*i1*(-(A45*cosa*r**2*(L - xa) + sina*(B26*r + D26)*(L*sina + r - sina*xa))*sin(pi*i1*xa/L) + (A45*cosa*r**2*(L - xb) + sina*(B26*r + D26)*(L*sina + r - sina*xb))*sin(pi*i1*xb/L))) + pi*i1*r*(B16*r + D16)*(L*sina*(sin(pi*i1*xa/L) - sin(pi*i1*xb/L)) + pi*i1*(r + sina*(L - xa))*cos(pi*i1*xa/L) - pi*i1*(L*sina + r - sina*xb)*cos(pi*i1*xb/L)))/(pi*L*i1**2*r**2)
                c += 1
                k0r[c] = 1
                k0c[c] = col+4
                k0v[c] += -2*r2*(L*(L*(A44*cosa*r**2 - sina**2*(B66*r + D66))*(sin(pi*i1*xa/L) - sin(pi*i1*xb/L)) + pi*i1*(A44*cosa*r**2*(-L + xb) + sina*(B66*r + D66)*(L*sina + r - sina*xb))*cos(pi*i1*xb/L) + pi*i1*(A44*cosa*r**2*(L - xa) - sina*(B66*r + D66)*(L*sina + r - sina*xa))*cos(pi*i1*xa/L)) + pi*i1*r*(B66*r + D66)*(L*sina*cos(pi*i1*xa/L) - L*sina*cos(pi*i1*xb/L) + pi*i1*(-(L*sina + r - sina*xa)*sin(pi*i1*xa/L) + (L*sina + r - sina*xb)*sin(pi*i1*xb/L))))/(pi*L*i1**2*r**2)
                c += 1
                k0r[c] = 2
                k0c[c] = col+0
                k0v[c] += (2*pi*A22*L*i1*sina**2*(-L + xb)*cos(pi*i1*xb/L) + 2*pi*A22*L*i1*sina**2*(L - xa)*cos(pi*i1*xa/L) + 2*(pi**2*A11*i1**2*r**2 + sina*(pi**2*A12*i1**2*r*(-L + xa) + A22*L**2*sina))*sin(pi*i1*xa/L) - 2*(pi**2*A11*i1**2*r**2 + sina*(pi**2*A12*i1**2*r*(-L + xb) + A22*L**2*sina))*sin(pi*i1*xb/L))/(pi*L*cosa*i1**2*r)
                c += 1
                k0r[c] = 2
                k0c[c] = col+1
                k0v[c] += (2*pi*L*i1*sina*(-B16*r - B26*(r + sina*(-L + xb)) - r*(A16*r + A26*(-L*sina + r + sina*xb)))*cos(pi*i1*xb/L) + 2*pi*L*i1*sina*(B16*r + B26*(r + sina*(-L + xa)) + r*(A16*r + A26*(-L*sina + r + sina*xa)))*cos(pi*i1*xa/L) + 2*(-pi**2*A16*i1**2*r**3 - pi**2*B16*i1**2*r**2 + sina*(A26*r + B26)*(L**2*sina + pi**2*i1**2*r*(L - xb)))*sin(pi*i1*xb/L) + 2*(pi**2*A16*i1**2*r**3 + pi**2*B16*i1**2*r**2 + sina*(A26*r + B26)*(-L**2*sina + pi**2*i1**2*r*(-L + xa)))*sin(pi*i1*xa/L))/(pi*L*cosa*i1**2*r**2)
                c += 1
                k0r[c] = 2
                k0c[c] = col+2
                k0v[c] += (2*A22*L*sina*(sin(pi*i1*xa/L) - sin(pi*i1*xb/L)) - 2*pi*i1*(A12*r + A22*sina*(-L + xa))*cos(pi*i1*xa/L) + 2*pi*i1*(A12*r + A22*sina*(-L + xb))*cos(pi*i1*xb/L))/(pi*i1**2*r)
                c += 1
                k0r[c] = 2
                k0c[c] = col+3
                k0v[c] += (2*pi*B22*L*i1*sina**2*((-L + xa)*sin(pi*i1*xa/L) + (L - xb)*sin(pi*i1*xb/L)) + 2*(pi**2*B11*i1**2*r**2 + sina*(pi**2*B12*i1**2*r*(-L + xa) + B22*L**2*sina))*cos(pi*i1*xa/L) - 2*(pi**2*B11*i1**2*r**2 + sina*(pi**2*B12*i1**2*r*(-L + xb) + B22*L**2*sina))*cos(pi*i1*xb/L))/(pi*L*cosa*i1**2*r)
                c += 1
                k0r[c] = 2
                k0c[c] = col+4
                k0v[c] += (2*pi*L*i1*sina*(-B16*r - B26*(-L*sina + r + sina*xb))*cos(pi*i1*xb/L) + 2*pi*L*i1*sina*(B16*r + B26*(-L*sina + r + sina*xa))*cos(pi*i1*xa/L) + 2*(-pi**2*B16*i1**2*r**2 + B26*sina*(L**2*sina + pi**2*i1**2*r*(L - xb)))*sin(pi*i1*xb/L) + 2*(pi**2*B16*i1**2*r**2 + B26*sina*(-L**2*sina + pi**2*i1**2*r*(-L + xa)))*sin(pi*i1*xa/L))/(pi*L*cosa*i1**2*r)

            else:
                # k0_01 cond_5
                c += 1
                k0r[c] = 0
                k0c[c] = col+3
                k0v[c] += pi*sina*(xa - xb)*(2*B12*r + B22*sina*(-2*L + xa + xb))/(L*cosa*r)
                c += 1
                k0r[c] = 1
                k0c[c] = col+3
                k0v[c] += -pi*r2*(xa - xb)*(A45*cosa*r**2*(-2*L + xa + xb) + sina*(-2*r + sina*(-2*L + xa + xb))*(B26*r + D26))/(L*r**2)
                c += 1
                k0r[c] = 2
                k0c[c] = col+3
                k0v[c] += pi*sina*(xa - xb)*(2*B12*r + B22*sina*(-2*L + xa + xb))/(L*cosa*r)

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
                        k0v[c] += 0.5*(2*L*(-2*pi*A12*L*i1*r*sina*sin(pi*i1*(xa + xb)/L) + (-pi**2*A11*i1**2*r**2 + A22*L**2*sina**2)*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - 2*pi*i1*(xa - xb)*(pi**2*A11*i1**2*r**2 + A22*L**2*sina**2))/(L**2*i1*r)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += 0.5*(2*L*(pi*L*i1*r*sina*(A16*r - A26*r + B16 - B26)*sin(pi*i1*(xa + xb)/L) - (L**2*sina**2*(A26*r + B26) + pi**2*i1**2*r**2*(A16*r + B16))*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - 2*pi*i1*(xa - xb)*(-L**2*sina**2*(A26*r + B26) + pi**2*i1**2*r**2*(A16*r + B16)))/(L**2*i1*r**2)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*cosa*(pi*A12*(cos(2*pi*i1*xa/L) - cos(2*pi*i1*xb/L)) + A22*sina*(L*sin(2*pi*i1*xa/L) - L*sin(2*pi*i1*xb/L) + 2*pi*i1*(-xa + xb))/(i1*r))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += (-2*pi*B12*L*i1*r*sina*cos(pi*i1*(xa + xb)/L) + (pi**2*B11*i1**2*r**2 - B22*L**2*sina**2)*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L)/(L*i1*r)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += (L*(pi*L*i1*r*sina*(B16 - B26)*sin(pi*i1*(xa + xb)/L) - (pi**2*B16*i1**2*r**2 + B26*L**2*sina**2)*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - pi*i1*(xa - xb)*(pi**2*B16*i1**2*r**2 - B26*L**2*sina**2))/(L**2*i1*r)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += 0.5*(2*L*(pi*L*i1*r*sina*(A16*r - A26*r + B16 - B26)*sin(pi*i1*(xa + xb)/L) - (L**2*sina**2*(A26*r + B26) + pi**2*i1**2*r**2*(A16*r + B16))*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - 2*pi*i1*(xa - xb)*(-L**2*sina**2*(A26*r + B26) + pi**2*i1**2*r**2*(A16*r + B16)))/(L**2*i1*r**2)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += pi*(L*(2*pi*L*i1*r*sina*(D66 + r*(A66*r + 2*B66))*sin(pi*i1*(xa + xb)/L) + (A44*L**2*cosa**2*r**2 - (-L*sina + pi*i1*r)*(L*sina + pi*i1*r)*(A66*r**2 + 2*B66*r + D66))*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L)/(pi*i1) - (xa - xb)*(A44*L**2*cosa**2*r**2 + (D66 + r*(A66*r + 2*B66))*(L**2*sina**2 + pi**2*i1**2*r**2)))/(L**2*r**3)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += 0.5*cosa*(2*pi*i1*sina*(xa - xb)*(A26*r + B26) + 2*(-L*sina*(A26*r + B26)*cos(pi*i1*(xa + xb)/L) - pi*i1*r*(B26 + r*(A26 - A45))*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L))/(i1*r**2)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += -0.5*pi*(2*pi*i1*r*sina*(xa - xb)*(D16 + D26 + r*(B16 + B26)) + 2*(pi*L*i1*r*sina*(-B16*r + B26*r - D16 + D26)*cos(pi*i1*(xa + xb)/L) - (L**2*sina**2*(B26*r + D26) + r**2*(A45*L**2*cosa + pi**2*i1**2*(B16*r + D16)))*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L)/(pi*i1))/(L*r**2)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += 0.5*(2*L*(2*pi*L*i1*r*sina*(B66*r + D66)*sin(pi*i1*(xa + xb)/L) - (A44*L**2*cosa*r**2 + (B66*r + D66)*(-L*sina + pi*i1*r)*(L*sina + pi*i1*r))*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - 2*pi*i1*(xa - xb)*(-A44*L**2*cosa*r**2 + (B66*r + D66)*(L**2*sina**2 + pi**2*i1**2*r**2)))/(L**2*i1*r**2)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 0.5*cosa*(pi*A12*(cos(2*pi*i1*xa/L) - cos(2*pi*i1*xb/L)) + A22*sina*(L*sin(2*pi*i1*xa/L) - L*sin(2*pi*i1*xb/L) + 2*pi*i1*(-xa + xb))/(i1*r))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += 0.5*cosa*(2*pi*i1*sina*(xa - xb)*(A26*r + B26) + 2*(-L*sina*(A26*r + B26)*cos(pi*i1*(xa + xb)/L) - pi*i1*r*(B26 + r*(A26 - A45))*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L))/(i1*r**2)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += 0.5*(L*(A22*L**2*cosa**2 - pi**2*A55*i1**2*r**2)*(sin(2*pi*i1*xa/L) - sin(2*pi*i1*xb/L)) - 2*pi*i1*(xa - xb)*(A22*L**2*cosa**2 + pi**2*A55*i1**2*r**2))/(L**2*i1*r)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += 0.5*(B22*L**2*cosa*sina*cos(2*pi*i1*xa/L) - B22*L**2*cosa*sina*cos(2*pi*i1*xb/L) + pi*i1*r*(L*(A55*r + B12*cosa)*(-sin(2*pi*i1*xa/L) + sin(2*pi*i1*xb/L)) + 2*pi*i1*(xa - xb)*(-A55*r + B12*cosa)))/(L*i1*r)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += (pi*B26*cosa*i1*sina*(xa - xb) - (B26*L*cosa*sina*cos(pi*i1*(xa + xb)/L) + pi*i1*r*(A45*r + B26*cosa)*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L))/(i1*r)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += (-2*pi*B12*L*i1*r*sina*cos(pi*i1*(xa + xb)/L) + (pi**2*B11*i1**2*r**2 - B22*L**2*sina**2)*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L)/(L*i1*r)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += -0.5*pi*(2*pi*i1*r*sina*(xa - xb)*(D16 + D26 + r*(B16 + B26)) + 2*(pi*L*i1*r*sina*(-B16*r + B26*r - D16 + D26)*cos(pi*i1*(xa + xb)/L) - (L**2*sina**2*(B26*r + D26) + r**2*(A45*L**2*cosa + pi**2*i1**2*(B16*r + D16)))*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L)/(pi*i1))/(L*r**2)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += 0.5*(B22*L**2*cosa*sina*cos(2*pi*i1*xa/L) - B22*L**2*cosa*sina*cos(2*pi*i1*xb/L) + pi*i1*r*(L*(A55*r + B12*cosa)*(-sin(2*pi*i1*xa/L) + sin(2*pi*i1*xb/L)) + 2*pi*i1*(xa - xb)*(-A55*r + B12*cosa)))/(L*i1*r)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += 0.5*(2*L*(2*pi*D12*L*i1*r*sina*sin(pi*i1*(xa + xb)/L) - (D22*L**2*sina**2 + r**2*(A55*L**2 - pi**2*D11*i1**2))*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - 2*pi*i1*(xa - xb)*(D22*L**2*sina**2 + r**2*(A55*L**2 + pi**2*D11*i1**2)))/(L**2*i1*r)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += (-pi**2*i1**2*r*sina*(D16 + D26)*(xa - xb) + (pi*L*i1*r*sina*(D16 - D26)*cos(pi*i1*(xa + xb)/L) + (-A45*L**2*r**2 + pi**2*D16*i1**2*r**2 + D26*L**2*sina**2)*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L))/(L*i1*r)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += (L*(pi*L*i1*r*sina*(B16 - B26)*sin(pi*i1*(xa + xb)/L) - (pi**2*B16*i1**2*r**2 + B26*L**2*sina**2)*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - pi*i1*(xa - xb)*(pi**2*B16*i1**2*r**2 - B26*L**2*sina**2))/(L**2*i1*r)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += 0.5*(2*L*(2*pi*L*i1*r*sina*(B66*r + D66)*sin(pi*i1*(xa + xb)/L) - (A44*L**2*cosa*r**2 + (B66*r + D66)*(-L*sina + pi*i1*r)*(L*sina + pi*i1*r))*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - 2*pi*i1*(xa - xb)*(-A44*L**2*cosa*r**2 + (B66*r + D66)*(L**2*sina**2 + pi**2*i1**2*r**2)))/(L**2*i1*r**2)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += (pi*B26*cosa*i1*sina*(xa - xb) - (B26*L*cosa*sina*cos(pi*i1*(xa + xb)/L) + pi*i1*r*(A45*r + B26*cosa)*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L))/(i1*r)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += (-pi**2*i1**2*r*sina*(D16 + D26)*(xa - xb) + (pi*L*i1*r*sina*(D16 - D26)*cos(pi*i1*(xa + xb)/L) + (-A45*L**2*r**2 + pi**2*D16*i1**2*r**2 + D26*L**2*sina**2)*sin(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L))/(L*i1*r)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += 0.5*(2*L*(2*pi*D66*L*i1*r*sina*sin(pi*i1*(xa + xb)/L) + (D66*L**2*sina**2 + r**2*(A44*L**2 - pi**2*D66*i1**2))*cos(pi*i1*(xa + xb)/L))*sin(pi*i1*(xa - xb)/L) - 2*pi*i1*(xa - xb)*(D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i1**2)))/(L**2*i1*r)

                    else:
                        # k0_11 cond_5
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += 2*pi*r*(A55 + D22*sina**2/r**2)*(-xa + xb)

                else:
                    # k0_11 cond_2
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += (-2*k1*(pi**2*A11*i1**2*r**2 + A22*L**2*sina**2)*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + 2*k1*(pi**2*A11*i1**2*r**2 + A22*L**2*sina**2)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*(pi*A12*L*r*sina*(-i1**2 + k1**2)*sin(pi*i1*xa/L) + i1*(pi**2*A11*k1**2*r**2 + A22*L**2*sina**2)*cos(pi*i1*xa/L))*sin(pi*k1*xa/L) - 2*(pi*A12*L*r*sina*(-i1**2 + k1**2)*sin(pi*i1*xb/L) + i1*(pi**2*A11*k1**2*r**2 + A22*L**2*sina**2)*cos(pi*i1*xb/L))*sin(pi*k1*xb/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+1
                    k0v[c] += (-2*pi*L*r*sina*(A16*i1**2*r + B16*i1**2 + k1**2*(A26*r + B26))*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*i1*(pi*L*k1*r*sina*(B16 + B26 + r*(A16 + A26))*cos(pi*k1*xa/L) + (-L**2*sina**2*(A26*r + B26) + pi**2*k1**2*r**2*(A16*r + B16))*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*i1*(-pi*L*k1*r*sina*(B16 + B26 + r*(A16 + A26))*cos(pi*k1*xb/L) + (L**2*sina**2*(A26*r + B26) - pi**2*k1**2*r**2*(A16*r + B16))*sin(pi*k1*xb/L))*cos(pi*i1*xb/L) + 2*k1*(-L**2*sina**2*(A26*r + B26) + pi**2*i1**2*r**2*(A16*r + B16))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*(pi*L*r*sina*(A16*i1**2*r + B16*i1**2 + k1**2*(A26*r + B26))*sin(pi*k1*xa/L) + k1*(L**2*sina**2*(A26*r + B26) - pi**2*i1**2*r**2*(A16*r + B16))*cos(pi*k1*xa/L))*sin(pi*i1*xa/L))/(L*r**2*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+2
                    k0v[c] += 2*cosa*(i1*(pi*A12*i1*r*sin(pi*i1*xb/L) - A22*L*sina*cos(pi*i1*xb/L))*sin(pi*k1*xb/L) + i1*(-pi*A12*k1*r*cos(pi*k1*xa/L) + A22*L*sina*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) + k1*(pi*A12*i1*r*cos(pi*i1*xb/L) + A22*L*sina*sin(pi*i1*xb/L))*cos(pi*k1*xb/L) - (pi*A12*i1**2*r*sin(pi*k1*xa/L) + A22*L*k1*sina*cos(pi*k1*xa/L))*sin(pi*i1*xa/L))/(r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+3
                    k0v[c] += (2*pi*B12*L*r*sina*(i1 - k1)*(i1 + k1)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*i1*(pi**2*B11*k1**2*r**2 + B22*L**2*sina**2)*cos(pi*i1*xa/L)*cos(pi*k1*xa/L) - 2*i1*(pi**2*B11*k1**2*r**2 + B22*L**2*sina**2)*cos(pi*i1*xb/L)*cos(pi*k1*xb/L) - 2*k1*(pi**2*B11*i1**2*r**2 + B22*L**2*sina**2)*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*(pi*B12*L*r*sina*(-i1**2 + k1**2)*cos(pi*k1*xa/L) + k1*(pi**2*B11*i1**2*r**2 + B22*L**2*sina**2)*sin(pi*k1*xa/L))*sin(pi*i1*xa/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+4
                    k0v[c] += (-2*pi*L*r*sina*(B16*i1**2 + B26*k1**2)*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*i1*(pi*L*k1*r*sina*(B16 + B26)*cos(pi*k1*xa/L) + (pi**2*B16*k1**2*r**2 - B26*L**2*sina**2)*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*i1*(-pi*L*k1*r*sina*(B16 + B26)*cos(pi*k1*xb/L) + (-pi**2*B16*k1**2*r**2 + B26*L**2*sina**2)*sin(pi*k1*xb/L))*cos(pi*i1*xb/L) + 2*k1*(pi**2*B16*i1**2*r**2 - B26*L**2*sina**2)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*(pi*L*r*sina*(B16*i1**2 + B26*k1**2)*sin(pi*k1*xa/L) + k1*(-pi**2*B16*i1**2*r**2 + B26*L**2*sina**2)*cos(pi*k1*xa/L))*sin(pi*i1*xa/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += (2*pi*L*r*sina*(A26*i1**2*r + B26*i1**2 + k1**2*(A16*r + B16))*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*i1*(-pi*L*k1*r*sina*(B16 + B26 + r*(A16 + A26))*cos(pi*k1*xa/L) + (-L**2*sina**2*(A26*r + B26) + pi**2*k1**2*r**2*(A16*r + B16))*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*i1*(pi*L*k1*r*sina*(B16 + B26 + r*(A16 + A26))*cos(pi*k1*xb/L) + (L**2*sina**2*(A26*r + B26) - pi**2*k1**2*r**2*(A16*r + B16))*sin(pi*k1*xb/L))*cos(pi*i1*xb/L) + 2*k1*(-L**2*sina**2*(A26*r + B26) + pi**2*i1**2*r**2*(A16*r + B16))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*(-pi*L*r*sina*(A26*i1**2*r + B26*i1**2 + k1**2*(A16*r + B16))*sin(pi*k1*xa/L) + k1*(L**2*sina**2*(A26*r + B26) - pi**2*i1**2*r**2*(A16*r + B16))*cos(pi*k1*xa/L))*sin(pi*i1*xa/L))/(L*r**2*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += (-2*k1*(A44*L**2*cosa**2*r**2 + (L**2*sina**2 + pi**2*i1**2*r**2)*(A66*r**2 + 2*B66*r + D66))*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + 2*k1*(A44*L**2*cosa**2*r**2 + (L**2*sina**2 + pi**2*i1**2*r**2)*(A66*r**2 + 2*B66*r + D66))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*(pi*L*r*sina*(D66 + r*(A66*r + 2*B66))*(i1 - k1)*(i1 + k1)*sin(pi*i1*xa/L) + i1*(A44*L**2*cosa**2*r**2 + (L**2*sina**2 + pi**2*k1**2*r**2)*(A66*r**2 + 2*B66*r + D66))*cos(pi*i1*xa/L))*sin(pi*k1*xa/L) - 2*(pi*L*r*sina*(D66 + r*(A66*r + 2*B66))*(i1 - k1)*(i1 + k1)*sin(pi*i1*xb/L) + i1*(A44*L**2*cosa**2*r**2 + (L**2*sina**2 + pi**2*k1**2*r**2)*(A66*r**2 + 2*B66*r + D66))*cos(pi*i1*xb/L))*sin(pi*k1*xb/L))/(L*r**3*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+2
                    k0v[c] += 2*cosa*(L*k1*sina*(A26*r + B26)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + i1*(L*sina*(A26*r + B26)*sin(pi*k1*xa/L) + pi*k1*r*(B26 + r*(A26 + A45))*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) + i1*(-L*sina*(A26*r + B26)*sin(pi*k1*xb/L) - pi*k1*r*(B26 + r*(A26 + A45))*cos(pi*k1*xb/L))*cos(pi*i1*xb/L) - pi*r*(A26*i1**2*r + A45*k1**2*r + B26*i1**2)*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + (-L*k1*sina*(A26*r + B26)*cos(pi*k1*xa/L) + pi*r*(A26*i1**2*r + A45*k1**2*r + B26*i1**2)*sin(pi*k1*xa/L))*sin(pi*i1*xa/L))/(r**2*(-i1 + k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+3
                    k0v[c] += (2*pi*L*r*sina*(B26*i1**2*r + D26*i1**2 + k1**2*(B16*r + D16))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) - 2*i1*(-pi*L*k1*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*k1*xa/L) + (A45*L**2*cosa*r**2 + L**2*sina**2*(B26*r + D26) - pi**2*k1**2*r**2*(B16*r + D16))*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) - 2*i1*(pi*L*k1*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*k1*xb/L) - (A45*L**2*cosa*r**2 + L**2*sina**2*(B26*r + D26) - pi**2*k1**2*r**2*(B16*r + D16))*cos(pi*k1*xb/L))*cos(pi*i1*xb/L) + 2*k1*(A45*L**2*cosa*r**2 + L**2*sina**2*(B26*r + D26) - pi**2*i1**2*r**2*(B16*r + D16))*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) - 2*(pi*L*r*sina*(B26*i1**2*r + D26*i1**2 + k1**2*(B16*r + D16))*cos(pi*k1*xa/L) + k1*(A45*L**2*cosa*r**2 + L**2*sina**2*(B26*r + D26) - pi**2*i1**2*r**2*(B16*r + D16))*sin(pi*k1*xa/L))*sin(pi*i1*xa/L))/(L*r**2*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+4
                    k0v[c] += (2*k1*(-A44*L**2*cosa*r**2 + (B66*r + D66)*(L**2*sina**2 + pi**2*i1**2*r**2))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*k1*(A44*L**2*cosa*r**2 - (B66*r + D66)*(L**2*sina**2 + pi**2*i1**2*r**2))*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + 2*(pi*L*r*sina*(i1 - k1)*(i1 + k1)*(B66*r + D66)*sin(pi*i1*xa/L) + i1*(-A44*L**2*cosa*r**2 + (B66*r + D66)*(L**2*sina**2 + pi**2*k1**2*r**2))*cos(pi*i1*xa/L))*sin(pi*k1*xa/L) - 2*(pi*L*r*sina*(i1 - k1)*(i1 + k1)*(B66*r + D66)*sin(pi*i1*xb/L) + i1*(-A44*L**2*cosa*r**2 + (B66*r + D66)*(L**2*sina**2 + pi**2*k1**2*r**2))*cos(pi*i1*xb/L))*sin(pi*k1*xb/L))/(L*r**2*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+0
                    k0v[c] += 2*cosa*(i1*(pi*A12*k1*r*cos(pi*k1*xa/L) + A22*L*sina*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) - i1*(pi*A12*k1*r*cos(pi*k1*xb/L) + A22*L*sina*sin(pi*k1*xb/L))*cos(pi*i1*xb/L) + k1*(pi*A12*k1*r*sin(pi*k1*xa/L) - A22*L*sina*cos(pi*k1*xa/L))*sin(pi*i1*xa/L) + k1*(-pi*A12*k1*r*sin(pi*k1*xb/L) + A22*L*sina*cos(pi*k1*xb/L))*sin(pi*i1*xb/L))/(r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+1
                    k0v[c] += 2*cosa*(-L*k1*sina*(A26*r + B26)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + i1*(-L*sina*(A26*r + B26)*sin(pi*k1*xa/L) + pi*k1*r*(B26 + r*(A26 + A45))*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) + i1*(L*sina*(A26*r + B26)*sin(pi*k1*xb/L) - pi*k1*r*(B26 + r*(A26 + A45))*cos(pi*k1*xb/L))*cos(pi*i1*xb/L) - pi*r*(A45*i1**2*r + k1**2*(A26*r + B26))*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + (L*k1*sina*(A26*r + B26)*cos(pi*k1*xa/L) + pi*r*(A45*i1**2*r + k1**2*(A26*r + B26))*sin(pi*k1*xa/L))*sin(pi*i1*xa/L))/(r**2*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += (2*i1*(A22*L**2*cosa**2 + pi**2*A55*k1**2*r**2)*sin(pi*k1*xa/L)*cos(pi*i1*xa/L) - 2*i1*(A22*L**2*cosa**2 + pi**2*A55*k1**2*r**2)*sin(pi*k1*xb/L)*cos(pi*i1*xb/L) - 2*k1*(A22*L**2*cosa**2 + pi**2*A55*i1**2*r**2)*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + 2*k1*(A22*L**2*cosa**2 + pi**2*A55*i1**2*r**2)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+3
                    k0v[c] += (-2*B22*L*cosa*k1*sina*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*i1*(B22*L*cosa*sina*cos(pi*k1*xa/L) + pi*k1*r*(A55*r - B12*cosa)*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*i1*(-B22*L*cosa*sina*cos(pi*k1*xb/L) + pi*k1*r*(-A55*r + B12*cosa)*sin(pi*k1*xb/L))*cos(pi*i1*xb/L) + 2*pi*r*(A55*i1**2*r - B12*cosa*k1**2)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*(B22*L*cosa*k1*sina*sin(pi*k1*xa/L) + pi*r*(-A55*i1**2*r + B12*cosa*k1**2)*cos(pi*k1*xa/L))*sin(pi*i1*xa/L))/(r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+4
                    k0v[c] += (-2*B26*L*cosa*k1*sina*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*i1*(-B26*L*cosa*sina*sin(pi*k1*xa/L) + pi*k1*r*(-A45*r + B26*cosa)*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*i1*(B26*L*cosa*sina*sin(pi*k1*xb/L) + pi*k1*r*(A45*r - B26*cosa)*cos(pi*k1*xb/L))*cos(pi*i1*xb/L) + 2*pi*r*(A45*i1**2*r - B26*cosa*k1**2)*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*(B26*L*cosa*k1*sina*cos(pi*k1*xa/L) + pi*r*(-A45*i1**2*r + B26*cosa*k1**2)*sin(pi*k1*xa/L))*sin(pi*i1*xa/L))/(r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+0
                    k0v[c] += (-2*i1*(sin(pi*i1*xa/L)*sin(pi*k1*xa/L) - sin(pi*i1*xb/L)*sin(pi*k1*xb/L))*(pi**2*B11*k1**2*r**2 + B22*L**2*sina**2) - 2*(pi*B12*L*r*sina*(i1 - k1)*(i1 + k1)*sin(pi*k1*xa/L) + k1*(pi**2*B11*i1**2*r**2 + B22*L**2*sina**2)*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*(pi*B12*L*r*sina*(i1 - k1)*(i1 + k1)*sin(pi*k1*xb/L) + k1*(pi**2*B11*i1**2*r**2 + B22*L**2*sina**2)*cos(pi*k1*xb/L))*cos(pi*i1*xb/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+1
                    k0v[c] += (-2*i1*(-pi*L*k1*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + (pi*L*k1*r*sina*(D16 + D26 + r*(B16 + B26))*cos(pi*k1*xa/L) - (A45*L**2*cosa*r**2 + L**2*sina**2*(B26*r + D26) - pi**2*k1**2*r**2*(B16*r + D16))*sin(pi*k1*xa/L))*sin(pi*i1*xa/L) + (A45*L**2*cosa*r**2 + L**2*sina**2*(B26*r + D26) - pi**2*k1**2*r**2*(B16*r + D16))*sin(pi*i1*xb/L)*sin(pi*k1*xb/L)) - 2*(-pi*L*r*sina*(B16*i1**2*r + D16*i1**2 + k1**2*(B26*r + D26))*sin(pi*k1*xa/L) - k1*(A45*L**2*cosa*r**2 + L**2*sina**2*(B26*r + D26) - pi**2*i1**2*r**2*(B16*r + D16))*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) - 2*(pi*L*r*sina*(B16*i1**2*r + D16*i1**2 + k1**2*(B26*r + D26))*sin(pi*k1*xb/L) + k1*(A45*L**2*cosa*r**2 + L**2*sina**2*(B26*r + D26) - pi**2*i1**2*r**2*(B16*r + D16))*cos(pi*k1*xb/L))*cos(pi*i1*xb/L))/(L*r**2*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+2
                    k0v[c] += (2*i1*(-B22*L*cosa*sina*sin(pi*i1*xa/L)*sin(pi*k1*xa/L) + pi*k1*r*(-A55*r + B12*cosa)*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + (B22*L*cosa*sina*sin(pi*k1*xb/L) + pi*k1*r*(A55*r - B12*cosa)*cos(pi*k1*xb/L))*sin(pi*i1*xb/L)) - 2*(B22*L*cosa*k1*sina*cos(pi*k1*xa/L) + pi*r*(-A55*k1**2*r + B12*cosa*i1**2)*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*(B22*L*cosa*k1*sina*cos(pi*k1*xb/L) + pi*r*(-A55*k1**2*r + B12*cosa*i1**2)*sin(pi*k1*xb/L))*cos(pi*i1*xb/L))/(r*(i1**2 - k1**2))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+3
                    k0v[c] += (2*i1*(-sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + sin(pi*i1*xb/L)*cos(pi*k1*xb/L))*(D22*L**2*sina**2 + r**2*(A55*L**2 + pi**2*D11*k1**2)) + 2*(pi*D12*L*r*sina*(-i1**2 + k1**2)*cos(pi*k1*xa/L) + k1*(D22*L**2*sina**2 + r**2*(A55*L**2 + pi**2*D11*i1**2))*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*(pi*D12*L*r*sina*(i1 - k1)*(i1 + k1)*cos(pi*k1*xb/L) - k1*(D22*L**2*sina**2 + r**2*(A55*L**2 + pi**2*D11*i1**2))*sin(pi*k1*xb/L))*cos(pi*i1*xb/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+4
                    k0v[c] += (2*i1*(pi*L*k1*r*sina*(D16 + D26)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + (-D26*L**2*sina**2 + r**2*(A45*L**2 + pi**2*D16*k1**2))*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + (-pi*L*k1*r*sina*(D16 + D26)*cos(pi*k1*xa/L) - (-D26*L**2*sina**2 + r**2*(A45*L**2 + pi**2*D16*k1**2))*sin(pi*k1*xa/L))*sin(pi*i1*xa/L)) + 2*(pi*L*r*sina*(D16*i1**2 + D26*k1**2)*sin(pi*k1*xa/L) - k1*(-D26*L**2*sina**2 + r**2*(A45*L**2 + pi**2*D16*i1**2))*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*(-pi*L*r*sina*(D16*i1**2 + D26*k1**2)*sin(pi*k1*xb/L) + k1*(-D26*L**2*sina**2 + r**2*(A45*L**2 + pi**2*D16*i1**2))*cos(pi*k1*xb/L))*cos(pi*i1*xb/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+0
                    k0v[c] += (2*pi*L*r*sina*(B16*k1**2 + B26*i1**2)*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*i1*(-pi*L*k1*r*sina*(B16 + B26)*cos(pi*k1*xa/L) + (pi**2*B16*k1**2*r**2 - B26*L**2*sina**2)*sin(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*i1*(pi*L*k1*r*sina*(B16 + B26)*cos(pi*k1*xb/L) + (-pi**2*B16*k1**2*r**2 + B26*L**2*sina**2)*sin(pi*k1*xb/L))*cos(pi*i1*xb/L) + 2*k1*(pi**2*B16*i1**2*r**2 - B26*L**2*sina**2)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*(-pi*L*r*sina*(B16*k1**2 + B26*i1**2)*sin(pi*k1*xa/L) + k1*(-pi**2*B16*i1**2*r**2 + B26*L**2*sina**2)*cos(pi*k1*xa/L))*sin(pi*i1*xa/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+1
                    k0v[c] += (2*k1*(-A44*L**2*cosa*r**2 + (B66*r + D66)*(L**2*sina**2 + pi**2*i1**2*r**2))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*k1*(A44*L**2*cosa*r**2 - (B66*r + D66)*(L**2*sina**2 + pi**2*i1**2*r**2))*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + 2*(pi*L*r*sina*(i1 - k1)*(i1 + k1)*(B66*r + D66)*sin(pi*i1*xa/L) + i1*(-A44*L**2*cosa*r**2 + (B66*r + D66)*(L**2*sina**2 + pi**2*k1**2*r**2))*cos(pi*i1*xa/L))*sin(pi*k1*xa/L) - 2*(pi*L*r*sina*(i1 - k1)*(i1 + k1)*(B66*r + D66)*sin(pi*i1*xb/L) + i1*(-A44*L**2*cosa*r**2 + (B66*r + D66)*(L**2*sina**2 + pi**2*k1**2*r**2))*cos(pi*i1*xb/L))*sin(pi*k1*xb/L))/(L*r**2*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+2
                    k0v[c] += (-2*B26*L*cosa*k1*sina*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) - 2*i1*(B26*L*cosa*sina*sin(pi*k1*xa/L) + pi*k1*r*(-A45*r + B26*cosa)*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*i1*(B26*L*cosa*sina*sin(pi*k1*xb/L) + pi*k1*r*(-A45*r + B26*cosa)*cos(pi*k1*xb/L))*cos(pi*i1*xb/L) + 2*pi*r*(-A45*k1**2*r + B26*cosa*i1**2)*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*(B26*L*cosa*k1*sina*cos(pi*k1*xa/L) + pi*r*(A45*k1**2*r - B26*cosa*i1**2)*sin(pi*k1*xa/L))*sin(pi*i1*xa/L))/(r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+3
                    k0v[c] += (2*pi*L*r*sina*(D16*k1**2 + D26*i1**2)*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*i1*(pi*L*k1*r*sina*(D16 + D26)*sin(pi*k1*xa/L) + (-D26*L**2*sina**2 + r**2*(A45*L**2 + pi**2*D16*k1**2))*cos(pi*k1*xa/L))*cos(pi*i1*xa/L) + 2*i1*(-pi*L*k1*r*sina*(D16 + D26)*sin(pi*k1*xb/L) - (-D26*L**2*sina**2 + r**2*(A45*L**2 + pi**2*D16*k1**2))*cos(pi*k1*xb/L))*cos(pi*i1*xb/L) - 2*k1*(-D26*L**2*sina**2 + r**2*(A45*L**2 + pi**2*D16*i1**2))*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) + 2*(-pi*L*r*sina*(D16*k1**2 + D26*i1**2)*cos(pi*k1*xa/L) + k1*(-D26*L**2*sina**2 + r**2*(A45*L**2 + pi**2*D16*i1**2))*sin(pi*k1*xa/L))*sin(pi*i1*xa/L))/(L*r*(i1 - k1)*(i1 + k1))
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+4
                    k0v[c] += (-2*k1*(D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i1**2))*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + 2*k1*(D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i1**2))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) + 2*(pi*D66*L*r*sina*(i1 - k1)*(i1 + k1)*sin(pi*i1*xa/L) + i1*(D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*k1**2))*cos(pi*i1*xa/L))*sin(pi*k1*xa/L) - 2*(pi*D66*L*r*sina*(i1 - k1)*(i1 + k1)*sin(pi*i1*xb/L) + i1*(D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*k1**2))*cos(pi*i1*xb/L))*sin(pi*k1*xb/L))/(L*r*(i1 - k1)*(i1 + k1))

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
                                k0v[c] += 0.25*(2*L*(2*pi*A12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (-pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += 0.25*pi*(L**2*(j2 - sina)*(j2 + sina)*(A26*r + B26)*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(pi*i2) - r*(2*L*(L*sina*(A16*r - A26*r + B16 - B26)*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(A16*r + B16)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(A16*r + B16)))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += -0.25*j2*(pi*i2*r*(-A12*r + A66*r - B12 + B66)*cos(2*pi*i2*xa/L) + pi*i2*r*(A12*r - A66*r + B12 - B66)*cos(2*pi*i2*xb/L) - sina*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb))*(B22 + B66 + r*(A22 + A66)))/(i2*r**2)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += 0.25*cosa*(pi*A12*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L)) + A22*sina*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(i2*r))
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+5
                                k0v[c] += 0.25*A26*cosa*j2*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(i2*r)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+6
                                k0v[c] += 0.25*(2*L*(2*pi*B12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (-pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+8
                                k0v[c] += 0.5*(L*(pi*L*i2*r*sina*(-B16 + B26)*sin(pi*i2*(xa + xb)/L) + (pi**2*B16*i2**2*r**2 + B26*L**2*(-j2**2 + sina**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - pi*i2*(xa - xb)*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+9
                                k0v[c] += 0.25*j2*(pi*i2*r*(-B12 + B66)*cos(2*pi*i2*xb/L) + pi*i2*r*(B12 - B66)*cos(2*pi*i2*xa/L) + sina*(B22 + B66)*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb)))/(i2*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += 0.25*(2*L*(2*pi*A12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (-pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += -0.25*j2*(pi*i2*r*(-A12*r + A66*r - B12 + B66)*cos(2*pi*i2*xb/L) + pi*i2*r*(A12*r - A66*r + B12 - B66)*cos(2*pi*i2*xa/L) + sina*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb))*(B22 + B66 + r*(A22 + A66)))/(i2*r**2)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+3
                                k0v[c] += 0.25*pi*(L**2*(j2 - sina)*(j2 + sina)*(A26*r + B26)*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(pi*i2) - r*(2*L*(L*sina*(A16*r - A26*r + B16 - B26)*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(A16*r + B16)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(A16*r + B16)))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+4
                                k0v[c] += 0.25*A26*cosa*j2*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+5
                                k0v[c] += 0.25*cosa*(pi*A12*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L)) + A22*sina*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(i2*r))
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+7
                                k0v[c] += 0.25*(2*L*(2*pi*B12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (-pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+8
                                k0v[c] += 0.25*j2*(pi*i2*r*(-B12 + B66)*cos(2*pi*i2*xa/L) + pi*i2*r*(B12 - B66)*cos(2*pi*i2*xb/L) + sina*(B22 + B66)*(L*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb)))/(i2*r)
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+9
                                k0v[c] += 0.5*(L*(pi*L*i2*r*sina*(-B16 + B26)*sin(pi*i2*(xa + xb)/L) + (pi**2*B16*i2**2*r**2 + B26*L**2*(-j2**2 + sina**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - pi*i2*(xa - xb)*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += 0.25*pi*(L**2*(j2 - sina)*(j2 + sina)*(A26*r + B26)*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(pi*i2) - r*(2*L*(L*sina*(A16*r - A26*r + B16 - B26)*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(A16*r + B16)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(A16*r + B16)))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += -0.25*j2*(pi*i2*r*(-A12*r + A66*r - B12 + B66)*cos(2*pi*i2*xb/L) + pi*i2*r*(A12*r - A66*r + B12 - B66)*cos(2*pi*i2*xa/L) + sina*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb))*(B22 + B66 + r*(A22 + A66)))/(i2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += 0.25*pi*(-L**2*(D22*j2**2 + r*(A44*cosa**2*r + j2**2*(A22*r + 2*B22)) + sina**2*(D66 + r*(A66*r + 2*B66)))*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(pi*i2) + r*(D66 + r*(A66*r + 2*B66))*(2*L**2*sina*cos(2*pi*i2*xa/L) - 2*L**2*sina*cos(2*pi*i2*xb/L) + pi*i2*r*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb))))/(L**2*r**3)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+4
                                k0v[c] += 0.25*cosa*(2*pi*i2*sina*(xa - xb)*(A26*r + B26) + 2*(L*sina*(A26*r + B26)*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(B26 + r*(A26 - A45))*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+5
                                k0v[c] += 0.25*cosa*j2*(B22 + r*(A22 + A44))*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(i2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+6
                                k0v[c] += 0.25*pi*(-L**2*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))*(-A45*cosa*r**2 + B26*j2**2*r + D26*j2**2 - sina**2*(B26*r + D26))/(pi*i2) - r*(2*L*(L*sina*(B16*r - B26*r + D16 - D26)*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(B16*r + D16)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(B16*r + D16)))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+7
                                k0v[c] += -0.25*j2*(pi*i2*r*(-B12*r + B66*r - D12 + D66)*cos(2*pi*i2*xb/L) + pi*i2*r*(B12*r - B66*r + D12 - D66)*cos(2*pi*i2*xa/L) + sina*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb))*(D22 + D66 + r*(B22 + B66)))/(i2*r**2)
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+8
                                k0v[c] += 0.25*pi*(L**2*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))*(-A44*cosa*r**2 + B22*j2**2*r + D22*j2**2 + sina**2*(B66*r + D66))/(pi*i2) + r*(B66*r + D66)*(2*L**2*sina*cos(2*pi*i2*xa/L) - 2*L**2*sina*cos(2*pi*i2*xb/L) + pi*i2*r*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb))))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += 0.25*j2*(pi*i2*r*(-A12*r + A66*r - B12 + B66)*cos(2*pi*i2*xb/L) + pi*i2*r*(A12*r - A66*r + B12 - B66)*cos(2*pi*i2*xa/L) + sina*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb))*(B22 + B66 + r*(A22 + A66)))/(i2*r**2)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+1
                                k0v[c] += 0.25*pi*(L**2*(j2 - sina)*(j2 + sina)*(A26*r + B26)*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(pi*i2) - r*(2*L*(L*sina*(A16*r - A26*r + B16 - B26)*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(A16*r + B16)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(A16*r + B16)))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += 0.25*pi*(-L**2*(D22*j2**2 + r*(A44*cosa**2*r + j2**2*(A22*r + 2*B22)) + sina**2*(D66 + r*(A66*r + 2*B66)))*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(pi*i2) + r*(D66 + r*(A66*r + 2*B66))*(2*L**2*sina*cos(2*pi*i2*xa/L) - 2*L**2*sina*cos(2*pi*i2*xb/L) + pi*i2*r*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb))))/(L**2*r**3)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += 0.25*cosa*j2*(B22 + r*(A22 + A44))*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r**2)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+5
                                k0v[c] += 0.25*cosa*(2*pi*i2*sina*(xa - xb)*(A26*r + B26) + 2*(L*sina*(A26*r + B26)*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(B26 + r*(A26 - A45))*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r**2)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+6
                                k0v[c] += 0.25*j2*(pi*i2*r*(-B12*r + B66*r - D12 + D66)*cos(2*pi*i2*xb/L) + pi*i2*r*(B12*r - B66*r + D12 - D66)*cos(2*pi*i2*xa/L) + sina*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb))*(D22 + D66 + r*(B22 + B66)))/(i2*r**2)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+7
                                k0v[c] += 0.25*pi*(-L**2*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))*(-A45*cosa*r**2 + B26*j2**2*r + D26*j2**2 - sina**2*(B26*r + D26))/(pi*i2) - r*(2*L*(L*sina*(B16*r - B26*r + D16 - D26)*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(B16*r + D16)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(B16*r + D16)))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+9
                                k0v[c] += 0.25*pi*(L**2*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))*(-A44*cosa*r**2 + B22*j2**2*r + D22*j2**2 + sina**2*(B66*r + D66))/(pi*i2) + r*(B66*r + D66)*(2*L**2*sina*cos(2*pi*i2*xa/L) - 2*L**2*sina*cos(2*pi*i2*xb/L) + pi*i2*r*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb))))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += 0.25*cosa*(pi*A12*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L)) + A22*sina*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(i2*r))
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+1
                                k0v[c] += 0.25*A26*cosa*j2*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+2
                                k0v[c] += 0.25*cosa*(2*pi*i2*sina*(xa - xb)*(A26*r + B26) + 2*(L*sina*(A26*r + B26)*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(B26 + r*(A26 - A45))*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r**2)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += 0.25*cosa*j2*(B22 + r*(A22 + A44))*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r**2)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += 0.25*(-L*(-pi**2*A55*i2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) - 2*pi*i2*(xa - xb)*(pi**2*A55*i2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+6
                                k0v[c] += 0.5*(pi*B22*cosa*i2*sina*(-xa + xb) + (-B22*L*cosa*sina*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(A55*r + B12*cosa)*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+7
                                k0v[c] += 0.25*j2*(-A45*r + B26*cosa)*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+8
                                k0v[c] += 0.5*(pi*B26*cosa*i2*sina*(xa - xb) + (B26*L*cosa*sina*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(A45*r + B26*cosa)*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r)
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+9
                                k0v[c] += 0.25*j2*(-A44*r + B22*cosa)*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+0
                                k0v[c] += 0.25*A26*cosa*j2*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(i2*r)
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+1
                                k0v[c] += 0.25*cosa*(pi*A12*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L)) + A22*sina*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(i2*r))
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+2
                                k0v[c] += 0.25*cosa*j2*(B22 + r*(A22 + A44))*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(i2*r**2)
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+3
                                k0v[c] += 0.25*cosa*(2*pi*i2*sina*(xa - xb)*(A26*r + B26) + 2*(L*sina*(A26*r + B26)*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(B26 + r*(A26 - A45))*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r**2)
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+5
                                k0v[c] += 0.25*(-L*(-pi**2*A55*i2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) - 2*pi*i2*(xa - xb)*(pi**2*A55*i2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+6
                                k0v[c] += 0.25*j2*(A45*r - B26*cosa)*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+7
                                k0v[c] += 0.5*(pi*B22*cosa*i2*sina*(-xa + xb) + (-B22*L*cosa*sina*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(A55*r + B12*cosa)*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r)
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+8
                                k0v[c] += 0.25*j2*(A44*r - B22*cosa)*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+9
                                k0v[c] += 0.5*(pi*B26*cosa*i2*sina*(xa - xb) + (B26*L*cosa*sina*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(A45*r + B26*cosa)*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r)
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+0
                                k0v[c] += 0.25*(2*L*(2*pi*B12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (-pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+2
                                k0v[c] += 0.25*pi*(-L**2*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))*(-A45*cosa*r**2 + B26*j2**2*r + D26*j2**2 - sina**2*(B26*r + D26))/(pi*i2) - r*(2*L*(L*sina*(B16*r - B26*r + D16 - D26)*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(B16*r + D16)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(B16*r + D16)))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+3
                                k0v[c] += 0.25*j2*(pi*i2*r*(-B12*r + B66*r - D12 + D66)*cos(2*pi*i2*xb/L) + pi*i2*r*(B12*r - B66*r + D12 - D66)*cos(2*pi*i2*xa/L) + sina*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb))*(D22 + D66 + r*(B22 + B66)))/(i2*r**2)
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+4
                                k0v[c] += 0.5*(pi*B22*cosa*i2*sina*(-xa + xb) + (-B22*L*cosa*sina*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(A55*r + B12*cosa)*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r)
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+5
                                k0v[c] += 0.25*j2*(A45*r - B26*cosa)*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+6
                                k0v[c] += 0.25*(2*L*(2*pi*D12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 - pi**2*D11*i2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 + pi**2*D11*i2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+8
                                k0v[c] += 0.25*(2*L*(pi*L*i2*r*sina*(-D16 + D26)*sin(pi*i2*(xa + xb)/L) + (D26*L**2*(-j2**2 + sina**2) + r**2*(-A45*L**2 + pi**2*D16*i2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+9
                                k0v[c] += 0.25*j2*(pi*i2*r*(-D12 + D66)*cos(2*pi*i2*xb/L) + pi*i2*r*(D12 - D66)*cos(2*pi*i2*xa/L) + sina*(D22 + D66)*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb)))/(i2*r)
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+1
                                k0v[c] += 0.25*(2*L*(2*pi*B12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (-pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+2
                                k0v[c] += -0.25*j2*(pi*i2*r*(-B12*r + B66*r - D12 + D66)*cos(2*pi*i2*xb/L) + pi*i2*r*(B12*r - B66*r + D12 - D66)*cos(2*pi*i2*xa/L) + sina*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb))*(D22 + D66 + r*(B22 + B66)))/(i2*r**2)
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+3
                                k0v[c] += 0.25*pi*(-L**2*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))*(-A45*cosa*r**2 + B26*j2**2*r + D26*j2**2 - sina**2*(B26*r + D26))/(pi*i2) - r*(2*L*(L*sina*(B16*r - B26*r + D16 - D26)*sin(pi*i2*(xa + xb)/L) - pi*i2*r*(B16*r + D16)*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) + 2*pi**2*i2**2*r*(xa - xb)*(B16*r + D16)))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+4
                                k0v[c] += 0.25*j2*(-A45*r + B26*cosa)*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+5
                                k0v[c] += 0.5*(pi*B22*cosa*i2*sina*(-xa + xb) + (-B22*L*cosa*sina*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(A55*r + B12*cosa)*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r)
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+7
                                k0v[c] += 0.25*(2*L*(2*pi*D12*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 - pi**2*D11*i2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 + pi**2*D11*i2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+8
                                k0v[c] += 0.25*j2*(pi*i2*r*(-D12 + D66)*cos(2*pi*i2*xa/L) + pi*i2*r*(D12 - D66)*cos(2*pi*i2*xb/L) + sina*(D22 + D66)*(L*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb)))/(i2*r)
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+9
                                k0v[c] += 0.25*(2*L*(pi*L*i2*r*sina*(-D16 + D26)*sin(pi*i2*(xa + xb)/L) + (D26*L**2*(-j2**2 + sina**2) + r**2*(-A45*L**2 + pi**2*D16*i2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+0
                                k0v[c] += 0.5*(L*(pi*L*i2*r*sina*(-B16 + B26)*sin(pi*i2*(xa + xb)/L) + (pi**2*B16*i2**2*r**2 + B26*L**2*(-j2**2 + sina**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - pi*i2*(xa - xb)*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+1
                                k0v[c] += 0.25*j2*(pi*i2*r*(-B12 + B66)*cos(2*pi*i2*xa/L) + pi*i2*r*(B12 - B66)*cos(2*pi*i2*xb/L) + sina*(B22 + B66)*(L*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb)))/(i2*r)
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+2
                                k0v[c] += 0.25*pi*(L**2*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))*(-A44*cosa*r**2 + B22*j2**2*r + D22*j2**2 + sina**2*(B66*r + D66))/(pi*i2) + r*(B66*r + D66)*(2*L**2*sina*cos(2*pi*i2*xa/L) - 2*L**2*sina*cos(2*pi*i2*xb/L) + pi*i2*r*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb))))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+4
                                k0v[c] += 0.5*(pi*B26*cosa*i2*sina*(xa - xb) + (B26*L*cosa*sina*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(A45*r + B26*cosa)*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r)
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+5
                                k0v[c] += 0.25*j2*(A44*r - B22*cosa)*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+6
                                k0v[c] += 0.25*(2*L*(pi*L*i2*r*sina*(-D16 + D26)*sin(pi*i2*(xa + xb)/L) + (D26*L**2*(-j2**2 + sina**2) + r**2*(-A45*L**2 + pi**2*D16*i2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+7
                                k0v[c] += 0.25*j2*(pi*i2*r*(-D12 + D66)*cos(2*pi*i2*xa/L) + pi*i2*r*(D12 - D66)*cos(2*pi*i2*xb/L) + sina*(D22 + D66)*(L*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb)))/(i2*r)
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+8
                                k0v[c] += 0.25*(2*L*(-2*pi*D66*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 - pi**2*D66*i2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+0
                                k0v[c] += 0.25*j2*(pi*i2*r*(-B12 + B66)*cos(2*pi*i2*xb/L) + pi*i2*r*(B12 - B66)*cos(2*pi*i2*xa/L) + sina*(B22 + B66)*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb)))/(i2*r)
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+1
                                k0v[c] += 0.5*(L*(pi*L*i2*r*sina*(-B16 + B26)*sin(pi*i2*(xa + xb)/L) + (pi**2*B16*i2**2*r**2 + B26*L**2*(-j2**2 + sina**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - pi*i2*(xa - xb)*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+3
                                k0v[c] += 0.25*pi*(L**2*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))*(-A44*cosa*r**2 + B22*j2**2*r + D22*j2**2 + sina**2*(B66*r + D66))/(pi*i2) + r*(B66*r + D66)*(2*L**2*sina*cos(2*pi*i2*xa/L) - 2*L**2*sina*cos(2*pi*i2*xb/L) + pi*i2*r*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(-xa + xb))))/(L**2*r**2)
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+4
                                k0v[c] += 0.25*j2*(-A44*r + B22*cosa)*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(i2*r)
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+5
                                k0v[c] += 0.5*(pi*B26*cosa*i2*sina*(xa - xb) + (B26*L*cosa*sina*cos(pi*i2*(xa + xb)/L) + pi*i2*r*(A45*r + B26*cosa)*sin(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L))/(i2*r)
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+6
                                k0v[c] += 0.25*j2*(pi*i2*r*(-D12 + D66)*cos(2*pi*i2*xb/L) + pi*i2*r*(D12 - D66)*cos(2*pi*i2*xa/L) + sina*(D22 + D66)*(L*(sin(2*pi*i2*xa/L) - sin(2*pi*i2*xb/L)) + 2*pi*i2*(xa - xb)))/(i2*r)
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+7
                                k0v[c] += 0.25*(2*L*(pi*L*i2*r*sina*(-D16 + D26)*sin(pi*i2*(xa + xb)/L) + (D26*L**2*(-j2**2 + sina**2) + r**2*(-A45*L**2 + pi**2*D16*i2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2)))/(L**2*i2*r)
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+9
                                k0v[c] += 0.25*(2*L*(-2*pi*D66*L*i2*r*sina*sin(pi*i2*(xa + xb)/L) - (D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 - pi**2*D66*i2**2))*cos(pi*i2*(xa + xb)/L))*sin(pi*i2*(xa - xb)/L) - 2*pi*i2*(xa - xb)*(D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i2**2)))/(L**2*i2*r)

                            else:
                                # k0_22 cond_5
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+0
                                k0v[c] += pi*(-xa + xb)*(A22*sina**2 + A66*j2**2)/r
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+2
                                k0v[c] += pi*(j2 - sina)*(j2 + sina)*(-xa + xb)*(A26*r + B26)/r**2
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+3
                                k0v[c] += pi*j2*sina*(xa - xb)*(B22 + B66 + r*(A22 + A66))/r**2
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+4
                                k0v[c] += pi*A22*cosa*sina*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+5
                                k0v[c] += pi*A26*cosa*j2*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+6
                                k0v[c] += pi*(-xa + xb)*(B22*sina**2 + B66*j2**2)/r
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+8
                                k0v[c] += pi*B26*(j2 - sina)*(j2 + sina)*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+0
                                k0c[c] = col+9
                                k0v[c] += pi*j2*sina*(B22 + B66)*(xa - xb)/r
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+1
                                k0v[c] += pi*(-xa + xb)*(A22*sina**2 + A66*j2**2)/r
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+2
                                k0v[c] += pi*j2*sina*(-xa + xb)*(B22 + B66 + r*(A22 + A66))/r**2
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+3
                                k0v[c] += -pi*(j2 - sina)*(j2 + sina)*(xa - xb)*(A26*r + B26)/r**2
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+4
                                k0v[c] += pi*A26*cosa*j2*(xa - xb)/r
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+5
                                k0v[c] += pi*A22*cosa*sina*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+7
                                k0v[c] += pi*(-xa + xb)*(B22*sina**2 + B66*j2**2)/r
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+8
                                k0v[c] += pi*j2*sina*(B22 + B66)*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+1
                                k0c[c] = col+9
                                k0v[c] += -pi*B26*(j2 - sina)*(j2 + sina)*(xa - xb)/r
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+0
                                k0v[c] += pi*(j2 - sina)*(j2 + sina)*(-xa + xb)*(A26*r + B26)/r**2
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+1
                                k0v[c] += pi*j2*sina*(-xa + xb)*(B22 + B66 + r*(A22 + A66))/r**2
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+2
                                k0v[c] += -pi*(xa - xb)*(D22*j2**2 + r*(A44*cosa**2*r + j2**2*(A22*r + 2*B22)) + sina**2*(D66 + r*(A66*r + 2*B66)))/r**3
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+4
                                k0v[c] += pi*cosa*sina*(xa - xb)*(A26*r + B26)/r**2
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+5
                                k0v[c] += pi*cosa*j2*(B22 + r*(A22 + A44))*(-xa + xb)/r**2
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+6
                                k0v[c] += pi*(xa - xb)*(A45*cosa*r**2 - B26*j2**2*r - D26*j2**2 + sina**2*(B26*r + D26))/r**2
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+7
                                k0v[c] += pi*j2*sina*(-xa + xb)*(D22 + D66 + r*(B22 + B66))/r**2
                                c += 1
                                k0r[c] = row+2
                                k0c[c] = col+8
                                k0v[c] += pi*(-xa + xb)*(-A44*cosa*r**2 + B22*j2**2*r + D22*j2**2 + sina**2*(B66*r + D66))/r**2
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+0
                                k0v[c] += pi*j2*sina*(xa - xb)*(B22 + B66 + r*(A22 + A66))/r**2
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+1
                                k0v[c] += -pi*(j2 - sina)*(j2 + sina)*(xa - xb)*(A26*r + B26)/r**2
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+3
                                k0v[c] += -pi*(xa - xb)*(D22*j2**2 + r*(A44*cosa**2*r + j2**2*(A22*r + 2*B22)) + sina**2*(D66 + r*(A66*r + 2*B66)))/r**3
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+4
                                k0v[c] += pi*cosa*j2*(B22 + r*(A22 + A44))*(xa - xb)/r**2
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+5
                                k0v[c] += pi*cosa*sina*(xa - xb)*(A26*r + B26)/r**2
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+6
                                k0v[c] += pi*j2*sina*(xa - xb)*(D22 + D66 + r*(B22 + B66))/r**2
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+7
                                k0v[c] += pi*(xa - xb)*(A45*cosa*r**2 - B26*j2**2*r - D26*j2**2 + sina**2*(B26*r + D26))/r**2
                                c += 1
                                k0r[c] = row+3
                                k0c[c] = col+9
                                k0v[c] += pi*(-xa + xb)*(-A44*cosa*r**2 + B22*j2**2*r + D22*j2**2 + sina**2*(B66*r + D66))/r**2
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+0
                                k0v[c] += pi*A22*cosa*sina*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+1
                                k0v[c] += pi*A26*cosa*j2*(xa - xb)/r
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+2
                                k0v[c] += pi*cosa*sina*(xa - xb)*(A26*r + B26)/r**2
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+3
                                k0v[c] += pi*cosa*j2*(B22 + r*(A22 + A44))*(xa - xb)/r**2
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+4
                                k0v[c] += pi*(-xa + xb)*(A22*cosa**2 + A44*j2**2)/r
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+6
                                k0v[c] += pi*B22*cosa*sina*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+7
                                k0v[c] += pi*j2*(-xa + xb)*(A45*r - B26*cosa)/r
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+8
                                k0v[c] += pi*B26*cosa*sina*(xa - xb)/r
                                c += 1
                                k0r[c] = row+4
                                k0c[c] = col+9
                                k0v[c] += pi*j2*(-xa + xb)*(A44*r - B22*cosa)/r
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+0
                                k0v[c] += pi*A26*cosa*j2*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+1
                                k0v[c] += pi*A22*cosa*sina*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+2
                                k0v[c] += pi*cosa*j2*(B22 + r*(A22 + A44))*(-xa + xb)/r**2
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+3
                                k0v[c] += pi*cosa*sina*(xa - xb)*(A26*r + B26)/r**2
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+5
                                k0v[c] += pi*(-xa + xb)*(A22*cosa**2 + A44*j2**2)/r
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+6
                                k0v[c] += pi*j2*(xa - xb)*(A45*r - B26*cosa)/r
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+7
                                k0v[c] += pi*B22*cosa*sina*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+8
                                k0v[c] += pi*j2*(xa - xb)*(A44*r - B22*cosa)/r
                                c += 1
                                k0r[c] = row+5
                                k0c[c] = col+9
                                k0v[c] += pi*B26*cosa*sina*(xa - xb)/r
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+0
                                k0v[c] += pi*(-xa + xb)*(B22*sina**2 + B66*j2**2)/r
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+2
                                k0v[c] += pi*(xa - xb)*(A45*cosa*r**2 - B26*j2**2*r - D26*j2**2 + sina**2*(B26*r + D26))/r**2
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+3
                                k0v[c] += pi*j2*sina*(xa - xb)*(D22 + D66 + r*(B22 + B66))/r**2
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+4
                                k0v[c] += pi*B22*cosa*sina*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+5
                                k0v[c] += pi*j2*(xa - xb)*(A45*r - B26*cosa)/r
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+6
                                k0v[c] += pi*(-xa + xb)*(A55*r**2 + D22*sina**2 + D66*j2**2)/r
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+8
                                k0v[c] += pi*(-xa + xb)*(A45*r**2 + D26*(j2 - sina)*(j2 + sina))/r
                                c += 1
                                k0r[c] = row+6
                                k0c[c] = col+9
                                k0v[c] += pi*j2*sina*(D22 + D66)*(xa - xb)/r
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+1
                                k0v[c] += pi*(-xa + xb)*(B22*sina**2 + B66*j2**2)/r
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+2
                                k0v[c] += pi*j2*sina*(-xa + xb)*(D22 + D66 + r*(B22 + B66))/r**2
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+3
                                k0v[c] += pi*(xa - xb)*(A45*cosa*r**2 - B26*j2**2*r - D26*j2**2 + sina**2*(B26*r + D26))/r**2
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+4
                                k0v[c] += pi*j2*(-xa + xb)*(A45*r - B26*cosa)/r
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+5
                                k0v[c] += pi*B22*cosa*sina*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+7
                                k0v[c] += pi*(-xa + xb)*(A55*r**2 + D22*sina**2 + D66*j2**2)/r
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+8
                                k0v[c] += pi*j2*sina*(D22 + D66)*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+7
                                k0c[c] = col+9
                                k0v[c] += pi*(-xa + xb)*(A45*r**2 + D26*(j2 - sina)*(j2 + sina))/r
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+0
                                k0v[c] += pi*B26*(j2 - sina)*(j2 + sina)*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+1
                                k0v[c] += pi*j2*sina*(B22 + B66)*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+2
                                k0v[c] += pi*(-xa + xb)*(-A44*cosa*r**2 + B22*j2**2*r + D22*j2**2 + sina**2*(B66*r + D66))/r**2
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+4
                                k0v[c] += pi*B26*cosa*sina*(xa - xb)/r
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+5
                                k0v[c] += pi*j2*(xa - xb)*(A44*r - B22*cosa)/r
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+6
                                k0v[c] += pi*(-xa + xb)*(A45*r**2 + D26*(j2 - sina)*(j2 + sina))/r
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+7
                                k0v[c] += pi*j2*sina*(D22 + D66)*(-xa + xb)/r
                                c += 1
                                k0r[c] = row+8
                                k0c[c] = col+8
                                k0v[c] += pi*(-xa + xb)*(A44*r**2 + D22*j2**2 + D66*sina**2)/r
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+0
                                k0v[c] += pi*j2*sina*(B22 + B66)*(xa - xb)/r
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+1
                                k0v[c] += -pi*B26*(j2 - sina)*(j2 + sina)*(xa - xb)/r
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+3
                                k0v[c] += pi*(-xa + xb)*(-A44*cosa*r**2 + B22*j2**2*r + D22*j2**2 + sina**2*(B66*r + D66))/r**2
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+4
                                k0v[c] += pi*j2*(-xa + xb)*(A44*r - B22*cosa)/r
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+5
                                k0v[c] += pi*B26*cosa*sina*(xa - xb)/r
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+6
                                k0v[c] += pi*j2*sina*(D22 + D66)*(xa - xb)/r
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+7
                                k0v[c] += pi*(-xa + xb)*(A45*r**2 + D26*(j2 - sina)*(j2 + sina))/r
                                c += 1
                                k0r[c] = row+9
                                k0c[c] = col+9
                                k0v[c] += pi*(-xa + xb)*(A44*r**2 + D22*j2**2 + D66*sina**2)/r

                        elif k2 != i2 and l2 == j2:
                            # k0_22 cond_2
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(pi**2*A11*k2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2)) + (pi*A12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*A12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += pi*A16*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += pi*(L*(A26*r + B26)*(i2*(L*(j2 - sina)*(j2 + sina)*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - pi*k2*r*sina*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (L*(-j2**2 + sina**2)*cos(pi*k2*xa/L) + pi*k2*r*sina*sin(pi*k2*xa/L))*sin(pi*i2*xa/L)) + k2*(L*(j2 - sina)*(j2 + sina)*sin(pi*k2*xa/L) + pi*k2*r*sina*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) - k2*(L*(j2 - sina)*(j2 + sina)*sin(pi*k2*xb/L) + pi*k2*r*sina*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/pi + i2*r*(A16*r + B16)*(i2*(L*sina*cos(pi*k2*xa/L) + pi*k2*r*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - i2*(L*sina*cos(pi*k2*xb/L) + pi*k2*r*sin(pi*k2*xb/L))*cos(pi*i2*xb/L) + k2*(L*sina*sin(pi*k2*xa/L) - pi*k2*r*cos(pi*k2*xa/L))*sin(pi*i2*xa/L) + k2*(-L*sina*sin(pi*k2*xb/L) + pi*k2*r*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += -j2*(i2*(-L*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(B12 + B66 + r*(A12 + A66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(B22 + B66 + r*(A22 + A66))*cos(pi*k2*xb/L) + pi*k2*r*(B12 + B66 + r*(A12 + A66))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*k2*xa/L) - pi*r*(A12*i2**2*r + B12*i2**2 + k2**2*(A66*r + B66))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-L*k2*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*k2*xb/L) + pi*r*(A12*i2**2*r + B12*i2**2 + k2**2*(A66*r + B66))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += cosa*(i2*(pi*A12*i2*r*cos(pi*i2*xb/L) + A22*L*sina*sin(pi*i2*xb/L))*cos(pi*k2*xb/L) - i2*(pi*A12*k2*r*sin(pi*k2*xa/L) + A22*L*sina*cos(pi*k2*xa/L))*sin(pi*i2*xa/L) + k2*(pi*A12*i2*r*sin(pi*i2*xb/L) - A22*L*sina*cos(pi*i2*xb/L))*sin(pi*k2*xb/L) + (-pi*A12*i2**2*r*cos(pi*k2*xa/L) + A22*L*k2*sina*sin(pi*k2*xa/L))*cos(pi*i2*xa/L))/(r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+5
                            k0v[c] += A26*L*cosa*j2*(-i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+6
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(pi**2*B11*k2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2)) + (pi*B12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*B12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+7
                            k0v[c] += pi*B16*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+8
                            k0v[c] += (i2*(pi*L*k2*r*sina*(B16 + B26)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (pi**2*B16*k2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + (-pi*L*k2*r*sina*(B16 + B26)*sin(pi*k2*xb/L) + (pi**2*B16*k2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (pi*L*r*sina*(B16*i2**2 + B26*k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - (pi*L*r*sina*(B16*i2**2 + B26*k2**2)*cos(pi*k2*xb/L) + k2*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+9
                            k0v[c] += j2*(i2*(L*sina*(B22 + B66)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(B12 + B66)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (L*sina*(B22 + B66)*cos(pi*k2*xb/L) + pi*k2*r*(B12 + B66)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (-L*k2*sina*(B22 + B66)*sin(pi*k2*xa/L) + pi*r*(B12*i2**2 + B66*k2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(B22 + B66)*sin(pi*k2*xb/L) - pi*r*(B12*i2**2 + B66*k2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += -pi*A16*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(pi**2*A11*k2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2)) + (pi*A12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*A12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(pi**2*A11*i2**2*r**2 + L**2*(A22*sina**2 + A66*j2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += j2*(i2*(-L*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(B12 + B66 + r*(A12 + A66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(B22 + B66 + r*(A22 + A66))*cos(pi*k2*xb/L) + pi*k2*r*(B12 + B66 + r*(A12 + A66))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*k2*xa/L) - pi*r*(A12*i2**2*r + B12*i2**2 + k2**2*(A66*r + B66))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-L*k2*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*k2*xb/L) + pi*r*(A12*i2**2*r + B12*i2**2 + k2**2*(A66*r + B66))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += pi*(L*(A26*r + B26)*(i2*(L*(j2 - sina)*(j2 + sina)*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - pi*k2*r*sina*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (L*(-j2**2 + sina**2)*cos(pi*k2*xa/L) + pi*k2*r*sina*sin(pi*k2*xa/L))*sin(pi*i2*xa/L)) + k2*(L*(j2 - sina)*(j2 + sina)*sin(pi*k2*xa/L) + pi*k2*r*sina*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) - k2*(L*(j2 - sina)*(j2 + sina)*sin(pi*k2*xb/L) + pi*k2*r*sina*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/pi + i2*r*(A16*r + B16)*(i2*(L*sina*cos(pi*k2*xa/L) + pi*k2*r*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - i2*(L*sina*cos(pi*k2*xb/L) + pi*k2*r*sin(pi*k2*xb/L))*cos(pi*i2*xb/L) + k2*(L*sina*sin(pi*k2*xa/L) - pi*k2*r*cos(pi*k2*xa/L))*sin(pi*i2*xa/L) + k2*(-L*sina*sin(pi*k2*xb/L) + pi*k2*r*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += A26*L*cosa*j2*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+5
                            k0v[c] += cosa*(i2*(pi*A12*i2*r*cos(pi*i2*xb/L) + A22*L*sina*sin(pi*i2*xb/L))*cos(pi*k2*xb/L) - i2*(pi*A12*k2*r*sin(pi*k2*xa/L) + A22*L*sina*cos(pi*k2*xa/L))*sin(pi*i2*xa/L) + k2*(pi*A12*i2*r*sin(pi*i2*xb/L) - A22*L*sina*cos(pi*i2*xb/L))*sin(pi*k2*xb/L) + (-pi*A12*i2**2*r*cos(pi*k2*xa/L) + A22*L*k2*sina*sin(pi*k2*xa/L))*cos(pi*i2*xa/L))/(r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+6
                            k0v[c] += -pi*B16*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+7
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(pi**2*B11*k2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2)) + (pi*B12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*B12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+8
                            k0v[c] += j2*(i2*(-L*sina*(B22 + B66)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(B12 + B66)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(B22 + B66)*cos(pi*k2*xb/L) + pi*k2*r*(B12 + B66)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(B22 + B66)*sin(pi*k2*xa/L) - pi*r*(B12*i2**2 + B66*k2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-L*k2*sina*(B22 + B66)*sin(pi*k2*xb/L) + pi*r*(B12*i2**2 + B66*k2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+9
                            k0v[c] += (i2*(pi*L*k2*r*sina*(B16 + B26)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (pi**2*B16*k2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + (-pi*L*k2*r*sina*(B16 + B26)*sin(pi*k2*xb/L) + (pi**2*B16*k2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (pi*L*r*sina*(B16*i2**2 + B26*k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - (pi*L*r*sina*(B16*i2**2 + B26*k2**2)*cos(pi*k2*xb/L) + k2*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += (i2*(pi*L*k2*r*sina*(B16 + B26 + r*(A16 + A26))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (B26*L**2*(j2 - sina)*(j2 + sina) + r*(A26*L**2*(j2 - sina)*(j2 + sina) + pi**2*k2**2*r*(A16*r + B16)))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (-pi*L*k2*r*sina*(B16 + B26 + r*(A16 + A26))*sin(pi*k2*xa/L) - (B26*L**2*(j2 - sina)*(j2 + sina) + r*(A26*L**2*(j2 - sina)*(j2 + sina) + pi**2*k2**2*r*(A16*r + B16)))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-pi*L*r*sina*(A26*i2**2*r + B26*i2**2 + k2**2*(A16*r + B16))*cos(pi*k2*xa/L) + k2*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(A26*L**2*(j2 - sina)*(j2 + sina) + pi**2*i2**2*r*(A16*r + B16)))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*L*r*sina*(A26*i2**2*r + B26*i2**2 + k2**2*(A16*r + B16))*cos(pi*k2*xb/L) - k2*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(A26*L**2*(j2 - sina)*(j2 + sina) + pi**2*i2**2*r*(A16*r + B16)))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += j2*(i2*(-L*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(B12 + B66 + r*(A12 + A66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(B22 + B66 + r*(A22 + A66))*cos(pi*k2*xb/L) - pi*k2*r*(B12 + B66 + r*(A12 + A66))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*k2*xa/L) + pi*r*(A66*i2**2*r + B66*i2**2 + k2**2*(A12*r + B12))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-L*k2*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*k2*xb/L) - pi*r*(A66*i2**2*r + B66*i2**2 + k2**2*(A12*r + B12))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += (-i2*(sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(2*B22*L**2*j2**2*r + D22*L**2*j2**2 + L**2*sina**2*(A66*r**2 + 2*B66*r + D66) + r**2*(L**2*(A22*j2**2 + A44*cosa**2) + pi**2*k2**2*(A66*r**2 + 2*B66*r + D66))) + (pi*L*r*sina*(D66 + r*(A66*r + 2*B66))*(i2 - k2)*(i2 + k2)*cos(pi*k2*xa/L) + k2*(D22*L**2*j2**2 + L**2*sina**2*(D66 + r*(A66*r + 2*B66)) + r*(2*B22*L**2*j2**2 + r*(L**2*(A22*j2**2 + A44*cosa**2) + pi**2*i2**2*(A66*r**2 + 2*B66*r + D66))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - (pi*L*r*sina*(D66 + r*(A66*r + 2*B66))*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) + k2*(D22*L**2*j2**2 + L**2*sina**2*(D66 + r*(A66*r + 2*B66)) + r*(2*B22*L**2*j2**2 + r*(L**2*(A22*j2**2 + A44*cosa**2) + pi**2*i2**2*(A66*r**2 + 2*B66*r + D66))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**3*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+3
                            k0v[c] += pi*j2*(D26 + r*(A26*r + 2*B26))*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r**2*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+4
                            k0v[c] += cosa*(i2*(L*sina*(A26*r + B26)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(B26 + r*(A26 + A45))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (-L*sina*(A26*r + B26)*cos(pi*k2*xb/L) + pi*k2*r*(B26 + r*(A26 + A45))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) - (L*k2*sina*(A26*r + B26)*sin(pi*k2*xa/L) + pi*r*(A26*i2**2*r + A45*k2**2*r + B26*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(A26*r + B26)*sin(pi*k2*xb/L) + pi*r*(A26*i2**2*r + A45*k2**2*r + B26*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+5
                            k0v[c] += L*cosa*j2*(B22 + r*(A22 + A44))*(-i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r**2*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+6
                            k0v[c] += (i2*(pi*L*k2*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*k2**2*(B16*r + D16))))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (-pi*L*k2*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*k2*xa/L) - (D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*k2**2*(B16*r + D16))))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-pi*L*r*sina*(B26*i2**2*r + D26*i2**2 + k2**2*(B16*r + D16))*cos(pi*k2*xa/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*i2**2*(B16*r + D16))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*L*r*sina*(B26*i2**2*r + D26*i2**2 + k2**2*(B16*r + D16))*cos(pi*k2*xb/L) - k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*i2**2*(B16*r + D16))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+7
                            k0v[c] += j2*(i2*(-L*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(D12 + D66 + r*(B12 + B66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(D22 + D66 + r*(B22 + B66))*cos(pi*k2*xb/L) - pi*k2*r*(D12 + D66 + r*(B12 + B66))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*k2*xa/L) + pi*r*(B66*i2**2*r + D66*i2**2 + k2**2*(B12*r + D12))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-L*k2*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*k2*xb/L) - pi*r*(B66*i2**2*r + D66*i2**2 + k2**2*(B12*r + D12))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+8
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*k2**2*(B66*r + D66)))) + (pi*L*r*sina*(i2 - k2)*(i2 + k2)*(B66*r + D66)*cos(pi*k2*xa/L) + k2*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*i2**2*(B66*r + D66))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-pi*L*r*sina*(i2 - k2)*(i2 + k2)*(B66*r + D66)*cos(pi*k2*xb/L) - k2*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*i2**2*(B66*r + D66))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+9
                            k0v[c] += pi*j2*(B26*r + D26)*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+0
                            k0v[c] += j2*(i2*(L*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(B12 + B66 + r*(A12 + A66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (-L*sina*(B22 + B66 + r*(A22 + A66))*cos(pi*k2*xb/L) + pi*k2*r*(B12 + B66 + r*(A12 + A66))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (-L*k2*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*k2*xa/L) - pi*r*(A66*i2**2*r + B66*i2**2 + k2**2*(A12*r + B12))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(B22 + B66 + r*(A22 + A66))*sin(pi*k2*xb/L) + pi*r*(A66*i2**2*r + B66*i2**2 + k2**2*(A12*r + B12))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+1
                            k0v[c] += (i2*(pi*L*k2*r*sina*(B16 + B26 + r*(A16 + A26))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (B26*L**2*(j2 - sina)*(j2 + sina) + r*(A26*L**2*(j2 - sina)*(j2 + sina) + pi**2*k2**2*r*(A16*r + B16)))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (-pi*L*k2*r*sina*(B16 + B26 + r*(A16 + A26))*sin(pi*k2*xa/L) - (B26*L**2*(j2 - sina)*(j2 + sina) + r*(A26*L**2*(j2 - sina)*(j2 + sina) + pi**2*k2**2*r*(A16*r + B16)))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-pi*L*r*sina*(A26*i2**2*r + B26*i2**2 + k2**2*(A16*r + B16))*cos(pi*k2*xa/L) + k2*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(A26*L**2*(j2 - sina)*(j2 + sina) + pi**2*i2**2*r*(A16*r + B16)))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*L*r*sina*(A26*i2**2*r + B26*i2**2 + k2**2*(A16*r + B16))*cos(pi*k2*xb/L) - k2*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(A26*L**2*(j2 - sina)*(j2 + sina) + pi**2*i2**2*r*(A16*r + B16)))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+2
                            k0v[c] += -pi*j2*(D26 + r*(A26*r + 2*B26))*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r**2*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += (-i2*(sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(2*B22*L**2*j2**2*r + D22*L**2*j2**2 + L**2*sina**2*(A66*r**2 + 2*B66*r + D66) + r**2*(L**2*(A22*j2**2 + A44*cosa**2) + pi**2*k2**2*(A66*r**2 + 2*B66*r + D66))) + (pi*L*r*sina*(D66 + r*(A66*r + 2*B66))*(i2 - k2)*(i2 + k2)*cos(pi*k2*xa/L) + k2*(D22*L**2*j2**2 + L**2*sina**2*(D66 + r*(A66*r + 2*B66)) + r*(2*B22*L**2*j2**2 + r*(L**2*(A22*j2**2 + A44*cosa**2) + pi**2*i2**2*(A66*r**2 + 2*B66*r + D66))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - (pi*L*r*sina*(D66 + r*(A66*r + 2*B66))*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) + k2*(D22*L**2*j2**2 + L**2*sina**2*(D66 + r*(A66*r + 2*B66)) + r*(2*B22*L**2*j2**2 + r*(L**2*(A22*j2**2 + A44*cosa**2) + pi**2*i2**2*(A66*r**2 + 2*B66*r + D66))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**3*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+4
                            k0v[c] += L*cosa*j2*(B22 + r*(A22 + A44))*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r**2*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+5
                            k0v[c] += cosa*(i2*(L*sina*(A26*r + B26)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(B26 + r*(A26 + A45))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (-L*sina*(A26*r + B26)*cos(pi*k2*xb/L) + pi*k2*r*(B26 + r*(A26 + A45))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) - (L*k2*sina*(A26*r + B26)*sin(pi*k2*xa/L) + pi*r*(A26*i2**2*r + A45*k2**2*r + B26*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(A26*r + B26)*sin(pi*k2*xb/L) + pi*r*(A26*i2**2*r + A45*k2**2*r + B26*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+6
                            k0v[c] += j2*(i2*(L*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(D12 + D66 + r*(B12 + B66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (-L*sina*(D22 + D66 + r*(B22 + B66))*cos(pi*k2*xb/L) + pi*k2*r*(D12 + D66 + r*(B12 + B66))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (-L*k2*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*k2*xa/L) - pi*r*(B66*i2**2*r + D66*i2**2 + k2**2*(B12*r + D12))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*k2*xb/L) + pi*r*(B66*i2**2*r + D66*i2**2 + k2**2*(B12*r + D12))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+7
                            k0v[c] += (i2*(pi*L*k2*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*k2**2*(B16*r + D16))))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (-pi*L*k2*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*k2*xa/L) - (D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*k2**2*(B16*r + D16))))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-pi*L*r*sina*(B26*i2**2*r + D26*i2**2 + k2**2*(B16*r + D16))*cos(pi*k2*xa/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*i2**2*(B16*r + D16))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*L*r*sina*(B26*i2**2*r + D26*i2**2 + k2**2*(B16*r + D16))*cos(pi*k2*xb/L) - k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*i2**2*(B16*r + D16))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+8
                            k0v[c] += -pi*j2*(B26*r + D26)*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+9
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*k2**2*(B66*r + D66)))) + (pi*L*r*sina*(i2 - k2)*(i2 + k2)*(B66*r + D66)*cos(pi*k2*xa/L) + k2*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*i2**2*(B66*r + D66))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-pi*L*r*sina*(i2 - k2)*(i2 + k2)*(B66*r + D66)*cos(pi*k2*xb/L) - k2*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*i2**2*(B66*r + D66))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+0
                            k0v[c] += cosa*(i2*((pi*A12*k2*r*sin(pi*k2*xa/L) - A22*L*sina*cos(pi*k2*xa/L))*sin(pi*i2*xa/L) + (-pi*A12*k2*r*sin(pi*k2*xb/L) + A22*L*sina*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)) + k2*(pi*A12*k2*r*cos(pi*k2*xa/L) + A22*L*sina*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - k2*(pi*A12*k2*r*cos(pi*k2*xb/L) + A22*L*sina*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+1
                            k0v[c] += A26*L*cosa*j2*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+2
                            k0v[c] += cosa*(i2*(-L*sina*(A26*r + B26)*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - pi*k2*r*(B26 + r*(A26 + A45))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (L*sina*(A26*r + B26)*cos(pi*k2*xa/L) + pi*k2*r*(B26 + r*(A26 + A45))*sin(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-L*k2*sina*(A26*r + B26)*sin(pi*k2*xa/L) + pi*r*(A45*i2**2*r + k2**2*(A26*r + B26))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(A26*r + B26)*sin(pi*k2*xb/L) - pi*r*(A45*i2**2*r + k2**2*(A26*r + B26))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+3
                            k0v[c] += L*cosa*j2*(B22 + r*(A22 + A44))*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r**2*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+4
                            k0v[c] += (-i2*(pi**2*A55*k2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*(pi**2*A55*k2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*(pi**2*A55*i2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*(pi**2*A55*i2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+5
                            k0v[c] += pi*A45*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+6
                            k0v[c] += (i2*(-B22*L*cosa*sina*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(-A55*r + B12*cosa)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (B22*L*cosa*sina*cos(pi*k2*xb/L) + pi*k2*r*(A55*r - B12*cosa)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (B22*L*cosa*k2*sina*sin(pi*k2*xa/L) + pi*r*(-A55*i2**2*r + B12*cosa*k2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) - (B22*L*cosa*k2*sina*sin(pi*k2*xb/L) + pi*r*(-A55*i2**2*r + B12*cosa*k2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+7
                            k0v[c] += L*j2*(-A45*r + B26*cosa)*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+8
                            k0v[c] += (i2*(B26*L*cosa*sina*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(-A45*r + B26*cosa)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (B26*L*cosa*sina*cos(pi*k2*xb/L) + pi*k2*r*(-A45*r + B26*cosa)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (-B26*L*cosa*k2*sina*sin(pi*k2*xa/L) + pi*r*(-A45*i2**2*r + B26*cosa*k2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (B26*L*cosa*k2*sina*sin(pi*k2*xb/L) + pi*r*(A45*i2**2*r - B26*cosa*k2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+9
                            k0v[c] += L*j2*(-A44*r + B22*cosa)*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+0
                            k0v[c] += A26*L*cosa*j2*(-i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+1
                            k0v[c] += cosa*(i2*((pi*A12*k2*r*sin(pi*k2*xa/L) - A22*L*sina*cos(pi*k2*xa/L))*sin(pi*i2*xa/L) + (-pi*A12*k2*r*sin(pi*k2*xb/L) + A22*L*sina*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)) + k2*(pi*A12*k2*r*cos(pi*k2*xa/L) + A22*L*sina*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - k2*(pi*A12*k2*r*cos(pi*k2*xb/L) + A22*L*sina*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+2
                            k0v[c] += L*cosa*j2*(B22 + r*(A22 + A44))*(-i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r**2*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+3
                            k0v[c] += cosa*(i2*(-L*sina*(A26*r + B26)*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - pi*k2*r*(B26 + r*(A26 + A45))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (L*sina*(A26*r + B26)*cos(pi*k2*xa/L) + pi*k2*r*(B26 + r*(A26 + A45))*sin(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-L*k2*sina*(A26*r + B26)*sin(pi*k2*xa/L) + pi*r*(A45*i2**2*r + k2**2*(A26*r + B26))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(A26*r + B26)*sin(pi*k2*xb/L) - pi*r*(A45*i2**2*r + k2**2*(A26*r + B26))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+4
                            k0v[c] += -pi*A45*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+5
                            k0v[c] += (-i2*(pi**2*A55*k2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*(pi**2*A55*k2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*(pi**2*A55*i2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*(pi**2*A55*i2**2*r**2 + L**2*(A22*cosa**2 + A44*j2**2))*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+6
                            k0v[c] += L*j2*(A45*r - B26*cosa)*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+7
                            k0v[c] += (i2*(-B22*L*cosa*sina*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(-A55*r + B12*cosa)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (B22*L*cosa*sina*cos(pi*k2*xb/L) + pi*k2*r*(A55*r - B12*cosa)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (B22*L*cosa*k2*sina*sin(pi*k2*xa/L) + pi*r*(-A55*i2**2*r + B12*cosa*k2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) - (B22*L*cosa*k2*sina*sin(pi*k2*xb/L) + pi*r*(-A55*i2**2*r + B12*cosa*k2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+8
                            k0v[c] += L*j2*(A44*r - B22*cosa)*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+9
                            k0v[c] += (i2*(B26*L*cosa*sina*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(-A45*r + B26*cosa)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (B26*L*cosa*sina*cos(pi*k2*xb/L) + pi*k2*r*(-A45*r + B26*cosa)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (-B26*L*cosa*k2*sina*sin(pi*k2*xa/L) + pi*r*(-A45*i2**2*r + B26*cosa*k2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (B26*L*cosa*k2*sina*sin(pi*k2*xb/L) + pi*r*(A45*i2**2*r - B26*cosa*k2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+0
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(pi**2*B11*k2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2)) + (pi*B12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*B12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+1
                            k0v[c] += pi*B16*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+2
                            k0v[c] += (i2*(-pi*L*k2*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*k2**2*(B16*r + D16))))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (pi*L*k2*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*k2*xa/L) - (D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*k2**2*(B16*r + D16))))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (pi*L*r*sina*(B16*i2**2*r + D16*i2**2 + k2**2*(B26*r + D26))*cos(pi*k2*xa/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*i2**2*(B16*r + D16))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-pi*L*r*sina*(B16*i2**2*r + D16*i2**2 + k2**2*(B26*r + D26))*cos(pi*k2*xb/L) - k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*i2**2*(B16*r + D16))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+3
                            k0v[c] += -j2*(i2*(-L*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(D12 + D66 + r*(B12 + B66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(D22 + D66 + r*(B22 + B66))*cos(pi*k2*xb/L) + pi*k2*r*(D12 + D66 + r*(B12 + B66))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*k2*xa/L) - pi*r*(B12*i2**2*r + D12*i2**2 + k2**2*(B66*r + D66))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-L*k2*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*k2*xb/L) + pi*r*(B12*i2**2*r + D12*i2**2 + k2**2*(B66*r + D66))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+4
                            k0v[c] += (i2*(-B22*L*cosa*sina*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(A55*r - B12*cosa)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (B22*L*cosa*sina*cos(pi*k2*xb/L) + pi*k2*r*(-A55*r + B12*cosa)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (B22*L*cosa*k2*sina*sin(pi*k2*xa/L) + pi*r*(A55*k2**2*r - B12*cosa*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-B22*L*cosa*k2*sina*sin(pi*k2*xb/L) + pi*r*(-A55*k2**2*r + B12*cosa*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+5
                            k0v[c] += L*j2*(A45*r - B26*cosa)*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+6
                            k0v[c] += (-i2*(sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 + pi**2*D11*k2**2)) + (pi*D12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 + pi**2*D11*i2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*D12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 + pi**2*D11*i2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+7
                            k0v[c] += pi*D16*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+8
                            k0v[c] += (i2*(pi*L*k2*r*sina*(D16 + D26)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*k2**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + (-pi*L*k2*r*sina*(D16 + D26)*sin(pi*k2*xb/L) + (D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*k2**2))*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (pi*L*r*sina*(D16*i2**2 + D26*k2**2)*cos(pi*k2*xa/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - (pi*L*r*sina*(D16*i2**2 + D26*k2**2)*cos(pi*k2*xb/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+9
                            k0v[c] += j2*(i2*(L*sina*(D22 + D66)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(D12 + D66)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (L*sina*(D22 + D66)*cos(pi*k2*xb/L) + pi*k2*r*(D12 + D66)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (-L*k2*sina*(D22 + D66)*sin(pi*k2*xa/L) + pi*r*(D12*i2**2 + D66*k2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(D22 + D66)*sin(pi*k2*xb/L) - pi*r*(D12*i2**2 + D66*k2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+0
                            k0v[c] += -pi*B16*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+1
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(pi**2*B11*k2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2)) + (pi*B12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*B12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(pi**2*B11*i2**2*r**2 + L**2*(B22*sina**2 + B66*j2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+2
                            k0v[c] += j2*(i2*(-L*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(D12 + D66 + r*(B12 + B66))*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(D22 + D66 + r*(B22 + B66))*cos(pi*k2*xb/L) + pi*k2*r*(D12 + D66 + r*(B12 + B66))*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*k2*xa/L) - pi*r*(B12*i2**2*r + D12*i2**2 + k2**2*(B66*r + D66))*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-L*k2*sina*(D22 + D66 + r*(B22 + B66))*sin(pi*k2*xb/L) + pi*r*(B12*i2**2*r + D12*i2**2 + k2**2*(B66*r + D66))*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+3
                            k0v[c] += (i2*(-pi*L*k2*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*k2**2*(B16*r + D16))))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (pi*L*k2*r*sina*(D16 + D26 + r*(B16 + B26))*sin(pi*k2*xa/L) - (D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*k2**2*(B16*r + D16))))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (pi*L*r*sina*(B16*i2**2*r + D16*i2**2 + k2**2*(B26*r + D26))*cos(pi*k2*xa/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*i2**2*(B16*r + D16))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-pi*L*r*sina*(B16*i2**2*r + D16*i2**2 + k2**2*(B26*r + D26))*cos(pi*k2*xb/L) - k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r*(B26*L**2*(j2 - sina)*(j2 + sina) + r*(-A45*L**2*cosa + pi**2*i2**2*(B16*r + D16))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+4
                            k0v[c] += L*j2*(-A45*r + B26*cosa)*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+5
                            k0v[c] += (i2*(-B22*L*cosa*sina*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(A55*r - B12*cosa)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (B22*L*cosa*sina*cos(pi*k2*xb/L) + pi*k2*r*(-A55*r + B12*cosa)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (B22*L*cosa*k2*sina*sin(pi*k2*xa/L) + pi*r*(A55*k2**2*r - B12*cosa*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-B22*L*cosa*k2*sina*sin(pi*k2*xb/L) + pi*r*(-A55*k2**2*r + B12*cosa*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+6
                            k0v[c] += -pi*D16*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+7
                            k0v[c] += (-i2*(sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 + pi**2*D11*k2**2)) + (pi*D12*L*r*sina*(-i2**2 + k2**2)*cos(pi*k2*xa/L) + k2*(D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 + pi**2*D11*i2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*D12*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) - k2*(D22*L**2*sina**2 + D66*L**2*j2**2 + r**2*(A55*L**2 + pi**2*D11*i2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+8
                            k0v[c] += j2*(i2*(-L*sina*(D22 + D66)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(D12 + D66)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(D22 + D66)*cos(pi*k2*xb/L) + pi*k2*r*(D12 + D66)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(D22 + D66)*sin(pi*k2*xa/L) - pi*r*(D12*i2**2 + D66*k2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-L*k2*sina*(D22 + D66)*sin(pi*k2*xb/L) + pi*r*(D12*i2**2 + D66*k2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+9
                            k0v[c] += (i2*(pi*L*k2*r*sina*(D16 + D26)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*k2**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + (-pi*L*k2*r*sina*(D16 + D26)*sin(pi*k2*xb/L) + (D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*k2**2))*cos(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (pi*L*r*sina*(D16*i2**2 + D26*k2**2)*cos(pi*k2*xa/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - (pi*L*r*sina*(D16*i2**2 + D26*k2**2)*cos(pi*k2*xb/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+0
                            k0v[c] += (i2*(pi*L*k2*r*sina*(B16 + B26)*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (pi**2*B16*k2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (-pi*L*k2*r*sina*(B16 + B26)*sin(pi*k2*xa/L) - (pi**2*B16*k2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-pi*L*r*sina*(B16*k2**2 + B26*i2**2)*cos(pi*k2*xa/L) + k2*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*L*r*sina*(B16*k2**2 + B26*i2**2)*cos(pi*k2*xb/L) - k2*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+1
                            k0v[c] += j2*(i2*(-L*sina*(B22 + B66)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(B12 + B66)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(B22 + B66)*cos(pi*k2*xb/L) - pi*k2*r*(B12 + B66)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(B22 + B66)*sin(pi*k2*xa/L) + pi*r*(B12*k2**2 + B66*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) - (L*k2*sina*(B22 + B66)*sin(pi*k2*xb/L) + pi*r*(B12*k2**2 + B66*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+2
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*k2**2*(B66*r + D66)))) + (pi*L*r*sina*(i2 - k2)*(i2 + k2)*(B66*r + D66)*cos(pi*k2*xa/L) + k2*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*i2**2*(B66*r + D66))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-pi*L*r*sina*(i2 - k2)*(i2 + k2)*(B66*r + D66)*cos(pi*k2*xb/L) - k2*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*i2**2*(B66*r + D66))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+3
                            k0v[c] += pi*j2*(B26*r + D26)*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+4
                            k0v[c] += (i2*(B26*L*cosa*sina*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(A45*r - B26*cosa)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (B26*L*cosa*sina*cos(pi*k2*xb/L) + pi*k2*r*(A45*r - B26*cosa)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) - (B26*L*cosa*k2*sina*sin(pi*k2*xa/L) + pi*r*(-A45*k2**2*r + B26*cosa*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (B26*L*cosa*k2*sina*sin(pi*k2*xb/L) + pi*r*(-A45*k2**2*r + B26*cosa*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+5
                            k0v[c] += L*j2*(A44*r - B22*cosa)*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+6
                            k0v[c] += (i2*(pi*L*k2*r*sina*(D16 + D26)*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*k2**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (-pi*L*k2*r*sina*(D16 + D26)*sin(pi*k2*xa/L) - (D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*k2**2))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-pi*L*r*sina*(D16*k2**2 + D26*i2**2)*cos(pi*k2*xa/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*L*r*sina*(D16*k2**2 + D26*i2**2)*cos(pi*k2*xb/L) - k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+7
                            k0v[c] += j2*(i2*(-L*sina*(D22 + D66)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(D12 + D66)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (L*sina*(D22 + D66)*cos(pi*k2*xb/L) - pi*k2*r*(D12 + D66)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) + (L*k2*sina*(D22 + D66)*sin(pi*k2*xa/L) + pi*r*(D12*k2**2 + D66*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) - (L*k2*sina*(D22 + D66)*sin(pi*k2*xb/L) + pi*r*(D12*k2**2 + D66*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+8
                            k0v[c] += (-i2*(sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*k2**2)) + (pi*D66*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xa/L) + k2*(D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - (pi*D66*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) + k2*(D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+9
                            k0v[c] += pi*D26*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+0
                            k0v[c] += j2*(i2*(L*sina*(B22 + B66)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(B12 + B66)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (-L*sina*(B22 + B66)*cos(pi*k2*xb/L) + pi*k2*r*(B12 + B66)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) - (L*k2*sina*(B22 + B66)*sin(pi*k2*xa/L) + pi*r*(B12*k2**2 + B66*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(B22 + B66)*sin(pi*k2*xb/L) + pi*r*(B12*k2**2 + B66*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+1
                            k0v[c] += (i2*(pi*L*k2*r*sina*(B16 + B26)*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (pi**2*B16*k2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (-pi*L*k2*r*sina*(B16 + B26)*sin(pi*k2*xa/L) - (pi**2*B16*k2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-pi*L*r*sina*(B16*k2**2 + B26*i2**2)*cos(pi*k2*xa/L) + k2*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*L*r*sina*(B16*k2**2 + B26*i2**2)*cos(pi*k2*xb/L) - k2*(pi**2*B16*i2**2*r**2 + B26*L**2*(j2 - sina)*(j2 + sina))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+2
                            k0v[c] += -pi*j2*(B26*r + D26)*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+3
                            k0v[c] += (i2*(-sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*k2**2*(B66*r + D66)))) + (pi*L*r*sina*(i2 - k2)*(i2 + k2)*(B66*r + D66)*cos(pi*k2*xa/L) + k2*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*i2**2*(B66*r + D66))))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (-pi*L*r*sina*(i2 - k2)*(i2 + k2)*(B66*r + D66)*cos(pi*k2*xb/L) - k2*(D22*L**2*j2**2 + L**2*sina**2*(B66*r + D66) + r*(B22*L**2*j2**2 + r*(-A44*L**2*cosa + pi**2*i2**2*(B66*r + D66))))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r**2*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+4
                            k0v[c] += L*j2*(-A44*r + B22*cosa)*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+5
                            k0v[c] += (i2*(B26*L*cosa*sina*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + pi*k2*r*(A45*r - B26*cosa)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - (B26*L*cosa*sina*cos(pi*k2*xb/L) + pi*k2*r*(A45*r - B26*cosa)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) - (B26*L*cosa*k2*sina*sin(pi*k2*xa/L) + pi*r*(-A45*k2**2*r + B26*cosa*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (B26*L*cosa*k2*sina*sin(pi*k2*xb/L) + pi*r*(-A45*k2**2*r + B26*cosa*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+6
                            k0v[c] += j2*(i2*(L*sina*(D22 + D66)*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - pi*k2*r*(D12 + D66)*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + (-L*sina*(D22 + D66)*cos(pi*k2*xb/L) + pi*k2*r*(D12 + D66)*sin(pi*k2*xb/L))*sin(pi*i2*xb/L)) - (L*k2*sina*(D22 + D66)*sin(pi*k2*xa/L) + pi*r*(D12*k2**2 + D66*i2**2)*cos(pi*k2*xa/L))*cos(pi*i2*xa/L) + (L*k2*sina*(D22 + D66)*sin(pi*k2*xb/L) + pi*r*(D12*k2**2 + D66*i2**2)*cos(pi*k2*xb/L))*cos(pi*i2*xb/L))/(r*(i2**2 - k2**2))
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+7
                            k0v[c] += (i2*(pi*L*k2*r*sina*(D16 + D26)*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) + (D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*k2**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + (-pi*L*k2*r*sina*(D16 + D26)*sin(pi*k2*xa/L) - (D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*k2**2))*cos(pi*k2*xa/L))*sin(pi*i2*xa/L)) + (-pi*L*r*sina*(D16*k2**2 + D26*i2**2)*cos(pi*k2*xa/L) + k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) + (pi*L*r*sina*(D16*k2**2 + D26*i2**2)*cos(pi*k2*xb/L) - k2*(D26*L**2*(j2 - sina)*(j2 + sina) + r**2*(A45*L**2 + pi**2*D16*i2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+8
                            k0v[c] += -pi*D26*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            k0r[c] = row+9
                            k0c[c] = col+9
                            k0v[c] += (-i2*(sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - sin(pi*i2*xb/L)*cos(pi*k2*xb/L))*(D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*k2**2)) + (pi*D66*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xa/L) + k2*(D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i2**2))*sin(pi*k2*xa/L))*cos(pi*i2*xa/L) - (pi*D66*L*r*sina*(i2 - k2)*(i2 + k2)*cos(pi*k2*xb/L) + k2*(D22*L**2*j2**2 + D66*L**2*sina**2 + r**2*(A44*L**2 + pi**2*D66*i2**2))*sin(pi*k2*xb/L))*cos(pi*i2*xb/L))/(L*r*(i2 - k2)*(i2 + k2))

    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0_cyl(double r2, double L, double [:, ::1] F,
            int m1, int m2, int n2):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r
    cdef long [:] k0r, k0c
    cdef double [:] k0v

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

    fdim = 9 + 8*m1 + k11_num + k22_num

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
    k0v[c] += 2*pi*r2*(A16*r + B16)/L
    c += 1
    k0r[c] = 0
    k0c[c] = 2
    k0v[c] += 2*pi*A11*r/L
    c += 1
    k0r[c] = 1
    k0c[c] = 0
    k0v[c] += 2*pi*r2*(A16*r + B16)/L
    c += 1
    k0r[c] = 1
    k0c[c] = 1
    k0v[c] += 0.666666666666667*pi*r2**2*(A44*L**2 + 3*D66 + 3*r*(A66*r + 2*B66))/(L*r)
    c += 1
    k0r[c] = 1
    k0c[c] = 2
    k0v[c] += 2*pi*r2*(A16*r + B16)/L
    c += 1
    k0r[c] = 2
    k0c[c] = 0
    k0v[c] += 2*pi*A11*r/L
    c += 1
    k0r[c] = 2
    k0c[c] = 1
    k0v[c] += 2*pi*r2*(A16*r + B16)/L
    c += 1
    k0r[c] = 2
    k0c[c] = 2
    k0v[c] += 3*pi*A11*r/L + 0.333333333333333*pi*A66*L/r

    for i1 in range(i0, m1+i0):
        col = (i1-i0)*num1 + num0
        row = col

        if i1 != 0:
            # k0_01
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
            k0v[c] += r2*(2*(-1)**i1 - 2)*(B26 + r*(A26 + A45))/(i1*r)
            c += 1
            k0r[c] = 1
            k0c[c] = col+3
            k0v[c] += (-1)**(i1 - 1)*r2*(-4*A45*L**2 + 4*pi**2*i1**2*(B16*r + D16))/(pi*L*i1**2)
            c += 1
            k0r[c] = 1
            k0c[c] = col+4
            k0v[c] += -2*A44*L*r2/i1
            c += 1
            k0r[c] = 2
            k0c[c] = col+2
            k0v[c] += A12*(2*(-1)**i1 - 2)/i1
            c += 1
            k0r[c] = 2
            k0c[c] = col+3
            k0v[c] += pi*B11*r*(-2*(-1)**i1 + 2)/L

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

            if k1 == i1:
                if i1 != 0:
                    # k0_11 cond_1
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+0
                    k0v[c] += pi**3*A11*i1**2*r/L
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+1
                    k0v[c] += pi**3*i1**2*(A16*r + B16)/L
                    c += 1
                    k0r[c] = row+0
                    k0c[c] = col+4
                    k0v[c] += pi**3*B16*i1**2*r/L
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+0
                    k0v[c] += pi**3*i1**2*(A16*r + B16)/L
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+1
                    k0v[c] += (pi*A44*L**2 + pi**3*i1**2*(D66 + r*(A66*r + 2*B66)))/(L*r)
                    c += 1
                    k0r[c] = row+1
                    k0c[c] = col+4
                    k0v[c] += (-pi*A44*L**2 + pi**3*i1**2*(B66*r + D66))/L
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+2
                    k0v[c] += pi*A22*L/r + pi**3*A55*i1**2*r/L
                    c += 1
                    k0r[c] = row+2
                    k0c[c] = col+3
                    k0v[c] += pi**2*i1*(A55*r - B12)
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+2
                    k0v[c] += pi**2*i1*(A55*r - B12)
                    c += 1
                    k0r[c] = row+3
                    k0c[c] = col+3
                    k0v[c] += pi*A55*L*r + pi**3*D11*i1**2*r/L
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+0
                    k0v[c] += pi**3*B16*i1**2*r/L
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+1
                    k0v[c] += (-pi*A44*L**2 + pi**3*i1**2*(B66*r + D66))/L
                    c += 1
                    k0r[c] = row+4
                    k0c[c] = col+4
                    k0v[c] += pi*A44*L*r + pi**3*D66*i1**2*r/L

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
                k0v[c] += pi*A12*i1*k1*(2*(-1)**(i1 + k1) - 2)/(i1**2 - k1**2)
                c += 1
                k0r[c] = row+0
                k0c[c] = col+3
                k0v[c] += pi**2*B11*i1*k1**2*r*(-2*(-1)**(i1 + k1) + 2)/(L*(i1**2 - k1**2))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+2
                k0v[c] += pi*i1*k1*(2*(-1)**(i1 + k1) - 2)*(B26 + r*(A26 + A45))/(r*(i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+1
                k0c[c] = col+3
                k0v[c] += i1*(-2*(-1)**(i1 + k1) + 2)*(-A45*L**2 + pi**2*k1**2*(B16*r + D16))/(L*(i1**2 - k1**2))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+0
                k0v[c] += pi*A12*i1*k1*(-2*(-1)**(i1 + k1) + 2)/(i1**2 - k1**2)
                c += 1
                k0r[c] = row+2
                k0c[c] = col+1
                k0v[c] += pi*i1*k1*(-2*(-1)**(i1 + k1) + 2)*(B26 + r*(A26 + A45))/(r*(i1**2 - k1**2))
                c += 1
                k0r[c] = row+2
                k0c[c] = col+4
                k0v[c] += pi*i1*k1*(-2*(-1)**(i1 + k1) + 2)*(-A45*r + B26)/((i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+0
                k0v[c] += pi**2*B11*i1**2*k1*r*(2*(-1)**(i1 + k1) - 2)/(L*(i1**2 - k1**2))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+1
                k0v[c] += k1*(-2*(-1)**(i1 + k1) + 2)*(-A45*L**2 + pi**2*i1**2*(B16*r + D16))/(L*(-i1**2 + k1**2))
                c += 1
                k0r[c] = row+3
                k0c[c] = col+4
                k0v[c] += k1*r*(2*(-1)**(i1 + k1) - 2)*(A45*L**2 + pi**2*D16*i1**2)/(L*(i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+2
                k0v[c] += pi*i1*k1*(2*(-1)**(i1 + k1) - 2)*(-A45*r + B26)/((i1 - k1)*(i1 + k1))
                c += 1
                k0r[c] = row+4
                k0c[c] = col+3
                k0v[c] += i1*r*(-2*(-1)**(i1 + k1) + 2)*(A45*L**2 + pi**2*D16*k1**2)/(L*(i1 - k1)*(i1 + k1))

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
                        # k0_22 cond_1
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += 0.5*pi**3*A11*i2**2*r/L + 0.5*pi*A66*L*j2**2/r
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*pi*(B26*L**2*j2**2 + r*(A26*L**2*j2**2 + pi**2*i2**2*r*(A16*r + B16)))/(L*r**2)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+5
                        k0v[c] += 0.5*pi*A26*L*j2/r
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+6
                        k0v[c] += 0.5*pi**3*B11*i2**2*r/L + 0.5*pi*B66*L*j2**2/r
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
                        k0v[c] += 0.5*pi*(B26*L**2*j2**2 + r*(A26*L**2*j2**2 + pi**2*i2**2*r*(A16*r + B16)))/(L*r**2)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += -0.5*pi*A26*L*j2/r
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+7
                        k0v[c] += 0.5*pi**3*B11*i2**2*r/L + 0.5*pi*B66*L*j2**2/r
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+9
                        k0v[c] += 0.5*pi**3*B16*i2**2*r/L + 0.5*pi*B26*L*j2**2/r
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 0.5*pi*(B26*L**2*j2**2 + r*(A26*L**2*j2**2 + pi**2*i2**2*r*(A16*r + B16)))/(L*r**2)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += 0.5*pi*(L**2*(D22*j2**2 + r*(A44*r + j2**2*(A22*r + 2*B22))) + pi**2*i2**2*r**2*(D66 + r*(A66*r + 2*B66)))/(L*r**3)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+5
                        k0v[c] += 0.5*pi*L*j2*(B22 + r*(A22 + A44))/r**2
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+6
                        k0v[c] += 0.5*pi*(D26*L**2*j2**2 + r*(B26*L**2*j2**2 + r*(-A45*L**2 + pi**2*i2**2*(B16*r + D16))))/(L*r**2)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+8
                        k0v[c] += 0.5*pi*(D22*L**2*j2**2 + r*(B22*L**2*j2**2 + r*(-A44*L**2 + pi**2*i2**2*(B66*r + D66))))/(L*r**2)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += 0.5*pi*(B26*L**2*j2**2 + r*(A26*L**2*j2**2 + pi**2*i2**2*r*(A16*r + B16)))/(L*r**2)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += 0.5*pi*(L**2*(D22*j2**2 + r*(A44*r + j2**2*(A22*r + 2*B22))) + pi**2*i2**2*r**2*(D66 + r*(A66*r + 2*B66)))/(L*r**3)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += -0.5*pi*L*j2*(B22 + r*(A22 + A44))/r**2
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+7
                        k0v[c] += 0.5*pi*(D26*L**2*j2**2 + r*(B26*L**2*j2**2 + r*(-A45*L**2 + pi**2*i2**2*(B16*r + D16))))/(L*r**2)
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+9
                        k0v[c] += 0.5*pi*(D22*L**2*j2**2 + r*(B22*L**2*j2**2 + r*(-A44*L**2 + pi**2*i2**2*(B66*r + D66))))/(L*r**2)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += -0.5*pi*A26*L*j2/r
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += -0.5*pi*L*j2*(B22 + r*(A22 + A44))/r**2
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += 0.5*pi**3*A55*i2**2*r/L + 0.5*pi*L*(A22 + A44*j2**2)/r
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+7
                        k0v[c] += 0.5*pi*L*j2*(A45*r - B26)/r
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
                        k0v[c] += 0.5*pi*L*j2*(B22 + r*(A22 + A44))/r**2
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+5
                        k0v[c] += 0.5*pi**3*A55*i2**2*r/L + 0.5*pi*L*(A22 + A44*j2**2)/r
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+6
                        k0v[c] += 0.5*pi*L*j2*(-A45*r + B26)/r
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+8
                        k0v[c] += 0.5*pi*L*j2*(-A44*r + B22)/r
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+0
                        k0v[c] += 0.5*pi**3*B11*i2**2*r/L + 0.5*pi*B66*L*j2**2/r
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+2
                        k0v[c] += 0.5*pi*(D26*L**2*j2**2 + r*(B26*L**2*j2**2 + r*(-A45*L**2 + pi**2*i2**2*(B16*r + D16))))/(L*r**2)
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+5
                        k0v[c] += 0.5*pi*L*j2*(-A45*r + B26)/r
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+6
                        k0v[c] += 0.5*pi*D66*L*j2**2/r + 0.5*pi*r*(A55*L**2 + pi**2*D11*i2**2)/L
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+8
                        k0v[c] += 0.5*pi*D26*L*j2**2/r + 0.5*pi*r*(A45*L**2 + pi**2*D16*i2**2)/L
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+1
                        k0v[c] += 0.5*pi**3*B11*i2**2*r/L + 0.5*pi*B66*L*j2**2/r
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+3
                        k0v[c] += 0.5*pi*(D26*L**2*j2**2 + r*(B26*L**2*j2**2 + r*(-A45*L**2 + pi**2*i2**2*(B16*r + D16))))/(L*r**2)
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+4
                        k0v[c] += 0.5*pi*L*j2*(A45*r - B26)/r
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+7
                        k0v[c] += 0.5*pi*D66*L*j2**2/r + 0.5*pi*r*(A55*L**2 + pi**2*D11*i2**2)/L
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+9
                        k0v[c] += 0.5*pi*D26*L*j2**2/r + 0.5*pi*r*(A45*L**2 + pi**2*D16*i2**2)/L
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+0
                        k0v[c] += 0.5*pi**3*B16*i2**2*r/L + 0.5*pi*B26*L*j2**2/r
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+2
                        k0v[c] += 0.5*pi*(D22*L**2*j2**2 + r*(B22*L**2*j2**2 + r*(-A44*L**2 + pi**2*i2**2*(B66*r + D66))))/(L*r**2)
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+5
                        k0v[c] += 0.5*pi*L*j2*(-A44*r + B22)/r
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+6
                        k0v[c] += 0.5*pi*D26*L*j2**2/r + 0.5*pi*r*(A45*L**2 + pi**2*D16*i2**2)/L
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
                        k0v[c] += 0.5*pi*(D22*L**2*j2**2 + r*(B22*L**2*j2**2 + r*(-A44*L**2 + pi**2*i2**2*(B66*r + D66))))/(L*r**2)
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+4
                        k0v[c] += 0.5*pi*L*j2*(A44*r - B22)/r
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+7
                        k0v[c] += 0.5*pi*D26*L*j2**2/r + 0.5*pi*r*(A45*L**2 + pi**2*D16*i2**2)/L
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+9
                        k0v[c] += 0.5*pi*D22*L*j2**2/r + 0.5*pi*r*(A44*L**2 + pi**2*D66*i2**2)/L

                    elif k2 != i2 and l2 == j2:
                        # k0_22 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += -pi*A16*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(A12*i2**2*r + B12*i2**2 + k2**2*(A66*r + B66))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += pi*A12*i2**2*((-1)**(i2 + k2) - 1)/(i2**2 - k2**2)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+7
                        k0v[c] += -pi*B16*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+9
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(B12*i2**2 + B66*k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += pi*A16*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(A12*i2**2*r + B12*i2**2 + k2**2*(A66*r + B66))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+5
                        k0v[c] += pi*A12*i2**2*((-1)**(i2 + k2) - 1)/(i2**2 - k2**2)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+6
                        k0v[c] += pi*B16*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+8
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(B12*i2**2 + B66*k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(A66*i2**2*r + B66*i2**2 + k2**2*(A12*r + B12))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(D26 + r*(A26*r + 2*B26))*(i2**2 + k2**2)/(r**2*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(A26*i2**2*r + A45*k2**2*r + B26*i2**2)/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+7
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(B66*i2**2*r + D66*i2**2 + k2**2*(B12*r + D12))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+9
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)*(B26*r + D26)/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(A66*i2**2*r + B66*i2**2 + k2**2*(A12*r + B12))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(D26 + r*(A26*r + 2*B26))*(i2**2 + k2**2)/(r**2*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+5
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(A26*i2**2*r + A45*k2**2*r + B26*i2**2)/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+6
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(B66*i2**2*r + D66*i2**2 + k2**2*(B12*r + D12))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+8
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)*(B26*r + D26)/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += pi*A12*k2**2*((-1)**(i2 + k2) - 1)/(-i2**2 + k2**2)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += -pi*((-1)**(i2 + k2) - 1)*(A45*i2**2*r + k2**2*(A26*r + B26))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+5
                        k0v[c] += -pi*A45*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+6
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(A55*i2**2*r - B12*k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+8
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(A45*i2**2*r - B26*k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+1
                        k0v[c] += pi*A12*k2**2*((-1)**(i2 + k2) - 1)/(-i2**2 + k2**2)
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+3
                        k0v[c] += -pi*((-1)**(i2 + k2) - 1)*(A45*i2**2*r + k2**2*(A26*r + B26))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+4
                        k0v[c] += pi*A45*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+7
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(A55*i2**2*r - B12*k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+5
                        k0c[c] = col+9
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(A45*i2**2*r - B26*k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+1
                        k0v[c] += -pi*B16*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+3
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(B12*i2**2*r + D12*i2**2 + k2**2*(B66*r + D66))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+4
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(-A55*k2**2*r + B12*i2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+7
                        k0v[c] += -pi*D16*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+6
                        k0c[c] = col+9
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(D12*i2**2 + D66*k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+0
                        k0v[c] += pi*B16*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+2
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(B12*i2**2*r + D12*i2**2 + k2**2*(B66*r + D66))/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+5
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(-A55*k2**2*r + B12*i2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+6
                        k0v[c] += pi*D16*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+7
                        k0c[c] = col+8
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(D12*i2**2 + D66*k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+1
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(B12*k2**2 + B66*i2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+3
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)*(B26*r + D26)/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+4
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(-A45*k2**2*r + B26*i2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+7
                        k0v[c] += -pi*j2*((-1)**(i2 + k2) - 1)*(D12*k2**2 + D66*i2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+8
                        k0c[c] = col+9
                        k0v[c] += -pi*D26*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+0
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(B12*k2**2 + B66*i2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+2
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)*(B26*r + D26)/(r*(i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+5
                        k0v[c] += pi*((-1)**(i2 + k2) - 1)*(-A45*k2**2*r + B26*i2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+6
                        k0v[c] += pi*j2*((-1)**(i2 + k2) - 1)*(D12*k2**2 + D66*i2**2)/((i2 - k2)*(i2 + k2))
                        c += 1
                        k0r[c] = row+9
                        k0c[c] = col+8
                        k0v[c] += pi*D26*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/((i2 - k2)*(i2 + k2))

    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0edges(int m1, int m2, int n2, double r1, double r2,
             double kuBot, double kuTop,
             double kvBot, double kvTop,
             double kwBot, double kwTop,
             double kphixBot, double kphixTop,
             double kphitBot, double kphitTop):
    cdef int i1, k1, i2, j2, k2, l2, row, col, c
    cdef long [:] k0edgesr, k0edgesc
    cdef double [:] k0edgesv

    k11_cond_1 = 1
    k11_cond_2 = 1
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 10
    k22_cond_2 = 10
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

            if k1 == i1:
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

                    if k2 == i2 and l2 == j2:
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
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += pi*(kvBot*r1 + kvTop*r2)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += pi*(kvBot*r1 + kvTop*r2)
                        c += 1
                        k0edgesr[c] = row+4
                        k0edgesc[c] = col+4
                        k0edgesv[c] += pi*(kwBot*r1 + kwTop*r2)
                        c += 1
                        k0edgesr[c] = row+5
                        k0edgesc[c] = col+5
                        k0edgesv[c] += pi*(kwBot*r1 + kwTop*r2)
                        c += 1
                        k0edgesr[c] = row+6
                        k0edgesc[c] = col+6
                        k0edgesv[c] += pi*(kphixBot*r1 + kphixTop*r2)
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+7
                        k0edgesv[c] += pi*(kphixBot*r1 + kphixTop*r2)
                        c += 1
                        k0edgesr[c] = row+8
                        k0edgesc[c] = col+8
                        k0edgesv[c] += pi*(kphitBot*r1 + kphitTop*r2)
                        c += 1
                        k0edgesr[c] = row+9
                        k0edgesc[c] = col+9
                        k0edgesv[c] += pi*(kphitBot*r1 + kphitTop*r2)

                    elif k2 != i2 and l2 == j2:
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
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kvBot*r1 + kvTop*r2)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kvBot*r1 + kvTop*r2)
                        c += 1
                        k0edgesr[c] = row+4
                        k0edgesc[c] = col+4
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kwBot*r1 + kwTop*r2)
                        c += 1
                        k0edgesr[c] = row+5
                        k0edgesc[c] = col+5
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kwBot*r1 + kwTop*r2)
                        c += 1
                        k0edgesr[c] = row+6
                        k0edgesc[c] = col+6
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kphixBot*r1 + kphixTop*r2)
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+7
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kphixBot*r1 + kphixTop*r2)
                        c += 1
                        k0edgesr[c] = row+8
                        k0edgesc[c] = col+8
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kphitBot*r1 + kphitTop*r2)
                        c += 1
                        k0edgesr[c] = row+9
                        k0edgesc[c] = col+9
                        k0edgesv[c] += pi*((-1)**(i2 + k2)*kphitBot*r1 + kphitTop*r2)

    size = num0 + num1*m1 + num2*m2*n2

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fc, double P, double T, double r2, double alpharad, double L,
        int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double sina, cosa, xa, xb, r
    cdef long [:] kG0r, kG0c
    cdef double [:] kG0v

    # sparse parameters
    k11_cond_1 = 6
    k11_cond_2 = 6
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 20
    k22_cond_2 = 22
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 5 + 5*m1 + k11_num + k22_num

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

        # kG0_00
        c += 1
        kG0r[c] = 0
        kG0c[c] = 1
        kG0v[c] += 0.5*T*r2*sina*(xa - xb)*(2*L - xa - xb)/(L**2*cosa*r**2)
        c += 1
        kG0r[c] = 1
        kG0c[c] = 0
        kG0v[c] += 0.5*T*r2*sina*(xa - xb)*(2*L - xa - xb)/(L**2*cosa*r**2)
        c += 1
        kG0r[c] = 1
        kG0c[c] = 1
        kG0v[c] += -0.333333333333333*r2**2*(-3*Fc + pi*P*(6*L**2 - 6*L*(xa + xb) + 3*r**2 + 2*xa**2 + 2*xa*xb + 2*xb**2))*(xa - xb)/(L**2*cosa)
        c += 1
        kG0r[c] = 1
        kG0c[c] = 2
        kG0v[c] += 0.5*T*r2*sina*(xa - xb)*(2*L - xa - xb)/(L**2*cosa*r**2)
        c += 1
        kG0r[c] = 2
        kG0c[c] = 1
        kG0v[c] += 0.5*T*r2*sina*(xa - xb)*(2*L - xa - xb)/(L**2*cosa*r**2)

        for i1 in range(i0, m1+i0):
            col = (i1-i0)*num1 + num0
            row = col

            if i1 != 0:
                # kG0_01 cond_1
                c += 1
                kG0r[c] = 0
                kG0c[c] = col+1
                kG0v[c] += T*sina*(-cos(pi*i1*xa/L) + cos(pi*i1*xb/L))/(pi*cosa*i1*r**2)
                c += 1
                kG0r[c] = 1
                kG0c[c] = col+0
                kG0v[c] += T*r2*sina*(L*cos(pi*i1*xa/L) - L*cos(pi*i1*xb/L) + pi*i1*((-L + xa)*sin(pi*i1*xa/L) + (L - xb)*sin(pi*i1*xb/L)))/(pi*L*i1*r**2)
                c += 1
                kG0r[c] = 1
                kG0c[c] = col+1
                kG0v[c] += r2*(2*pi*L*P*i1*(-L + xb)*cos(pi*i1*xb/L) + 2*pi*L*P*i1*(L - xa)*cos(pi*i1*xa/L) + (2*L**2*P + pi*i1**2*(-Fc + pi*P*r**2))*(sin(pi*i1*xa/L) - sin(pi*i1*xb/L)))/(pi*L*cosa*i1**2)
                c += 1
                kG0r[c] = 1
                kG0c[c] = col+2
                kG0v[c] += T*cosa*r2*(L*cos(pi*i1*xa/L) - L*cos(pi*i1*xb/L) + pi*i1*((-L + xa)*sin(pi*i1*xa/L) + (L - xb)*sin(pi*i1*xb/L)))/(pi*L*i1*r**2)
                c += 1
                kG0r[c] = 2
                kG0c[c] = col+1
                kG0v[c] += T*sina*(-cos(pi*i1*xa/L) + cos(pi*i1*xb/L))/(pi*cosa*i1*r**2)

            for k1 in range(i0, m1+i0):
                col = (k1-i0)*num1 + num0

                #NOTE symmetry
                if row > col:
                    continue

                if k1 == i1:
                    if i1 != 0:
                        # kG0_11 cond_1
                        c += 1
                        kG0r[c] = row+0
                        kG0c[c] = col+1
                        kG0v[c] += 0.25*T*sina*(cos(2*pi*i1*xa/L) - cos(2*pi*i1*xb/L))/r**2
                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+0
                        kG0v[c] += 0.25*T*sina*(cos(2*pi*i1*xa/L) - cos(2*pi*i1*xb/L))/r**2
                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+1
                        kG0v[c] += 0.25*(L*(2*L**2*P + pi*i1**2*(Fc - pi*P*r**2))*(sin(2*pi*i1*xa/L) - sin(2*pi*i1*xb/L)) - 2*pi*i1*(xa - xb)*(2*L**2*P + pi*i1**2*(-Fc + pi*P*r**2)))/(L**2*cosa*i1)
                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+2
                        kG0v[c] += 0.25*T*cosa*(cos(2*pi*i1*xa/L) - cos(2*pi*i1*xb/L))/r**2
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+1
                        kG0v[c] += 0.25*T*cosa*(cos(2*pi*i1*xa/L) - cos(2*pi*i1*xb/L))/r**2
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += 0.25*pi*i1*(Fc - pi*P*r**2)*(L*sin(2*pi*i1*xa/L) - L*sin(2*pi*i1*xb/L) + 2*pi*i1*(xa - xb))/(L**2*cosa)

                else:
                    # kG0_11 cond_2
                    c += 1
                    kG0r[c] = row+0
                    kG0c[c] = col+1
                    kG0v[c] += T*i1*sina*(-i1*sin(pi*i1*xa/L)*sin(pi*k1*xa/L) + i1*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) - k1*cos(pi*i1*xa/L)*cos(pi*k1*xa/L) + k1*cos(pi*i1*xb/L)*cos(pi*k1*xb/L))/(r**2*(i1**2 - k1**2))
                    c += 1
                    kG0r[c] = row+1
                    kG0c[c] = col+0
                    kG0v[c] += T*k1*sina*(i1*cos(pi*i1*xa/L)*cos(pi*k1*xa/L) - i1*cos(pi*i1*xb/L)*cos(pi*k1*xb/L) + k1*sin(pi*i1*xa/L)*sin(pi*k1*xa/L) - k1*sin(pi*i1*xb/L)*sin(pi*k1*xb/L))/(r**2*(i1**2 - k1**2))
                    c += 1
                    kG0r[c] = row+1
                    kG0c[c] = col+1
                    kG0v[c] += (i1*(-2*L**2*P + pi*k1**2*(Fc - pi*P*r**2))*sin(pi*k1*xb/L)*cos(pi*i1*xb/L) + i1*(2*L**2*P + pi*k1**2*(-Fc + pi*P*r**2))*sin(pi*k1*xa/L)*cos(pi*i1*xa/L) + k1*(-2*L**2*P + pi*i1**2*(Fc - pi*P*r**2))*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) + k1*(2*L**2*P + pi*i1**2*(-Fc + pi*P*r**2))*sin(pi*i1*xb/L)*cos(pi*k1*xb/L))/(L*cosa*(i1 - k1)*(i1 + k1))
                    c += 1
                    kG0r[c] = row+1
                    kG0c[c] = col+2
                    kG0v[c] += T*cosa*k1*(i1*cos(pi*i1*xa/L)*cos(pi*k1*xa/L) - i1*cos(pi*i1*xb/L)*cos(pi*k1*xb/L) + k1*sin(pi*i1*xa/L)*sin(pi*k1*xa/L) - k1*sin(pi*i1*xb/L)*sin(pi*k1*xb/L))/(r**2*(i1**2 - k1**2))
                    c += 1
                    kG0r[c] = row+2
                    kG0c[c] = col+1
                    kG0v[c] += T*cosa*i1*(-i1*sin(pi*i1*xa/L)*sin(pi*k1*xa/L) + i1*sin(pi*i1*xb/L)*sin(pi*k1*xb/L) - k1*cos(pi*i1*xa/L)*cos(pi*k1*xa/L) + k1*cos(pi*i1*xb/L)*cos(pi*k1*xb/L))/(r**2*(i1**2 - k1**2))
                    c += 1
                    kG0r[c] = row+2
                    kG0c[c] = col+2
                    kG0v[c] += pi*i1*k1*(Fc - pi*P*r**2)*(i1*sin(pi*i1*xa/L)*cos(pi*k1*xa/L) - i1*sin(pi*i1*xb/L)*cos(pi*k1*xb/L) - k1*sin(pi*k1*xa/L)*cos(pi*i1*xa/L) + k1*sin(pi*k1*xb/L)*cos(pi*i1*xb/L))/(L*cosa*(i1 - k1)*(i1 + k1))

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
                                # kG0_22 cond_1
                                c += 1
                                kG0r[c] = row+0
                                kG0c[c] = col+2
                                kG0v[c] += 0.125*T*sina*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L))/r**2
                                c += 1
                                kG0r[c] = row+0
                                kG0c[c] = col+3
                                kG0v[c] += 0.25*P*j2*sina*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(cosa*i2)
                                c += 1
                                kG0r[c] = row+1
                                kG0c[c] = col+2
                                kG0v[c] += 0.25*P*j2*sina*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(cosa*i2)
                                c += 1
                                kG0r[c] = row+1
                                kG0c[c] = col+3
                                kG0v[c] += 0.125*T*sina*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L))/r**2
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+0
                                kG0v[c] += 0.125*T*sina*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L))/r**2
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+1
                                kG0v[c] += 0.25*P*j2*sina*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/(cosa*i2)
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+2
                                kG0v[c] += 0.125*(L*(2*L**2*P + pi*i2**2*(Fc - pi*P*r**2))*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) - 2*pi*i2*(xa - xb)*(2*L**2*P + pi*i2**2*(-Fc + pi*P*r**2)))/(L**2*cosa*i2)
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+4
                                kG0v[c] += 0.125*T*cosa*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L))/r**2
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+5
                                kG0v[c] += 0.25*P*j2*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/i2
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+0
                                kG0v[c] += 0.25*P*j2*sina*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/(cosa*i2)
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+1
                                kG0v[c] += 0.125*T*sina*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L))/r**2
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+3
                                kG0v[c] += 0.125*(L*(2*L**2*P + pi*i2**2*(Fc - pi*P*r**2))*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) - 2*pi*i2*(xa - xb)*(2*L**2*P + pi*i2**2*(-Fc + pi*P*r**2)))/(L**2*cosa*i2)
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+4
                                kG0v[c] += 0.25*P*j2*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/i2
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+5
                                kG0v[c] += 0.125*T*cosa*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L))/r**2
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+2
                                kG0v[c] += 0.125*T*cosa*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L))/r**2
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+3
                                kG0v[c] += 0.25*P*j2*(L*sin(2*pi*i2*xa/L) - L*sin(2*pi*i2*xb/L) + 2*pi*i2*(xa - xb))/i2
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+4
                                kG0v[c] += 0.125*(L*(2*L**2*P*j2**2 + pi*i2**2*(Fc - pi*P*r**2))*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) - 2*pi*i2*(xa - xb)*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2)))/(L**2*cosa*i2)
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+2
                                kG0v[c] += 0.25*P*j2*(-L*sin(2*pi*i2*xa/L) + L*sin(2*pi*i2*xb/L) + 2*pi*i2*(-xa + xb))/i2
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+3
                                kG0v[c] += 0.125*T*cosa*(-cos(2*pi*i2*xa/L) + cos(2*pi*i2*xb/L))/r**2
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+5
                                kG0v[c] += 0.125*(L*(2*L**2*P*j2**2 + pi*i2**2*(Fc - pi*P*r**2))*(-sin(2*pi*i2*xa/L) + sin(2*pi*i2*xb/L)) - 2*pi*i2*(xa - xb)*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2)))/(L**2*cosa*i2)

                            else:
                                # kG0_22 cond_5
                                c += 1
                                kG0r[c] = row+0
                                kG0c[c] = col+3
                                kG0v[c] += pi*P*j2*sina*(xa - xb)/cosa
                                c += 1
                                kG0r[c] = row+1
                                kG0c[c] = col+2
                                kG0v[c] += pi*P*j2*sina*(-xa + xb)/cosa
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+1
                                kG0v[c] += pi*P*j2*sina*(-xa + xb)/cosa
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+2
                                kG0v[c] += pi*P*(-xa + xb)/cosa
                                c += 1
                                kG0r[c] = row+2
                                kG0c[c] = col+5
                                kG0v[c] += pi*P*j2*(-xa + xb)
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+0
                                kG0v[c] += pi*P*j2*sina*(xa - xb)/cosa
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+3
                                kG0v[c] += pi*P*(-xa + xb)/cosa
                                c += 1
                                kG0r[c] = row+3
                                kG0c[c] = col+4
                                kG0v[c] += pi*P*j2*(xa - xb)
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+3
                                kG0v[c] += pi*P*j2*(xa - xb)
                                c += 1
                                kG0r[c] = row+4
                                kG0c[c] = col+4
                                kG0v[c] += pi*P*j2**2*(-xa + xb)/cosa
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+2
                                kG0v[c] += pi*P*j2*(-xa + xb)
                                c += 1
                                kG0r[c] = row+5
                                kG0c[c] = col+5
                                kG0v[c] += pi*P*j2**2*(-xa + xb)/cosa

                        elif k2 != i2 and l2 == j2:
                            # kG0_22 cond_2
                            c += 1
                            kG0r[c] = row+0
                            kG0c[c] = col+2
                            kG0v[c] += T*i2*sina*(-i2*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*cos(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + k2*sin(pi*i2*xb/L)*sin(pi*k2*xb/L))/(r**2*(2.0*i2**2 - 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+0
                            kG0c[c] = col+3
                            kG0v[c] += L*P*j2*sina*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(cosa*(i2**2 - k2**2))
                            c += 1
                            kG0r[c] = row+1
                            kG0c[c] = col+2
                            kG0v[c] += L*P*j2*sina*(-i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(cosa*(i2**2 - k2**2))
                            c += 1
                            kG0r[c] = row+1
                            kG0c[c] = col+3
                            kG0v[c] += T*i2*sina*(-i2*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*cos(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + k2*sin(pi*i2*xb/L)*sin(pi*k2*xb/L))/(r**2*(2.0*i2**2 - 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+0
                            kG0v[c] += T*k2*sina*(-i2*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) - k2*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) + k2*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r**2*(-2.0*i2**2 + 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+1
                            kG0v[c] += L*P*j2*sina*(-i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(cosa*(i2**2 - k2**2))
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += 0.5*(i2*(-2*L**2*P + pi*k2**2*(Fc - pi*P*r**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*(2*L**2*P + pi*k2**2*(-Fc + pi*P*r**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*(-2*L**2*P + pi*i2**2*(Fc - pi*P*r**2))*sin(pi*k2*xb/L)*cos(pi*i2*xb/L) + k2*(2*L**2*P + pi*i2**2*(-Fc + pi*P*r**2))*sin(pi*k2*xa/L)*cos(pi*i2*xa/L))/(L*cosa*(i2 - k2)*(i2 + k2))
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+4
                            kG0v[c] += T*cosa*k2*(-i2*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) - k2*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) + k2*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r**2*(-2.0*i2**2 + 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+5
                            kG0v[c] += L*P*j2*(-i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            kG0r[c] = row+3
                            kG0c[c] = col+0
                            kG0v[c] += L*P*j2*sina*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(cosa*(i2**2 - k2**2))
                            c += 1
                            kG0r[c] = row+3
                            kG0c[c] = col+1
                            kG0v[c] += T*k2*sina*(-i2*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) - k2*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) + k2*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r**2*(-2.0*i2**2 + 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+3
                            kG0c[c] = col+3
                            kG0v[c] += 0.5*(i2*(-2*L**2*P + pi*k2**2*(Fc - pi*P*r**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*(2*L**2*P + pi*k2**2*(-Fc + pi*P*r**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*(-2*L**2*P + pi*i2**2*(Fc - pi*P*r**2))*sin(pi*k2*xb/L)*cos(pi*i2*xb/L) + k2*(2*L**2*P + pi*i2**2*(-Fc + pi*P*r**2))*sin(pi*k2*xa/L)*cos(pi*i2*xa/L))/(L*cosa*(i2 - k2)*(i2 + k2))
                            c += 1
                            kG0r[c] = row+3
                            kG0c[c] = col+4
                            kG0v[c] += L*P*j2*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            kG0r[c] = row+3
                            kG0c[c] = col+5
                            kG0v[c] += T*cosa*k2*(-i2*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*sin(pi*k2*xb/L) - k2*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) + k2*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r**2*(-2.0*i2**2 + 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+4
                            kG0c[c] = col+2
                            kG0v[c] += T*cosa*i2*(-i2*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*cos(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + k2*sin(pi*i2*xb/L)*sin(pi*k2*xb/L))/(r**2*(2.0*i2**2 - 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+4
                            kG0c[c] = col+3
                            kG0v[c] += L*P*j2*(i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) - i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) + k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            kG0r[c] = row+4
                            kG0c[c] = col+4
                            kG0v[c] += 0.5*(i2*(-2*L**2*P*j2**2 + pi*k2**2*(Fc - pi*P*r**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*(2*L**2*P*j2**2 + pi*k2**2*(-Fc + pi*P*r**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*(-2*L**2*P*j2**2 + pi*i2**2*(Fc - pi*P*r**2))*sin(pi*k2*xb/L)*cos(pi*i2*xb/L) + k2*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2))*sin(pi*k2*xa/L)*cos(pi*i2*xa/L))/(L*cosa*(i2 - k2)*(i2 + k2))
                            c += 1
                            kG0r[c] = row+4
                            kG0c[c] = col+5
                            kG0v[c] += -T*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r**2*(2.0*i2**2 - 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+5
                            kG0c[c] = col+2
                            kG0v[c] += L*P*j2*(-i2*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*sin(pi*k2*xa/L)*cos(pi*i2*xa/L) - k2*sin(pi*k2*xb/L)*cos(pi*i2*xb/L))/(i2**2 - k2**2)
                            c += 1
                            kG0r[c] = row+5
                            kG0c[c] = col+3
                            kG0v[c] += T*cosa*i2*(-i2*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*cos(pi*i2*xb/L)*cos(pi*k2*xb/L) - k2*sin(pi*i2*xa/L)*sin(pi*k2*xa/L) + k2*sin(pi*i2*xb/L)*sin(pi*k2*xb/L))/(r**2*(2.0*i2**2 - 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+5
                            kG0c[c] = col+4
                            kG0v[c] += T*j2*(2*i2*k2*(sin(pi*i2*xa/L)*sin(pi*k2*xa/L) - sin(pi*i2*xb/L)*sin(pi*k2*xb/L)) + (i2**2 + k2**2)*cos(pi*i2*xa/L)*cos(pi*k2*xa/L) - (i2**2 + k2**2)*cos(pi*i2*xb/L)*cos(pi*k2*xb/L))/(r**2*(2.0*i2**2 - 2.0*k2**2))
                            c += 1
                            kG0r[c] = row+5
                            kG0c[c] = col+5
                            kG0v[c] += 0.5*(i2*(-2*L**2*P*j2**2 + pi*k2**2*(Fc - pi*P*r**2))*sin(pi*i2*xa/L)*cos(pi*k2*xa/L) + i2*(2*L**2*P*j2**2 + pi*k2**2*(-Fc + pi*P*r**2))*sin(pi*i2*xb/L)*cos(pi*k2*xb/L) + k2*(-2*L**2*P*j2**2 + pi*i2**2*(Fc - pi*P*r**2))*sin(pi*k2*xb/L)*cos(pi*i2*xb/L) + k2*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2))*sin(pi*k2*xa/L)*cos(pi*i2*xa/L))/(L*cosa*(i2 - k2)*(i2 + k2))

    size = num0 + num1*m1 + num2*m2*n2

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkG0_cyl(double Fc, double P, double T, double r2, double L,
            int m1, int m2, int n2):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col
    cdef double r=r2
    cdef long [:] kG0r, kG0c
    cdef double [:] kG0v

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

    fdim = 1 + 2*m1 + k11_num + k22_num

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kG0_00
    c += 1
    kG0r[c] = 1
    kG0c[c] = 1
    kG0v[c] += 0.333333333333333*r2**2*(-3*Fc + pi*P*(2*L**2 + 3*r**2))/L


    for i1 in range(i0, m1+i0):
        col = (i1-i0)*num1 + num0
        row = col

        if i1 != 0:
            # kG0_01 cond_1
            c += 1
            kG0r[c] = 1
            kG0c[c] = col+1
            kG0v[c] += 2*L*P*r2/i1
            c += 1
            kG0r[c] = 1
            kG0c[c] = col+2
            kG0v[c] += -T*r2*((-1)**i1 - 1)/(pi*i1*r**2)

        for k1 in range(i0, m1+i0):
            col = (k1-i0)*num1 + num0

            #NOTE symmetry
            if row > col:
                continue

            if k1 == i1:
                # kG0_11 cond_1
                c += 1
                kG0r[c] = row+1
                kG0c[c] = col+1
                kG0v[c] += 0.5*pi*(2*L**2*P + pi*i1**2*(-Fc + pi*P*r**2))/L
                c += 1
                kG0r[c] = row+2
                kG0c[c] = col+2
                kG0v[c] += 0.5*pi**2*i1**2*(-Fc + pi*P*r**2)/L

            else:
                # kG0_11 cond_2
                c += 1
                kG0r[c] = row+1
                kG0c[c] = col+2
                kG0v[c] += -T*i1*k1*((-1)**(i1 + k1) - 1)/(r**2*(i1**2 - k1**2))
                c += 1
                kG0r[c] = row+2
                kG0c[c] = col+1
                kG0v[c] += T*i1*k1*((-1)**(i1 + k1) - 1)/(r**2*(i1**2 - k1**2))

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
                        # kG0_22 cond_1
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += 0.25*pi*(2*L**2*P + pi*i2**2*(-Fc + pi*P*r**2))/L
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+5
                        kG0v[c] += 0.5*pi*L*P*j2
                        c += 1
                        kG0r[c] = row+3
                        kG0c[c] = col+3
                        kG0v[c] += 0.25*pi*(2*L**2*P + pi*i2**2*(-Fc + pi*P*r**2))/L
                        c += 1
                        kG0r[c] = row+3
                        kG0c[c] = col+4
                        kG0v[c] += -0.5*pi*L*P*j2
                        c += 1
                        kG0r[c] = row+4
                        kG0c[c] = col+3
                        kG0v[c] += -0.5*pi*L*P*j2
                        c += 1
                        kG0r[c] = row+4
                        kG0c[c] = col+4
                        kG0v[c] += 0.25*pi*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2))/L
                        c += 1
                        kG0r[c] = row+5
                        kG0c[c] = col+2
                        kG0v[c] += 0.5*pi*L*P*j2
                        c += 1
                        kG0r[c] = row+5
                        kG0c[c] = col+5
                        kG0v[c] += 0.25*pi*(2*L**2*P*j2**2 + pi*i2**2*(-Fc + pi*P*r**2))/L

                    elif k2 != i2 and l2 == j2:
                        # kG0_22 cond_2
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+4
                        kG0v[c] += T*k2**2*((-1)**(i2 + k2) - 1)/(r**2*(-2.0*i2**2 + 2.0*k2**2))
                        c += 1
                        kG0r[c] = row+3
                        kG0c[c] = col+5
                        kG0v[c] += T*k2**2*((-1)**(i2 + k2) - 1)/(r**2*(-2.0*i2**2 + 2.0*k2**2))
                        c += 1
                        kG0r[c] = row+4
                        kG0c[c] = col+2
                        kG0v[c] += T*i2**2*((-1)**(i2 + k2) - 1)/(r**2*(2.0*i2**2 - 2.0*k2**2))
                        c += 1
                        kG0r[c] = row+4
                        kG0c[c] = col+5
                        kG0v[c] += T*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/(r**2*(i2 + k2)*(2.0*i2 - 2.0*k2))
                        c += 1
                        kG0r[c] = row+5
                        kG0c[c] = col+3
                        kG0v[c] += T*i2**2*((-1)**(i2 + k2) - 1)/(r**2*(2.0*i2**2 - 2.0*k2**2))
                        c += 1
                        kG0r[c] = row+5
                        kG0c[c] = col+4
                        kG0v[c] += -T*j2*((-1)**(i2 + k2) - 1)*(i2**2 + k2**2)/(r**2*(i2 + k2)*(2.0*i2 - 2.0*k2))

    size = num0 + num1*m1 + num2*m2*n2

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
