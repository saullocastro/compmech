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
    double cos(double theta) nogil
    double sin(double theta) nogil

cdef int num = 3
cdef int num1 = 4
cdef double pi = 3.141592653589793


def fk0y1y2(double y1, double y2, double a, double b, double r,
            int m, int n, np.ndarray[cDOUBLE, ndim=2] F):
    cdef int i, j, k, l, row, col, c
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef np.ndarray[cINT, ndim=1] k0y1y2r, k0y1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] k0y1y2v

    fdim = 9*m*n*m*n

    k0y1y2r = np.zeros((fdim,), dtype=INT)
    k0y1y2c = np.zeros((fdim,), dtype=INT)
    k0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

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

    # k0y1y2_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # k0y1y2_11 cond_1
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += A16*k*((-1)**(k + 1) - 1)*(2*j*l*cos(pi*j*y1/b)*cos(pi*l*y1/b) - 2*j*l*cos(pi*j*y2/b)*cos(pi*l*y2/b) + ((j*j) + (l*l))*(sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b)))/(((j*j) - (l*l))*(-(k*k) + 1))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += k*(2*(-1)**(k + 2) - 2)*(j*l*(A12 + A66)*cos(pi*j*y1/b)*cos(pi*l*y1/b) - j*l*(A12 + A66)*cos(pi*j*y2/b)*cos(pi*l*y2/b) + (A12*(l*l) + A66*(j*j))*(sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b)))/(((j*j) - (l*l))*(-(k*k) + 4))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += k*(3*(-1)**(k + 3) - 3)*(2*(pi*pi)*B66*(a*a)*j*l*r*(-j*sin(pi*j*y1/b)*cos(pi*l*y1/b) + j*sin(pi*j*y2/b)*cos(pi*l*y2/b) + l*sin(pi*l*y1/b)*cos(pi*j*y1/b) - l*sin(pi*l*y2/b)*cos(pi*j*y2/b)) + ((pi*pi)*B11*(b*b)*(k*k)*r + (a*a)*(A12*(b*b) + (pi*pi)*B12*(l*l)*r))*(j*sin(pi*l*y1/b)*cos(pi*j*y1/b) - j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - l*sin(pi*j*y1/b)*cos(pi*l*y1/b) + l*sin(pi*j*y2/b)*cos(pi*l*y2/b)))/(pi*(a*a)*b*r*((j*j) - (l*l))*(-(k*k) + 9))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += k*(4*(-1)**(k + 4) - 4)*(j*l*(A12 + A66)*cos(pi*j*y1/b)*cos(pi*l*y1/b) - j*l*(A12 + A66)*cos(pi*j*y2/b)*cos(pi*l*y2/b) + (A12*(j*j) + A66*(l*l))*(sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b)))/(((j*j) - (l*l))*(-(k*k) + 16))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += A26*k*(5*(-1)**(k + 5) - 5)*(2*j*l*cos(pi*j*y1/b)*cos(pi*l*y1/b) - 2*j*l*cos(pi*j*y2/b)*cos(pi*l*y2/b) + ((j*j) + (l*l))*(sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b)))/(((j*j) - (l*l))*(-(k*k) + 25))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += k*(6*(-1)**(k + 6) - 6)*(2*(pi*pi)*B26*(a*a)*j*l*r*(-j*sin(pi*j*y1/b)*cos(pi*l*y1/b) + j*sin(pi*j*y2/b)*cos(pi*l*y2/b) + l*sin(pi*l*y1/b)*cos(pi*j*y1/b) - l*sin(pi*l*y2/b)*cos(pi*j*y2/b)) + ((pi*pi)*B16*(b*b)*(k*k)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(l*l)*r))*(j*sin(pi*l*y1/b)*cos(pi*j*y1/b) - j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - l*sin(pi*j*y1/b)*cos(pi*l*y1/b) + l*sin(pi*j*y2/b)*cos(pi*l*y2/b)))/(pi*(a*a)*b*r*((j*j) - (l*l))*(-(k*k) + 36))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += k*(-7*(-1)**(k + 7) + 7)*(2*(pi*pi)*B66*(a*a)*j*l*r*(-j*sin(pi*j*y1/b)*cos(pi*l*y1/b) + j*sin(pi*j*y2/b)*cos(pi*l*y2/b) + l*sin(pi*l*y1/b)*cos(pi*j*y1/b) - l*sin(pi*l*y2/b)*cos(pi*j*y2/b)) + (49*(pi*pi)*B11*(b*b)*r + (a*a)*(A12*(b*b) + (pi*pi)*B12*(j*j)*r))*(j*sin(pi*l*y1/b)*cos(pi*j*y1/b) - j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - l*sin(pi*j*y1/b)*cos(pi*l*y1/b) + l*sin(pi*j*y2/b)*cos(pi*l*y2/b)))/(pi*(a*a)*b*r*((j*j) - (l*l))*(-(k*k) + 49))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += k*(-8*(-1)**(k + 8) + 8)*(2*(pi*pi)*B26*(a*a)*j*l*r*(-j*sin(pi*j*y1/b)*cos(pi*l*y1/b) + j*sin(pi*j*y2/b)*cos(pi*l*y2/b) + l*sin(pi*l*y1/b)*cos(pi*j*y1/b) - l*sin(pi*l*y2/b)*cos(pi*j*y2/b)) + (64*(pi*pi)*B16*(b*b)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j*j)*r))*(j*sin(pi*l*y1/b)*cos(pi*j*y1/b) - j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - l*sin(pi*j*y1/b)*cos(pi*l*y1/b) + l*sin(pi*j*y2/b)*cos(pi*l*y2/b)))/(pi*(a*a)*b*r*((j*j) - (l*l))*(-(k*k) + 64))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += k*(18*(-1)**(k + 9) - 18)*(j*((pi*pi)*D16*(b*b)*(k*k)*r + (a*a)*(B26*(b*b) + (pi*pi)*D26*(l*l)*r))*(j*sin(pi*j*y1/b)*sin(pi*l*y1/b) - j*sin(pi*j*y2/b)*sin(pi*l*y2/b) + l*cos(pi*j*y1/b)*cos(pi*l*y1/b) - l*cos(pi*j*y2/b)*cos(pi*l*y2/b)) + l*(81*(pi*pi)*D16*(b*b)*r + (a*a)*(B26*(b*b) + (pi*pi)*D26*(j*j)*r))*(j*cos(pi*j*y1/b)*cos(pi*l*y1/b) - j*cos(pi*j*y2/b)*cos(pi*l*y2/b) + l*sin(pi*j*y1/b)*sin(pi*l*y1/b) - l*sin(pi*j*y2/b)*sin(pi*l*y2/b)))/((a*a)*(b*b)*r*((j*j) - (l*l))*(-(k*k) + 81))

                    elif k == i and l != j:
                        # k0y1y2_11 cond_2
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += -0.5*pi*(19321*A11*(b*b)*j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - 19321*A11*(b*b)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) - A66*(a*a)*(j*j)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) + A66*(a*a)*j*(l*l)*sin(pi*l*y2/b)*cos(pi*j*y2/b) - j*(19321*A11*(b*b) + A66*(a*a)*(l*l))*sin(pi*l*y1/b)*cos(pi*j*y1/b) + l*(19321*A11*(b*b) + A66*(a*a)*(j*j))*sin(pi*j*y1/b)*cos(pi*l*y1/b))/(a*b*((j*j) - (l*l)))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += -0.5*pi*(19600*A16*(b*b)*j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - 19600*A16*(b*b)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) - A26*(a*a)*(j*j)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) + A26*(a*a)*j*(l*l)*sin(pi*l*y2/b)*cos(pi*j*y2/b) - j*(19600*A16*(b*b) + A26*(a*a)*(l*l))*sin(pi*l*y1/b)*cos(pi*j*y1/b) + l*(19600*A16*(b*b) + A26*(a*a)*(j*j))*sin(pi*j*y1/b)*cos(pi*l*y1/b))/(a*b*((j*j) - (l*l)))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.5*(-j*l*(59643*(pi*pi)*B16*(b*b)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(l*l)*r))*cos(pi*j*y1/b)*cos(pi*l*y1/b) + j*l*(59643*(pi*pi)*B16*(b*b)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(l*l)*r))*cos(pi*j*y2/b)*cos(pi*l*y2/b) - (sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b))*(19881*(pi*pi)*B16*(b*b)*r*((j*j) + 2*(l*l)) + (a*a)*(j*j)*(A26*(b*b) + (pi*pi)*B26*(l*l)*r)))/(a*(b*b)*r*((j*j) - (l*l)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += -0.5*pi*(20164*A16*(b*b)*j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - 20164*A16*(b*b)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) - A26*(a*a)*(j*j)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) + A26*(a*a)*j*(l*l)*sin(pi*l*y2/b)*cos(pi*j*y2/b) - j*(20164*A16*(b*b) + A26*(a*a)*(l*l))*sin(pi*l*y1/b)*cos(pi*j*y1/b) + l*(20164*A16*(b*b) + A26*(a*a)*(j*j))*sin(pi*j*y1/b)*cos(pi*l*y1/b))/(a*b*((j*j) - (l*l)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += -0.5*pi*(-A22*(a*a)*(j*j)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) + A22*(a*a)*j*(l*l)*sin(pi*l*y2/b)*cos(pi*j*y2/b) + 20449*A66*(b*b)*j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - 20449*A66*(b*b)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) - j*(A22*(a*a)*(l*l) + 20449*A66*(b*b))*sin(pi*l*y1/b)*cos(pi*j*y1/b) + l*(A22*(a*a)*(j*j) + 20449*A66*(b*b))*sin(pi*j*y1/b)*cos(pi*l*y1/b))/(a*b*((j*j) - (l*l)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.5*(-j*l*((a*a)*(A22*(b*b) + (pi*pi)*B22*(l*l)*r) + 20736*(pi*pi)*(b*b)*r*(B12 + 2*B66))*cos(pi*j*y1/b)*cos(pi*l*y1/b) + j*l*((a*a)*(A22*(b*b) + (pi*pi)*B22*(l*l)*r) + 20736*(pi*pi)*(b*b)*r*(B12 + 2*B66))*cos(pi*j*y2/b)*cos(pi*l*y2/b) - (sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b))*((a*a)*(j*j)*(A22*(b*b) + (pi*pi)*B22*(l*l)*r) + 20736*(pi*pi)*(b*b)*r*(B12*(j*j) + 2*B66*(l*l))))/(a*(b*b)*r*((j*j) - (l*l)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 0.5*(j*l*(63075*(pi*pi)*B16*(b*b)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j*j)*r))*cos(pi*j*y1/b)*cos(pi*l*y1/b) - j*l*(63075*(pi*pi)*B16*(b*b)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j*j)*r))*cos(pi*j*y2/b)*cos(pi*l*y2/b) + (sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b))*(21025*(pi*pi)*B16*(b*b)*r*(2*(j*j) + (l*l)) + (a*a)*(l*l)*(A26*(b*b) + (pi*pi)*B26*(j*j)*r)))/(a*(b*b)*r*((j*j) - (l*l)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += 0.5*(j*l*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j*j)*r) + 21316*(pi*pi)*(b*b)*r*(B12 + 2*B66))*cos(pi*j*y1/b)*cos(pi*l*y1/b) - j*l*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j*j)*r) + 21316*(pi*pi)*(b*b)*r*(B12 + 2*B66))*cos(pi*j*y2/b)*cos(pi*l*y2/b) + (sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b))*((a*a)*(l*l)*(A22*(b*b) + (pi*pi)*B22*(j*j)*r) + 21316*(pi*pi)*(b*b)*r*(B12*(l*l) + 2*B66*(j*j))))/(a*(b*b)*r*((j*j) - (l*l)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.5*(86436*(pi*pi*pi*pi)*D66*(a*a)*(b*b)*j*l*(r*r)*(-j*sin(pi*j*y1/b)*cos(pi*l*y1/b) + j*sin(pi*j*y2/b)*cos(pi*l*y2/b) + l*sin(pi*l*y1/b)*cos(pi*j*y1/b) - l*sin(pi*l*y2/b)*cos(pi*j*y2/b)) + (466948881*(pi*pi*pi*pi)*D11*(b*b*b*b)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*r*(B22*(b*b)*((j*j) + (l*l)) + (pi*pi)*D22*(j*j)*(l*l)*r)) + 21609*(pi*pi)*(a*a)*(b*b)*r*(2*B12*(b*b) + (pi*pi)*D12*r*((j*j) + (l*l))))*(j*sin(pi*l*y1/b)*cos(pi*j*y1/b) - j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - l*sin(pi*j*y1/b)*cos(pi*l*y1/b) + l*sin(pi*j*y2/b)*cos(pi*l*y2/b)))/(pi*(a*a*a)*(b*b*b)*(r*r)*((j*j) - (l*l)))

                    elif k != i and l == j:
                        # k0y1y2_11 cond_3
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += k*(139*(-1)**(k + 278) - 139)*(A12 - A66)*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))/(-2.0*(k*k) + 154568.0)
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += k*(279*(-1)**(k + 279) - 279)*(0.5*pi*B66*(a*a)*j*r*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2)) + 0.25*((pi*pi)*B11*(b*b)*(k*k)*r + (a*a)*(A12*(b*b) + (pi*pi)*B12*(j*j)*r))*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(pi*j))/((a*a)*(b*b)*r*(-(k*k) + 77841))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += k*(70*(-1)**(k + 280) - 70)*(-A12 + A66)*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))/(-(k*k) + 78400)
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += k*(282*(-1)**(k + 282) - 282)*(0.5*pi*B26*(a*a)*j*r*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2)) + 0.25*((pi*pi)*B16*(b*b)*(k*k)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j*j)*r))*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(pi*j))/((a*a)*(b*b)*r*(-(k*k) + 79524))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += k*(-283*(-1)**(k + 283) + 283)*(0.5*pi*B66*(a*a)*j*r*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2)) + 0.25*(80089*(pi*pi)*B11*(b*b)*r + (a*a)*(A12*(b*b) + (pi*pi)*B12*(j*j)*r))*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(pi*j))/((a*a)*(b*b)*r*(-(k*k) + 80089))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += k*(-284*(-1)**(k + 284) + 284)*(0.5*pi*B26*(a*a)*j*r*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2)) + 0.25*(80656*(pi*pi)*B16*(b*b)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j*j)*r))*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(pi*j))/((a*a)*(b*b)*r*(-(k*k) + 80656))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.5*(pi*pi)*D16*k*(285*(-1)**(k + 285) - 285)*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))/(a*a)

                    elif k == i and l == j:
                        # k0y1y2_11 cond_4
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 21528.125*pi*A11*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(a*j) + 0.125*pi*A66*a*j*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(b*b)
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += 21632*pi*A16*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(a*j) + 0.125*pi*A26*a*j*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(b*b)
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.125*(-173889*(pi*pi)*B16*(b*b)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j*j)*r))*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))/(a*(b*b)*r)
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 21840.5*pi*A16*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(a*j) + 0.125*pi*A26*a*j*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(b*b)
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += 0.125*pi*A22*a*j*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(b*b) + 21945.125*pi*A66*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(a*j)
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.125*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j*j)*r) + 176400*(pi*pi)*(b*b)*r*(B12 - 2*B66))*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))/(a*(b*b)*r)
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 0.125*(-177241*(pi*pi)*B16*(b*b)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j*j)*r))*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))/(a*(b*b)*r)
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += 0.125*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j*j)*r) + 178084*(pi*pi)*(b*b)*r*(B12 - 2*B66))*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))/(a*(b*b)*r)
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.125*(b*(32015587041*(pi*pi*pi*pi)*D11*(b*b*b*b)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*(j*j)*r*(2*B22*(b*b) + (pi*pi)*D22*(j*j)*r)) + 357858*(pi*pi)*(a*a)*(b*b)*r*(B12*(b*b) + (pi*pi)*(j*j)*r*(D12 - 2*D66)))*sin(2*pi*j*y1/b) - b*(32015587041*(pi*pi*pi*pi)*D11*(b*b*b*b)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*(j*j)*r*(2*B22*(b*b) + (pi*pi)*D22*(j*j)*r)) + 357858*(pi*pi)*(a*a)*(b*b)*r*(B12*(b*b) + (pi*pi)*(j*j)*r*(D12 - 2*D66)))*sin(2*pi*j*y2/b) - 2*pi*j*(y1 - y2)*(32015587041*(pi*pi*pi*pi)*D11*(b*b*b*b)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*(j*j)*r*(2*B22*(b*b) + (pi*pi)*D22*(j*j)*r)) + 357858*(pi*pi)*(a*a)*(b*b)*r*(B12*(b*b) + (pi*pi)*(j*j)*r*(D12 + 2*D66))))/(pi*(a*a*a)*(b*b*b*b)*j*(r*r))

    size = num*m*n

    k0 = coo_matrix((k0y1y2v, (k0y1y2r, k0y1y2c)), shape=(size, size))

    return k0


def fkG0y1y2(double y1, double y2, double Nxx, double Nyy, double Nxy,
             double a, double b, double r, int m, int n):
    cdef int i, k, j, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kG0y1y2r, kG0y1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0y1y2v

    fdim = 1*m*n*m*n

    kG0y1y2r = np.zeros((fdim,), dtype=INT)
    kG0y1y2c = np.zeros((fdim,), dtype=INT)
    kG0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kG0_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # kG0y1y2_11 cond_1
                        c += 1
                        kG0y1y2r[c] = row+2
                        kG0y1y2c[c] = col+2
                        kG0y1y2v[c] += Nxy*i*k*((-1)**(i + k) - 1)*(2*j*l*cos(pi*j*y1/b)*cos(pi*l*y1/b) - 2*j*l*cos(pi*j*y2/b)*cos(pi*l*y2/b) + ((j*j) + (l*l))*(sin(pi*j*y1/b)*sin(pi*l*y1/b) - sin(pi*j*y2/b)*sin(pi*l*y2/b)))/(((i*i) - (k*k))*((j*j) - (l*l)))

                    elif k == i and l != j:
                        # kG0y1y2_11 cond_2
                        c += 1
                        kG0y1y2r[c] = row+2
                        kG0y1y2c[c] = col+2
                        kG0y1y2v[c] += -0.5*pi*(Nxx*(b*b)*(i*i)*j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - Nxx*(b*b)*(i*i)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) - Nyy*(a*a)*(j*j)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) + Nyy*(a*a)*j*(l*l)*sin(pi*l*y2/b)*cos(pi*j*y2/b) - j*(Nxx*(b*b)*(i*i) + Nyy*(a*a)*(l*l))*sin(pi*l*y1/b)*cos(pi*j*y1/b) + l*(Nxx*(b*b)*(i*i) + Nyy*(a*a)*(j*j))*sin(pi*j*y1/b)*cos(pi*l*y1/b))/(a*b*((j*j) - (l*l)))

                    elif k != i and l == j:
                        # kG0y1y2_11 cond_3
                        pass

                    elif k == i and l == j:
                        # kG0y1y2_11 cond_4
                        c += 1
                        kG0y1y2r[c] = row+2
                        kG0y1y2c[c] = col+2
                        kG0y1y2v[c] += 0.125*pi*Nxx*(i*i)*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(a*j) + 0.125*pi*Nyy*a*j*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(b*b)

    size = num*m*n

    kG0 = coo_matrix((kG0y1y2v, (kG0y1y2r, kG0y1y2c)), shape=(size, size))

    return kG0


def fkMy1y2(double y1, double y2, double mu, double d, double h,
            double a, double b, int m, int n):
    cdef int i, k, j, l, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMy1y2r, kMy1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] kMy1y2v

    fdim = 5*m*n*m*n

    kMy1y2r = np.zeros((fdim,), dtype=INT)
    kMy1y2c = np.zeros((fdim,), dtype=INT)
    kMy1y2v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kMy1y2_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # kMy1y2_11 cond_1
                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += b*d*h*i*k*mu*((-1)**(i + k) - 1)*(-j*sin(pi*l*y1/b)*cos(pi*j*y1/b) + j*sin(pi*l*y2/b)*cos(pi*j*y2/b) + l*sin(pi*j*y1/b)*cos(pi*l*y1/b) - l*sin(pi*j*y2/b)*cos(pi*l*y2/b))/(pi*((i*i) - (k*k))*((j*j) - (l*l)))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += b*d*h*i*k*mu*((-1)**(i + k) - 1)*(j*sin(pi*l*y1/b)*cos(pi*j*y1/b) - j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - l*sin(pi*j*y1/b)*cos(pi*l*y1/b) + l*sin(pi*j*y2/b)*cos(pi*l*y2/b))/(pi*((i*i) - (k*k))*((j*j) - (l*l)))

                    elif k == i and l != j:
                        # kMy1y2_11 cond_2
                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += a*b*h*mu*(j*sin(pi*l*y1/b)*cos(pi*j*y1/b) - j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - l*sin(pi*j*y1/b)*cos(pi*l*y1/b) + l*sin(pi*j*y2/b)*cos(pi*l*y2/b))/(pi*(2.0*(j*j) - 2.0*(l*l)))
                        c += 1
                        kMy1y2r[c] = row+1
                        kMy1y2c[c] = col+1
                        kMy1y2v[c] += a*b*h*mu*(j*sin(pi*l*y1/b)*cos(pi*j*y1/b) - j*sin(pi*l*y2/b)*cos(pi*j*y2/b) - l*sin(pi*j*y1/b)*cos(pi*l*y1/b) + l*sin(pi*j*y2/b)*cos(pi*l*y2/b))/(pi*(2.0*(j*j) - 2.0*(l*l)))
                        c += 1
                        kMy1y2r[c] = row+1
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += a*d*h*l*mu*(j*cos(pi*j*y1/b)*cos(pi*l*y1/b) - j*cos(pi*j*y2/b)*cos(pi*l*y2/b) + l*sin(pi*j*y1/b)*sin(pi*l*y1/b) - l*sin(pi*j*y2/b)*sin(pi*l*y2/b))/(2.0*(j*j) - 2.0*(l*l))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+1
                        kMy1y2v[c] += a*d*h*j*mu*(-j*sin(pi*j*y1/b)*sin(pi*l*y1/b) + j*sin(pi*j*y2/b)*sin(pi*l*y2/b) - l*cos(pi*j*y1/b)*cos(pi*l*y1/b) + l*cos(pi*j*y2/b)*cos(pi*l*y2/b))/(2.0*(j*j) - 2.0*(l*l))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += 0.0416666666666667*h*mu*(-12*(a*a)*(b*b)*j*sin(pi*l*y2/b)*cos(pi*j*y2/b) + 12*(a*a)*(b*b)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) + 12*(pi*pi)*(a*a)*(d*d)*(j*j)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) - 12*(pi*pi)*(a*a)*(d*d)*j*(l*l)*sin(pi*l*y2/b)*cos(pi*j*y2/b) + (pi*pi)*(a*a)*(h*h)*(j*j)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) - (pi*pi)*(a*a)*(h*h)*j*(l*l)*sin(pi*l*y2/b)*cos(pi*j*y2/b) - 12*(pi*pi)*(b*b)*(d*d)*(i*i)*j*sin(pi*l*y2/b)*cos(pi*j*y2/b) + 12*(pi*pi)*(b*b)*(d*d)*(i*i)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) - (pi*pi)*(b*b)*(h*h)*(i*i)*j*sin(pi*l*y2/b)*cos(pi*j*y2/b) + (pi*pi)*(b*b)*(h*h)*(i*i)*l*sin(pi*j*y2/b)*cos(pi*l*y2/b) + j*((a*a)*(12*(b*b) + (pi*pi)*(l*l)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i*i)*(12*(d*d) + (h*h)))*sin(pi*l*y1/b)*cos(pi*j*y1/b) - l*((a*a)*(12*(b*b) + (pi*pi)*(j*j)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i*i)*(12*(d*d) + (h*h)))*sin(pi*j*y1/b)*cos(pi*l*y1/b))/(pi*a*b*((j*j) - (l*l)))

                    elif k != i and l == j:
                        # kMy1y2_11 cond_3
                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += 0.25*d*h*i*k*mu*((-1)**(i + k) - 1)*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(y1 - y2))/(pi*j*((i*i) - (k*k)))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += -0.25*d*h*i*k*mu*((-1)**(i + k) - 1)*(-b*sin(2*pi*j*y1/b) + b*sin(2*pi*j*y2/b) + 2*pi*j*(y1 - y2))/(pi*j*((i*i) - (k*k)))

                    elif k == i and l == j:
                        # kMy1y2_11 cond_4
                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += 0.125*a*h*mu*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(pi*j)
                        c += 1
                        kMy1y2r[c] = row+1
                        kMy1y2c[c] = col+1
                        kMy1y2v[c] += 0.125*a*h*mu*(b*sin(2*pi*j*y1/b) - b*sin(2*pi*j*y2/b) + 2*pi*j*(-y1 + y2))/(pi*j)
                        c += 1
                        kMy1y2r[c] = row+1
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += 0.125*a*d*h*mu*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+1
                        kMy1y2v[c] += 0.125*a*d*h*mu*(cos(2*pi*j*y1/b) - cos(2*pi*j*y2/b))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += 0.0104166666666667*h*mu*(b*((a*a)*(12*(b*b) - (pi*pi)*(j*j)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i*i)*(12*(d*d) + (h*h)))*sin(2*pi*j*y1/b) - b*((a*a)*(12*(b*b) - (pi*pi)*(j*j)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i*i)*(12*(d*d) + (h*h)))*sin(2*pi*j*y2/b) - 2*pi*j*(y1 - y2)*((a*a)*(12*(b*b) + (pi*pi)*(j*j)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i*i)*(12*(d*d) + (h*h))))/(pi*a*(b*b)*j)

    size = num*m*n

    kM = coo_matrix((kMy1y2v, (kMy1y2r, kMy1y2c)), shape=(size, size))

    return kM


def fk0edges(int m, int n, double a, double b,
             double kphixBot, double kphixTop,
             double kphiyLeft, double kphiyRight):
    cdef int i, j, k, l, row, col, c
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv

    fdim = 1*m*n*m*n + 1*m*n*m*n

    k0edgesr = np.zeros((fdim,), dtype=INT)
    k0edgesc = np.zeros((fdim,), dtype=INT)
    k0edgesv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # k0edgesBT_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # k0edgesBT_11 cond_1
                        pass

                    elif k == i and l != j:
                        # k0edgesBT_11 cond_2
                        pass

                    elif k != i and l == j:
                        # k0edgesBT_11 cond_3
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*b*i*k*((-1)**(i + k)*kphixTop + kphixBot)/(a*a)

                    elif k == i and l == j:
                        # k0edgesBT_11 cond_4
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*b*(i*i)*(kphixBot + kphixTop)/(a*a)

    # k0edgesLR_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # k0edgesLR_11 cond_1
                        pass

                    elif k == i and l != j:
                        # k0edgesLR_11 cond_2
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*a*j*l*((-1)**(j + l)*kphiyLeft + kphiyRight)/(b*b)

                    elif k != i and l == j:
                        # k0edgesLR_11 cond_3
                        pass

                    elif k == i and l == j:
                        # k0edgesLR_11 cond_4
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*a*(j*j)*(kphiyLeft + kphiyRight)/(b*b)

    size = num*m*n

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkAx(double beta, double gamma, double a, double b, int m, int n):
    cdef int i, k, j, l, c, row, col
    cdef np.ndarray[cINT, ndim=1] kAxr, kAxc
    cdef np.ndarray[cDOUBLE, ndim=1] kAxv

    fdim = 1*m*n*m*n

    kAxr = np.zeros((fdim,), dtype=INT)
    kAxc = np.zeros((fdim,), dtype=INT)
    kAxv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kAx_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # kAx_11 cond_1
                        pass

                    elif k == i and l != j:
                        # kAx_11 cond_2
                        pass

                    elif k != i and l == j:
                        # kAx_11 cond_3
                        c += 1
                        kAxr[c] = row+2
                        kAxc[c] = col+2
                        kAxv[c] += -b*beta*i*k*((-1)**(i + k) - 1)/(2.0*(i*i) - 2.0*(k*k))

                    elif k == i and l == j:
                        # kAx_11 cond_4
                        c += 1
                        kAxr[c] = row+2
                        kAxc[c] = col+2
                        kAxv[c] += 0.25*a*b*gamma

    size = num*m*n

    kAx = coo_matrix((kAxv, (kAxr, kAxc)), shape=(size, size))

    return kAx


def fkAy(double beta, double a, double b, int m, int n):
    cdef int i, k, j, l, c, row, col
    cdef np.ndarray[cINT, ndim=1] kAyr, kAyc
    cdef np.ndarray[cDOUBLE, ndim=1] kAyv

    fdim = 1*m*n*m*n

    kAyr = np.zeros((fdim,), dtype=INT)
    kAyc = np.zeros((fdim,), dtype=INT)
    kAyv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kAy_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # kAy_11 cond_1
                        pass

                    elif k == i and l != j:
                        # kAy_11 cond_2
                        c += 1
                        kAyr[c] = row+2
                        kAyc[c] = col+2
                        kAyv[c] += -a*beta*j*l*((-1)**(j + l) - 1)/(2.0*(j*j) - 2.0*(l*l))

                    elif k != i and l == j:
                        # kAy_11 cond_3
                        pass

                    elif k == i and l == j:
                        # kAy_11 cond_4
                        pass

    size = num*m*n

    kAy = coo_matrix((kAyv, (kAyr, kAyc)), shape=(size, size))

    return kAy


def fcA(double aeromu, double a, double b, int m, int n):
    cdef int i, k, j, l, c, row, col
    cdef np.ndarray[cINT, ndim=1] cAr, cAc
    cdef np.ndarray[cDOUBLE, ndim=1] cAv

    fdim = 1*m*n*m*n

    cAr = np.zeros((fdim,), dtype=INT)
    cAc = np.zeros((fdim,), dtype=INT)
    cAv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # cA_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # cA_11 cond_1
                        pass

                    elif k == i and l != j:
                        # cA_11 cond_2
                        pass

                    elif k != i and l == j:
                        pass

                    elif k == i and l == j:
                        # cA_11 cond_4
                        c += 1
                        cAr[c] = row+2
                        cAc[c] = col+2
                        cAv[c] += -0.25*a*aeromu*b

    size = num*m*n

    cA = coo_matrix((cAv, (cAr, cAc)), shape=(size, size))

    return cA


def fk0f(double a, double bf, int m1, int n1, np.ndarray[cDOUBLE, ndim=2] F):
    cdef int i1, k1, j1, l1, c, row, col
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef np.ndarray[cINT, ndim=1] k0fr, k0fc
    cdef np.ndarray[cDOUBLE, ndim=1] k0fv

    fdim = 8*m1*n1*m1*n1

    k0fr = np.zeros((fdim,), dtype=INT)
    k0fc = np.zeros((fdim,), dtype=INT)
    k0fv = np.zeros((fdim,), dtype=DOUBLE)

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

    # k0f_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0f_11 cond_1
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+0
                        k0fv[c] += -A16*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+1
                        k0fv[c] += -((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(i1*i1)*(l1*l1) + A66*(j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0fr[c] = row+1
                        k0fc[c] = col+0
                        k0fv[c] += -((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(j1*j1)*(k1*k1) + A66*(i1*i1)*(l1*l1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0fr[c] = row+1
                        k0fc[c] = col+1
                        k0fv[c] += -A26*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+2
                        k0fv[c] += (pi*pi)*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*(D16*(bf*bf)*((i1*i1) + (k1*k1)) + D26*(a*a)*((j1*j1) + (l1*l1)))/((a*a)*(bf*bf)*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+3
                        k0fv[c] += (pi*pi)*i1*j1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(D11*(bf*bf*bf*bf)*(i1*i1)*(k1*k1) + D22*(a*a*a*a)*(j1*j1)*(l1*l1) + (a*a)*(bf*bf)*(D12*(i1*i1)*(l1*l1) + D12*(j1*j1)*(k1*k1) + 4*D66*(k1*k1)*(l1*l1)))/((a*a*a)*(bf*bf*bf)*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0fr[c] = row+3
                        k0fc[c] = col+2
                        k0fv[c] += (pi*pi)*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(D11*(bf*bf*bf*bf)*(i1*i1)*(k1*k1) + D22*(a*a*a*a)*(j1*j1)*(l1*l1) + (a*a)*(bf*bf)*(D12*(i1*i1)*(l1*l1) + D12*(j1*j1)*(k1*k1) + 4*D66*(i1*i1)*(j1*j1)))/((a*a*a)*(bf*bf*bf)*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0fr[c] = row+3
                        k0fc[c] = col+3
                        k0fv[c] += (pi*pi)*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*(D16*(bf*bf)*(i1*i1)*(k1*k1)*((j1*j1) + (l1*l1)) + D26*(a*a)*(j1*j1)*(l1*l1)*((i1*i1) + (k1*k1)))/((a*a)*(bf*bf)*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # k0f_11 cond_2
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+2
                        k0fv[c] += 0.5*(pi*pi)*i1*l1*((-1)**(j1 + l1) - 1)*(B11*(bf*bf)*(i1*i1) + (a*a)*(B12*(l1*l1) + 2*B66*(j1*j1)))/((a*a)*bf*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+3
                        k0fv[c] += 0.5*(pi*pi)*((-1)**(j1 + l1) - 1)*(B16*(bf*bf)*(i1*i1)*((j1*j1) + 2*(l1*l1)) + B26*(a*a)*(j1*j1)*(l1*l1))/(a*(bf*bf)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0fr[c] = row+1
                        k0fc[c] = col+2
                        k0fv[c] += -0.5*(pi*pi)*i1*l1*((-1)**(j1 + l1) - 1)*(B16*(bf*bf)*(i1*i1) + B26*(a*a)*(2*(j1*j1) + (l1*l1)))/((a*a)*bf*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0fr[c] = row+1
                        k0fc[c] = col+3
                        k0fv[c] += 0.5*(pi*pi)*((-1)**(j1 + l1) - 1)*(B12*(bf*bf)*(i1*i1)*(j1*j1) + (l1*l1)*(B22*(a*a)*(j1*j1) + 2*B66*(bf*bf)*(i1*i1)))/(a*(bf*bf)*((j1*j1) - (l1*l1)))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+0
                        k0fv[c] += 0.5*(pi*pi)*i1*j1*((-1)**(j1 + l1) - 1)*(B11*(bf*bf)*(i1*i1) + (a*a)*(B12*(j1*j1) + 2*B66*(l1*l1)))/((a*a)*bf*((j1*j1) - (l1*l1)))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+1
                        k0fv[c] += 0.5*(pi*pi)*i1*j1*((-1)**(j1 + l1) - 1)*(B16*(bf*bf)*(i1*i1) + B26*(a*a)*((j1*j1) + 2*(l1*l1)))/((a*a)*bf*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0fr[c] = row+3
                        k0fc[c] = col+0
                        k0fv[c] += -0.5*(pi*pi)*((-1)**(j1 + l1) - 1)*(B16*(bf*bf)*(i1*i1)*(2*(j1*j1) + (l1*l1)) + B26*(a*a)*(j1*j1)*(l1*l1))/(a*(bf*bf)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0fr[c] = row+3
                        k0fc[c] = col+1
                        k0fv[c] += -0.5*(pi*pi)*((-1)**(j1 + l1) - 1)*(2*B66*(bf*bf)*(i1*i1)*(j1*j1) + (l1*l1)*(B12*(bf*bf)*(i1*i1) + B22*(a*a)*(j1*j1)))/(a*(bf*bf)*((j1*j1) - (l1*l1)))

                    elif k1 != i1 and l1 == j1:
                        # k0f_11 cond_3
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+2
                        k0fv[c] += 0.5*(pi*pi)*j1*k1*((-1)**(i1 + k1) - 1)*(B16*(bf*bf)*(2*(i1*i1) + (k1*k1)) + B26*(a*a)*(j1*j1))/(a*(bf*bf)*(-(i1*i1) + (k1*k1)))
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+3
                        k0fv[c] += 0.5*(pi*pi)*((-1)**(i1 + k1) - 1)*(B11*(bf*bf)*(i1*i1)*(k1*k1) + (a*a)*(j1*j1)*(B12*(i1*i1) + 2*B66*(k1*k1)))/((a*a)*bf*((i1*i1) - (k1*k1)))
                        c += 1
                        k0fr[c] = row+1
                        k0fc[c] = col+2
                        k0fv[c] += 0.5*(pi*pi)*j1*k1*((-1)**(i1 + k1) - 1)*(B22*(a*a)*(j1*j1) + (bf*bf)*(B12*(k1*k1) + 2*B66*(i1*i1)))/(a*(bf*bf)*(-(i1*i1) + (k1*k1)))
                        c += 1
                        k0fr[c] = row+1
                        k0fc[c] = col+3
                        k0fv[c] += 0.5*(pi*pi)*((-1)**(i1 + k1) - 1)*(B16*(bf*bf)*(i1*i1)*(k1*k1) + B26*(a*a)*(j1*j1)*((i1*i1) + 2*(k1*k1)))/((a*a)*bf*((i1*i1) - (k1*k1)))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+0
                        k0fv[c] += 0.5*(pi*pi)*i1*j1*((-1)**(i1 + k1) - 1)*(B16*(bf*bf)*((i1*i1) + 2*(k1*k1)) + B26*(a*a)*(j1*j1))/(a*(bf*bf)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+1
                        k0fv[c] += 0.5*(pi*pi)*i1*j1*((-1)**(i1 + k1) - 1)*(B12*(bf*bf)*(i1*i1) + B22*(a*a)*(j1*j1) + 2*B66*(bf*bf)*(k1*k1))/(a*(bf*bf)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0fr[c] = row+3
                        k0fc[c] = col+0
                        k0fv[c] += -0.5*(pi*pi)*((-1)**(i1 + k1) - 1)*(B11*(bf*bf)*(i1*i1)*(k1*k1) + (a*a)*(j1*j1)*(B12*(k1*k1) + 2*B66*(i1*i1)))/((a*a)*bf*((i1*i1) - (k1*k1)))
                        c += 1
                        k0fr[c] = row+3
                        k0fc[c] = col+1
                        k0fv[c] += -0.5*(pi*pi)*((-1)**(i1 + k1) - 1)*(B16*(bf*bf)*(i1*i1)*(k1*k1) + B26*(a*a)*(j1*j1)*(2*(i1*i1) + (k1*k1)))/((a*a)*bf*((i1*i1) - (k1*k1)))

                    elif k1 == i1 and l1 == j1:
                        # k0f_11 cond_4
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+0
                        k0fv[c] += 0.25*(pi*pi)*(A11*(bf*bf)*(i1*i1) + A66*(a*a)*(j1*j1))/(a*bf)
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+1
                        k0fv[c] += 0.25*(pi*pi)*(A16*(bf*bf)*(i1*i1) + A26*(a*a)*(j1*j1))/(a*bf)
                        c += 1
                        k0fr[c] = row+1
                        k0fc[c] = col+0
                        k0fv[c] += 0.25*(pi*pi)*(A16*(bf*bf)*(i1*i1) + A26*(a*a)*(j1*j1))/(a*bf)
                        c += 1
                        k0fr[c] = row+1
                        k0fc[c] = col+1
                        k0fv[c] += 0.25*(pi*pi)*(A22*(a*a)*(j1*j1) + A66*(bf*bf)*(i1*i1))/(a*bf)
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+2
                        k0fv[c] += 0.25*(pi*pi*pi*pi)*(D11*(bf*bf*bf*bf)*(i1*i1*i1*i1) + D22*(a*a*a*a)*(j1*j1*j1*j1) + 2*(a*a)*(bf*bf)*(i1*i1)*(j1*j1)*(D12 + 2*D66))/((a*a*a)*(bf*bf*bf))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+3
                        k0fv[c] += (pi*pi*pi*pi)*i1*j1*(-D16*(i1*i1)/(a*a) - D26*(j1*j1)/(bf*bf))
                        c += 1
                        k0fr[c] = row+3
                        k0fc[c] = col+2
                        k0fv[c] += (pi*pi*pi*pi)*i1*j1*(-D16*(i1*i1)/(a*a) - D26*(j1*j1)/(bf*bf))
                        c += 1
                        k0fr[c] = row+3
                        k0fc[c] = col+3
                        k0fv[c] += 0.25*(pi*pi*pi*pi)*(D11*(bf*bf*bf*bf)*(i1*i1*i1*i1) + D22*(a*a*a*a)*(j1*j1*j1*j1) + 2*(a*a)*(bf*bf)*(i1*i1)*(j1*j1)*(D12 + 2*D66))/((a*a*a)*(bf*bf*bf))

    size = num1*m1*n1

    k0f = coo_matrix((k0fv, (k0fr, k0fc)), shape=(size, size))

    return k0f


def fkMsf(double mu, double ys, double df, double Asf, double a, double b,
          double Iyy, double Jxx, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMsfr, kMsfc
    cdef np.ndarray[cDOUBLE, ndim=1] kMsfv

    fdim = 5*m1*n1*m1*n1

    kMsfr = np.zeros((fdim,), dtype=INT)
    kMsfc = np.zeros((fdim,), dtype=INT)
    kMsfv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kMsf_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kMsf_11 cond_1
                        c += 1
                        kMsfr[c] = row+0
                        kMsfc[c] = col+2
                        kMsfv[c] += -Asf*df*i1*k1*mu*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((i1*i1) - (k1*k1))
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+0
                        kMsfv[c] += Asf*df*i1*k1*mu*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((i1*i1) - (k1*k1))

                    elif k1 == i1 and l1 != j1:
                        # kMsf_11 cond_2
                        c += 1
                        kMsfr[c] = row+0
                        kMsfc[c] = col+0
                        kMsfv[c] += 0.5*Asf*a*mu*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)
                        c += 1
                        kMsfr[c] = row+1
                        kMsfc[c] = col+1
                        kMsfv[c] += 0.5*Asf*a*mu*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)
                        c += 1
                        kMsfr[c] = row+1
                        kMsfc[c] = col+2
                        kMsfv[c] += 0.5*pi*Asf*a*df*l1*mu*sin(pi*j1*ys/b)*cos(pi*l1*ys/b)/b
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+1
                        kMsfv[c] += 0.5*pi*Asf*a*df*j1*mu*sin(pi*l1*ys/b)*cos(pi*j1*ys/b)/b
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+2
                        kMsfv[c] += 0.5*mu*((pi*pi)*(a*a)*j1*l1*(Asf*(df*df) + Jxx)*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + (b*b)*(Asf*(a*a) + (pi*pi)*(i1*i1)*(Asf*(df*df) + Iyy))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))/(a*(b*b))

                    elif k1 != i1 and l1 == j1:
                        # kMsf_11 cond_3
                        c += 1
                        kMsfr[c] = row+0
                        kMsfc[c] = col+2
                        kMsfv[c] += -Asf*df*i1*k1*mu*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)**2/((i1*i1) - (k1*k1))
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+0
                        kMsfv[c] += Asf*df*i1*k1*mu*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)**2/((i1*i1) - (k1*k1))

                    elif k1 == i1 and l1 == j1:
                        # kMsf_11 cond_4
                        c += 1
                        kMsfr[c] = row+0
                        kMsfc[c] = col+0
                        kMsfv[c] += 0.5*Asf*a*mu*sin(pi*j1*ys/b)**2
                        c += 1
                        kMsfr[c] = row+1
                        kMsfc[c] = col+1
                        kMsfv[c] += 0.5*Asf*a*mu*sin(pi*j1*ys/b)**2
                        c += 1
                        kMsfr[c] = row+1
                        kMsfc[c] = col+2
                        kMsfv[c] += 0.25*pi*Asf*a*df*j1*mu*sin(2*pi*j1*ys/b)/b
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+1
                        kMsfv[c] += 0.25*pi*Asf*a*df*j1*mu*sin(2*pi*j1*ys/b)/b
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+2
                        kMsfv[c] += 0.5*mu*((pi*pi)*(a*a)*(j1*j1)*(Asf*(df*df) + Jxx)*cos(pi*j1*ys/b)**2 + (b*b)*(Asf*(a*a) + (pi*pi)*(i1*i1)*(Asf*(df*df) + Iyy))*sin(pi*j1*ys/b)**2)/(a*(b*b))

    size = num1*m1*n1

    kMsf = coo_matrix((kMsfv, (kMsfr, kMsfc)), shape=(size, size))

    return kMsf


def kCff(double kt, double a, double bf, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kCffr, kCffc
    cdef np.ndarray[cDOUBLE, ndim=1] kCffv

    fdim = 4*m1*n1*m1*n1

    kCffr = np.zeros((fdim,), dtype=INT)
    kCffc = np.zeros((fdim,), dtype=INT)
    kCffv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kCff_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kCff_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # kCff_11 cond_2
                        c += 1
                        kCffr[c] = row+0
                        kCffc[c] = col+0
                        kCffv[c] += 0.5*a*kt
                        c += 1
                        kCffr[c] = row+1
                        kCffc[c] = col+1
                        kCffv[c] += 0.5*a*kt
                        c += 1
                        kCffr[c] = row+2
                        kCffc[c] = col+2
                        kCffv[c] += 0.5*(pi*pi)*a*j1*kt*l1/(bf*bf)
                        c += 1
                        kCffr[c] = row+3
                        kCffc[c] = col+3
                        kCffv[c] += 0.5*a*kt

                    elif k1 != i1 and l1 == j1:
                        # kCff_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kCff_11 cond_4
                        c += 1
                        kCffr[c] = row+0
                        kCffc[c] = col+0
                        kCffv[c] += 0.5*a*kt
                        c += 1
                        kCffr[c] = row+1
                        kCffc[c] = col+1
                        kCffv[c] += 0.5*a*kt
                        c += 1
                        kCffr[c] = row+2
                        kCffc[c] = col+2
                        kCffv[c] += 0.5*(pi*pi)*a*(j1*j1)*kt/(bf*bf)
                        c += 1
                        kCffr[c] = row+3
                        kCffc[c] = col+3
                        kCffv[c] += 0.5*a*kt

    size = num1*m1*n1

    kCff = coo_matrix((kCffv, (kCffr, kCffc)), shape=(size, size))

    return kCff


def kCsf(double kt, double ys, double a, double b, double bf,
         int m, int n, int m1, int n1):
    cdef int i, j, k1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kCsfr, kCsfc
    cdef np.ndarray[cDOUBLE, ndim=1] kCsfv

    fdim = 3*m*n*m1*n1

    kCsfr = np.zeros((fdim,), dtype=INT)
    kCsfc = np.zeros((fdim,), dtype=INT)
    kCsfv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kCsf_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num1*((j-1)*m + (i-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i and l1 != j:
                        # kCsf_11 cond_1
                        c += 1
                        kCsfr[c] = row+0
                        kCsfc[c] = col+0
                        kCsfv[c] += a*kt*(10*(-1)**(k1 + 10) - 10)*sin(pi*j*ys/b)/(pi*(-(k1*k1) + 100))
                        c += 1
                        kCsfr[c] = row+1
                        kCsfc[c] = col+3
                        kCsfv[c] += a*kt*(-17*(-1)**(k1 + 17) + 17)*sin(pi*j*ys/b)/(pi*(-(k1*k1) + 289))
                        c += 1
                        kCsfr[c] = row+2
                        kCsfc[c] = col+1
                        kCsfv[c] += a*kt*(19*(-1)**(k1 + 19) - 19)*sin(pi*j*ys/b)/(pi*(-(k1*k1) + 361))

                    elif k1 == i and l1 != j:
                        # kCsf_11 cond_2
                        c += 1
                        kCsfr[c] = row+2
                        kCsfc[c] = col+2
                        kCsfv[c] += -0.5*(pi*pi)*a*j*kt*l1*cos(pi*j*ys/b)/(b*bf)

                    elif k1 != i and l1 == j:
                        # kCsf_11 cond_3
                        c += 1
                        kCsfr[c] = row+0
                        kCsfc[c] = col+0
                        kCsfv[c] += a*kt*(84*(-1)**(k1 + 84) - 84)*sin(pi*j*ys/b)/(pi*(-(k1*k1) + 7056))
                        c += 1
                        kCsfr[c] = row+1
                        kCsfc[c] = col+3
                        kCsfv[c] += a*kt*(-91*(-1)**(k1 + 91) + 91)*sin(pi*j*ys/b)/(pi*(-(k1*k1) + 8281))
                        c += 1
                        kCsfr[c] = row+2
                        kCsfc[c] = col+1
                        kCsfv[c] += a*kt*(93*(-1)**(k1 + 93) - 93)*sin(pi*j*ys/b)/(pi*(-(k1*k1) + 8649))

                    elif k1 == i and l1 == j:
                        # kCsf_11 cond_4
                        c += 1
                        kCsfr[c] = row+2
                        kCsfc[c] = col+2
                        kCsfv[c] += -0.5*(pi*pi)*a*(j*j)*kt*cos(pi*j*ys/b)/(b*bf)

    size = num1*m1*n1

    kCsf = coo_matrix((kCsfv, (kCsfr, kCsfc)), shape=(size, size))

    return kCsf


def kCss(double kt, double ys, double a, double b, int m, int m):
    cdef int i, k, j, l, c, row, col
    cdef np.ndarray[cINT, ndim=1] kCssr, kCssc
    cdef np.ndarray[cDOUBLE, ndim=1] kCssv

    fdim = 4*m*n*m*n

    kCssr = np.zeros((fdim,), dtype=INT)
    kCssc = np.zeros((fdim,), dtype=INT)
    kCssv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kCss_11
    for i in range(1, m+1):
        for j in range(1, n+1):
            row = num*((j-1)*m + (i-1))
            for k in range(1, m+1):
                for l in range(1, n+1):
                    col = num*((l-1)*m + (k-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k != i and l != j:
                        # kCss_11 cond_1
                        pass

                    elif k == i and l != j:
                        # kCss_11 cond_2
                        c += 1
                        kCssr[c] = row+0
                        kCssc[c] = col+0
                        kCssv[c] += 0.5*a*kt*sin(pi*j*ys/b)*sin(pi*l*ys/b)
                        c += 1
                        kCssr[c] = row+1
                        kCssc[c] = col+1
                        kCssv[c] += 0.5*a*kt*sin(pi*j*ys/b)*sin(pi*l*ys/b)
                        c += 1
                        kCssr[c] = row+2
                        kCssc[c] = col+2
                        kCssv[c] += 0.5*a*kt*((b*b)*sin(pi*j*ys/b)*sin(pi*l*ys/b) + (pi*pi)*j*l*cos(pi*j*ys/b)*cos(pi*l*ys/b))/(b*b)

                    elif k != i and l == j:
                        # kCss_11 cond_3
                        pass

                    elif k == i and l == j:
                        # kCss_11 cond_4
                        c += 1
                        kCssr[c] = row+0
                        kCssc[c] = col+0
                        kCssv[c] += 0.5*a*kt*sin(pi*j*ys/b)**2
                        c += 1
                        kCssr[c] = row+1
                        kCssc[c] = col+1
                        kCssv[c] += 0.5*a*kt*sin(pi*j*ys/b)**2
                        c += 1
                        kCssr[c] = row+2
                        kCssc[c] = col+2
                        kCssv[c] += 0.5*a*kt*((b*b)*sin(pi*j*ys/b)**2 + (pi*pi)*(j*j)*cos(pi*j*ys/b)**2)/(b*b)

    size = num*m*n

    kCss = coo_matrix((kCssv, (kCssr, kCssc)), shape=(size, size))

    return kCss


def kG0f(double Nxx, double Nxy, double a, double bf, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kG0fr, kG0fc
    cdef np.ndarray[cDOUBLE, ndim=1] kG0fv

    fdim = 4*m1*n1*m1*n1

    kG0fr = np.zeros((fdim,), dtype=INT)
    kG0fc = np.zeros((fdim,), dtype=INT)
    kG0fv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kG0f_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kG0f_11 cond_1
                        c += 1
                        kG0fr[c] = row+2
                        kG0fc[c] = col+2
                        kG0fv[c] += Nxy*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        kG0fr[c] = row+2
                        kG0fc[c] = col+3
                        kG0fv[c] += Nxx*bf*i1*j1*(k1*k1)*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)/(a*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        kG0fr[c] = row+3
                        kG0fc[c] = col+2
                        kG0fv[c] += -Nxx*bf*(i1*i1)*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)/(a*((i1*i1) - (k1*k1))*(-(j1*j1) + (l1*l1)))
                        c += 1
                        kG0fr[c] = row+3
                        kG0fc[c] = col+3
                        kG0fv[c] += -Nxy*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))

                    elif k1 == i1 and l1 != j1:
                        # kG0f_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1:
                        # kG0f_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kG0f_11 cond_4
                        c += 1
                        kG0fr[c] = row+2
                        kG0fc[c] = col+2
                        kG0fv[c] += 0.25*(pi*pi)*Nxx*bf*(i1*i1)/a
                        c += 1
                        kG0fr[c] = row+2
                        kG0fc[c] = col+3
                        kG0fv[c] += -0.5*(pi*pi)*Nxy*i1*j1
                        c += 1
                        kG0fr[c] = row+3
                        kG0fc[c] = col+2
                        kG0fv[c] += -0.5*(pi*pi)*Nxy*i1*j1
                        c += 1
                        kG0fr[c] = row+3
                        kG0fc[c] = col+3
                        kG0fv[c] += 0.25*(pi*pi)*Nxx*bf*(i1*i1)/a

    size = num1*m1*n1

    kG0f = coo_matrix((kG0fv, (kG0fr, kG0fc)), shape=(size, size))

    return kCff


def kMf(double mu, double hf, double a, double bf, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMfr, kMfc
    cdef np.ndarray[cDOUBLE, ndim=1] kMfv

    fdim = 4*m1*n1*m1*n1

    kMfr = np.zeros((fdim,), dtype=INT)
    kMfc = np.zeros((fdim,), dtype=INT)
    kMfv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kMf_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kMf_11 cond_1
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+3
                        kMfv[c] += 0.0833333333333333*hf*i1*j1*mu*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((a*a)*(12*(bf*bf) + (pi*pi)*(hf*hf)*(l1*l1)) + (pi*pi)*(bf*bf)*(hf*hf)*(k1*k1))/((pi*pi)*a*bf*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        kMfr[c] = row+3
                        kMfc[c] = col+2
                        kMfv[c] += 0.0833333333333333*hf*k1*l1*mu*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((a*a)*(12*(bf*bf) + (pi*pi)*(hf*hf)*(j1*j1)) + (pi*pi)*(bf*bf)*(hf*hf)*(i1*i1))/((pi*pi)*a*bf*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # kMf_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1:
                        # kMf_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kMf_11 cond_4
                        c += 1
                        kMfr[c] = row+0
                        kMfc[c] = col+0
                        kMfv[c] += 0.25*a*bf*hf*mu
                        c += 1
                        kMfr[c] = row+1
                        kMfc[c] = col+1
                        kMfv[c] += 0.25*a*bf*hf*mu
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+2
                        kMfv[c] += 0.25*a*bf*hf*mu + 0.0208333333333333*(pi*pi)*a*(hf*hf*hf)*(j1*j1)*mu/bf + 0.0208333333333333*(pi*pi)*bf*(hf*hf*hf)*(i1*i1)*mu/a
                        c += 1
                        kMfr[c] = row+3
                        kMfc[c] = col+3
                        kMfv[c] += 0.25*a*bf*hf*mu + 0.0208333333333333*(pi*pi)*a*(hf*hf*hf)*(j1*j1)*mu/bf + 0.0208333333333333*(pi*pi)*bf*(hf*hf*hf)*(i1*i1)*mu/a

    size = num1*m1*n1

    kMf = coo_matrix((kMfv, (kMfr, kMfc)), shape=(size, size))

    return kCff
