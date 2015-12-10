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

cdef int num0 = 0
cdef int num1 = 3
cdef double pi = 3.141592653589793


def fk0y1y2(double y1, double y2, double a, double b, double r,
            int m1, int n1, np.ndarray[cDOUBLE, ndim=2] F):
    cdef int i1, j1, k1, l1, row, col, c
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef np.ndarray[cINT, ndim=1] k0y1y2r, k0y1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] k0y1y2v

    fdim = 9*m1*n1*m1*n1

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
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0y1y2_11 cond_1
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += A16*i1*k1*((-1)**(i1 + k1) - 1)*(2*j1*l1*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - 2*j1*l1*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + ((j1*j1) + (l1*l1))*(sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b)))/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(j1*l1*(A12 + A66)*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - j1*l1*(A12 + A66)*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + (A12*(l1*l1) + A66*(j1*j1))*(sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b)))/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(2*(pi*pi)*B66*(a*a)*j1*l1*r*(-j1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + j1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + l1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - l1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b)) + ((pi*pi)*B11*(b*b)*(k1*k1)*r + (a*a)*(A12*(b*b) + (pi*pi)*B12*(l1*l1)*r))*(j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b)))/(pi*(a*a)*b*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(j1*l1*(A12 + A66)*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - j1*l1*(A12 + A66)*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + (A12*(j1*j1) + A66*(l1*l1))*(sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b)))/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += A26*i1*k1*((-1)**(i1 + k1) - 1)*(2*j1*l1*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - 2*j1*l1*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + ((j1*j1) + (l1*l1))*(sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b)))/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(2*(pi*pi)*B26*(a*a)*j1*l1*r*(-j1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + j1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + l1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - l1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b)) + ((pi*pi)*B16*(b*b)*(k1*k1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(l1*l1)*r))*(j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b)))/(pi*(a*a)*b*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += -i1*k1*((-1)**(i1 + k1) - 1)*(2*(pi*pi)*B66*(a*a)*j1*l1*r*(-j1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + j1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + l1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - l1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b)) + ((pi*pi)*B11*(b*b)*(i1*i1)*r + (a*a)*(A12*(b*b) + (pi*pi)*B12*(j1*j1)*r))*(j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b)))/(pi*(a*a)*b*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += -i1*k1*((-1)**(i1 + k1) - 1)*(2*(pi*pi)*B26*(a*a)*j1*l1*r*(-j1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + j1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + l1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - l1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b)) + ((pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r))*(j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b)))/(pi*(a*a)*b*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += i1*k1*(2*(-1)**(i1 + k1) - 2)*(j1*((pi*pi)*D16*(b*b)*(k1*k1)*r + (a*a)*(B26*(b*b) + (pi*pi)*D26*(l1*l1)*r))*(j1*sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - j1*sin(pi*j1*y2/b)*sin(pi*l1*y2/b) + l1*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - l1*cos(pi*j1*y2/b)*cos(pi*l1*y2/b)) + l1*((pi*pi)*D16*(b*b)*(i1*i1)*r + (a*a)*(B26*(b*b) + (pi*pi)*D26*(j1*j1)*r))*(j1*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - j1*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + l1*sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - l1*sin(pi*j1*y2/b)*sin(pi*l1*y2/b)))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # k0y1y2_11 cond_2
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += -0.5*pi*(A11*(b*b)*(i1*i1)*j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - A11*(b*b)*(i1*i1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) - A66*(a*a)*(j1*j1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + A66*(a*a)*j1*(l1*l1)*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - j1*(A11*(b*b)*(i1*i1) + A66*(a*a)*(l1*l1))*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) + l1*(A11*(b*b)*(i1*i1) + A66*(a*a)*(j1*j1))*sin(pi*j1*y1/b)*cos(pi*l1*y1/b))/(a*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += -0.5*pi*(A16*(b*b)*(i1*i1)*j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - A16*(b*b)*(i1*i1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) - A26*(a*a)*(j1*j1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + A26*(a*a)*j1*(l1*l1)*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - j1*(A16*(b*b)*(i1*i1) + A26*(a*a)*(l1*l1))*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) + l1*(A16*(b*b)*(i1*i1) + A26*(a*a)*(j1*j1))*sin(pi*j1*y1/b)*cos(pi*l1*y1/b))/(a*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.5*(-j1*l1*(3*(pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(l1*l1)*r))*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) + j1*l1*(3*(pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(l1*l1)*r))*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) - (sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b))*((pi*pi)*B16*(b*b)*(i1*i1)*r*((j1*j1) + 2*(l1*l1)) + (a*a)*(j1*j1)*(A26*(b*b) + (pi*pi)*B26*(l1*l1)*r)))/(a*(b*b)*r*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += -0.5*pi*(A16*(b*b)*(i1*i1)*j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - A16*(b*b)*(i1*i1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) - A26*(a*a)*(j1*j1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + A26*(a*a)*j1*(l1*l1)*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - j1*(A16*(b*b)*(i1*i1) + A26*(a*a)*(l1*l1))*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) + l1*(A16*(b*b)*(i1*i1) + A26*(a*a)*(j1*j1))*sin(pi*j1*y1/b)*cos(pi*l1*y1/b))/(a*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += -0.5*pi*(-A22*(a*a)*(j1*j1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + A22*(a*a)*j1*(l1*l1)*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) + A66*(b*b)*(i1*i1)*j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - A66*(b*b)*(i1*i1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) - j1*(A22*(a*a)*(l1*l1) + A66*(b*b)*(i1*i1))*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) + l1*(A22*(a*a)*(j1*j1) + A66*(b*b)*(i1*i1))*sin(pi*j1*y1/b)*cos(pi*l1*y1/b))/(a*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.5*(-j1*l1*((a*a)*(A22*(b*b) + (pi*pi)*B22*(l1*l1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12 + 2*B66))*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) + j1*l1*((a*a)*(A22*(b*b) + (pi*pi)*B22*(l1*l1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12 + 2*B66))*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) - (sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b))*((a*a)*(j1*j1)*(A22*(b*b) + (pi*pi)*B22*(l1*l1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12*(j1*j1) + 2*B66*(l1*l1))))/(a*(b*b)*r*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 0.5*(j1*l1*(3*(pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r))*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - j1*l1*(3*(pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r))*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + (sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b))*((pi*pi)*B16*(b*b)*(i1*i1)*r*(2*(j1*j1) + (l1*l1)) + (a*a)*(l1*l1)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r)))/(a*(b*b)*r*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += 0.5*(j1*l1*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12 + 2*B66))*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - j1*l1*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12 + 2*B66))*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + (sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b))*((a*a)*(l1*l1)*(A22*(b*b) + (pi*pi)*B22*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12*(l1*l1) + 2*B66*(j1*j1))))/(a*(b*b)*r*((j1*j1) - (l1*l1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.5*(4*(pi*pi*pi*pi)*D66*(a*a)*(b*b)*(i1*i1)*j1*l1*(r*r)*(-j1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + j1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + l1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - l1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b)) + ((pi*pi*pi*pi)*D11*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*r*(B22*(b*b)*((j1*j1) + (l1*l1)) + (pi*pi)*D22*(j1*j1)*(l1*l1)*r)) + (pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(2*B12*(b*b) + (pi*pi)*D12*r*((j1*j1) + (l1*l1))))*(j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b)))/(pi*(a*a*a)*(b*b*b)*(r*r)*((j1*j1) - (l1*l1)))

                    elif k1 != i1 and l1 == j1:
                        # k0y1y2_11 cond_3
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(A12 - A66)*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))/(4.0*(i1*i1) - 4.0*(k1*k1))
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(0.5*pi*B66*(a*a)*j1*r*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2)) + 0.25*((pi*pi)*B11*(b*b)*(k1*k1)*r + (a*a)*(A12*(b*b) + (pi*pi)*B12*(j1*j1)*r))*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(pi*j1))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(-A12 + A66)*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))/(4.0*(i1*i1) - 4.0*(k1*k1))
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(0.5*pi*B26*(a*a)*j1*r*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2)) + 0.25*((pi*pi)*B16*(b*b)*(k1*k1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r))*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(pi*j1))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += -i1*k1*((-1)**(i1 + k1) - 1)*(0.5*pi*B66*(a*a)*j1*r*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2)) + 0.25*((pi*pi)*B11*(b*b)*(i1*i1)*r + (a*a)*(A12*(b*b) + (pi*pi)*B12*(j1*j1)*r))*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(pi*j1))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += -i1*k1*((-1)**(i1 + k1) - 1)*(0.5*pi*B26*(a*a)*j1*r*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2)) + 0.25*((pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r))*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(pi*j1))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.5*(pi*pi)*D16*i1*k1*((-1)**(i1 + k1) - 1)*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))/(a*a)

                    elif k1 == i1 and l1 == j1:
                        # k0y1y2_11 cond_4
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 0.125*pi*A11*(i1*i1)*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(a*j1) + 0.125*pi*A66*a*j1*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(b*b)
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += 0.125*pi*A16*(i1*i1)*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(a*j1) + 0.125*pi*A26*a*j1*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(b*b)
                        c += 1
                        k0y1y2r[c] = row+0
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.125*(-(pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r))*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))/(a*(b*b)*r)
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 0.125*pi*A16*(i1*i1)*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(a*j1) + 0.125*pi*A26*a*j1*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(b*b)
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += 0.125*pi*A22*a*j1*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(b*b) + 0.125*pi*A66*(i1*i1)*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(a*j1)
                        c += 1
                        k0y1y2r[c] = row+1
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.125*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12 - 2*B66))*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))/(a*(b*b)*r)
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+0
                        k0y1y2v[c] += 0.125*(-(pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r))*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))/(a*(b*b)*r)
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+1
                        k0y1y2v[c] += 0.125*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12 - 2*B66))*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))/(a*(b*b)*r)
                        c += 1
                        k0y1y2r[c] = row+2
                        k0y1y2c[c] = col+2
                        k0y1y2v[c] += 0.125*(b*((pi*pi*pi*pi)*D11*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*(j1*j1)*r*(2*B22*(b*b) + (pi*pi)*D22*(j1*j1)*r)) + 2*(pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(B12*(b*b) + (pi*pi)*(j1*j1)*r*(D12 - 2*D66)))*sin(2*pi*j1*y1/b) - b*((pi*pi*pi*pi)*D11*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*(j1*j1)*r*(2*B22*(b*b) + (pi*pi)*D22*(j1*j1)*r)) + 2*(pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(B12*(b*b) + (pi*pi)*(j1*j1)*r*(D12 - 2*D66)))*sin(2*pi*j1*y2/b) - 2*pi*j1*(y1 - y2)*((pi*pi*pi*pi)*D11*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*(j1*j1)*r*(2*B22*(b*b) + (pi*pi)*D22*(j1*j1)*r)) + 2*(pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(B12*(b*b) + (pi*pi)*(j1*j1)*r*(D12 + 2*D66))))/(pi*(a*a*a)*(b*b*b*b)*j1*(r*r))

    size = num0 + num1*m1*n1

    k0 = coo_matrix((k0y1y2v, (k0y1y2r, k0y1y2c)), shape=(size, size))

    return k0


def fkG0y1y2(double y1, double y2, double Nxx, double Nyy, double Nxy,
             double a, double b, double r, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kG0y1y2r, kG0y1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0y1y2v

    fdim = 1*m1*n1*m1*n1

    kG0y1y2r = np.zeros((fdim,), dtype=INT)
    kG0y1y2c = np.zeros((fdim,), dtype=INT)
    kG0y1y2v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kG0_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kG0y1y2_11 cond_1
                        c += 1
                        kG0y1y2r[c] = row+2
                        kG0y1y2c[c] = col+2
                        kG0y1y2v[c] += Nxy*i1*k1*((-1)**(i1 + k1) - 1)*(2*j1*l1*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - 2*j1*l1*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + ((j1*j1) + (l1*l1))*(sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - sin(pi*j1*y2/b)*sin(pi*l1*y2/b)))/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # kG0y1y2_11 cond_2
                        c += 1
                        kG0y1y2r[c] = row+2
                        kG0y1y2c[c] = col+2
                        kG0y1y2v[c] += -0.5*pi*(Nxx*(b*b)*(i1*i1)*j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - Nxx*(b*b)*(i1*i1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) - Nyy*(a*a)*(j1*j1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + Nyy*(a*a)*j1*(l1*l1)*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - j1*(Nxx*(b*b)*(i1*i1) + Nyy*(a*a)*(l1*l1))*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) + l1*(Nxx*(b*b)*(i1*i1) + Nyy*(a*a)*(j1*j1))*sin(pi*j1*y1/b)*cos(pi*l1*y1/b))/(a*b*((j1*j1) - (l1*l1)))

                    elif k1 != i1 and l1 == j1:
                        # kG0y1y2_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kG0y1y2_11 cond_4
                        c += 1
                        kG0y1y2r[c] = row+2
                        kG0y1y2c[c] = col+2
                        kG0y1y2v[c] += 0.125*pi*Nxx*(i1*i1)*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(a*j1) + 0.125*pi*Nyy*a*j1*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(b*b)

    size = num0 + num1*m1*n1

    kG0 = coo_matrix((kG0y1y2v, (kG0y1y2r, kG0y1y2c)), shape=(size, size))

    return kG0


def fkMy1y2(double y1, double y2, double mu, double d, double h,
            double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMy1y2r, kMy1y2c
    cdef np.ndarray[cDOUBLE, ndim=1] kMy1y2v

    fdim = 5*m1*n1*m1*n1

    kMy1y2r = np.zeros((fdim,), dtype=INT)
    kMy1y2c = np.zeros((fdim,), dtype=INT)
    kMy1y2v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kMy1y2_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kMy1y2_11 cond_1
                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += b*d*h*i1*k1*mu*((-1)**(i1 + k1) - 1)*(-j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) + j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) + l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) - l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b))/(pi*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += b*d*h*i1*k1*mu*((-1)**(i1 + k1) - 1)*(j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b))/(pi*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # kMy1y2_11 cond_2
                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += a*b*h*mu*(j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b))/(pi*(2.0*(j1*j1) - 2.0*(l1*l1)))
                        c += 1
                        kMy1y2r[c] = row+1
                        kMy1y2c[c] = col+1
                        kMy1y2v[c] += a*b*h*mu*(j1*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - l1*sin(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b))/(pi*(2.0*(j1*j1) - 2.0*(l1*l1)))
                        c += 1
                        kMy1y2r[c] = row+1
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += a*d*h*l1*mu*(j1*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) - j1*cos(pi*j1*y2/b)*cos(pi*l1*y2/b) + l1*sin(pi*j1*y1/b)*sin(pi*l1*y1/b) - l1*sin(pi*j1*y2/b)*sin(pi*l1*y2/b))/(2.0*(j1*j1) - 2.0*(l1*l1))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+1
                        kMy1y2v[c] += a*d*h*j1*mu*(-j1*sin(pi*j1*y1/b)*sin(pi*l1*y1/b) + j1*sin(pi*j1*y2/b)*sin(pi*l1*y2/b) - l1*cos(pi*j1*y1/b)*cos(pi*l1*y1/b) + l1*cos(pi*j1*y2/b)*cos(pi*l1*y2/b))/(2.0*(j1*j1) - 2.0*(l1*l1))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += 0.0416666666666667*h*mu*(-12*(a*a)*(b*b)*j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) + 12*(a*a)*(b*b)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + 12*(pi*pi)*(a*a)*(d*d)*(j1*j1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) - 12*(pi*pi)*(a*a)*(d*d)*j1*(l1*l1)*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) + (pi*pi)*(a*a)*(h*h)*(j1*j1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) - (pi*pi)*(a*a)*(h*h)*j1*(l1*l1)*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) - 12*(pi*pi)*(b*b)*(d*d)*(i1*i1)*j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) + 12*(pi*pi)*(b*b)*(d*d)*(i1*i1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) - (pi*pi)*(b*b)*(h*h)*(i1*i1)*j1*sin(pi*l1*y2/b)*cos(pi*j1*y2/b) + (pi*pi)*(b*b)*(h*h)*(i1*i1)*l1*sin(pi*j1*y2/b)*cos(pi*l1*y2/b) + j1*((a*a)*(12*(b*b) + (pi*pi)*(l1*l1)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i1*i1)*(12*(d*d) + (h*h)))*sin(pi*l1*y1/b)*cos(pi*j1*y1/b) - l1*((a*a)*(12*(b*b) + (pi*pi)*(j1*j1)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i1*i1)*(12*(d*d) + (h*h)))*sin(pi*j1*y1/b)*cos(pi*l1*y1/b))/(pi*a*b*((j1*j1) - (l1*l1)))

                    elif k1 != i1 and l1 == j1:
                        # kMy1y2_11 cond_3
                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += 0.25*d*h*i1*k1*mu*((-1)**(i1 + k1) - 1)*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(y1 - y2))/(pi*j1*((i1*i1) - (k1*k1)))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += -0.25*d*h*i1*k1*mu*((-1)**(i1 + k1) - 1)*(-b*sin(2*pi*j1*y1/b) + b*sin(2*pi*j1*y2/b) + 2*pi*j1*(y1 - y2))/(pi*j1*((i1*i1) - (k1*k1)))

                    elif k1 == i1 and l1 == j1:
                        # kMy1y2_11 cond_4
                        c += 1
                        kMy1y2r[c] = row+0
                        kMy1y2c[c] = col+0
                        kMy1y2v[c] += 0.125*a*h*mu*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(pi*j1)
                        c += 1
                        kMy1y2r[c] = row+1
                        kMy1y2c[c] = col+1
                        kMy1y2v[c] += 0.125*a*h*mu*(b*sin(2*pi*j1*y1/b) - b*sin(2*pi*j1*y2/b) + 2*pi*j1*(-y1 + y2))/(pi*j1)
                        c += 1
                        kMy1y2r[c] = row+1
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += 0.125*a*d*h*mu*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+1
                        kMy1y2v[c] += 0.125*a*d*h*mu*(cos(2*pi*j1*y1/b) - cos(2*pi*j1*y2/b))
                        c += 1
                        kMy1y2r[c] = row+2
                        kMy1y2c[c] = col+2
                        kMy1y2v[c] += 0.0104166666666667*h*mu*(b*((a*a)*(12*(b*b) - (pi*pi)*(j1*j1)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i1*i1)*(12*(d*d) + (h*h)))*sin(2*pi*j1*y1/b) - b*((a*a)*(12*(b*b) - (pi*pi)*(j1*j1)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i1*i1)*(12*(d*d) + (h*h)))*sin(2*pi*j1*y2/b) - 2*pi*j1*(y1 - y2)*((a*a)*(12*(b*b) + (pi*pi)*(j1*j1)*(12*(d*d) + (h*h))) + (pi*pi)*(b*b)*(i1*i1)*(12*(d*d) + (h*h))))/(pi*a*(b*b)*j1)

    size = num0 + num1*m1*n1

    kM = coo_matrix((kMy1y2v, (kMy1y2r, kMy1y2c)), shape=(size, size))

    return kM


def fk0edges(int m1, int n1, double a, double b,
             double kphixBot, double kphixTop,
             double kphiyLeft, double kphiyRight):
    cdef int i1, j1, k1, l1, row, col, c
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv

    fdim = 1*m1*n1*m1*n1 + 1*m1*n1*m1*n1

    k0edgesr = np.zeros((fdim,), dtype=INT)
    k0edgesc = np.zeros((fdim,), dtype=INT)
    k0edgesv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # k0edgesBT_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0edgesBT_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # k0edgesBT_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1:
                        # k0edgesBT_11 cond_3
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*b*i1*k1*((-1)**(i1 + k1)*kphixTop + kphixBot)/(a*a)

                    elif k1 == i1 and l1 == j1:
                        # k0edgesBT_11 cond_4
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*b*(i1*i1)*(kphixBot + kphixTop)/(a*a)

    # k0edgesLR_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0edgesLR_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # k0edgesLR_11 cond_2
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*a*j1*l1*((-1)**(j1 + l1)*kphiyLeft + kphiyRight)/(b*b)

                    elif k1 != i1 and l1 == j1:
                        # k0edgesLR_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # k0edgesLR_11 cond_4
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*a*(j1*j1)*(kphiyLeft + kphiyRight)/(b*b)

    size = num0 + num1*m1*n1

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fk0sf(double bf, double df, double ys, double a, double b, double r,
          int m1, int n1, double E1, double F1, double S1, double Jxx):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] k0sfr, k0sfc
    cdef np.ndarray[cDOUBLE, ndim=1] k0sfv

    fdim = 4*m1*n1*m1*n1

    k0sfr = np.zeros((fdim,), dtype=INT)
    k0sfc = np.zeros((fdim,), dtype=INT)
    k0sfv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # k0sf_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0sf_11 cond_1
                        c += 1
                        k0sfr[c] = row+0
                        k0sfc[c] = col+2
                        k0sfv[c] += -(pi*pi)*E1*bf*df*i1*(k1*k1*k1)*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((a*a)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+0
                        k0sfv[c] += (pi*pi)*E1*bf*df*(i1*i1*i1)*k1*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((a*a)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+2
                        k0sfv[c] += -(pi*pi*pi)*S1*bf*df*i1*k1*((-1)**(i1 + k1) - 1)*((i1*i1)*l1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) - j1*(k1*k1)*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))/((a*a)*b*((i1*i1) - (k1*k1)))

                    elif k1 == i1 and l1 != j1:
                        # k0sf_11 cond_2
                        c += 1
                        k0sfr[c] = row+0
                        k0sfc[c] = col+0
                        k0sfv[c] += 0.5*(pi*pi)*E1*bf*(i1*i1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/a
                        c += 1
                        k0sfr[c] = row+0
                        k0sfc[c] = col+2
                        k0sfv[c] += -0.5*(pi*pi*pi)*S1*bf*(i1*i1)*l1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b)/(a*b)
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+0
                        k0sfv[c] += -0.5*(pi*pi*pi)*S1*bf*(i1*i1)*j1*sin(pi*l1*ys/b)*cos(pi*j1*ys/b)/(a*b)
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+2
                        k0sfv[c] += 0.5*(pi*pi*pi*pi)*bf*(i1*i1)*(Jxx*(a*a)*j1*l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + (b*b)*(i1*i1)*(E1*(df*df) + F1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))/((a*a*a)*(b*b))

                    elif k1 != i1 and l1 == j1:
                        # k0sf_11 cond_3
                        c += 1
                        k0sfr[c] = row+0
                        k0sfc[c] = col+2
                        k0sfv[c] += -(pi*pi)*E1*bf*df*i1*(k1*k1*k1)*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)**2/((a*a)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+0
                        k0sfv[c] += (pi*pi)*E1*bf*df*(i1*i1*i1)*k1*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)**2/((a*a)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+2
                        k0sfv[c] += -0.5*(pi*pi*pi)*S1*bf*df*i1*j1*k1*((-1)**(i1 + k1) - 1)*sin(2*pi*j1*ys/b)/((a*a)*b)

                    elif k1 == i1 and l1 == j1:
                        # k0sf_11 cond_4
                        c += 1
                        k0sfr[c] = row+0
                        k0sfc[c] = col+0
                        k0sfv[c] += 0.5*(pi*pi)*E1*bf*(i1*i1)*sin(pi*j1*ys/b)**2/a
                        c += 1
                        k0sfr[c] = row+0
                        k0sfc[c] = col+2
                        k0sfv[c] += -0.25*(pi*pi*pi)*S1*bf*(i1*i1)*j1*sin(2*pi*j1*ys/b)/(a*b)
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+0
                        k0sfv[c] += -0.25*(pi*pi*pi)*S1*bf*(i1*i1)*j1*sin(2*pi*j1*ys/b)/(a*b)
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+2
                        k0sfv[c] += 0.5*(pi*pi*pi*pi)*bf*(i1*i1)*(Jxx*(a*a)*(j1*j1)*cos(pi*j1*ys/b)**2 + (b*b)*(i1*i1)*(E1*(df*df) + F1)*sin(pi*j1*ys/b)**2)/((a*a*a)*(b*b))

    size = num0 + num1*m1*n1

    k0sf = coo_matrix((k0sfv, (k0sfr, k0sfc)), shape=(size, size))

    return k0sf


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
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

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

    size = num0 + num1*m1*n1

    kMsf = coo_matrix((kMsfv, (kMsfr, kMsfc)), shape=(size, size))

    return kMsf


def fkAx(double beta, double gamma, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kAxr, kAxc
    cdef np.ndarray[cDOUBLE, ndim=1] kAxv

    fdim = 1*m1*n1*m1*n1

    kAxr = np.zeros((fdim,), dtype=INT)
    kAxc = np.zeros((fdim,), dtype=INT)
    kAxv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kAx_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kAx_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # kAx_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1:
                        # kAx_11 cond_3
                        c += 1
                        kAxr[c] = row+2
                        kAxc[c] = col+2
                        kAxv[c] += -b*beta*i1*k1*((-1)**(i1 + k1) - 1)/(2.0*(i1*i1) - 2.0*(k1*k1))

                    elif k1 == i1 and l1 == j1:
                        # kAx_11 cond_4
                        c += 1
                        kAxr[c] = row+2
                        kAxc[c] = col+2
                        kAxv[c] += 0.25*a*b*gamma

    size = num0 + num1*m1*n1

    kAx = coo_matrix((kAxv, (kAxr, kAxc)), shape=(size, size))

    return kAx


def fkAy(double beta, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kAyr, kAyc
    cdef np.ndarray[cDOUBLE, ndim=1] kAyv

    fdim = 1*m1*n1*m1*n1

    kAyr = np.zeros((fdim,), dtype=INT)
    kAyc = np.zeros((fdim,), dtype=INT)
    kAyv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kAy_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kAy_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # kAy_11 cond_2
                        c += 1
                        kAyr[c] = row+2
                        kAyc[c] = col+2
                        kAyv[c] += -a*beta*j1*l1*((-1)**(j1 + l1) - 1)/(2.0*(j1*j1) - 2.0*(l1*l1))

                    elif k1 != i1 and l1 == j1:
                        # kAy_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kAy_11 cond_4
                        pass

    size = num0 + num1*m1*n1

    kAy = coo_matrix((kAyv, (kAyr, kAyc)), shape=(size, size))

    return kAy


def fcA(double aeromu, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] cAr, cAc
    cdef np.ndarray[cDOUBLE, ndim=1] cAv

    fdim = 1*m1*n1*m1*n1

    cAr = np.zeros((fdim,), dtype=INT)
    cAc = np.zeros((fdim,), dtype=INT)
    cAv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # cA_11
    for i1 in range(1, m1+1):
        for j1 in range(1, n1+1):
            row = num0 + num1*((j1-1)*m1 + (i1-1))
            for k1 in range(1, m1+1):
                for l1 in range(1, n1+1):
                    col = num0 + num1*((l1-1)*m1 + (k1-1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # cA_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # cA_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1:
                        pass

                    elif k1 == i1 and l1 == j1:
                        # cA_11 cond_4
                        c += 1
                        cAr[c] = row+2
                        cAc[c] = col+2
                        cAv[c] += -0.25*a*aeromu*b

    size = num0 + num1*m1*n1

    cA = coo_matrix((cAv, (cAr, cAc)), shape=(size, size))

    return cA
