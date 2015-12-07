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

cdef int num0 = 0
cdef int num1 = 3
cdef double pi = 3.141592653589793

def fk0(double r1, double L, double tmin, double tmax,
        np.ndarray[cDOUBLE, ndim=2] F,
        int m1, int n1, double alpharad, int s):
    cdef int i1, j1, k1, l1, c, row, col, section
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r, sina, cosa, xa, xb

    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    sina = sin(alpharad)
    cosa = cos(alpharad)

    fdim = 9*m1*n1*m1*n1//2

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

    for section in range(s):
        c = -1

        xa = -L/2. + L*float(section)/s
        xb = -L/2. + L*float(section+1)/s

        r = r1 - sina*((xa+xb)/2. + L/2.)

        # k0_11
        for i1 in range(m1):
            for j1 in range(n1):
                row = num0 + num1*((j1)*m1 + (i1))
                for k1 in range(m1):
                    for l1 in range(n1):
                        col = num0 + num1*((l1)*m1 + (k1))

                        #NOTE symmetry
                        if row > col:
                            continue

                        if k1 != i1 and l1 != j1:
                            # k0_11 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += ((-1)**(j1 + l1) - 1)*(A16*i1*(l1*l1)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) - A16*(j1*j1)*k1*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) - A26*L*(j1*j1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) + A26*L*(l1*l1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r))/((i1 + k1)*(2.0*i1 - 2.0*k1)*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += ((-1)**(j1 + l1) - 1)*(A66*L*(j1*j1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/pi + A66*(j1*j1)*k1*r*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) + (l1*l1)*(-pi*A12*(i1*i1)*r*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - pi*A12*(i1*i1)*r*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - pi*A12*i1*k1*r*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + pi*A12*i1*k1*r*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + pi*A12*i1*r*(i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + pi*A12*i1*r*(i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + A22*L*i1*sina*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - A22*L*i1*sina*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + A22*L*i1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - A22*L*i1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + A22*L*k1*sina*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - A22*L*k1*sina*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - A22*L*k1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + A22*L*k1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L))/pi)/(r*(i1 + k1)*(2.0*i1 - 2.0*k1)*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += l1*((-1)**(j1 + l1) - 1)*(-B66*(j1*j1)*(L*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) - 2*pi*k1*r*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))) + cosa*r*(tmax - tmin)**2*(pi*A12*(i1*i1)*r*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - pi*A12*(i1*i1)*r*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - pi*A12*(i1*i1)*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + pi*A12*(i1*i1)*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + pi*A12*i1*k1*r*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - pi*A12*i1*k1*r*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + pi*A12*i1*k1*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - pi*A12*i1*k1*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + A22*L*i1*sina*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - A22*L*i1*sina*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + A22*L*k1*sina*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + A22*L*k1*sina*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + A22*L*sina*(i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - A22*L*sina*(i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(pi*pi) - (r*r)*(-pi*B12*i1*(l1*l1)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r - B12*i1*k1*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/L + B22*L*(l1*l1)*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - B22*k1*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r)) + (k1*k1)*(r*r)*(tmax - tmin)**2*(pi*B11*(i1*i1)*r*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - pi*B11*(i1*i1)*r*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - pi*B11*(i1*i1)*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + pi*B11*(i1*i1)*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + pi*B11*i1*k1*r*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - pi*B11*i1*k1*r*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + pi*B11*i1*k1*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - pi*B11*i1*k1*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + B12*L*i1*sina*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - B12*L*i1*sina*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + B12*L*k1*sina*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + B12*L*k1*sina*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + B12*L*sina*(i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - B12*L*sina*(i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(L*L))/((r*r)*(i1 + k1)*(2.0*i1 - 2.0*k1)*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += ((-1)**(j1 + l1) - 1)*(-A12*(j1*j1)*k1*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) - A22*L*(j1*j1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) - A66*L*(l1*l1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) + A66*i1*(l1*l1)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)))/((i1 + k1)*(2.0*i1 - 2.0*k1)*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += A26*((-1)**(j1 + l1) - 1)*(L*(j1*j1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) - L*(l1*l1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) + i1*(l1*l1)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) - (j1*j1)*k1*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)))/((i1 + k1)*(2.0*i1 - 2.0*k1)*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += l1*r*((-1)**(j1 + l1) - 1)*(A26*cosa*(tmax - tmin)**2*(L*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/r - pi*i1*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L))))/((pi*pi)*r) + B16*(k1*k1)*(tmax - tmin)**2*(L*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/r - pi*i1*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L))))/(L*L) - B26*(j1*j1)*(L*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) - 2*pi*k1*r*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L))))/(r*r*r) - B26*(-L*(l1*l1)*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - pi*i1*(l1*l1)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r + k1*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) - i1*k1*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/L)/r)/((i1 + k1)*(2.0*i1 - 2.0*k1)*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += j1*((-1)**(j1 + l1) - 1)*(-A12*cosa*k1*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/pi - A22*L*cosa*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/((pi*pi)*r) + pi*B11*(i1*i1)*k1*r*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(L*L) - pi*B12*(j1*j1)*k1*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r - B12*(i1*i1)*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/L + B12*i1*k1*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/L - B22*L*(j1*j1)*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - B22*i1*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) - B66*L*(l1*l1)*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - 2*pi*B66*i1*(l1*l1)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r)/((i1 + k1)*(2.0*i1 - 2.0*k1)*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += j1*r*((-1)**(j1 + l1) - 1)*(A26*L*cosa*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/((pi*pi)*(r*r)*((i1*i1) - (k1*k1))*(j1 - l1)) + A26*cosa*k1*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r*(i1 - k1)*(i1 + k1)*(j1 - l1)) + B16*(i1*i1)*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(L*r*(i1 - k1)*(i1 + k1)*(j1 - l1)) + pi*B16*(i1*i1)*k1*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/((L*L)*(i1 - k1)*(i1 + k1)*(j1 - l1)) + B26*L*(j1*j1)*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/((r*r*r)*(i1 - k1)*(i1 + k1)*(j1 - l1)) + B26*i1*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*(r*r)*(i1 - k1)*(i1 + k1)*(j1 - l1)) + pi*B26*(j1*j1)*k1*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/((r*r)*(i1 - k1)*(i1 + k1)*(j1 - l1)) + B26*(l1*l1)*(-L*i1*sina*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - L*i1*sina*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - L*k1*sina*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + L*k1*sina*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + L*sina*(i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + L*sina*(i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + 2*pi*(i1*i1)*r*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - 2*pi*(i1*i1)*r*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + 2*pi*(i1*i1)*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - 2*pi*(i1*i1)*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + 2*pi*i1*k1*r*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - 2*pi*i1*k1*r*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - 2*pi*i1*k1*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + 2*pi*i1*k1*r*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L))/((r*r*r)*(i1 - k1)*(i1 + k1)*(-j1 + l1)) + B26*i1*k1*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(L*r*(i1 - k1)*(i1 + k1)*(j1 - l1)))/((2.0*j1 + 2.0*l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += j1*l1*((-1)**(j1 + l1) - 1)*(B26*cosa*r*(-L*i1*sina*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + L*i1*sina*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + L*i1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - L*i1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) - L*k1*sina*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + L*k1*sina*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - L*k1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + L*k1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) - 2*pi*(i1*i1)*r*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + 2*pi*(i1*i1)*r*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - 2*pi*i1*k1*r*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - 2*pi*i1*k1*r*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - 2*pi*i1*r*(i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + 2*pi*i1*r*(i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(pi*(i1 - k1)) - pi*D16*(k1*k1)*(r*r)*(-L*i1*sina*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + L*i1*sina*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + L*i1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - L*i1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) - L*k1*sina*sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + L*k1*sina*sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - L*k1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + L*k1*sina*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) - 2*pi*(i1*i1)*r*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + 2*pi*(i1*i1)*r*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - 2*pi*i1*k1*r*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - 2*pi*i1*k1*r*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - 2*pi*i1*r*(i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + 2*pi*i1*r*(i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/((L*L)*(-i1 + k1)) - D26*(pi*L*(l1*l1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(tmax - tmin)**2 - 2*(pi*pi)*i1*(l1*l1)*r*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(tmax - tmin)**2 + k1*r*(sina*sina)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) + 2*pi*i1*k1*(r*r)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L)/(i1 - k1) - (r*r*r)*(-B26*L*cosa*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*(r*r)) - 2*B26*cosa*k1*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/r - pi*D16*(i1*i1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(L*r) - 2*(pi*pi)*D16*(i1*i1)*k1*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(L*L) - pi*D26*L*(j1*j1)*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/((r*r*r)*(tmax - tmin)**2) + D26*i1*(sina*sina)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - 2*(pi*pi)*D26*(j1*j1)*k1*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/((r*r)*(tmax - tmin)**2) - 2*pi*D26*i1*k1*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(L*r))/(i1 - k1))/((r*r*r)*(2.0*i1 + 2.0*k1)*(j1 - l1)*(j1 + l1))

                        elif k1 == i1 and l1 != j1 and i1 != 0:
                            # k0_11 cond_2
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += -0.25*((-1)**(j1 + l1) - 1)*(pi*A16*i1*r*cos(pi*i1*(L + 2*xa)/L) - pi*A16*i1*r*cos(pi*i1*(L + 2*xb)/L) + A26*sina*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)))/(pi*i1*r)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += 0.25*((-1)**(j1 + l1) - 1)*(-pi*i1*r*(-A12*(l1*l1) + A66*(j1*j1))*cos(pi*i1*(L + 2*xa)/L) + pi*i1*r*(-A12*(l1*l1) + A66*(j1*j1))*cos(pi*i1*(L + 2*xb)/L) + sina*(A22*(l1*l1) + A66*(j1*j1))*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)))/(pi*i1*r*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += 0.25*l1*((-1)**(j1 + l1) - 1)*(A12*(L*L)*cosa*(tmax - tmin)**2*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/pi + A22*(L*L*L*L)*cosa*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/((pi*pi)*i1*r) + pi*B11*(i1*i1)*r*(tmax - tmin)**2*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)) + 2*B12*(L*L)*i1*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + pi*B12*(L*L)*(l1*l1)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/r + B22*(L*L*L*L)*(l1*l1)*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(i1*(r*r)) + B22*(L*L)*(sina*sina)*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*r) + B66*(L*L*L*L)*(j1*j1)*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(i1*(r*r)) + 2*pi*B66*(L*L)*(j1*j1)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/r)/((L*L*L)*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += -0.25*((-1)**(j1 + l1) - 1)*(pi*i1*r*(A12*(j1*j1) - A66*(l1*l1))*cos(pi*i1*(L + 2*xa)/L) - pi*i1*r*(A12*(j1*j1) - A66*(l1*l1))*cos(pi*i1*(L + 2*xb)/L) + sina*(A22*(j1*j1) + A66*(l1*l1))*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)))/(pi*i1*r*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.25*A26*((-1)**(j1 + l1) - 1)*(-pi*i1*r*cos(pi*i1*(L + 2*xa)/L) + pi*i1*r*cos(pi*i1*(L + 2*xb)/L) + sina*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)))/(pi*i1*r)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += 0.25*l1*((-1)**(j1 + l1) - 1)*(-A26*(L*L*L*L)*cosa*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/((pi*pi)*i1*r) + A26*(L*L)*cosa*(tmax - tmin)**2*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/pi - B16*(L*L)*i1*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + pi*B16*(i1*i1)*r*(tmax - tmin)**2*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)) + B26*(L*L*L*L)*(j1*j1)*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(i1*(r*r)) - B26*(L*L*L*L)*(l1*l1)*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(i1*(r*r)) + B26*(L*L)*i1*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + 2*pi*B26*(L*L)*(j1*j1)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/r + pi*B26*(L*L)*(l1*l1)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/r - B26*(L*L)*(sina*sina)*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*r))/((L*L*L)*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += 0.25*j1*((-1)**(j1 + l1) - 1)*(A12*(L*L)*cosa*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/pi - A22*(L*L*L*L)*cosa*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/((pi*pi)*i1*r) - pi*B11*(i1*i1)*r*(tmax - tmin)**2*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)) - 2*B12*(L*L)*i1*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) - pi*B12*(L*L)*(j1*j1)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/r - B22*(L*L*L*L)*(j1*j1)*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(i1*(r*r)) - B22*(L*L)*(sina*sina)*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*r) - B66*(L*L*L*L)*(l1*l1)*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(i1*(r*r)) - 2*pi*B66*(L*L)*(l1*l1)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/r)/((L*L*L)*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += 0.25*j1*((-1)**(j1 + l1) - 1)*(-(-1)**i1*A26*(L*L*L*L)*cosa*r*sina*(tmax*tmax)*cos(2*pi*i1*xb/L) + 2*(-1)**i1*A26*(L*L*L*L)*cosa*r*sina*tmax*tmin*cos(2*pi*i1*xb/L) - (-1)**i1*A26*(L*L*L*L)*cosa*r*sina*(tmin*tmin)*cos(2*pi*i1*xb/L) + (-1)**i1*pi*A26*(L*L*L)*cosa*i1*(r*r)*(tmax*tmax)*sin(2*pi*i1*xa/L) - (-1)**i1*pi*A26*(L*L*L)*cosa*i1*(r*r)*(tmax*tmax)*sin(2*pi*i1*xb/L) - 2*(-1)**i1*pi*A26*(L*L*L)*cosa*i1*(r*r)*tmax*tmin*sin(2*pi*i1*xa/L) + 2*(-1)**i1*pi*A26*(L*L*L)*cosa*i1*(r*r)*tmax*tmin*sin(2*pi*i1*xb/L) + (-1)**i1*pi*A26*(L*L*L)*cosa*i1*(r*r)*(tmin*tmin)*sin(2*pi*i1*xa/L) - (-1)**i1*pi*A26*(L*L*L)*cosa*i1*(r*r)*(tmin*tmin)*sin(2*pi*i1*xb/L) - (-1)**i1*(pi*pi)*B16*(L*L)*(i1*i1)*(r*r)*sina*(tmax*tmax)*cos(2*pi*i1*xb/L) + 2*(-1)**i1*(pi*pi)*B16*(L*L)*(i1*i1)*(r*r)*sina*tmax*tmin*cos(2*pi*i1*xb/L) - (-1)**i1*(pi*pi)*B16*(L*L)*(i1*i1)*(r*r)*sina*(tmin*tmin)*cos(2*pi*i1*xb/L) + (-1)**i1*(pi*pi*pi)*B16*L*(i1*i1*i1)*(r*r*r)*(tmax*tmax)*sin(2*pi*i1*xa/L) - (-1)**i1*(pi*pi*pi)*B16*L*(i1*i1*i1)*(r*r*r)*(tmax*tmax)*sin(2*pi*i1*xb/L) - 2*(-1)**i1*(pi*pi*pi)*B16*L*(i1*i1*i1)*(r*r*r)*tmax*tmin*sin(2*pi*i1*xa/L) + 2*(-1)**i1*(pi*pi*pi)*B16*L*(i1*i1*i1)*(r*r*r)*tmax*tmin*sin(2*pi*i1*xb/L) + (-1)**i1*(pi*pi*pi)*B16*L*(i1*i1*i1)*(r*r*r)*(tmin*tmin)*sin(2*pi*i1*xa/L) - (-1)**i1*(pi*pi*pi)*B16*L*(i1*i1*i1)*(r*r*r)*(tmin*tmin)*sin(2*pi*i1*xb/L) - (-1)**i1*(pi*pi)*B26*(L*L*L*L)*(j1*j1)*sina*cos(2*pi*i1*xb/L) + (-1)**i1*(pi*pi*pi)*B26*(L*L*L)*i1*(j1*j1)*r*sin(2*pi*i1*xa/L) - (-1)**i1*(pi*pi*pi)*B26*(L*L*L)*i1*(j1*j1)*r*sin(2*pi*i1*xb/L) + (-1)**i1*pi*B26*(L*L*L)*i1*r*(sina*sina)*(tmax*tmax)*sin(2*pi*i1*xa/L) - (-1)**i1*pi*B26*(L*L*L)*i1*r*(sina*sina)*(tmax*tmax)*sin(2*pi*i1*xb/L) - 2*(-1)**i1*pi*B26*(L*L*L)*i1*r*(sina*sina)*tmax*tmin*sin(2*pi*i1*xa/L) + 2*(-1)**i1*pi*B26*(L*L*L)*i1*r*(sina*sina)*tmax*tmin*sin(2*pi*i1*xb/L) + (-1)**i1*pi*B26*(L*L*L)*i1*r*(sina*sina)*(tmin*tmin)*sin(2*pi*i1*xa/L) - (-1)**i1*pi*B26*(L*L*L)*i1*r*(sina*sina)*(tmin*tmin)*sin(2*pi*i1*xb/L) + (-1)**i1*(pi*pi)*B26*(L*L)*(i1*i1)*(r*r)*sina*(tmax*tmax)*cos(2*pi*i1*xb/L) - 2*(-1)**i1*(pi*pi)*B26*(L*L)*(i1*i1)*(r*r)*sina*tmax*tmin*cos(2*pi*i1*xb/L) + (-1)**i1*(pi*pi)*B26*(L*L)*(i1*i1)*(r*r)*sina*(tmin*tmin)*cos(2*pi*i1*xb/L) + (-1)**i1*(L*L)*sina*((pi*pi)*B26*((L*L)*(j1*j1) - (i1*i1)*(r*r)*(tmax - tmin)**2) + r*(tmax - tmin)**2*(A26*(L*L)*cosa + (pi*pi)*B16*(i1*i1)*r))*cos(2*pi*i1*xa/L) - 2*(pi*pi)*A26*(L*L)*cosa*(i1*i1)*(r*r)*(tmax*tmax)*xa + 2*(pi*pi)*A26*(L*L)*cosa*(i1*i1)*(r*r)*(tmax*tmax)*xb + 4*(pi*pi)*A26*(L*L)*cosa*(i1*i1)*(r*r)*tmax*tmin*xa - 4*(pi*pi)*A26*(L*L)*cosa*(i1*i1)*(r*r)*tmax*tmin*xb - 2*(pi*pi)*A26*(L*L)*cosa*(i1*i1)*(r*r)*(tmin*tmin)*xa + 2*(pi*pi)*A26*(L*L)*cosa*(i1*i1)*(r*r)*(tmin*tmin)*xb - 2*(pi*pi*pi*pi)*B16*(i1*i1*i1*i1)*(r*r*r)*(tmax*tmax)*xa + 2*(pi*pi*pi*pi)*B16*(i1*i1*i1*i1)*(r*r*r)*(tmax*tmax)*xb + 4*(pi*pi*pi*pi)*B16*(i1*i1*i1*i1)*(r*r*r)*tmax*tmin*xa - 4*(pi*pi*pi*pi)*B16*(i1*i1*i1*i1)*(r*r*r)*tmax*tmin*xb - 2*(pi*pi*pi*pi)*B16*(i1*i1*i1*i1)*(r*r*r)*(tmin*tmin)*xa + 2*(pi*pi*pi*pi)*B16*(i1*i1*i1*i1)*(r*r*r)*(tmin*tmin)*xb - (pi*pi)*B26*(L*L*L*L)*(l1*l1)*sina*cos(pi*i1*(L + 2*xa)/L) + (pi*pi)*B26*(L*L*L*L)*(l1*l1)*sina*cos(pi*i1*(L + 2*xb)/L) - 2*(pi*pi*pi)*B26*(L*L*L)*i1*(l1*l1)*r*sin(pi*i1*(L + 2*xa)/L) + 2*(pi*pi*pi)*B26*(L*L*L)*i1*(l1*l1)*r*sin(pi*i1*(L + 2*xb)/L) - 2*(pi*pi*pi*pi)*B26*(L*L)*(i1*i1)*(j1*j1)*r*xa + 2*(pi*pi*pi*pi)*B26*(L*L)*(i1*i1)*(j1*j1)*r*xb - 4*(pi*pi*pi*pi)*B26*(L*L)*(i1*i1)*(l1*l1)*r*xa + 4*(pi*pi*pi*pi)*B26*(L*L)*(i1*i1)*(l1*l1)*r*xb + 2*(pi*pi)*B26*(L*L)*(i1*i1)*r*(sina*sina)*(tmax*tmax)*xa - 2*(pi*pi)*B26*(L*L)*(i1*i1)*r*(sina*sina)*(tmax*tmax)*xb - 4*(pi*pi)*B26*(L*L)*(i1*i1)*r*(sina*sina)*tmax*tmin*xa + 4*(pi*pi)*B26*(L*L)*(i1*i1)*r*(sina*sina)*tmax*tmin*xb + 2*(pi*pi)*B26*(L*L)*(i1*i1)*r*(sina*sina)*(tmin*tmin)*xa - 2*(pi*pi)*B26*(L*L)*(i1*i1)*r*(sina*sina)*(tmin*tmin)*xb)/((pi*pi)*(L*L*L)*i1*(r*r)*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.25*pi*D26*j1*l1*((-1)**(j1 + l1) - 1)*(2*pi*i1*r*cos(pi*i1*(L + 2*xa)/L) - 2*pi*i1*r*cos(pi*i1*(L + 2*xb)/L) + sina*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)))/(i1*(r*r*r)*(tmax - tmin)**2)

                        elif k1 != i1 and l1 == j1 and j1 != 0:
                            # k0_11 cond_3
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += (pi*A11*i1*k1*r*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L - A12*i1*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) - A12*k1*sina*(tmax - tmin)**2*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) - A22*L*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) - pi*A66*L*(j1*j1)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r)/((i1 + k1)*(4.0*i1 - 4.0*k1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += (A16*i1*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) + pi*A16*i1*k1*r*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L - pi*A26*L*(j1*j1)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r + A26*L*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) - A26*k1*sina*(tmax - tmin)**2*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)))/((i1 + k1)*(4.0*i1 - 4.0*k1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += j1*(A26*L*cosa*((-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L))/r - pi*B16*i1*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r + 2*(pi*pi)*B16*i1*k1*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/L + (pi*pi)*B16*(k1*k1)*((-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L))/L - (pi*pi)*B26*L*(j1*j1)*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/((r*r)*(tmax - tmin)**2) + B26*L*(sina*sina)*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) + pi*B26*k1*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r)/((i1 + k1)*(4.0*i1 - 4.0*k1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += (A16*k1*sina*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L)) + pi*A16*i1*k1*r*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L - pi*A26*L*(j1*j1)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r + A26*L*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) + A26*i1*sina*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/((i1 + k1)*(4.0*i1 - 4.0*k1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += (pi*A22*L*(j1*j1)*((-i1 + k1)*cos(pi*(0.5*i1 + 0.5*k1 - 0.5 + 0.5*xa*(2*i1 + 2*k1)/L)) + (i1 - k1)*cos(pi*(0.5*i1 + 0.5*k1 - 0.5 + 0.5*xb*(2*i1 + 2*k1)/L)) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r - A66*L*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) + A66*i1*sina*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L)) + A66*k1*sina*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L)) + pi*A66*i1*k1*r*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L)/((i1 + k1)*(4.0*i1 - 4.0*k1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += j1*(A22*L*cosa*((-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L))/r + (pi*pi)*B12*(k1*k1)*((-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L))/L - (pi*pi)*B22*L*(j1*j1)*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/((r*r)*(tmax - tmin)**2) - pi*B22*k1*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r - B66*L*(sina*sina)*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - pi*B66*i1*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r - 2*pi*B66*k1*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r + 2*(pi*pi)*B66*i1*k1*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/L)/((i1 + k1)*(4.0*i1 - 4.0*k1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += j1*(-A26*L*cosa*((-i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L))/r + pi*B16*k1*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r - (pi*pi)*B16*(i1*i1)*((-i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L))/L + 2*(pi*pi)*B16*i1*k1*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/L - (pi*pi)*B26*L*(j1*j1)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/((r*r)*(tmax - tmin)**2) + B26*L*(sina*sina)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) + pi*B26*i1*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r)/((i1 + k1)*(4.0*i1 - 4.0*k1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += -j1*(B66*L*(sina*sina)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) + 2*pi*B66*i1*r*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L))) - pi*B66*k1*r*sina*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L))) + 2*(pi*pi)*B66*i1*k1*(r*r)*((-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L))/L + pi*(r*r)*(A22*L*cosa*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(pi*r) + pi*B12*(i1*i1)*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/L + pi*B22*L*(j1*j1)*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) + B22*i1*sina*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/r)/(tmax - tmin)**2)/((r*r)*(i1 + k1)*(4.0*i1 - 4.0*k1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += (A22*L*(cosa*cosa)*(tmax - tmin)**4*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r) + pi*B12*cosa*(i1*i1)*(tmax - tmin)**4*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L - pi*B12*cosa*(k1*k1)*(tmax - tmin)**4*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L + 2*pi*B22*L*cosa*(j1*j1)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(r*r) - B22*cosa*i1*sina*(tmax - tmin)**4*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/r - B22*cosa*k1*sina*(tmax - tmin)**4*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/r + (pi*pi*pi)*D11*(i1*i1)*(k1*k1)*r*(tmax - tmin)**4*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(L*L*L) + (pi*pi*pi)*D12*(i1*i1)*(j1*j1)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(L*r) + (pi*pi*pi)*D12*(j1*j1)*(k1*k1)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(L*r) - (pi*pi)*D12*(i1*i1)*k1*sina*(tmax - tmin)**4*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(L*L) - (pi*pi)*D12*i1*(k1*k1)*sina*(tmax - tmin)**4*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(L*L) + (pi*pi*pi)*D22*L*(j1*j1*j1*j1)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(r*r*r) + (pi*pi)*D22*i1*(j1*j1)*sina*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L))/(r*r) - (pi*pi)*D22*(j1*j1)*k1*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - pi*D22*i1*k1*(sina*sina)*(tmax - tmin)**4*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(L*r) + pi*D66*L*(j1*j1)*(sina*sina)*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(r*r*r) - 2*(pi*pi)*D66*i1*(j1*j1)*sina*(tmax - tmin)**2*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - 2*(pi*pi)*D66*(j1*j1)*k1*sina*(tmax - tmin)**2*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(r*r) - 4*(pi*pi*pi)*D66*i1*(j1*j1)*k1*(tmax - tmin)**2*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(L*r))/((i1 + k1)*(4.0*i1 - 4.0*k1)*(tmax - tmin)**3)

                        elif k1 == i1 and l1 == j1 and i1 != 0 and j1 != 0:
                            # k0_11 cond_4
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += (pi*A11*i1*r*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(L*L) - 2*A12*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + A22*(sina*sina)*(tmax - tmin)**2*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(pi*i1*r) + pi*A66*(j1*j1)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(i1*r))/(8.0*tmax - 8.0*tmin)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += (A16*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + pi*A16*i1*r*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(L*L) - A26*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + pi*A26*(j1*j1)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(i1*r) + A26*(sina*sina)*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*i1*r))/(8.0*tmax - 8.0*tmin)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += 0.125*j1*(pi*sina*(-L*(B16 - B26)*sin(pi*i1*(L + 2*xa)/L) + L*(B16 - B26)*sin(pi*i1*(L + 2*xb)/L) + pi*i1*(2*B16 + 2*B26)*(xa - xb))/r + (-(pi*pi)*B16*(i1*i1) + (L*L)*(A26*cosa*r + (pi*pi)*B26*(j1*j1)/(tmax - tmin)**2 - B26*(sina*sina))/(r*r))*cos(pi*i1*(L + 2*xb)/L)/i1 + ((pi*pi)*B16*(i1*i1) + (L*L)*(-A26*cosa*r + B26*(-(pi*pi)*(j1*j1)/(tmax - tmin)**2 + (sina*sina)))/(r*r))*cos(pi*i1*(L + 2*xa)/L)/i1)/L
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += (A16*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + pi*A16*i1*r*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(L*L) - A26*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + pi*A26*(j1*j1)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(i1*r) + A26*(sina*sina)*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*i1*r))/(8.0*tmax - 8.0*tmin)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += (pi*A22*(j1*j1)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(i1*r) + 2*A66*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + A66*(sina*sina)*(tmax - tmin)**2*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(pi*i1*r) + pi*A66*i1*r*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(L*L))/(8.0*tmax - 8.0*tmin)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += 0.125*j1*(-pi*sina*(L*(B22 + 3*B66)*sin(pi*i1*(L + 2*xa)/L) - L*(B22 + 3*B66)*sin(pi*i1*(L + 2*xb)/L) + pi*i1*(2*B22 + 2*B66)*(xa - xb))/r - ((pi*pi)*B12*(i1*i1) + B66*((L*L)*(sina*sina)/(r*r) - 2*(pi*pi)*(i1*i1)) + (L*L)*(A22*cosa*r + (pi*pi)*B22*(j1*j1)/(tmax - tmin)**2)/(r*r))*cos(pi*i1*(L + 2*xa)/L)/i1 + ((pi*pi)*B12*(i1*i1) + B66*((L*L)*(sina*sina)/(r*r) - 2*(pi*pi)*(i1*i1)) + (L*L)*(A22*cosa*r + (pi*pi)*B22*(j1*j1)/(tmax - tmin)**2)/(r*r))*cos(pi*i1*(L + 2*xb)/L)/i1)/L
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += 0.125*j1*(pi*sina*(-L*(B16 - B26)*sin(pi*i1*(L + 2*xa)/L) + L*(B16 - B26)*sin(pi*i1*(L + 2*xb)/L) + pi*i1*(2*B16 + 2*B26)*(xa - xb))/r + (-(pi*pi)*B16*(i1*i1) + (L*L)*(A26*cosa*r + (pi*pi)*B26*(j1*j1)/(tmax - tmin)**2 - B26*(sina*sina))/(r*r))*cos(pi*i1*(L + 2*xb)/L)/i1 + ((pi*pi)*B16*(i1*i1) + (L*L)*(-A26*cosa*r + B26*(-(pi*pi)*(j1*j1)/(tmax - tmin)**2 + (sina*sina)))/(r*r))*cos(pi*i1*(L + 2*xa)/L)/i1)/L
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += 0.125*j1*(-pi*sina*(L*(B22 + 3*B66)*sin(pi*i1*(L + 2*xa)/L) - L*(B22 + 3*B66)*sin(pi*i1*(L + 2*xb)/L) + pi*i1*(2*B22 + 2*B66)*(xa - xb))/r - ((pi*pi)*B12*(i1*i1) + B66*((L*L)*(sina*sina)/(r*r) - 2*(pi*pi)*(i1*i1)) + (L*L)*(A22*cosa*r + (pi*pi)*B22*(j1*j1)/(tmax - tmin)**2)/(r*r))*cos(pi*i1*(L + 2*xa)/L)/i1 + ((pi*pi)*B12*(i1*i1) + B66*((L*L)*(sina*sina)/(r*r) - 2*(pi*pi)*(i1*i1)) + (L*L)*(A22*cosa*r + (pi*pi)*B22*(j1*j1)/(tmax - tmin)**2)/(r*r))*cos(pi*i1*(L + 2*xb)/L)/i1)/L
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.125*(-A22*(cosa*cosa)*(tmax - tmin)**4*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*i1*r) - 2*pi*B12*cosa*i1*(tmax - tmin)**4*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(L*L) - 2*B22*cosa*sina*(tmax - tmin)**4*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/r + 2*pi*B22*cosa*(j1*j1)*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(i1*(r*r)) + (pi*pi*pi)*D11*(i1*i1*i1)*r*(tmax - tmin)**4*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(L*L*L*L) - 2*(pi*pi)*D12*(i1*i1)*sina*(tmax - tmin)**4*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(L*L) + 2*(pi*pi*pi)*D12*i1*(j1*j1)*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/((L*L)*r) - 2*(pi*pi)*D22*(j1*j1)*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(r*r) + (pi*pi*pi)*D22*(j1*j1*j1*j1)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(i1*(r*r*r)) - pi*D22*i1*(sina*sina)*(tmax - tmin)**4*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/((L*L)*r) - 4*(pi*pi)*D66*(j1*j1)*sina*(tmax - tmin)**2*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L))/(r*r) + pi*D66*(j1*j1)*(sina*sina)*(tmax - tmin)**2*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(i1*(r*r*r)) + 4*(pi*pi*pi)*D66*i1*(j1*j1)*(tmax - tmin)**2*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/((L*L)*r))/(tmax - tmin)**3

                        elif k1 == i1 and l1 != j1 and i1 == 0:
                            # k0_11 cond_5
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += -A26*sina*((-1)**(j1 + l1) - 1)*(xa - xb)/r
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += sina*((-1)**(j1 + l1) - 1)*(xa - xb)*(A22*(l1*l1) + A66*(j1*j1))/(r*((j1*j1) - (l1*l1)))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += -sina*((-1)**(j1 + l1) - 1)*(xa - xb)*(A22*(j1*j1) + A66*(l1*l1))/(r*((j1*j1) - (l1*l1)))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += A26*sina*((-1)**(j1 + l1) - 1)*(xa - xb)/r

                        elif k1 != i1 and l1 == j1 and j1 == 0:
                            # k0_11 cond_6
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += -(tmax - tmin)*(-pi*A11*i1*k1*r*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L + A12*i1*sina*(-i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) + A12*k1*sina*(i1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - i1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) + (i1 - k1)*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) - (i1 + k1)*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L)) + A22*L*(sina*sina)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*r))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += (tmax - tmin)*(i1*(pi*L*k1*r*sina*(A16 + A26)*sin(0.5*pi*i1*(L + 2*xa)/L)*sin(0.5*pi*k1*(L + 2*xa)/L) - ((pi*pi)*A16*(k1*k1)*(r*r) - A26*(L*L)*(sina*sina))*sin(0.5*pi*i1*(L + 2*xa)/L)*cos(0.5*pi*k1*(L + 2*xa)/L) + (-pi*L*k1*r*sina*(A16 + A26)*sin(0.5*pi*k1*(L + 2*xb)/L) + ((pi*pi)*A16*(k1*k1)*(r*r) - A26*(L*L)*(sina*sina))*cos(0.5*pi*k1*(L + 2*xb)/L))*sin(0.5*pi*i1*(L + 2*xb)/L)) + (pi*L*r*sina*(A16*(i1*i1) + A26*(k1*k1))*cos(0.5*pi*k1*(L + 2*xa)/L) + k1*((pi*pi)*A16*(i1*i1)*(r*r) - A26*(L*L)*(sina*sina))*sin(0.5*pi*k1*(L + 2*xa)/L))*cos(0.5*pi*i1*(L + 2*xa)/L) - (pi*L*r*sina*(A16*(i1*i1) + A26*(k1*k1))*cos(0.5*pi*k1*(L + 2*xb)/L) + k1*((pi*pi)*A16*(i1*i1)*(r*r) - A26*(L*L)*(sina*sina))*sin(0.5*pi*k1*(L + 2*xb)/L))*cos(0.5*pi*i1*(L + 2*xb)/L))/(pi*L*r*(i1 - k1)*(i1 + k1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += (tmax - tmin)*(i1*(-pi*L*k1*r*sina*(A16 + A26)*sin(0.5*pi*i1*(L + 2*xa)/L)*sin(0.5*pi*k1*(L + 2*xa)/L) - ((pi*pi)*A16*(k1*k1)*(r*r) - A26*(L*L)*(sina*sina))*sin(0.5*pi*i1*(L + 2*xa)/L)*cos(0.5*pi*k1*(L + 2*xa)/L) + (pi*L*k1*r*sina*(A16 + A26)*sin(0.5*pi*k1*(L + 2*xb)/L) + ((pi*pi)*A16*(k1*k1)*(r*r) - A26*(L*L)*(sina*sina))*cos(0.5*pi*k1*(L + 2*xb)/L))*sin(0.5*pi*i1*(L + 2*xb)/L)) + (-pi*L*r*sina*(A16*(k1*k1) + A26*(i1*i1))*cos(0.5*pi*k1*(L + 2*xa)/L) + k1*((pi*pi)*A16*(i1*i1)*(r*r) - A26*(L*L)*(sina*sina))*sin(0.5*pi*k1*(L + 2*xa)/L))*cos(0.5*pi*i1*(L + 2*xa)/L) + (pi*L*r*sina*(A16*(k1*k1) + A26*(i1*i1))*cos(0.5*pi*k1*(L + 2*xb)/L) + k1*(-(pi*pi)*A16*(i1*i1)*(r*r) + A26*(L*L)*(sina*sina))*sin(0.5*pi*k1*(L + 2*xb)/L))*cos(0.5*pi*i1*(L + 2*xb)/L))/(pi*L*r*(i1 - k1)*(i1 + k1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += A66*(tmax - tmin)*(-i1*((L*L)*(sina*sina) + (pi*pi)*(k1*k1)*(r*r))*(sin(0.5*pi*i1*(L + 2*xa)/L)*cos(0.5*pi*k1*(L + 2*xa)/L) - sin(0.5*pi*i1*(L + 2*xb)/L)*cos(0.5*pi*k1*(L + 2*xb)/L)) + (pi*L*r*sina*((i1*i1) - (k1*k1))*cos(0.5*pi*k1*(L + 2*xa)/L) + k1*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r))*sin(0.5*pi*k1*(L + 2*xa)/L))*cos(0.5*pi*i1*(L + 2*xa)/L) - (pi*L*r*sina*((i1*i1) - (k1*k1))*cos(0.5*pi*k1*(L + 2*xb)/L) + k1*((L*L)*(sina*sina) + (pi*pi)*(i1*i1)*(r*r))*sin(0.5*pi*k1*(L + 2*xb)/L))*cos(0.5*pi*i1*(L + 2*xb)/L))/(pi*L*r*(i1 - k1)*(i1 + k1))

                        elif k1 == i1 and l1 == j1 and i1 == 0 and j1 != 0:
                            # k0_11 cond_7
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += -0.5*(xa - xb)*(A22*(sina*sina)*(tmax - tmin)**2 + (pi*pi)*A66*(j1*j1))/(r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += 0.5*A26*(xa - xb)*(-(pi*pi)*(j1*j1) + (sina*sina)*(tmax - tmin)**2)/(r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += 0.5*A26*(xa - xb)*(-(pi*pi)*(j1*j1) + (sina*sina)*(tmax - tmin)**2)/(r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += -0.5*(xa - xb)*((pi*pi)*A22*(j1*j1) + A66*(sina*sina)*(tmax - tmin)**2)/(r*(tmax - tmin))

                        elif k1 == i1 and l1 == j1 and i1 != 0 and j1 == 0:
                            # k0_11 cond_8
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.25*(tmax - tmin)*(pi*A11*i1*r*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(L*L) - 2*A12*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + A22*(sina*sina)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(pi*i1*r))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += 0.25*(tmax - tmin)*(A16*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + pi*A16*i1*r*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(L*L) - A26*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + A26*(sina*sina)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*i1*r))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += 0.25*(tmax - tmin)*(A16*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + pi*A16*i1*r*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(L*L) - A26*sina*(cos(pi*i1*(L + 2*xa)/L) - cos(pi*i1*(L + 2*xb)/L)) + A26*(sina*sina)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*i1*r))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.25*A66*(tmax - tmin)*(-(L*L*L)*(sina*sina)*sin(pi*i1*(L + 2*xa)/L) + (L*L*L)*(sina*sina)*sin(pi*i1*(L + 2*xb)/L) + 2*pi*(L*L)*i1*r*sina*cos(pi*i1*(L + 2*xa)/L) - 2*pi*(L*L)*i1*r*sina*cos(pi*i1*(L + 2*xb)/L) - 2*pi*(L*L)*i1*(sina*sina)*xa + 2*pi*(L*L)*i1*(sina*sina)*xb + (pi*pi)*L*(i1*i1)*(r*r)*sin(pi*i1*(L + 2*xa)/L) - (pi*pi)*L*(i1*i1)*(r*r)*sin(pi*i1*(L + 2*xb)/L) - 2*(pi*pi*pi)*(i1*i1*i1)*(r*r)*xa + 2*(pi*pi*pi)*(i1*i1*i1)*(r*r)*xb)/(pi*(L*L)*i1*r)

                        elif k1 == i1 and l1 == j1 and i1 == 0 and j1 == 0:
                            # k0_11 cond_9
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += A22*(sina*sina)*(tmax - tmin)*(-xa + xb)/r
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += A26*(sina*sina)*(tmax - tmin)*(xa - xb)/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += A26*(sina*sina)*(tmax - tmin)*(xa - xb)/r
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += A66*(sina*sina)*(tmax - tmin)*(-xa + xb)/r

    size = num0 + num1*m1*n1

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0_cyl(double r1, double L, double tmin, double tmax,
            np.ndarray[cDOUBLE, ndim=2] F, int m1, int n1):
    cdef int i1, j1, k1, l1, c, row, col
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    fdim = 5*m1*n1*m1*n1//2

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
    r = r1

    # k0_11
    for i1 in range(m1):
        for j1 in range(n1):
            row = num0 + num1*((j1)*m1 + (i1))
            for k1 in range(m1):
                for l1 in range(n1):
                    col = num0 + num1*((l1)*m1 + (k1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0_11 cond_1
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += -A16*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += -((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(i1*i1)*(l1*l1) + A66*(j1*j1)*(k1*k1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += -((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(j1*j1)*(k1*k1) + A66*(i1*i1)*(l1*l1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += -A26*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*((pi*pi)*D26*(L*L)*((j1*j1) + (l1*l1)) + r*(tmax - tmin)**2*(2*B26*(L*L) + (pi*pi)*D16*r*((i1*i1) + (k1*k1))))/((L*L)*(r*r)*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin)**2)

                    elif k1 == i1 and l1 != j1 and i1 != 0:
                        # k0_11 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += -0.5*i1*l1*((-1)**(j1 + l1) - 1)*((pi*pi)*B12*(L*L)*(l1*l1) + 2*(pi*pi)*B66*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r))/((L*L)*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += -0.5*i1*l1*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*(2*(j1*j1) + (l1*l1)) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(i1*i1)*r))/((L*L)*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 0.5*i1*j1*((-1)**(j1 + l1) - 1)*((pi*pi)*B12*(L*L)*(j1*j1) + 2*(pi*pi)*B66*(L*L)*(l1*l1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r))/((L*L)*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += 0.5*i1*j1*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*((j1*j1) + 2*(l1*l1)) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(i1*i1)*r))/((L*L)*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))

                    elif k1 != i1 and l1 == j1 and j1 != 0:
                        # k0_11 cond_3
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*r*(2*(i1*i1) + (k1*k1))))/(L*(r*r)*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin)**2)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B22*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A22*(L*L) + (pi*pi)*r*(B12*(k1*k1) + 2*B66*(i1*i1))))/(L*(r*r)*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin)**2)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += i1*j1*((-1)**(i1 + k1) - 1)*((pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*r*((i1*i1) + 2*(k1*k1))))/(L*(r*r)*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin)**2)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += i1*j1*((-1)**(i1 + k1) - 1)*((pi*pi)*B22*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A22*(L*L) + (pi*pi)*r*(B12*(i1*i1) + 2*B66*(k1*k1))))/(L*(r*r)*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin)**2)

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 != 0:
                        # k0_11 cond_4
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += 0.25*(pi*pi)*A11*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A66*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += (pi*pi)*A22*L*(j1*j1)/(4*r*tmax - 4*r*tmin) + 0.25*(pi*pi)*A66*(i1*i1)*r*(tmax - tmin)/L
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += 0.25*(A22*(L*L*L*L)*(tmax - tmin)**4/r + 2*(pi*pi)*B12*(L*L)*(i1*i1)*(tmax - tmin)**4 + 2*(pi*pi)*B22*(L*L*L*L)*(j1*j1)*(tmax - tmin)**2/(r*r) + (pi*pi*pi*pi)*D11*(i1*i1*i1*i1)*r*(tmax - tmin)**4 + 2*(pi*pi*pi*pi)*D12*(L*L)*(i1*i1)*(j1*j1)*(tmax - tmin)**2/r + (pi*pi*pi*pi)*D22*(L*L*L*L)*(j1*j1*j1*j1)/(r*r*r) + 4*(pi*pi*pi*pi)*D66*(L*L)*(i1*i1)*(j1*j1)*(tmax - tmin)**2/r)/((L*L*L)*(tmax - tmin)**3)

                    elif k1 == i1 and l1 != j1 and i1 == 0:
                        # k0_11 cond_5
                        pass

                    elif k1 != i1 and l1 == j1 and j1 == 0:
                        # k0_11 cond_6
                        pass

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 != 0:
                        # k0_11 cond_7
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += (pi*pi)*A66*L*(j1*j1)/(2*r*tmax - 2*r*tmin)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += (pi*pi)*A26*L*(j1*j1)/(2*r*tmax - 2*r*tmin)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += (pi*pi)*A26*L*(j1*j1)/(2*r*tmax - 2*r*tmin)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += (pi*pi)*A22*L*(j1*j1)/(2*r*tmax - 2*r*tmin)

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 == 0:
                        # k0_11 cond_8
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += 0.5*(pi*pi)*A11*(i1*i1)*r*(tmax - tmin)/L
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += 0.5*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += 0.5*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += 0.5*(pi*pi)*A66*(i1*i1)*r*(tmax - tmin)/L

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 == 0:
                        # k0_11 cond_9
                        pass

    size = num0 + num1*m1*n1

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0edges(int m1, int n1, double alpharad, int s,
             double r1, double r2, double L,
             double tmin, double tmax,
             double kuBot, double kuTop,
             double kvBot, double kvTop,
             double kphixBot, double kphixTop,
             double kphitBot, double kphitTop,
             double kuLeft, double kuRight,
             double kvLeft, double kvRight,
             double kphixLeft, double kphixRight,
             double kphitLeft, double kphitRight):
    cdef int i1, j1, k1, l1, row, col, c, cbkp
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv
    cdef double xa, xb, r, sina
    cdef int section

    sina = sin(alpharad)

    fdim = 3*m1*n1*m1*n1//2 + 3*m1*n1*m1*n1//2

    k0edgesr = np.zeros((fdim,), dtype=INT)
    k0edgesc = np.zeros((fdim,), dtype=INT)
    k0edgesv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # k0edgesBT_11
    for i1 in range(m1):
        for j1 in range(n1):
            row = num0 + num1*((j1)*m1 + (i1))
            for k1 in range(m1):
                for l1 in range(n1):
                    col = num0 + num1*((l1)*m1 + (k1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0edgesBT_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1 and i1 != 0:
                        # k0edgesBT_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1 and j1 != 0:
                        # k0edgesBT_11 cond_3
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i1 + k1)*kuTop*r2 + kuBot*r1)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i1 + k1)*kvTop*r2 + kvBot*r1)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*i1*k1*(tmax - tmin)*((-1)**(i1 + k1)*kphixTop*r2 + kphixBot*r1)/(L*L)

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 != 0:
                        # k0edgesBT_11 cond_4
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
                        k0edgesv[c] += 0.5*(pi*pi)*(i1*i1)*(tmax - tmin)*(kphixBot*r1 + kphixTop*r2)/(L*L)

                    elif k1 == i1 and l1 != j1 and i1 == 0:
                        # k0edgesBT_11 cond_5
                        pass

                    elif k1 != i1 and l1 == j1 and j1 == 0:
                        # k0edgesBT_11 cond_6
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += (tmax - tmin)*((-1)**(i1 + k1)*kuTop*r2 + kuBot*r1)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += (tmax - tmin)*((-1)**(i1 + k1)*kvTop*r2 + kvBot*r1)

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 != 0:
                        # k0edgesBT_11 cond_7
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kvBot*r1 + kvTop*r2)

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 == 0:
                        # k0edgesBT_11 cond_8
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += (tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += (tmax - tmin)*(kvBot*r1 + kvTop*r2)

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 == 0:
                        # k0edgesBT_11 cond_9
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += (tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += (tmax - tmin)*(kvBot*r1 + kvTop*r2)

    cbkp = c

    for section in range(s):
        c = cbkp

        xa = -L/2. + L*float(section)/s
        xb = -L/2. + L*float(section+1)/s

        r = r1 - sina*((xa+xb)/2. + L/2.)

        # k0edgesLR_11
        for i1 in range(m1):
            for j1 in range(n1):
                row = num0 + num1*((j1)*m1 + (i1))
                for k1 in range(m1):
                    for l1 in range(n1):
                        col = num0 + num1*((l1)*m1 + (k1))

                        #NOTE symmetry
                        if row > col:
                            continue

                        if k1 != i1 and l1 != j1:
                            # k0edgesLR_11 cond_1
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += -L*((-1)**(j1 + l1)*kuLeft + kuRight)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*(i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += -L*((-1)**(j1 + l1)*kvLeft + kvRight)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*(i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0edgesr[c] = row+2
                            k0edgesc[c] = col+2
                            k0edgesv[c] += pi*L*j1*l1*((-1)**(j1 + l1)*kphitLeft + kphitRight)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/((r*r)*(i1 + k1)*(2.0*i1 - 2.0*k1)*(tmax - tmin)**2)

                        elif k1 == i1 and l1 != j1 and i1 != 0:
                            # k0edgesLR_11 cond_2
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += -0.25*((-1)**(j1 + l1)*kuLeft + kuRight)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*i1)
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += -0.25*((-1)**(j1 + l1)*kvLeft + kvRight)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(pi*i1)
                            c += 1
                            k0edgesr[c] = row+2
                            k0edgesc[c] = col+2
                            k0edgesv[c] += -0.25*pi*j1*l1*((-1)**(j1 + l1)*kphitLeft + kphitRight)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb))/(i1*(r*r)*(tmax - tmin)**2)

                        elif k1 != i1 and l1 == j1 and j1 != 0:
                            # k0edgesLR_11 cond_3
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += -L*(kuLeft + kuRight)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*(i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += -L*(kvLeft + kvRight)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*(i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0edgesr[c] = row+2
                            k0edgesc[c] = col+2
                            k0edgesv[c] += pi*L*(j1*j1)*(kphitLeft + kphitRight)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/((r*r)*(i1 + k1)*(2.0*i1 - 2.0*k1)*(tmax - tmin)**2)

                        elif k1 == i1 and l1 == j1 and i1 != 0 and j1 != 0:
                            # k0edgesLR_11 cond_4
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += 0.25*(kuLeft + kuRight)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(pi*i1)
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += 0.25*(kvLeft + kvRight)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(pi*i1)
                            c += 1
                            k0edgesr[c] = row+2
                            k0edgesc[c] = col+2
                            k0edgesv[c] += 0.25*pi*(j1*j1)*(kphitLeft + kphitRight)*(L*sin(pi*i1*(L + 2*xa)/L) - L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(i1*(r*r)*(tmax - tmin)**2)

                        elif k1 == i1 and l1 != j1 and i1 == 0:
                            # k0edgesBT_11 cond_5
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += (-xa + xb)*((-1)**(j1 + l1)*kuLeft + kuRight)
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += (-xa + xb)*((-1)**(j1 + l1)*kvLeft + kvRight)

                        elif k1 != i1 and l1 == j1 and j1 == 0:
                            # k0edgesBT_11 cond_6
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += -L*(kuLeft + kuRight)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*(i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += -L*(kvLeft + kvRight)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(pi*(i1 + k1)*(2.0*i1 - 2.0*k1))

                        elif k1 == i1 and l1 == j1 and i1 == 0 and j1 != 0:
                            # k0edgesBT_11 cond_7
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += (kuLeft + kuRight)*(-xa + xb)
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += (kvLeft + kvRight)*(-xa + xb)

                        elif k1 == i1 and l1 == j1 and i1 != 0 and j1 == 0:
                            # k0edgesBT_11 cond_8
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += 0.25*(kuLeft + kuRight)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(pi*i1)
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += 0.25*(kvLeft + kvRight)*(-L*sin(pi*i1*(L + 2*xa)/L) + L*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(-xa + xb))/(pi*i1)

                        elif k1 == i1 and l1 == j1 and i1 == 0 and j1 == 0:
                            # k0edgesBT_11 cond_9
                            c += 1
                            k0edgesr[c] = row+0
                            k0edgesc[c] = col+0
                            k0edgesv[c] += (kuLeft + kuRight)*(-xa + xb)
                            c += 1
                            k0edgesr[c] = row+1
                            k0edgesc[c] = col+1
                            k0edgesv[c] += (kvLeft + kvRight)*(-xa + xb)

    size = num0 + num1*m1*n1

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fk0edges_cyl(int m1, int n1, double r1, double L,
             double tmin, double tmax,
             double kuBot, double kuTop,
             double kvBot, double kvTop,
             double kphixBot, double kphixTop,
             double kphitBot, double kphitTop,
             double kuLeft, double kuRight,
             double kvLeft, double kvRight,
             double kphixLeft, double kphixRight,
             double kphitLeft, double kphitRight):
    cdef int i1, j1, k1, l1, row, col, c, cbkp
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv
    cdef double r2 = r1
    cdef double r = r1

    fdim = 3*m1*n1*m1*n1//2 + 3*m1*n1*m1*n1//2

    k0edgesr = np.zeros((fdim,), dtype=INT)
    k0edgesc = np.zeros((fdim,), dtype=INT)
    k0edgesv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # k0edgesBT_11
    for i1 in range(m1):
        for j1 in range(n1):
            row = num0 + num1*((j1)*m1 + (i1))
            for k1 in range(m1):
                for l1 in range(n1):
                    col = num0 + num1*((l1)*m1 + (k1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0edgesBT_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1 and i1 != 0:
                        # k0edgesBT_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1 and j1 != 0:
                        # k0edgesBT_11 cond_3
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i1 + k1)*kuTop*r2 + kuBot*r1)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i1 + k1)*kvTop*r2 + kvBot*r1)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*i1*k1*(tmax - tmin)*((-1)**(i1 + k1)*kphixTop*r2 + kphixBot*r1)/(L*L)

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 != 0:
                        # k0edgesBT_11 cond_4
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
                        k0edgesv[c] += 0.5*(pi*pi)*(i1*i1)*(tmax - tmin)*(kphixBot*r1 + kphixTop*r2)/(L*L)

                    elif k1 == i1 and l1 != j1 and i1 == 0:
                        # k0edgesBT_11 cond_5
                        pass

                    elif k1 != i1 and l1 == j1 and j1 == 0:
                        # k0edgesBT_11 cond_6
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += (tmax - tmin)*((-1)**(i1 + k1)*kuTop*r2 + kuBot*r1)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += (tmax - tmin)*((-1)**(i1 + k1)*kvTop*r2 + kvBot*r1)

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 != 0:
                        # k0edgesBT_11 cond_7
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kvBot*r1 + kvTop*r2)

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 == 0:
                        # k0edgesBT_11 cond_8
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += (tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += (tmax - tmin)*(kvBot*r1 + kvTop*r2)

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 == 0:
                        # k0edgesBT_11 cond_9
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += (tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += (tmax - tmin)*(kvBot*r1 + kvTop*r2)

    # k0edgesLR_11
    for i1 in range(m1):
        for j1 in range(n1):
            row = num0 + num1*((j1)*m1 + (i1))
            for k1 in range(m1):
                for l1 in range(n1):
                    col = num0 + num1*((l1)*m1 + (k1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # k0edgesLR_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1 and i1 != 0:
                        # k0edgesLR_11 cond_2
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*L*((-1)**(j1 + l1)*kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*L*((-1)**(j1 + l1)*kvLeft + kvRight)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*L*j1*l1*((-1)**(j1 + l1)*kphitLeft + kphitRight)/((r*r)*(tmax - tmin)**2)

                    elif k1 != i1 and l1 == j1 and j1 != 0:
                        # k0edgesLR_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 != 0:
                        # k0edgesLR_11 cond_4
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
                        k0edgesv[c] += 0.5*(pi*pi)*L*(j1*j1)*(kphitLeft + kphitRight)/((r*r)*(tmax - tmin)**2)

                    elif k1 == i1 and l1 != j1 and i1 == 0:
                        # k0edgesLR_11 cond_5
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += L*((-1)**(j1 + l1)*kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += L*((-1)**(j1 + l1)*kvLeft + kvRight)

                    elif k1 != i1 and l1 == j1 and j1 == 0:
                        # k0edgesLR_11 cond_6
                        pass

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 != 0:
                        # k0edgesLR_11 cond_7
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += L*(kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += L*(kvLeft + kvRight)

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 == 0:
                        # k0edgesLR_11 cond_8
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*L*(kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*L*(kvLeft + kvRight)

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 == 0:
                        # k0edgesLR_11 cond_9
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += L*(kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += L*(kvLeft + kvRight)

    size = num0 + num1*m1*n1

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fx, double Ft, double Fxt, double Ftx, double r1, double L,
        double tmin, double tmax, int m1, int n1, double alpharad, int s):
    cdef int i1, j1, k1, l1, c, row, col, section
    cdef double xa, xb, r, sina

    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    fdim = 1*m1*n1*m1*n1//2

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)

    for section in range(s):
        c = -1

        xa = -L/2. + L*float(section)/s
        xb = -L/2. + L*float(section+1)/s

        r = r1 - sina*((xa+xb)/2. + L/2.)

        # kG0_11
        for i1 in range(m1):
            for j1 in range(n1):
                row = num0 + num1*((j1)*m1 + (i1))
                for k1 in range(m1):
                    for l1 in range(n1):
                        col = num0 + num1*((l1)*m1 + (k1))

                        #NOTE symmetry
                        if row > col:
                            continue

                        if k1 != i1 and l1 != j1:
                            # kG0_11 cond_1
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += j1*l1*((-1)**(j1 + l1) - 1)*(Ftx*r*(tmax - tmin) + Fxt*L)*(-(i1*i1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + (i1*i1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - 2*i1*k1*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) - 2*i1*k1*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - (k1*k1)*cos(0.5*pi*(L + 2*xb)*(i1 - k1)/L) + (k1*k1)*cos(0.5*pi*(L + 2*xb)*(i1 + k1)/L) - (i1 - k1)**2*cos(0.5*pi*(L + 2*xa)*(i1 + k1)/L) + (i1 + k1)**2*cos(0.5*pi*(L + 2*xa)*(i1 - k1)/L))/(L*r*(i1 + k1)*(2.0*i1 - 2.0*k1)*(j1 - l1)*(j1 + l1)*(tmax - tmin))

                        elif k1 == i1 and l1 != j1 and i1 != 0:
                            # kG0_11 cond_2
                            pass

                        elif k1 != i1 and l1 == j1 and j1 != 0:
                            # kG0_11 cond_3
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += pi*(Ft*(j1*j1)*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) - (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/(r*(tmax - tmin)) - Fx*i1*k1*((-i1 + k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xb*(2*i1 + 2*k1))/L) + (i1 - k1)*cos(0.5*pi*(L*(i1 + k1 - 1) + xa*(2*i1 + 2*k1))/L) + (i1 + k1)*(sin(0.5*pi*(L + 2*xa)*(i1 - k1)/L) - sin(0.5*pi*(L + 2*xb)*(i1 - k1)/L)))/L)/((i1 + k1)*(4.0*i1 - 4.0*k1))

                        elif k1 == i1 and l1 == j1 and i1 != 0 and j1 != 0:
                            # kG0_11 cond_4
                            c += 1
                            kG0r[c] = row+2
                            kG0c[c] = col+2
                            kG0v[c] += -0.125*pi*(-L*(Ft*L*(j1*j1) + Fx*(i1*i1)*r*(-tmax + tmin))*sin(pi*i1*(L + 2*xa)/L) + L*(Ft*L*(j1*j1) + Fx*(i1*i1)*r*(-tmax + tmin))*sin(pi*i1*(L + 2*xb)/L) + 2*pi*i1*(xa - xb)*(Ft*L*(j1*j1) + Fx*(i1*i1)*r*(tmax - tmin)))/((L*L)*i1*r*(tmax - tmin))

                        elif k1 == i1 and l1 != j1 and i1 == 0:
                            # kG0_11 cond_5
                            pass

                        elif k1 != i1 and l1 == j1 and j1 == 0:
                            # kG0_11 cond_6
                            pass

                        elif k1 == i1 and l1 == j1 and i1 == 0 and j1 != 0:
                            # kG0_11 cond_7
                            pass

                        elif k1 == i1 and l1 == j1 and i1 != 0 and j1 == 0:
                            # kG0_11 cond_8
                            pass

                        elif k1 == i1 and l1 == j1 and i1 == 0 and j1 == 0:
                            # kG0_11 cond_9
                            pass

    size = num0 + num1*m1*n1

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkG0_cyl(double Fx, double Ft, double Fxt, double Ftx, double r1,
        double L, double tmin, double tmax, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef double r=r1
    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    fdim = 1*m1*n1*m1*n1//2

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kG0_11
    for i1 in range(m1):
        for j1 in range(n1):
            row = num0 + num1*((j1)*m1 + (i1))
            for k1 in range(m1):
                for l1 in range(n1):
                    col = num0 + num1*((l1)*m1 + (k1))

                    #NOTE symmetry
                    if row > col:
                        continue

                    if k1 != i1 and l1 != j1:
                        # kG0_11 cond_1
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*(Ftx*r*(tmax - tmin) + Fxt*L)/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))

                    elif k1 == i1 and l1 != j1 and i1 != 0:
                        # kG0_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1 and j1 != 0:
                        # kG0_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 != 0:
                        # kG0_11 cond_4
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += 0.25*(pi*pi)*(Ft*L*(j1*j1) + Fx*(i1*i1)*r*(tmax - tmin))/(L*r*(tmax - tmin))

                    elif k1 == i1 and l1 != j1 and i1 == 0:
                        # kG0_11 cond_5
                        pass

                    elif k1 != i1 and l1 == j1 and j1 == 0:
                        # kG0_11 cond_6
                        pass

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 != 0:
                        # kG0_11 cond_7
                        pass

                    elif k1 == i1 and l1 == j1 and i1 != 0 and j1 == 0:
                        # kG0_11 cond_8
                        pass

                    elif k1 == i1 and l1 == j1 and i1 == 0 and j1 == 0:
                        # kG0_11 cond_9
                        pass

    size = num0 + num1*m1*n1

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
