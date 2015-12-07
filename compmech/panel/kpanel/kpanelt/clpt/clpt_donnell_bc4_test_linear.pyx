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
cdef int num1 = 9
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

    fdim = ( 0 )

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

            xa = -L/2. + L*float(section)/s
            xb = -L/2. + L*float(section+1)/s

            r = r1 - sina*((xa+xb)/2. + L/2.)

            # k0_11
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
                                # k0_11 cond_1
                                pass


                            elif k1 == i1 and l1 != j1:
                                # k0_11 cond_2
                                pass


                            elif k1 != i1 and l1 == j1:
                                # k0_11 cond_3
                                pass


                            elif k1 == i1 and l1 == j1:
                                # k0_11 cond_4
                                pass


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

    fdim = 42*m1*n1*m1*n1//2

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

    with nogil:
        # k0_11
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
                            # k0_11 cond_1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += A16*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += i1*j1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A11*(k1*k1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(l1*l1))/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += -i1*j1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12 + A66)/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+7
                            k0v[c] += i1*j1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A16*(k1*k1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(l1*l1))/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += -A16*i1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((j1*j1) + (l1*l1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += -i1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A11*(k1*k1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j1*j1))/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+5
                            k0v[c] += -i1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(l1*l1) + A66*(j1*j1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+6
                            k0v[c] += -i1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A16*(k1*k1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+8
                            k0v[c] += i1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*B12*(L*L)*(l1*l1) + 2*(pi*pi)*B66*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(k1*k1)*r))/(pi*(L*L)*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A11*(i1*i1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(l1*l1))/(L*r*(-(i1*i1) + (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += -A16*j1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1) + (k1*k1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+5
                            k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(l1*l1))/(L*r*(-(i1*i1) + (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+6
                            k0v[c] += -j1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(i1*i1) + A66*(k1*k1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+8
                            k0v[c] += j1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*(l1*l1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*r*(2*(i1*i1) + (k1*k1))))/(pi*L*(r*r)*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+0
                            k0v[c] += -k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A11*(i1*i1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j1*j1))/(L*r*(-(i1*i1) + (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += -A16*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+4
                            k0v[c] += -k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(-(i1*i1) + (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+7
                            k0v[c] += -((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(i1*i1)*(l1*l1) + A66*(j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+0
                            k0v[c] += -i1*j1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12 + A66)/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+3
                            k0v[c] += i1*j1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A16*(k1*k1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(l1*l1))/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+4
                            k0v[c] += A26*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+7
                            k0v[c] += i1*j1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A22*(L*L)*(l1*l1) + A66*(k1*k1)*(r*r)*(tmax - tmin)**2)/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+1
                            k0v[c] += -i1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(j1*j1) + A66*(l1*l1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+2
                            k0v[c] += -i1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A16*(k1*k1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+5
                            k0v[c] += -A26*i1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((j1*j1) + (l1*l1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+6
                            k0v[c] += -i1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A22*(L*L)*(j1*j1) + A66*(k1*k1)*(r*r)*(tmax - tmin)**2)/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+8
                            k0v[c] += i1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*(2*(j1*j1) + (l1*l1)) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(k1*k1)*r))/(pi*(L*L)*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+1
                            k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(l1*l1))/(L*r*(-(i1*i1) + (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+2
                            k0v[c] += -j1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(k1*k1) + A66*(i1*i1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+5
                            k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A22*(L*L)*(l1*l1) + A66*(i1*i1)*(r*r)*(tmax - tmin)**2)/(L*r*(-(i1*i1) + (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+6
                            k0v[c] += -A26*j1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1) + (k1*k1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+8
                            k0v[c] += j1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*B22*(L*L)*(l1*l1) + r*(tmax - tmin)**2*(A22*(L*L) + (pi*pi)*r*(B12*(k1*k1) + 2*B66*(i1*i1))))/(pi*L*(r*r)*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+0
                            k0v[c] += -k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(-(i1*i1) + (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+3
                            k0v[c] += -((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(j1*j1)*(k1*k1) + A66*(i1*i1)*(l1*l1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+4
                            k0v[c] += -k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A22*(L*L)*(j1*j1) + A66*(i1*i1)*(r*r)*(tmax - tmin)**2)/(L*r*(-(i1*i1) + (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+7
                            k0v[c] += -A26*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/(((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+1
                            k0v[c] += i1*j1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*B12*(L*L)*(j1*j1) + 2*(pi*pi)*B66*(L*L)*(l1*l1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r))/(pi*(L*L)*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+2
                            k0v[c] += i1*j1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*B16*((i1*i1) + 2*(k1*k1)) + (L*L)*(A26*r + (pi*pi)*B26*(j1*j1)/(tmax - tmin)**2)/(r*r))/(pi*L*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+5
                            k0v[c] += i1*j1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*((j1*j1) + 2*(l1*l1)) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(i1*i1)*r))/(pi*(L*L)*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+6
                            k0v[c] += i1*j1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*B12*(i1*i1) + 2*(pi*pi)*B66*(k1*k1) + (L*L)*(A22*r + (pi*pi)*B22*(j1*j1)/(tmax - tmin)**2)/(r*r))/(pi*L*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+8
                            k0v[c] += i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*((pi*pi)*D26*(L*L)*((j1*j1) + (l1*l1)) + r*(tmax - tmin)**2*(2*B26*(L*L) + (pi*pi)*D16*r*((i1*i1) + (k1*k1))))/((L*L)*(r*r)*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin)**2)

                        elif k1 == i1 and l1 != j1:
                            # k0_11 cond_2
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(A11*(i1*i1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(l1*l1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += -pi*A16*i1*j1*l1*((-1)**(j1 + l1) - 1)/((j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+5
                            k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(l1*l1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+6
                            k0v[c] += -pi*i1*j1*l1*((-1)**(j1 + l1) - 1)*(A12 + A66)/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+8
                            k0v[c] += 0.5*j1*l1*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*(l1*l1) + r*(tmax - tmin)**2*(A26*(L*L) + 3*(pi*pi)*B16*(i1*i1)*r))/(L*(r*r)*(j1 - l1)*(j1 + l1)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(A11*(i1*i1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j1*j1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += -pi*A16*i1*((-1)**(j1 + l1) - 1)*((j1*j1) + (l1*l1))/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+7
                            k0v[c] += -pi*i1*((-1)**(j1 + l1) - 1)*(A12*(l1*l1) + A66*(j1*j1))/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += pi*A16*i1*j1*l1*((-1)**(j1 + l1) - 1)/((j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+3
                            k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(A11*(i1*i1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(l1*l1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+4
                            k0v[c] += pi*i1*j1*l1*((-1)**(j1 + l1) - 1)*(A12 + A66)/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+7
                            k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(l1*l1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+1
                            k0v[c] += pi*A16*i1*((-1)**(j1 + l1) - 1)*((j1*j1) + (l1*l1))/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(A11*(i1*i1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j1*j1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+5
                            k0v[c] += pi*i1*((-1)**(j1 + l1) - 1)*(A12*(l1*l1) + A66*(j1*j1))/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+6
                            k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+8
                            k0v[c] += -0.5*i1*l1*((-1)**(j1 + l1) - 1)*((pi*pi)*B12*(L*L)*(l1*l1) + 2*(pi*pi)*B66*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r))/((L*L)*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+1
                            k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(l1*l1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+2
                            k0v[c] += -pi*i1*j1*l1*((-1)**(j1 + l1) - 1)*(A12 + A66)/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+5
                            k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(A22*(L*L)*(l1*l1) + A66*(i1*i1)*(r*r)*(tmax - tmin)**2)/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+6
                            k0v[c] += -pi*A26*i1*j1*l1*((-1)**(j1 + l1) - 1)/((j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+8
                            k0v[c] += 0.5*j1*l1*((-1)**(j1 + l1) - 1)*((pi*pi)*B22*(L*L)*(l1*l1) + r*(tmax - tmin)**2*(A22*(L*L) + (pi*pi)*(i1*i1)*r*(B12 + 2*B66)))/(L*(r*r)*(j1 - l1)*(j1 + l1)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+0
                            k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+3
                            k0v[c] += -pi*i1*((-1)**(j1 + l1) - 1)*(A12*(j1*j1) + A66*(l1*l1))/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+4
                            k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(A22*(L*L)*(j1*j1) + A66*(i1*i1)*(r*r)*(tmax - tmin)**2)/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+7
                            k0v[c] += -pi*A26*i1*((-1)**(j1 + l1) - 1)*((j1*j1) + (l1*l1))/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+0
                            k0v[c] += pi*i1*j1*l1*((-1)**(j1 + l1) - 1)*(A12 + A66)/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+3
                            k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(l1*l1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+4
                            k0v[c] += pi*A26*i1*j1*l1*((-1)**(j1 + l1) - 1)/((j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+7
                            k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(A22*(L*L)*(l1*l1) + A66*(i1*i1)*(r*r)*(tmax - tmin)**2)/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+1
                            k0v[c] += pi*i1*((-1)**(j1 + l1) - 1)*(A12*(j1*j1) + A66*(l1*l1))/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+2
                            k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+5
                            k0v[c] += pi*A26*i1*((-1)**(j1 + l1) - 1)*((j1*j1) + (l1*l1))/((j1 + l1)*(2.0*j1 - 2.0*l1))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+6
                            k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(A22*(L*L)*(j1*j1) + A66*(i1*i1)*(r*r)*(tmax - tmin)**2)/(L*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+8
                            k0v[c] += -0.5*i1*l1*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*(2*(j1*j1) + (l1*l1)) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(i1*i1)*r))/((L*L)*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+0
                            k0v[c] += -0.5*j1*l1*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + 3*(pi*pi)*B16*(i1*i1)*r))/(L*(r*r)*(j1 - l1)*(j1 + l1)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+3
                            k0v[c] += 0.5*i1*j1*((-1)**(j1 + l1) - 1)*((pi*pi)*B12*(L*L)*(j1*j1) + 2*(pi*pi)*B66*(L*L)*(l1*l1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r))/((L*L)*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+4
                            k0v[c] += 0.5*j1*l1*((-1)**(j1 + l1) - 1)*(-(pi*pi)*B12*(i1*i1) - 2*(pi*pi)*B66*(i1*i1) + (L*L)*(-A22*r - (pi*pi)*B22*(j1*j1)/(tmax - tmin)**2)/(r*r))/(L*(j1 - l1)*(j1 + l1))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+7
                            k0v[c] += 0.5*i1*j1*((-1)**(j1 + l1) - 1)*((pi*pi)*B26*(L*L)*((j1*j1) + 2*(l1*l1)) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(i1*i1)*r))/((L*L)*r*(j1 - l1)*(j1 + l1)*(tmax - tmin))

                        elif k1 != i1 and l1 == j1:
                            # k0_11 cond_3
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+1
                            k0v[c] += -pi*A16*i1*j1*k1*((-1)**(i1 + k1) - 1)/((i1*i1) - (k1*k1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+2
                            k0v[c] += -pi*i1*((-1)**(i1 + k1) - 1)*(A11*(k1*k1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j1*j1))/(L*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+5
                            k0v[c] += -pi*i1*j1*k1*((-1)**(i1 + k1) - 1)*(A12 + A66)/(2.0*(i1*i1) - 2.0*(k1*k1))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+6
                            k0v[c] += -pi*i1*((-1)**(i1 + k1) - 1)*(A16*(k1*k1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+8
                            k0v[c] += i1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B12*(L*L)*(j1*j1) + 2*(pi*pi)*B66*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(k1*k1)*r))/((L*L)*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+0
                            k0v[c] += pi*A16*i1*j1*k1*((-1)**(i1 + k1) - 1)/((i1*i1) - (k1*k1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+3
                            k0v[c] += -pi*i1*((-1)**(i1 + k1) - 1)*(A11*(k1*k1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j1*j1))/(L*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+4
                            k0v[c] += pi*i1*j1*k1*((-1)**(i1 + k1) - 1)*(A12 + A66)/(2.0*(i1*i1) - 2.0*(k1*k1))
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+7
                            k0v[c] += -pi*i1*((-1)**(i1 + k1) - 1)*(A16*(k1*k1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+0
                            k0v[c] += -pi*k1*((-1)**(i1 + k1) - 1)*(A11*(i1*i1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j1*j1))/(L*r*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+3
                            k0v[c] += -pi*A16*j1*((-1)**(i1 + k1) - 1)*((i1*i1) + (k1*k1))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+4
                            k0v[c] += -pi*k1*((-1)**(i1 + k1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+7
                            k0v[c] += -pi*j1*((-1)**(i1 + k1) - 1)*(A12*(i1*i1) + A66*(k1*k1))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+1
                            k0v[c] += -pi*k1*((-1)**(i1 + k1) - 1)*(A11*(i1*i1)*(r*r)*(tmax - tmin)**2 + A66*(L*L)*(j1*j1))/(L*r*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+2
                            k0v[c] += pi*A16*j1*((-1)**(i1 + k1) - 1)*((i1*i1) + (k1*k1))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+5
                            k0v[c] += -pi*k1*((-1)**(i1 + k1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+6
                            k0v[c] += pi*j1*((-1)**(i1 + k1) - 1)*(A12*(i1*i1) + A66*(k1*k1))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+8
                            k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*r*(2*(i1*i1) + (k1*k1))))/(L*(r*r)*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+1
                            k0v[c] += -pi*i1*j1*k1*((-1)**(i1 + k1) - 1)*(A12 + A66)/(2.0*(i1*i1) - 2.0*(k1*k1))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+2
                            k0v[c] += -pi*i1*((-1)**(i1 + k1) - 1)*(A16*(k1*k1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+5
                            k0v[c] += -pi*A26*i1*j1*k1*((-1)**(i1 + k1) - 1)/((i1*i1) - (k1*k1))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+6
                            k0v[c] += -pi*i1*((-1)**(i1 + k1) - 1)*(A22*(L*L)*(j1*j1) + A66*(k1*k1)*(r*r)*(tmax - tmin)**2)/(L*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+8
                            k0v[c] += i1*k1*((-1)**(i1 + k1) - 1)*(3*(pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(k1*k1)*r))/((L*L)*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+0
                            k0v[c] += pi*i1*j1*k1*((-1)**(i1 + k1) - 1)*(A12 + A66)/(2.0*(i1*i1) - 2.0*(k1*k1))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+3
                            k0v[c] += -pi*i1*((-1)**(i1 + k1) - 1)*(A16*(k1*k1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+4
                            k0v[c] += pi*A26*i1*j1*k1*((-1)**(i1 + k1) - 1)/((i1*i1) - (k1*k1))
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+7
                            k0v[c] += -pi*i1*((-1)**(i1 + k1) - 1)*(A22*(L*L)*(j1*j1) + A66*(k1*k1)*(r*r)*(tmax - tmin)**2)/(L*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+0
                            k0v[c] += -pi*k1*((-1)**(i1 + k1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+3
                            k0v[c] += -pi*j1*((-1)**(i1 + k1) - 1)*(A12*(k1*k1) + A66*(i1*i1))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+4
                            k0v[c] += -pi*k1*((-1)**(i1 + k1) - 1)*(A22*(L*L)*(j1*j1) + A66*(i1*i1)*(r*r)*(tmax - tmin)**2)/(L*r*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+7
                            k0v[c] += -pi*A26*j1*((-1)**(i1 + k1) - 1)*((i1*i1) + (k1*k1))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+1
                            k0v[c] += -pi*k1*((-1)**(i1 + k1) - 1)*(A16*(i1*i1)*(r*r)*(tmax - tmin)**2 + A26*(L*L)*(j1*j1))/(L*r*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+2
                            k0v[c] += pi*j1*((-1)**(i1 + k1) - 1)*(A12*(k1*k1) + A66*(i1*i1))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+5
                            k0v[c] += -pi*k1*((-1)**(i1 + k1) - 1)*(A22*(L*L)*(j1*j1) + A66*(i1*i1)*(r*r)*(tmax - tmin)**2)/(L*r*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+6
                            k0v[c] += pi*A26*j1*((-1)**(i1 + k1) - 1)*((i1*i1) + (k1*k1))/((i1 + k1)*(2.0*i1 - 2.0*k1))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+8
                            k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B22*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A22*(L*L) + (pi*pi)*r*(B12*(k1*k1) + 2*B66*(i1*i1))))/(L*(r*r)*(-2.0*(i1*i1) + 2.0*(k1*k1))*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+0
                            k0v[c] += -i1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B12*(L*L)*(j1*j1) + 2*(pi*pi)*B66*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r))/((L*L)*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+3
                            k0v[c] += i1*j1*((-1)**(i1 + k1) - 1)*((pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*r*((i1*i1) + 2*(k1*k1))))/(L*(r*r)*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+4
                            k0v[c] += -i1*k1*((-1)**(i1 + k1) - 1)*(3*(pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(i1*i1)*r))/((L*L)*r*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+7
                            k0v[c] += i1*j1*((-1)**(i1 + k1) - 1)*((pi*pi)*B22*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A22*(L*L) + (pi*pi)*r*(B12*(i1*i1) + 2*B66*(k1*k1))))/(L*(r*r)*(2.0*(i1*i1) - 2.0*(k1*k1))*(tmax - tmin)**2)

                        elif k1 == i1 and l1 == j1:
                            # k0_11 cond_4
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+0
                            k0v[c] += 0.25*(pi*pi)*A11*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A66*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+3
                            k0v[c] += -0.5*(pi*pi)*A16*i1*j1
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+4
                            k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+0
                            k0c[c] = col+7
                            k0v[c] += -0.25*(pi*pi)*i1*j1*(A12 + A66)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+1
                            k0v[c] += 0.25*(pi*pi)*A11*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A66*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+2
                            k0v[c] += 0.5*(pi*pi)*A16*i1*j1
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+5
                            k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+6
                            k0v[c] += 0.25*(pi*pi)*i1*j1*(A12 + A66)
                            c += 1
                            k0r[c] = row+1
                            k0c[c] = col+8
                            k0v[c] += -0.25*pi*j1*((pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + 3*(pi*pi)*B16*(i1*i1)*r))/(L*(r*r)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+1
                            k0v[c] += 0.5*(pi*pi)*A16*i1*j1
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+2
                            k0v[c] += 0.25*(pi*pi)*A11*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A66*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+5
                            k0v[c] += 0.25*(pi*pi)*i1*j1*(A12 + A66)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+6
                            k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+2
                            k0c[c] = col+8
                            k0v[c] += -0.25*pi*i1*((pi*pi)*B12*(L*L)*(j1*j1) + 2*(pi*pi)*B66*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r))/((L*L)*r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+0
                            k0v[c] += -0.5*(pi*pi)*A16*i1*j1
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+3
                            k0v[c] += 0.25*(pi*pi)*A11*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A66*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+4
                            k0v[c] += -0.25*(pi*pi)*i1*j1*(A12 + A66)
                            c += 1
                            k0r[c] = row+3
                            k0c[c] = col+7
                            k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+0
                            k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+3
                            k0v[c] += -0.25*(pi*pi)*i1*j1*(A12 + A66)
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+4
                            k0v[c] += (pi*pi)*A22*L*(j1*j1)/(4*r*tmax - 4*r*tmin) + 0.25*(pi*pi)*A66*(i1*i1)*r*(tmax - tmin)/L
                            c += 1
                            k0r[c] = row+4
                            k0c[c] = col+7
                            k0v[c] += -0.5*(pi*pi)*A26*i1*j1
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+1
                            k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+2
                            k0v[c] += 0.25*(pi*pi)*i1*j1*(A12 + A66)
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+5
                            k0v[c] += (pi*pi)*A22*L*(j1*j1)/(4*r*tmax - 4*r*tmin) + 0.25*(pi*pi)*A66*(i1*i1)*r*(tmax - tmin)/L
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+6
                            k0v[c] += 0.5*(pi*pi)*A26*i1*j1
                            c += 1
                            k0r[c] = row+5
                            k0c[c] = col+8
                            k0v[c] += -0.25*pi*j1*((pi*pi)*B22*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A22*(L*L) + (pi*pi)*(i1*i1)*r*(B12 + 2*B66)))/(L*(r*r)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+1
                            k0v[c] += 0.25*(pi*pi)*i1*j1*(A12 + A66)
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+2
                            k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+5
                            k0v[c] += 0.5*(pi*pi)*A26*i1*j1
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+6
                            k0v[c] += (pi*pi)*A22*L*(j1*j1)/(4*r*tmax - 4*r*tmin) + 0.25*(pi*pi)*A66*(i1*i1)*r*(tmax - tmin)/L
                            c += 1
                            k0r[c] = row+6
                            k0c[c] = col+8
                            k0v[c] += -0.25*pi*i1*(3*(pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(i1*i1)*r))/((L*L)*r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+0
                            k0v[c] += -0.25*(pi*pi)*i1*j1*(A12 + A66)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+3
                            k0v[c] += 0.25*(pi*pi)*A16*(i1*i1)*r*(tmax - tmin)/L + (pi*pi)*A26*L*(j1*j1)/(4*r*tmax - 4*r*tmin)
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+4
                            k0v[c] += -0.5*(pi*pi)*A26*i1*j1
                            c += 1
                            k0r[c] = row+7
                            k0c[c] = col+7
                            k0v[c] += (pi*pi)*A22*L*(j1*j1)/(4*r*tmax - 4*r*tmin) + 0.25*(pi*pi)*A66*(i1*i1)*r*(tmax - tmin)/L
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+1
                            k0v[c] += -0.25*pi*j1*((pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + 3*(pi*pi)*B16*(i1*i1)*r))/(L*(r*r)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+2
                            k0v[c] += -0.25*pi*i1*((pi*pi)*B12*(L*L)*(j1*j1) + 2*(pi*pi)*B66*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A12*(L*L) + (pi*pi)*B11*(i1*i1)*r))/((L*L)*r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+5
                            k0v[c] += -0.25*pi*j1*((pi*pi)*B22*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A22*(L*L) + (pi*pi)*(i1*i1)*r*(B12 + 2*B66)))/(L*(r*r)*(tmax - tmin)**2)
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+6
                            k0v[c] += -0.25*pi*i1*(3*(pi*pi)*B26*(L*L)*(j1*j1) + r*(tmax - tmin)**2*(A26*(L*L) + (pi*pi)*B16*(i1*i1)*r))/((L*L)*r*(tmax - tmin))
                            c += 1
                            k0r[c] = row+8
                            k0c[c] = col+8
                            k0v[c] += 0.25*(A22*(L*L*L*L)*(tmax - tmin)**4/r + 2*(pi*pi)*B12*(L*L)*(i1*i1)*(tmax - tmin)**4 + 2*(pi*pi)*B22*(L*L*L*L)*(j1*j1)*(tmax - tmin)**2/(r*r) + (pi*pi*pi*pi)*D11*(i1*i1*i1*i1)*r*(tmax - tmin)**4 + 2*(pi*pi*pi*pi)*D12*(L*L)*(i1*i1)*(j1*j1)*(tmax - tmin)**2/r + (pi*pi*pi*pi)*D22*(L*L*L*L)*(j1*j1*j1*j1)/(r*r*r) + 4*(pi*pi*pi*pi)*D66*(L*L)*(i1*i1)*(j1*j1)*(tmax - tmin)**2/r)/((L*L*L)*(tmax - tmin)**3)

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

    fdim = 3*m1*n1*m1*n1 + 3*m1*n1*m1*n1

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
                        pass

                    elif k1 == i1 and l1 == j1:
                        # k0edgesBT_11 cond_4
                        pass

    cbkp = c

    with nogil:
        for section in range(s):
            c = cbkp

            xa = -L/2. + L*float(section)/s
            xb = -L/2. + L*float(section+1)/s

            r = r1 - sina*((xa+xb)/2. + L/2.)

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
                                pass

                            elif k1 != i1 and l1 == j1:
                                # k0edgesLR_11 cond_3
                                pass

                            elif k1 == i1 and l1 == j1:
                                # k0edgesLR_11 cond_4
                                pass

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

    fdim = 5*m1*n1*m1*n1//2 + 5*m1*n1*m1*n1//2

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
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+3
                        k0edgesv[c] += -j1*((-1)**(j1 + l1) - 1)*(tmax - tmin)*((-1)**(i1 + k1)*kuTop*r2 + kuBot*r1)/(pi*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+2
                        k0edgesv[c] += l1*((-1)**(j1 + l1) - 1)*(tmax - tmin)*((-1)**(i1 + k1)*kuTop*r2 + kuBot*r1)/(pi*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0edgesr[c] = row+6
                        k0edgesc[c] = col+7
                        k0edgesv[c] += -j1*((-1)**(j1 + l1) - 1)*(tmax - tmin)*((-1)**(i1 + k1)*kvTop*r2 + kvBot*r1)/(pi*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+6
                        k0edgesv[c] += l1*((-1)**(j1 + l1) - 1)*(tmax - tmin)*((-1)**(i1 + k1)*kvTop*r2 + kvBot*r1)/(pi*(j1 - l1)*(j1 + l1))

                    elif k1 == i1 and l1 != j1:
                        # k0edgesBT_11 cond_2
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+3
                        k0edgesv[c] += -j1*((-1)**(j1 + l1) - 1)*(tmax - tmin)*(kuBot*r1 + kuTop*r2)/(pi*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+2
                        k0edgesv[c] += l1*((-1)**(j1 + l1) - 1)*(tmax - tmin)*(kuBot*r1 + kuTop*r2)/(pi*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0edgesr[c] = row+6
                        k0edgesc[c] = col+7
                        k0edgesv[c] += -j1*((-1)**(j1 + l1) - 1)*(tmax - tmin)*(kvBot*r1 + kvTop*r2)/(pi*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+6
                        k0edgesv[c] += l1*((-1)**(j1 + l1) - 1)*(tmax - tmin)*(kvBot*r1 + kvTop*r2)/(pi*(j1 - l1)*(j1 + l1))

                    elif k1 != i1 and l1 == j1:
                        # k0edgesBT_11 cond_3
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i1 + k1)*kuTop*r2 + kuBot*r1)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i1 + k1)*kuTop*r2 + kuBot*r1)
                        c += 1
                        k0edgesr[c] = row+6
                        k0edgesc[c] = col+6
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i1 + k1)*kvTop*r2 + kvBot*r1)
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+7
                        k0edgesv[c] += 0.5*(tmax - tmin)*((-1)**(i1 + k1)*kvTop*r2 + kvBot*r1)
                        c += 1
                        k0edgesr[c] = row+8
                        k0edgesc[c] = col+8
                        k0edgesv[c] += 0.5*(pi*pi)*i1*k1*(tmax - tmin)*((-1)**(i1 + k1)*kphixTop*r2 + kphixBot*r1)/(L*L)

                    elif k1 == i1 and l1 == j1:
                        # k0edgesBT_11 cond_4
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kuBot*r1 + kuTop*r2)
                        c += 1
                        k0edgesr[c] = row+6
                        k0edgesc[c] = col+6
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kvBot*r1 + kvTop*r2)
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+7
                        k0edgesv[c] += 0.5*(tmax - tmin)*(kvBot*r1 + kvTop*r2)
                        c += 1
                        k0edgesr[c] = row+8
                        k0edgesc[c] = col+8
                        k0edgesv[c] += 0.5*(pi*pi)*(i1*i1)*(tmax - tmin)*(kphixBot*r1 + kphixTop*r2)/(L*L)

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
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+3
                        k0edgesv[c] += -L*i1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1)*kuLeft + kuRight)/(pi*((i1*i1) - (k1*k1)))
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+1
                        k0edgesv[c] += -L*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1)*kuLeft + kuRight)/(pi*(-(i1*i1) + (k1*k1)))
                        c += 1
                        k0edgesr[c] = row+5
                        k0edgesc[c] = col+7
                        k0edgesv[c] += -L*i1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1)*kvLeft + kvRight)/(pi*((i1*i1) - (k1*k1)))
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+5
                        k0edgesv[c] += -L*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1)*kvLeft + kvRight)/(pi*(-(i1*i1) + (k1*k1)))

                    elif k1 == i1 and l1 != j1:
                        # k0edgesLR_11 cond_2
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*L*((-1)**(j1 + l1)*kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += 0.5*L*((-1)**(j1 + l1)*kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+5
                        k0edgesc[c] = col+5
                        k0edgesv[c] += 0.5*L*((-1)**(j1 + l1)*kvLeft + kvRight)
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+7
                        k0edgesv[c] += 0.5*L*((-1)**(j1 + l1)*kvLeft + kvRight)
                        c += 1
                        k0edgesr[c] = row+8
                        k0edgesc[c] = col+8
                        k0edgesv[c] += 0.5*(pi*pi)*L*j1*l1*((-1)**(j1 + l1)*kphitLeft + kphitRight)/((r*r)*(tmax - tmin)**2)

                    elif k1 != i1 and l1 == j1:
                        # k0edgesLR_11 cond_3
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+3
                        k0edgesv[c] += -L*i1*((-1)**(i1 + k1) - 1)*(kuLeft + kuRight)/(pi*((i1*i1) - (k1*k1)))
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+1
                        k0edgesv[c] += -L*k1*((-1)**(i1 + k1) - 1)*(kuLeft + kuRight)/(pi*(-(i1*i1) + (k1*k1)))
                        c += 1
                        k0edgesr[c] = row+5
                        k0edgesc[c] = col+7
                        k0edgesv[c] += -L*i1*((-1)**(i1 + k1) - 1)*(kvLeft + kvRight)/(pi*((i1*i1) - (k1*k1)))
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+5
                        k0edgesv[c] += -L*k1*((-1)**(i1 + k1) - 1)*(kvLeft + kvRight)/(pi*(-(i1*i1) + (k1*k1)))

                    elif k1 == i1 and l1 == j1:
                        # k0edgesLR_11 cond_4
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*L*(kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+3
                        k0edgesc[c] = col+3
                        k0edgesv[c] += 0.5*L*(kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+5
                        k0edgesc[c] = col+5
                        k0edgesv[c] += 0.5*L*(kvLeft + kvRight)
                        c += 1
                        k0edgesr[c] = row+7
                        k0edgesc[c] = col+7
                        k0edgesv[c] += 0.5*L*(kvLeft + kvRight)
                        c += 1
                        k0edgesr[c] = row+8
                        k0edgesc[c] = col+8
                        k0edgesv[c] += 0.5*(pi*pi)*L*(j1*j1)*(kphitLeft + kphitRight)/((r*r)*(tmax - tmin)**2)

    size = num0 + num1*m1*n1

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fx, double Ft, double Fxt, double Ftx, double r1, double L,
        double tmin, double tmax, int m1, int n1, double alpharad, int s):
    cdef int i1, j1, k1, l1, c, row, col, section
    cdef double xa, xb, r, sina

    cdef np.ndarray[cINT, ndim=1] kG0r, kG0c
    cdef np.ndarray[cDOUBLE, ndim=1] kG0v

    fdim = 1*m1*n1*m1*n1

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

                            if k1 == i1 and l1 == j1:
                                # kG0_11 cond_1
                                pass

                            elif k1 != i1 and l1 == j1:
                                # kG0_11 cond_2
                                pass

                            elif k1 != i1 and l1 != j1:
                                # kG0_11 cond_3
                                pass

                            elif k1 == i1 and l1 != j1:
                                # kG0_11 cond_4
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

    fdim = 1*m1*n1*m1*n1

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

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
                        # kG0_11 cond_1
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*(Ftx*r*(tmax - tmin) + Fxt*L)/(L*r*((i1*i1) - (k1*k1))*(j1 - l1)*(j1 + l1)*(tmax - tmin))

                    elif k1 == i1 and l1 != j1:
                        # kG0_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1:
                        # kG0_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kG0_11 cond_4
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += 0.25*(pi*pi)*(Ft*L*(j1*j1) + Fx*(i1*i1)*r*(tmax - tmin))/(L*r*(tmax - tmin))

    size = num0 + num1*m1*n1

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
