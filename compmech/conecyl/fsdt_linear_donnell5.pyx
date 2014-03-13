#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
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
    double cos(double t)
    double sin(double t)
    double log(double t)

cdef int num0 = 3
cdef int num1 = 6
cdef int num2 = 12
cdef double pi = 3.141592653589793

def fk0(double alpharad, double r2, double L, np.ndarray[cDOUBLE, ndim=2] F,
           int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double r, sina, cosa
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    sina = sin(alpharad)
    cosa = cos(alpharad)

    # sparse parameters
    k11_cond_1 = 36
    k11_cond_2 = 36
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 124
    k22_cond_2 = 144
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 9 + 2*18*m1 + k11_num + k22_num

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

        for i1 in range(1, m1+1):
            col = (i1-1)*num1 + num0
            row = col

            # k0_01

            # k0_10 (using symmetry)

            for k1 in range(1, m1+1):
                col = (k1-1)*num1 + num0
                if k1==i1:
                    # k0_11 cond_1

                elif k1!=i1:
                    # k0_11 cond_2

        for i2 in range(1, m2+1):
            for j2 in range(1, n2+1):
                row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
                for k2 in range(1, m2+1):
                    for l2 in range(1, n2+1):
                        col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                        if k2==i2 and l2==j2:
                            # k0_22 cond_1

                        elif k2!=i2 and l2==j2:
                            # k0_22 cond_2


    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(num0 + num1*m1 + num2*m2*n2,
                                              num0 + num1*m1 + num2*m2*n2))

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
    k11_cond_1 = 20
    k11_cond_2 = 16
    k11_num = k11_cond_1*m1 + k11_cond_2*(m1-1)*m1
    k22_cond_1 = 72
    k22_cond_2 = 72
    k22_cond_3 = 0
    k22_cond_4 = 0
    k22_num = k22_cond_1*m2*n2 + k22_cond_2*(m2-1)*m2*n2 \
            + k22_cond_3*(m2-1)*m2*(n2-1)*n2 + k22_cond_4*m2*(n2-1)*n2

    fdim = 9 + 2*9*m1 + 0*m2*n2 + k11_num + k22_num

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

    for i1 in range(1, m1+1):
        col = (i1-1)*num1 + num0
        row = col

        # k0_01

        # k0_10 (using symmetry)

        for k1 in range(1, m1+1):
            col = (k1-1)*num1 + num0
            if k1==i1:
                # k0_11 cond_1

            elif k1!=i1:
                # k0_11 cond_2

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                    if k2==i2 and l2==j2:
                        # k0_22 cond_1

                    elif k2!=i2 and l2==j2:
                        # k0_22 cond_2


    size = num0 + num1*m1 + num2*m2*n2

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0edges(int m1, int m2, int n2, double r1, double r2,
             double kuBot, double kuTop, double kphixBot, double kphixTop):
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

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        for k1 in range(1, m1+1):
            col = (k1-1)*num1 + num0
            if k1==i1:
                # k0edges_11 cond_1

            elif k1!=i1:
                # k0edges_11 cond_2

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                    if k2==i2 and l2==j2:
                        # k0edges_22 cond_1

                    elif k2!=i2 and l2==j2:
                        # k0edges_22 cond_2


    size = num0 + num1*m1 + num2*m2*n2

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fc, double P, double T, double r2, double alpharad, double L,
        int m1, int m2, int n2, int s):
    cdef int i1, k1, i2, j2, k2, l2, c, row, col, section
    cdef double sina, cosa, xa, xb
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

    for section in range(s):
        c = -1

        xa = L*float(section)/s
        xb = L*float(section+1)/s

        r = r2 + sina*((xa+xb)/2.)

        for i1 in range(1, m1+1):
            row = (i1-1)*num1 + num0
            for k1 in range(1, m1+1):
                col = (k1-1)*num1 + num0
                if k1==i1:
                    # kG0_11 cond_1

                elif k1!=i1:
                    # kG0_11 cond_2

        for i2 in range(1, m2+1):
            for j2 in range(1, n2+1):
                row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
                for k2 in range(1, m2+1):
                    for l2 in range(1, n2+1):
                        col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                        if k2==i2 and l2==j2:
                            # kG0_22 cond_1

                        elif k2!=i2 and l2==j2:
                            # kG0_22 cond_2


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

    for i1 in range(1, m1+1):
        row = (i1-1)*num1 + num0
        for k1 in range(1, m1+1):
            col = (k1-1)*num1 + num0
            if k1==i1:
                # kG0_11 cond_1

    for i2 in range(1, m2+1):
        for j2 in range(1, n2+1):
            row = (i2-1)*num2 + (j2-1)*num2*m2 + num0 + num1*m1
            for k2 in range(1, m2+1):
                for l2 in range(1, n2+1):
                    col = (k2-1)*num2 + (l2-1)*num2*m2 + num0 + num1*m1
                    if k2==i2 and l2==j2:
                        # kG0_22 cond_1

                    elif k2!=i2 and l2==j2:
                        # kG0_22 cond_2


    size = num0 + num1*m1 + num2*m2*n2

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
