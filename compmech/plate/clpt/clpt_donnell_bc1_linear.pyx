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

cdef int num0 = 2
cdef int num1 = 3
cdef double pi = 3.141592653589793


def fk0(double a, double b, np.ndarray[cDOUBLE, ndim=2] F, int m1, int n1):
    cdef int i1, j1, k1, l1, c, row, col
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    fdim = 5*m1*n1*m1*n1

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

    # k0_00
    c += 1
    k0r[c] = 0
    k0c[c] = 0
    k0v[c] += A11*b/a
    c += 1
    k0r[c] = 0
    k0c[c] = 1
    k0v[c] += A12
    c += 1
    k0r[c] = 1
    k0c[c] = 0
    k0v[c] += A12
    c += 1
    k0r[c] = 1
    k0c[c] = 1
    k0v[c] += A22*a/b

    # k0_01
    for k1 in range(1, m1+1):
        for l1 in range(1, n1+1):
            col = num0 + num1*((l1-1)*m1 + (k1-1))
            c += 1
            k0r[c] = 0
            k0c[c] = col+2
            k0v[c] += (4*B11*(b*b)*(k1*k1) + 4*B12*(a*a)*(l1*l1))*sin(0.5*pi*k1)**2*sin(0.5*pi*l1)**2/((a*a)*b*k1*l1)
            c += 1
            k0r[c] = 1
            k0c[c] = col+2
            k0v[c] += (-2*(-1)**l1 + 2)*(B12*(b*b)*(k1*k1) + B22*(a*a)*(l1*l1))*sin(0.5*pi*k1)**2/(a*(b*b)*k1*l1)

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
                        k0v[c] += A16*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += -i1*j1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12 + A66)/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += -i1*j1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12 + A66)/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += A26*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += (pi*pi)*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*(D16*(b*b)*((i1*i1) + (k1*k1)) + D26*(a*a)*((j1*j1) + (l1*l1)))/((a*a)*(b*b)*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # k0_11 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*(pi*pi)*j1*l1*((-1)**(j1 + l1) - 1)*(3*B16*(b*b)*(i1*i1) + B26*(a*a)*(l1*l1))/(a*(b*b)*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += 0.5*(pi*pi)*j1*l1*((-1)**(j1 + l1) - 1)*(B22*(a*a)*(l1*l1) + (b*b)*(i1*i1)*(B12 + 2*B66))/(a*(b*b)*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += -0.5*(pi*pi)*j1*l1*((-1)**(j1 + l1) - 1)*(3*B16*(b*b)*(i1*i1) + B26*(a*a)*(j1*j1))/(a*(b*b)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -0.5*(pi*pi)*j1*l1*((-1)**(j1 + l1) - 1)*(B22*(a*a)*(j1*j1) + (b*b)*(i1*i1)*(B12 + 2*B66))/(a*(b*b)*((j1*j1) - (l1*l1)))

                    elif k1 != i1 and l1 == j1:
                        # k0_11 cond_3
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*(pi*pi)*i1*k1*((-1)**(i1 + k1) - 1)*(B11*(b*b)*(k1*k1) + (a*a)*(j1*j1)*(B12 + 2*B66))/((a*a)*b*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += 0.5*(pi*pi)*i1*k1*((-1)**(i1 + k1) - 1)*(B16*(b*b)*(k1*k1) + 3*B26*(a*a)*(j1*j1))/((a*a)*b*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += -0.5*(pi*pi)*i1*k1*((-1)**(i1 + k1) - 1)*(B11*(b*b)*(i1*i1) + (a*a)*(j1*j1)*(B12 + 2*B66))/((a*a)*b*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -0.5*(pi*pi)*i1*k1*((-1)**(i1 + k1) - 1)*(B16*(b*b)*(i1*i1) + 3*B26*(a*a)*(j1*j1))/((a*a)*b*((i1*i1) - (k1*k1)))

                    elif k1 == i1 and l1 == j1:
                        # k0_11 cond_4
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += 0.25*(pi*pi)*(A11*(b*b)*(i1*i1) + A66*(a*a)*(j1*j1))/(a*b)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += 0.25*(pi*pi)*(A16*(b*b)*(i1*i1) + A26*(a*a)*(j1*j1))/(a*b)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += 0.25*(pi*pi)*(A16*(b*b)*(i1*i1) + A26*(a*a)*(j1*j1))/(a*b)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += 0.25*(pi*pi)*(A22*(a*a)*(j1*j1) + A66*(b*b)*(i1*i1))/(a*b)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += 0.25*(pi*pi*pi*pi)*(D11*(b*b*b*b)*(i1*i1*i1*i1) + D22*(a*a*a*a)*(j1*j1*j1*j1) + 2*(a*a)*(b*b)*(i1*i1)*(j1*j1)*(D12 + 2*D66))/((a*a*a)*(b*b*b))

    size = num0 + num1*m1*n1

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0edges(int m1, int n1, double a, double b,
             double kphixBot, double kphixTop,
             double kphiyLeft, double kphiyRight):
    cdef int i1, j1, k1, l1, row, col, c, cbkp
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


def fkG0(double Nxx, double Nyy, double Nxy,
         double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
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
                        kG0v[c] += Nxy*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

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
                        kG0v[c] += 0.25*(pi*pi)*(Nxx*(b*b)*(i1*i1) + Nyy*(a*a)*(j1*j1))/(a*b)

    size = num0 + num1*m1*n1

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
