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
                        k0v[c] += -A16*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += -((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(i1*i1)*(l1*l1) + A66*(j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += -((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A12*(j1*j1)*(k1*k1) + A66*(i1*i1)*(l1*l1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += -A26*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1)*(l1*l1) + (j1*j1)*(k1*k1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += (pi*pi)*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*(D16*(b*b)*((i1*i1) + (k1*k1)) + D26*(a*a)*((j1*j1) + (l1*l1)))/((a*a)*(b*b)*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # k0_11 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*(pi*pi)*i1*l1*((-1)**(j1 + l1) - 1)*(B11*(b*b)*(i1*i1) + (a*a)*(B12*(l1*l1) + 2*B66*(j1*j1)))/((a*a)*b*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += -0.5*(pi*pi)*i1*l1*((-1)**(j1 + l1) - 1)*(B16*(b*b)*(i1*i1) + B26*(a*a)*(2*(j1*j1) + (l1*l1)))/((a*a)*b*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 0.5*(pi*pi)*i1*j1*((-1)**(j1 + l1) - 1)*(B11*(b*b)*(i1*i1) + (a*a)*(B12*(j1*j1) + 2*B66*(l1*l1)))/((a*a)*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += 0.5*(pi*pi)*i1*j1*((-1)**(j1 + l1) - 1)*(B16*(b*b)*(i1*i1) + B26*(a*a)*((j1*j1) + 2*(l1*l1)))/((a*a)*b*(j1 - l1)*(j1 + l1))

                    elif k1 != i1 and l1 == j1:
                        # k0_11 cond_3
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*(pi*pi)*j1*k1*((-1)**(i1 + k1) - 1)*(B16*(b*b)*(2*(i1*i1) + (k1*k1)) + B26*(a*a)*(j1*j1))/(a*(b*b)*(-(i1*i1) + (k1*k1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += 0.5*(pi*pi)*j1*k1*((-1)**(i1 + k1) - 1)*(B22*(a*a)*(j1*j1) + (b*b)*(B12*(k1*k1) + 2*B66*(i1*i1)))/(a*(b*b)*(-(i1*i1) + (k1*k1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 0.5*(pi*pi)*i1*j1*((-1)**(i1 + k1) - 1)*(B16*(b*b)*((i1*i1) + 2*(k1*k1)) + B26*(a*a)*(j1*j1))/(a*(b*b)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += 0.5*(pi*pi)*i1*j1*((-1)**(i1 + k1) - 1)*(B22*(a*a)*(j1*j1) + (b*b)*(B12*(i1*i1) + 2*B66*(k1*k1)))/(a*(b*b)*((i1*i1) - (k1*k1)))

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
             double kuBot, double kuTop,
             double kvBot, double kvTop,
             double kphixBot, double kphixTop,
             double kuLeft, double kuRight,
             double kvLeft, double kvRight,
             double kphiyLeft, double kphiyRight):
    cdef int i1, j1, k1, l1, row, col, c, cbkp
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv

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
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*b*((-1)**(i1 + k1)*kuTop + kuBot)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*b*((-1)**(i1 + k1)*kvTop + kvBot)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*b*i1*k1*((-1)**(i1 + k1)*kphixTop + kphixBot)/(a*a)

                    elif k1 == i1 and l1 == j1:
                        # k0edgesBT_11 cond_4
                        c += 1
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*b*(kuBot + kuTop)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*b*(kvBot + kvTop)
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
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*a*((-1)**(j1 + l1)*kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*a*((-1)**(j1 + l1)*kvLeft + kvRight)
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
                        k0edgesr[c] = row+0
                        k0edgesc[c] = col+0
                        k0edgesv[c] += 0.5*a*(kuLeft + kuRight)
                        c += 1
                        k0edgesr[c] = row+1
                        k0edgesc[c] = col+1
                        k0edgesv[c] += 0.5*a*(kvLeft + kvRight)
                        c += 1
                        k0edgesr[c] = row+2
                        k0edgesc[c] = col+2
                        k0edgesv[c] += 0.5*(pi*pi)*a*(j1*j1)*(kphiyLeft + kphiyRight)/(b*b)

    size = num0 + num1*m1*n1

    k0edges = coo_matrix((k0edgesv, (k0edgesr, k0edgesc)), shape=(size, size))

    return k0edges


def fkG0(double Fx, double Fy, double Fxy, double Fyx,
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
                        kG0v[c] += i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*(Fxy*a + Fyx*b)/(a*b*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

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
                        kG0v[c] += 0.25*(pi*pi)*(Fx*b*(i1*i1) + Fy*a*(j1*j1))/(a*b)

    size = num0 + num1*m1*n1

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkA(double beta, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kAr, kAc
    cdef np.ndarray[cDOUBLE, ndim=1] kAv

    fdim = 1*m1*n1*m1*n1

    kAr = np.zeros((fdim,), dtype=INT)
    kAc = np.zeros((fdim,), dtype=INT)
    kAv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kA_11
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
                        # kA_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # kA_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1:
                        # kA_11 cond_3
                        c += 1
                        kAr[c] = row+2
                        kAc[c] = col+2
                        kAv[c] += beta*b*i1*k1*((-1)**(i1 + k1) - 1)/(2.0*(i1*i1) - 2.0*(k1*k1))

                    elif k1 == i1 and l1 == j1:
                        # kA_11 cond_4
                        pass

    size = num0 + num1*m1*n1

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA


def fkM(double mu, double h, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMr, kMc
    cdef np.ndarray[cDOUBLE, ndim=1] kMv

    fdim = 1*m1*n1*m1*n1

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kM_11
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
                        # kM_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # kM_11 cond_2
                        pass

                    elif k1 != i1 and l1 == j1:
                        # kM_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kM_11 cond_4
                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+0
                        kMv[c] += 0.25*a*b*h*mu
                        c += 1
                        kMr[c] = row+1
                        kMc[c] = col+1
                        kMv[c] += 0.25*a*b*h*mu
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+2
                        kMv[c] += 0.25*a*b*h*mu

    size = num0 + num1*m1*n1

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM
