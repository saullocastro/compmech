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


def fk0(double a, double b, double r, np.ndarray[cDOUBLE, ndim=2] F,
        int m1, int n1):
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
                        k0v[c] += i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)*((pi*pi)*D16*(b*b)*r*((i1*i1) + (k1*k1)) + (a*a)*(2*B26*(b*b) + (pi*pi)*D26*r*((j1*j1) + (l1*l1))))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # k0_11 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*j1*l1*((-1)**(j1 + l1) - 1)*(A26*(a*a)*(b*b) + (pi*pi)*r*(3*B16*(b*b)*(i1*i1) + B26*(a*a)*(l1*l1)))/(a*(b*b)*r*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += 0.5*j1*l1*((-1)**(j1 + l1) - 1)*((a*a)*(A22*(b*b) + (pi*pi)*B22*(l1*l1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12 + 2*B66))/(a*(b*b)*r*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += -0.5*j1*l1*((-1)**(j1 + l1) - 1)*(3*(pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + (pi*pi)*B26*(j1*j1)*r))/(a*(b*b)*r*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -0.5*j1*l1*((-1)**(j1 + l1) - 1)*((a*a)*(A22*(b*b) + (pi*pi)*B22*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12 + 2*B66))/(a*(b*b)*r*((j1*j1) - (l1*l1)))

                    elif k1 != i1 and l1 == j1:
                        # k0_11 cond_3
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*i1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B11*(b*b)*(k1*k1)*r + (a*a)*(A12*(b*b) + (pi*pi)*(j1*j1)*r*(B12 + 2*B66)))/((a*a)*b*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += 0.5*i1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B16*(b*b)*(k1*k1)*r + (a*a)*(A26*(b*b) + 3*(pi*pi)*B26*(j1*j1)*r))/((a*a)*b*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += -0.5*i1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B11*(b*b)*(i1*i1)*r + (a*a)*(A12*(b*b) + (pi*pi)*(j1*j1)*r*(B12 + 2*B66)))/((a*a)*b*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -0.5*i1*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(A26*(b*b) + 3*(pi*pi)*B26*(j1*j1)*r))/((a*a)*b*r*((i1*i1) - (k1*k1)))

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
                        k0v[c] += 0.25*((pi*pi*pi*pi)*D11*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22*(b*b*b*b) + (pi*pi)*(j1*j1)*r*(2*B22*(b*b) + (pi*pi)*D22*(j1*j1)*r)) + 2*(pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(B12*(b*b) + (pi*pi)*(j1*j1)*r*(D12 + 2*D66)))/((a*a*a)*(b*b*b)*(r*r))

    size = num0 + num1*m1*n1

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


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


def fk0sb(double ys, double bb, double a, double b, double r,
          int m1, int n1,
          np.ndarray[cDOUBLE, ndim=2] Fsb):
    cdef int i1, j1, k1, l1, row, col, c
    cdef double A11sb, A12sb, A16sb, A22sb, A26sb, A66sb
    cdef double B11sb, B12sb, B16sb, B22sb, B26sb, B66sb
    cdef double D11sb, D12sb, D16sb, D22sb, D26sb, D66sb
    cdef np.ndarray[cINT, ndim=1] k0sbr, k0sbc
    cdef np.ndarray[cDOUBLE, ndim=1] k0sbv

    fdim = 9*m1*n1*m1*n1

    k0sbr = np.zeros((fdim,), dtype=INT)
    k0sbc = np.zeros((fdim,), dtype=INT)
    k0sbv = np.zeros((fdim,), dtype=DOUBLE)

    A11sb = Fsb[0,0]
    A12sb = Fsb[0,1]
    A16sb = Fsb[0,2]
    A22sb = Fsb[1,1]
    A26sb = Fsb[1,2]
    A66sb = Fsb[2,2]

    B11sb = Fsb[0,3]
    B12sb = Fsb[0,4]
    B16sb = Fsb[0,5]
    B22sb = Fsb[1,4]
    B26sb = Fsb[1,5]
    B66sb = Fsb[2,5]

    D11sb = Fsb[3,3]
    D12sb = Fsb[3,4]
    D16sb = Fsb[3,5]
    D22sb = Fsb[4,4]
    D26sb = Fsb[4,5]
    D66sb = Fsb[5,5]

    c = -1

    # k0sb_11
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
                        # k0sb_11 cond_1
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+0
                        k0sbv[c] += A16sb*i1*k1*(-2*(-1)**(i1 + k1) + 2)*((-2*j1*l1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) + ((j1*j1) + (l1*l1))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (-2*j1*l1*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) + ((j1*j1) + (l1*l1))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+1
                        k0sbv[c] += i1*k1*(2*(-1)**(i1 + k1) - 2)*((-j1*l1*(A12sb + A66sb)*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) + (A12sb*(l1*l1) + A66sb*(j1*j1))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (-j1*l1*(A12sb + A66sb)*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) + (A12sb*(l1*l1) + A66sb*(j1*j1))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(((i1*i1) - (k1*k1))*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+2
                        k0sbv[c] += i1*k1*(2*(-1)**(i1 + k1) - 2)*(2*(pi*pi)*B66sb*(a*a)*j1*l1*r*(-(j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b) + (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b)) + ((j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))*((pi*pi)*B11sb*(b*b)*(k1*k1)*r + (a*a)*(A12sb*(b*b) + (pi*pi)*B12sb*(l1*l1)*r)))/(pi*(a*a)*b*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+0
                        k0sbv[c] += i1*k1*(2*(-1)**(i1 + k1) - 2)*((-j1*l1*(A12sb + A66sb)*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) + (A12sb*(j1*j1) + A66sb*(l1*l1))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (-j1*l1*(A12sb + A66sb)*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) + (A12sb*(j1*j1) + A66sb*(l1*l1))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(((i1*i1) - (k1*k1))*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+1
                        k0sbv[c] += A26sb*i1*k1*(-2*(-1)**(i1 + k1) + 2)*((-2*j1*l1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) + ((j1*j1) + (l1*l1))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (-2*j1*l1*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) + ((j1*j1) + (l1*l1))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+2
                        k0sbv[c] += i1*k1*(2*(-1)**(i1 + k1) - 2)*(2*(pi*pi)*B26sb*(a*a)*j1*l1*r*(-(j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b) + (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b)) + ((j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))*((pi*pi)*B16sb*(b*b)*(k1*k1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(l1*l1)*r)))/(pi*(a*a)*b*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+0
                        k0sbv[c] += i1*k1*(-2*(-1)**(i1 + k1) + 2)*(2*(pi*pi)*B66sb*(a*a)*j1*l1*r*(-(j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b) + (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b)) + ((j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))*((pi*pi)*B11sb*(b*b)*(i1*i1)*r + (a*a)*(A12sb*(b*b) + (pi*pi)*B12sb*(j1*j1)*r)))/(pi*(a*a)*b*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+1
                        k0sbv[c] += i1*k1*(-2*(-1)**(i1 + k1) + 2)*(2*(pi*pi)*B26sb*(a*a)*j1*l1*r*(-(j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b) + (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b)) + ((j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))*((pi*pi)*B16sb*(b*b)*(i1*i1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r)))/(pi*(a*a)*b*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+2
                        k0sbv[c] += i1*k1*(4*(-1)**(i1 + k1) - 4)*(-j1*((j1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) - l1*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b) + (j1*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) - l1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b))*((pi*pi)*D16sb*(b*b)*(k1*k1)*r + (a*a)*(B26sb*(b*b) + (pi*pi)*D26sb*(l1*l1)*r)) + l1*((j1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) - l1*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (j1*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) - l1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))*((pi*pi)*D16sb*(b*b)*(i1*i1)*r + (a*a)*(B26sb*(b*b) + (pi*pi)*D26sb*(j1*j1)*r)))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))

                    elif k1 == i1 and l1 != j1:
                        # k0sb_11 cond_2
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+0
                        k0sbv[c] += pi*((j1*(A11sb*(b*b)*(i1*i1) + A66sb*(a*a)*(l1*l1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*(A11sb*(b*b)*(i1*i1) + A66sb*(a*a)*(j1*j1))*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*(A11sb*(b*b)*(i1*i1) + A66sb*(a*a)*(l1*l1))*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*(A11sb*(b*b)*(i1*i1) + A66sb*(a*a)*(j1*j1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(a*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+1
                        k0sbv[c] += pi*((j1*(A16sb*(b*b)*(i1*i1) + A26sb*(a*a)*(l1*l1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*(A16sb*(b*b)*(i1*i1) + A26sb*(a*a)*(j1*j1))*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*(A16sb*(b*b)*(i1*i1) + A26sb*(a*a)*(l1*l1))*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*(A16sb*(b*b)*(i1*i1) + A26sb*(a*a)*(j1*j1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(a*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+2
                        k0sbv[c] += -((-j1*l1*(3*(pi*pi)*B16sb*(b*b)*(i1*i1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(l1*l1)*r))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) + ((pi*pi)*B16sb*(b*b)*(i1*i1)*r*((j1*j1) + 2*(l1*l1)) + (a*a)*(j1*j1)*(A26sb*(b*b) + (pi*pi)*B26sb*(l1*l1)*r))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (-j1*l1*(3*(pi*pi)*B16sb*(b*b)*(i1*i1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(l1*l1)*r))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) + ((pi*pi)*B16sb*(b*b)*(i1*i1)*r*((j1*j1) + 2*(l1*l1)) + (a*a)*(j1*j1)*(A26sb*(b*b) + (pi*pi)*B26sb*(l1*l1)*r))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(a*(b*b)*r*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+0
                        k0sbv[c] += pi*((j1*(A16sb*(b*b)*(i1*i1) + A26sb*(a*a)*(l1*l1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*(A16sb*(b*b)*(i1*i1) + A26sb*(a*a)*(j1*j1))*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*(A16sb*(b*b)*(i1*i1) + A26sb*(a*a)*(l1*l1))*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*(A16sb*(b*b)*(i1*i1) + A26sb*(a*a)*(j1*j1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(a*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+1
                        k0sbv[c] += pi*((j1*(A22sb*(a*a)*(l1*l1) + A66sb*(b*b)*(i1*i1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*(A22sb*(a*a)*(j1*j1) + A66sb*(b*b)*(i1*i1))*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*(A22sb*(a*a)*(l1*l1) + A66sb*(b*b)*(i1*i1))*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*(A22sb*(a*a)*(j1*j1) + A66sb*(b*b)*(i1*i1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(a*b*((j1*j1) - (l1*l1)))
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+2
                        k0sbv[c] += ((j1*l1*((a*a)*(A22sb*(b*b) + (pi*pi)*B22sb*(l1*l1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb + 2*B66sb))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) - ((a*a)*(j1*j1)*(A22sb*(b*b) + (pi*pi)*B22sb*(l1*l1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb*(j1*j1) + 2*B66sb*(l1*l1)))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (j1*l1*((a*a)*(A22sb*(b*b) + (pi*pi)*B22sb*(l1*l1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb + 2*B66sb))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) - ((a*a)*(j1*j1)*(A22sb*(b*b) + (pi*pi)*B22sb*(l1*l1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb*(j1*j1) + 2*B66sb*(l1*l1)))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(a*(b*b)*r*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+0
                        k0sbv[c] += ((-j1*l1*(3*(pi*pi)*B16sb*(b*b)*(i1*i1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) + ((pi*pi)*B16sb*(b*b)*(i1*i1)*r*(2*(j1*j1) + (l1*l1)) + (a*a)*(l1*l1)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (-j1*l1*(3*(pi*pi)*B16sb*(b*b)*(i1*i1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) + ((pi*pi)*B16sb*(b*b)*(i1*i1)*r*(2*(j1*j1) + (l1*l1)) + (a*a)*(l1*l1)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(a*(b*b)*r*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+1
                        k0sbv[c] += ((-j1*l1*((a*a)*(A22sb*(b*b) + (pi*pi)*B22sb*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb + 2*B66sb))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) + ((a*a)*(l1*l1)*(A22sb*(b*b) + (pi*pi)*B22sb*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb*(l1*l1) + 2*B66sb*(j1*j1)))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) + (-j1*l1*((a*a)*(A22sb*(b*b) + (pi*pi)*B22sb*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb + 2*B66sb))*sin(pi*l1*ys/b)*cos(pi*j1*ys/b) + ((a*a)*(l1*l1)*(A22sb*(b*b) + (pi*pi)*B22sb*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb*(l1*l1) + 2*B66sb*(j1*j1)))*sin(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))/(a*(b*b)*r*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+2
                        k0sbv[c] += (4*(pi*pi*pi*pi)*D66sb*(a*a)*(b*b)*(i1*i1)*j1*l1*(r*r)*(-(j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b) + (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b)) + ((j1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b) + l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b))*sin(0.5*pi*bb*j1/b)*cos(0.5*pi*bb*l1/b) - (j1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + l1*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))*sin(0.5*pi*bb*l1/b)*cos(0.5*pi*bb*j1/b))*((pi*pi*pi*pi)*D11sb*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22sb*(b*b*b*b) + (pi*pi)*r*(B22sb*(b*b)*((j1*j1) + (l1*l1)) + (pi*pi)*D22sb*(j1*j1)*(l1*l1)*r)) + (pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(2*B12sb*(b*b) + (pi*pi)*D12sb*r*((j1*j1) + (l1*l1)))))/(pi*(a*a*a)*(b*b*b)*(r*r)*((j1*j1) - (l1*l1)))

                    elif k1 != i1 and l1 == j1:
                        # k0sb_11 cond_3
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+1
                        k0sbv[c] += i1*k1*((-1)**(i1 + k1) - 1)*(A12sb - A66sb)*sin(pi*bb*j1/b)*sin(2*pi*j1*ys/b)/(2.0*(i1*i1) - 2.0*(k1*k1))
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+2
                        k0sbv[c] += i1*k1*((-1)**(i1 + k1) - 1)*(0.5*pi*B66sb*(a*a)*j1*r*(b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1) + 0.25*(-b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)*((pi*pi)*B11sb*(b*b)*(k1*k1)*r + (a*a)*(A12sb*(b*b) + (pi*pi)*B12sb*(j1*j1)*r))/(pi*j1))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+0
                        k0sbv[c] += i1*k1*((-1)**(i1 + k1) - 1)*(-A12sb + A66sb)*sin(pi*bb*j1/b)*sin(2*pi*j1*ys/b)/(2.0*(i1*i1) - 2.0*(k1*k1))
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+2
                        k0sbv[c] += i1*k1*((-1)**(i1 + k1) - 1)*(0.5*pi*B26sb*(a*a)*j1*r*(b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1) + 0.25*(-b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)*((pi*pi)*B16sb*(b*b)*(k1*k1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r))/(pi*j1))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+0
                        k0sbv[c] += -i1*k1*((-1)**(i1 + k1) - 1)*(0.5*pi*B66sb*(a*a)*j1*r*(b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1) + 0.25*(-b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)*((pi*pi)*B11sb*(b*b)*(i1*i1)*r + (a*a)*(A12sb*(b*b) + (pi*pi)*B12sb*(j1*j1)*r))/(pi*j1))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+1
                        k0sbv[c] += -i1*k1*((-1)**(i1 + k1) - 1)*(0.5*pi*B26sb*(a*a)*j1*r*(b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1) + 0.25*(-b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)*((pi*pi)*B16sb*(b*b)*(i1*i1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r))/(pi*j1))/((a*a)*(b*b)*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+2
                        k0sbv[c] += (pi*pi)*D16sb*i1*k1*((-1)**(i1 + k1) - 1)*sin(pi*bb*j1/b)*sin(2*pi*j1*ys/b)/(a*a)

                    elif k1 == i1 and l1 == j1:
                        # k0sb_11 cond_4
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+0
                        k0sbv[c] += 0.125*pi*A11sb*(i1*i1)*(-b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)/(a*j1) + 0.125*pi*A66sb*a*j1*(b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)/(b*b)
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+1
                        k0sbv[c] += 0.125*pi*A16sb*(i1*i1)*(-b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)/(a*j1) + 0.125*pi*A26sb*a*j1*(b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)/(b*b)
                        c += 1
                        k0sbr[c] = row+0
                        k0sbc[c] = col+2
                        k0sbv[c] += 0.25*(-(pi*pi)*B16sb*(b*b)*(i1*i1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r))*sin(pi*bb*j1/b)*sin(2*pi*j1*ys/b)/(a*(b*b)*r)
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+0
                        k0sbv[c] += 0.125*pi*A16sb*(i1*i1)*(-b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)/(a*j1) + 0.125*pi*A26sb*a*j1*(b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)/(b*b)
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+1
                        k0sbv[c] += 0.125*pi*A22sb*a*j1*(b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)/(b*b) + 0.125*pi*A66sb*(i1*i1)*(-b*(sin(pi*j1*(bb - 2*ys)/b) + sin(pi*j1*(bb + 2*ys)/b)) + 2*pi*bb*j1)/(a*j1)
                        c += 1
                        k0sbr[c] = row+1
                        k0sbc[c] = col+2
                        k0sbv[c] += 0.25*((a*a)*(A22sb*(b*b) + (pi*pi)*B22sb*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb - 2*B66sb))*sin(pi*bb*j1/b)*sin(2*pi*j1*ys/b)/(a*(b*b)*r)
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+0
                        k0sbv[c] += 0.25*(-(pi*pi)*B16sb*(b*b)*(i1*i1)*r + (a*a)*(A26sb*(b*b) + (pi*pi)*B26sb*(j1*j1)*r))*sin(pi*bb*j1/b)*sin(2*pi*j1*ys/b)/(a*(b*b)*r)
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+1
                        k0sbv[c] += 0.25*((a*a)*(A22sb*(b*b) + (pi*pi)*B22sb*(j1*j1)*r) + (pi*pi)*(b*b)*(i1*i1)*r*(B12sb - 2*B66sb))*sin(pi*bb*j1/b)*sin(2*pi*j1*ys/b)/(a*(b*b)*r)
                        c += 1
                        k0sbr[c] = row+2
                        k0sbc[c] = col+2
                        k0sbv[c] += 0.125*(-b*((pi*pi*pi*pi)*D11sb*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22sb*(b*b*b*b) + (pi*pi)*(j1*j1)*r*(2*B22sb*(b*b) + (pi*pi)*D22sb*(j1*j1)*r)) + 2*(pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(B12sb*(b*b) + (pi*pi)*(j1*j1)*r*(D12sb - 2*D66sb)))*sin(pi*j1*(bb - 2*ys)/b) - b*((pi*pi*pi*pi)*D11sb*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22sb*(b*b*b*b) + (pi*pi)*(j1*j1)*r*(2*B22sb*(b*b) + (pi*pi)*D22sb*(j1*j1)*r)) + 2*(pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(B12sb*(b*b) + (pi*pi)*(j1*j1)*r*(D12sb - 2*D66sb)))*sin(pi*j1*(bb + 2*ys)/b) + 2*pi*bb*j1*((pi*pi*pi*pi)*D11sb*(b*b*b*b)*(i1*i1*i1*i1)*(r*r) + (a*a*a*a)*(A22sb*(b*b*b*b) + (pi*pi)*(j1*j1)*r*(2*B22sb*(b*b) + (pi*pi)*D22sb*(j1*j1)*r)) + 2*(pi*pi)*(a*a)*(b*b)*(i1*i1)*r*(B12sb*(b*b) + (pi*pi)*(j1*j1)*r*(D12sb + 2*D66sb))))/(pi*(a*a*a)*(b*b*b*b)*j1*(r*r))

    size = num0 + num1*m1*n1

    k0sb = coo_matrix((k0sbv, (k0sbr, k0sbc)), shape=(size, size))

    return k0sb


def fk0sf(double ys, double a, double b, double r, int m1, int n1,
          double Exx, double Gxy, double Jxx, double Iyy):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] k0sfr, k0sfc
    cdef np.ndarray[cDOUBLE, ndim=1] k0sfv

    fdim = 1*m1*n1*m1*n1

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
                        pass

                    elif k1 == i1 and l1 != j1:
                        # k0sf_11 cond_2
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+2
                        k0sfv[c] += 0.5*(pi*pi*pi*pi)*(Exx*Iyy*(b*b*b*b)*(i1*i1*i1*i1) + Gxy*Jxx*(a*a*a*a)*(j1*j1)*(l1*l1))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((a*a*a)*(b*b*b*b))

                    elif k1 != i1 and l1 == j1:
                        # k0sf_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # k0sf_11 cond_4
                        c += 1
                        k0sfr[c] = row+2
                        k0sfc[c] = col+2
                        k0sfv[c] += 0.5*(pi*pi*pi*pi)*(Exx*Iyy*(b*b*b*b)*(i1*i1*i1*i1) + Gxy*Jxx*(a*a*a*a)*(j1*j1*j1*j1))*sin(pi*j1*ys/b)**2/((a*a*a)*(b*b*b*b))

    size = num0 + num1*m1*n1

    k0sf = coo_matrix((k0sfv, (k0sfr, k0sfc)), shape=(size, size))

    return k0sf


def fk0sf2(double bf, double df, double ys, double a, double b, double r,
           int m1, int n1, double E1, double F1, double S1, double Jxx):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] k0sf2r, k0sf2c
    cdef np.ndarray[cDOUBLE, ndim=1] k0sf2v

    fdim = 4*m1*n1*m1*n1

    k0sf2r = np.zeros((fdim,), dtype=INT)
    k0sf2c = np.zeros((fdim,), dtype=INT)
    k0sf2v = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # k0sf2_11
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
                        # k0sf2_11 cond_1
                        c += 1
                        k0sf2r[c] = row+0
                        k0sf2c[c] = col+2
                        k0sf2v[c] += -(pi*pi)*E1*bf*df*i1*(k1*k1*k1)*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((a*a)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sf2r[c] = row+2
                        k0sf2c[c] = col+0
                        k0sf2v[c] += (pi*pi)*E1*bf*df*(i1*i1*i1)*k1*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((a*a)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sf2r[c] = row+2
                        k0sf2c[c] = col+2
                        k0sf2v[c] += -(pi*pi*pi)*S1*bf*df*i1*k1*((-1)**(i1 + k1) - 1)*((i1*i1)*l1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b) - j1*(k1*k1)*sin(pi*l1*ys/b)*cos(pi*j1*ys/b))/((a*a)*b*((i1*i1) - (k1*k1)))

                    elif k1 == i1 and l1 != j1:
                        # k0sf2_11 cond_2
                        c += 1
                        k0sf2r[c] = row+0
                        k0sf2c[c] = col+0
                        k0sf2v[c] += 0.5*(pi*pi)*E1*bf*(i1*i1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/a
                        c += 1
                        k0sf2r[c] = row+0
                        k0sf2c[c] = col+2
                        k0sf2v[c] += -0.5*(pi*pi*pi)*S1*bf*(i1*i1)*l1*sin(pi*j1*ys/b)*cos(pi*l1*ys/b)/(a*b)
                        c += 1
                        k0sf2r[c] = row+2
                        k0sf2c[c] = col+0
                        k0sf2v[c] += -0.5*(pi*pi*pi)*S1*bf*(i1*i1)*j1*sin(pi*l1*ys/b)*cos(pi*j1*ys/b)/(a*b)
                        c += 1
                        k0sf2r[c] = row+2
                        k0sf2c[c] = col+2
                        k0sf2v[c] += 0.5*(pi*pi*pi*pi)*bf*(i1*i1)*(Jxx*(a*a)*j1*l1*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + (b*b)*(i1*i1)*(E1*(df*df) + F1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))/((a*a*a)*(b*b))

                    elif k1 != i1 and l1 == j1:
                        # k0sf2_11 cond_3
                        c += 1
                        k0sf2r[c] = row+0
                        k0sf2c[c] = col+2
                        k0sf2v[c] += -(pi*pi)*E1*bf*df*i1*(k1*k1*k1)*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)**2/((a*a)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sf2r[c] = row+2
                        k0sf2c[c] = col+0
                        k0sf2v[c] += (pi*pi)*E1*bf*df*(i1*i1*i1)*k1*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)**2/((a*a)*((i1*i1) - (k1*k1)))
                        c += 1
                        k0sf2r[c] = row+2
                        k0sf2c[c] = col+2
                        k0sf2v[c] += -0.5*(pi*pi*pi)*S1*bf*df*i1*j1*k1*((-1)**(i1 + k1) - 1)*sin(2*pi*j1*ys/b)/((a*a)*b)

                    elif k1 == i1 and l1 == j1:
                        # k0sf2_11 cond_4
                        c += 1
                        k0sf2r[c] = row+0
                        k0sf2c[c] = col+0
                        k0sf2v[c] += 0.5*(pi*pi)*E1*bf*(i1*i1)*sin(pi*j1*ys/b)**2/a
                        c += 1
                        k0sf2r[c] = row+0
                        k0sf2c[c] = col+2
                        k0sf2v[c] += -0.25*(pi*pi*pi)*S1*bf*(i1*i1)*j1*sin(2*pi*j1*ys/b)/(a*b)
                        c += 1
                        k0sf2r[c] = row+2
                        k0sf2c[c] = col+0
                        k0sf2v[c] += -0.25*(pi*pi*pi)*S1*bf*(i1*i1)*j1*sin(2*pi*j1*ys/b)/(a*b)
                        c += 1
                        k0sf2r[c] = row+2
                        k0sf2c[c] = col+2
                        k0sf2v[c] += 0.5*(pi*pi*pi*pi)*bf*(i1*i1)*(Jxx*(a*a)*(j1*j1)*cos(pi*j1*ys/b)**2 + (b*b)*(i1*i1)*(E1*(df*df) + F1)*sin(pi*j1*ys/b)**2)/((a*a*a)*(b*b))

    size = num0 + num1*m1*n1

    k0sf2 = coo_matrix((k0sf2v, (k0sf2r, k0sf2c)), shape=(size, size))

    return k0sf2


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


def fkG0(double Fx, double Fy, double Fxy, double Fyx,
         double a, double b, double r, int m1, int n1):
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


def fkM(double mu, double h, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMr, kMc
    cdef np.ndarray[cDOUBLE, ndim=1] kMv

    fdim = 3*m1*n1*m1*n1

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
                        kMv[c] += 0.25*a*b*h*mu - 0.0208333333333333*(pi*pi)*a*(h*h*h)*(j1*j1)*mu/b - 0.0208333333333333*(pi*pi)*b*(h*h*h)*(i1*i1)*mu/a

    size = num0 + num1*m1*n1

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkMsb(double mus, double ys, double db, double hb, double a, double b,
          int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMsbr, kMsbc
    cdef np.ndarray[cDOUBLE, ndim=1] kMsbv

    fdim = 3*m1*n1*m1*n1

    kMsbr = np.zeros((fdim,), dtype=INT)
    kMsbc = np.zeros((fdim,), dtype=INT)
    kMsbv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kMsb_11
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
                        # kMsb_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # kMsb_11 cond_2
                        c += 1
                        kMsbr[c] = row+1
                        kMsbc[c] = col+2
                        kMsbv[c] += -a*db*hb*j1*l1*mus*((-1)**(j1 + l1) - 1)/(2.0*(j1*j1) - 2.0*(l1*l1))
                        c += 1
                        kMsbr[c] = row+2
                        kMsbc[c] = col+1
                        kMsbv[c] += -a*db*hb*j1*l1*mus*((-1)**(j1 + l1) - 1)/(2.0*(j1*j1) - 2.0*(l1*l1))

                    elif k1 != i1 and l1 == j1:
                        # kMsb_11 cond_3
                        c += 1
                        kMsbr[c] = row+0
                        kMsbc[c] = col+2
                        kMsbv[c] += -b*db*hb*i1*k1*mus*((-1)**(i1 + k1) - 1)/(2.0*(i1*i1) - 2.0*(k1*k1))
                        c += 1
                        kMsbr[c] = row+2
                        kMsbc[c] = col+0
                        kMsbv[c] += -b*db*hb*i1*k1*mus*((-1)**(i1 + k1) - 1)/(2.0*(i1*i1) - 2.0*(k1*k1))

                    elif k1 == i1 and l1 == j1:
                        # kMsb_11 cond_4
                        c += 1
                        kMsbr[c] = row+0
                        kMsbc[c] = col+0
                        kMsbv[c] += 0.25*a*b*hb*mus
                        c += 1
                        kMsbr[c] = row+1
                        kMsbc[c] = col+1
                        kMsbv[c] += 0.25*a*b*hb*mus
                        c += 1
                        kMsbr[c] = row+2
                        kMsbc[c] = col+2
                        kMsbv[c] += -0.0208333333333333*hb*mus*((a*a)*(-12*(b*b) + (pi*pi)*(j1*j1)*(12*(db*db) + (hb*hb))) + (pi*pi)*(b*b)*(i1*i1)*(12*(db*db) + (hb*hb)))/(a*b)

    size = num0 + num1*m1*n1

    kMsb = coo_matrix((kMsbv, (kMsbr, kMsbc)), shape=(size, size))

    return kMsb


def fkMsf(double mus, double ys, double df, double Asf, double a, double b,
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
                        kMsfv[c] += -Asf*df*i1*k1*mus*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((i1*i1) - (k1*k1))
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+0
                        kMsfv[c] += -Asf*df*i1*k1*mus*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)/((i1*i1) - (k1*k1))

                    elif k1 == i1 and l1 != j1:
                        # kMsf_11 cond_2
                        c += 1
                        kMsfr[c] = row+0
                        kMsfc[c] = col+0
                        kMsfv[c] += 0.5*Asf*a*mus*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)
                        c += 1
                        kMsfr[c] = row+1
                        kMsfc[c] = col+1
                        kMsfv[c] += 0.5*Asf*a*mus*sin(pi*j1*ys/b)*sin(pi*l1*ys/b)
                        c += 1
                        kMsfr[c] = row+1
                        kMsfc[c] = col+2
                        kMsfv[c] += 0.5*pi*Asf*a*df*l1*mus*sin(pi*j1*ys/b)*cos(pi*l1*ys/b)/b
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+1
                        kMsfv[c] += -0.5*pi*Asf*a*df*j1*mus*sin(pi*l1*ys/b)*cos(pi*j1*ys/b)/b
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+2
                        kMsfv[c] += -0.5*mus*((pi*pi)*(a*a)*j1*l1*(Asf*(df*df) + Jxx)*cos(pi*j1*ys/b)*cos(pi*l1*ys/b) + (b*b)*(-Asf*(a*a) + (pi*pi)*(i1*i1)*(Asf*(df*df) + Iyy))*sin(pi*j1*ys/b)*sin(pi*l1*ys/b))/(a*(b*b))

                    elif k1 != i1 and l1 == j1:
                        # kMsf_11 cond_3
                        c += 1
                        kMsfr[c] = row+0
                        kMsfc[c] = col+2
                        kMsfv[c] += -Asf*df*i1*k1*mus*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)**2/((i1*i1) - (k1*k1))
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+0
                        kMsfv[c] += -Asf*df*i1*k1*mus*((-1)**(i1 + k1) - 1)*sin(pi*j1*ys/b)**2/((i1*i1) - (k1*k1))

                    elif k1 == i1 and l1 == j1:
                        # kMsf_11 cond_4
                        c += 1
                        kMsfr[c] = row+0
                        kMsfc[c] = col+0
                        kMsfv[c] += 0.5*Asf*a*mus*sin(pi*j1*ys/b)**2
                        c += 1
                        kMsfr[c] = row+1
                        kMsfc[c] = col+1
                        kMsfv[c] += 0.5*Asf*a*mus*sin(pi*j1*ys/b)**2
                        c += 1
                        kMsfr[c] = row+1
                        kMsfc[c] = col+2
                        kMsfv[c] += 0.25*pi*Asf*a*df*j1*mus*sin(2*pi*j1*ys/b)/b
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+1
                        kMsfv[c] += -0.25*pi*Asf*a*df*j1*mus*sin(2*pi*j1*ys/b)/b
                        c += 1
                        kMsfr[c] = row+2
                        kMsfc[c] = col+2
                        kMsfv[c] += -0.5*mus*((pi*pi)*(a*a)*(j1*j1)*(Asf*(df*df) + Jxx)*cos(pi*j1*ys/b)**2 + (b*b)*(-Asf*(a*a) + (pi*pi)*(i1*i1)*(Asf*(df*df) + Iyy))*sin(pi*j1*ys/b)**2)/(a*(b*b))

    size = num0 + num1*m1*n1

    kMsf = coo_matrix((kMsfv, (kMsfr, kMsfc)), shape=(size, size))

    return kMsf
