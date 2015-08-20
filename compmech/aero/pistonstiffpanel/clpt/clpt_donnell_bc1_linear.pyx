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


def fk0stiff(int m1, int n1, double ystiff, double a, double b,
             double kustiff, double kvstiff, double kwstiff,
             double kphixstiff, double kphiystiff):
    cdef int i1, j1, k1, l1, row, col, c
    cdef np.ndarray[cINT, ndim=1] k0stiffr, k0stiffc
    cdef np.ndarray[cDOUBLE, ndim=1] k0stiffv

    fdim = 1*m1*n1*m1*n1 + 1*m1*n1*m1*n1

    k0stiffr = np.zeros((fdim,), dtype=INT)
    k0stiffc = np.zeros((fdim,), dtype=INT)
    k0stiffv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # k0stiff_11
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
                        # k0stiff_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # k0stiff_11 cond_2
                        c += 1
                        k0stiffr[c] = row+0
                        k0stiffc[c] = col+0
                        k0stiffv[c] += 0.5*a*kustiff*sin(pi*j1*ystiff/b)*sin(pi*l1*ystiff/b)
                        c += 1
                        k0stiffr[c] = row+1
                        k0stiffc[c] = col+1
                        k0stiffv[c] += 0.5*a*kvstiff*sin(pi*j1*ystiff/b)*sin(pi*l1*ystiff/b)
                        c += 1
                        k0stiffr[c] = row+2
                        k0stiffc[c] = col+2
                        k0stiffv[c] += 0.5*((pi*pi)*(a*a)*j1*kphiystiff*l1*cos(pi*j1*ystiff/b)*cos(pi*l1*ystiff/b) + (b*b)*((a*a)*kwstiff + (pi*pi)*(i1*i1)*kphixstiff)*sin(pi*j1*ystiff/b)*sin(pi*l1*ystiff/b))/(a*(b*b))

                    elif k1 != i1 and l1 == j1:
                        # k0stiff_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # k0stiff_11 cond_4
                        c += 1
                        k0stiffr[c] = row+0
                        k0stiffc[c] = col+0
                        k0stiffv[c] += 0.5*a*kustiff*sin(pi*j1*ystiff/b)**2
                        c += 1
                        k0stiffr[c] = row+1
                        k0stiffc[c] = col+1
                        k0stiffv[c] += 0.5*a*kvstiff*sin(pi*j1*ystiff/b)**2
                        c += 1
                        k0stiffr[c] = row+2
                        k0stiffc[c] = col+2
                        k0stiffv[c] += 0.5*((pi*pi)*(a*a)*(j1*j1)*kphiystiff*cos(pi*j1*ystiff/b)**2 + (b*b)*((a*a)*kwstiff + (pi*pi)*(i1*i1)*kphixstiff)*sin(pi*j1*ystiff/b)**2)/(a*(b*b))

    size = num0 + num1*m1*n1

    k0stiff = coo_matrix((k0stiffv, (k0stiffr, k0stiffc)), shape=(size, size))

    return k0stiff


def fkAx(double beta, double a, double b, int m1, int n1):
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
                        pass

    size = num0 + num1*m1*n1

    kAx = coo_matrix((kAxv, (kAxr, kAxc)), shape=(size, size))

    return kAx


def fkAy(double beta, double gamma, double a, double b, int m1, int n1):
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
                        c += 1
                        kAyr[c] = row+2
                        kAyc[c] = col+2
                        kAyv[c] += 0.25*a*b*gamma

    size = num0 + num1*m1*n1

    kAy = coo_matrix((kAyv, (kAyr, kAyc)), shape=(size, size))

    return kAy


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
                        kMv[c] += 0.25*a*b*h*mu

    size = num0 + num1*m1*n1

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkMstiff(double mustiff, double ystiff, double Astiff, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMstiffr, kMstiffc
    cdef np.ndarray[cDOUBLE, ndim=1] kMstiffv

    fdim = 3*m1*n1*m1*n1

    kMstiffr = np.zeros((fdim,), dtype=INT)
    kMstiffc = np.zeros((fdim,), dtype=INT)
    kMstiffv = np.zeros((fdim,), dtype=DOUBLE)

    c = -1

    # kMstiff_11
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
                        # kMstiff_11 cond_1
                        pass

                    elif k1 == i1 and l1 != j1:
                        # kMstiff_11 cond_2
                        c += 1
                        kMstiffr[c] = row+0
                        kMstiffc[c] = col+0
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*sin(pi*j1*ystiff/b)*sin(pi*l1*ystiff/b)
                        c += 1
                        kMstiffr[c] = row+1
                        kMstiffc[c] = col+1
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*sin(pi*j1*ystiff/b)*sin(pi*l1*ystiff/b)
                        c += 1
                        kMstiffr[c] = row+2
                        kMstiffc[c] = col+2
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*sin(pi*j1*ystiff/b)*sin(pi*l1*ystiff/b)

                    elif k1 != i1 and l1 == j1:
                        # kMstiff_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kMstiff_11 cond_4
                        c += 1
                        kMstiffr[c] = row+0
                        kMstiffc[c] = col+0
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*sin(pi*j1*ystiff/b)**2
                        c += 1
                        kMstiffr[c] = row+1
                        kMstiffc[c] = col+1
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*sin(pi*j1*ystiff/b)**2
                        c += 1
                        kMstiffr[c] = row+2
                        kMstiffc[c] = col+2
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*sin(pi*j1*ystiff/b)**2

    size = num0 + num1*m1*n1

    kMstiff = coo_matrix((kMstiffv, (kMstiffr, kMstiffc)), shape=(size, size))

    return kMstiff
