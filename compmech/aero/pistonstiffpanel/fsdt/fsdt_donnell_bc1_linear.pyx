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
cdef int num1 = 5
cdef double pi = 3.141592653589793


def fk0(double a, double b, double r, np.ndarray[cDOUBLE, ndim=2] F,
        int m1, int n1):
    cdef int i1, j1, k1, l1, c, row, col
    cdef double A11, A12, A16, A22, A26, A66, A44, A45, A55
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef np.ndarray[cINT, ndim=1] k0r, k0c
    cdef np.ndarray[cDOUBLE, ndim=1] k0v

    fdim = 13*m1*n1*m1*n1

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
                        k0v[c] += A45*i1*j1*k1*l1*(-2*(-1)**(i1 + k1) + 2)*((-1)**(j1 + l1) - 1)/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += a*i1*j1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(-A45*r + B26)/(pi*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += b*i1*j1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(-A45*r + B26)/(pi*r*((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += a*j1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A45*r - B26)/(pi*r*(-(i1*i1) + (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += -D16*j1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((i1*i1) + (k1*k1))/(((i1*i1) - (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += j1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*D16*(b*b)*(i1*i1) + (a*a)*(A45*(b*b) + (pi*pi)*D26*(l1*l1)))/((pi*pi)*a*b*(-(i1*i1) + (k1*k1))*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += b*i1*k1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*(A45*r - B26)/(pi*r*((i1*i1) - (k1*k1))*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += i1*l1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((pi*pi)*D16*(b*b)*(k1*k1) + (a*a)*(A45*(b*b) + (pi*pi)*D26*(j1*j1)))/((pi*pi)*a*b*((i1*i1) - (k1*k1))*(-(j1*j1) + (l1*l1)))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += -D26*i1*k1*((-1)**(i1 + k1) - 1)*((-1)**(j1 + l1) - 1)*((j1*j1) + (l1*l1))/((i1 - k1)*(i1 + k1)*(j1 - l1)*(j1 + l1))

                    elif k1 == i1 and l1 != j1:
                        # k0_11 cond_2
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += A26*a*j1*l1*((-1)**(j1 + l1) - 1)/(r*(2.0*(j1*j1) - 2.0*(l1*l1)))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += -pi*B16*i1*j1*l1*((-1)**(j1 + l1) - 1)/((j1*j1) - (l1*l1))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += -0.5*pi*j1*((-1)**(j1 + l1) - 1)*(B16*(b*b)*(i1*i1) + B26*(a*a)*(l1*l1))/(a*b*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += a*j1*l1*((-1)**(j1 + l1) - 1)*(A22 + A44)/(r*(2.0*(j1*j1) - 2.0*(l1*l1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += -pi*i1*j1*l1*((-1)**(j1 + l1) - 1)*(B12 + B66)/((j1 + l1)*(2.0*j1 - 2.0*l1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += -0.5*j1*((-1)**(j1 + l1) - 1)*((pi*pi)*B66*(b*b)*(i1*i1)*r + (a*a)*(-A44*(b*b) + (pi*pi)*B22*(l1*l1)*r))/(pi*a*b*r*((j1*j1) - (l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += -A26*a*j1*l1*((-1)**(j1 + l1) - 1)/(r*(2.0*(j1*j1) - 2.0*(l1*l1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -a*j1*l1*((-1)**(j1 + l1) - 1)*(A22 + A44)/(r*(j1 + l1)*(2.0*j1 - 2.0*l1))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += pi*B16*i1*j1*l1*((-1)**(j1 + l1) - 1)/((j1*j1) - (l1*l1))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += pi*i1*j1*l1*((-1)**(j1 + l1) - 1)*(B12 + B66)/((j1 + l1)*(2.0*j1 - 2.0*l1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += 0.5*pi*l1*((-1)**(j1 + l1) - 1)*(B16*(b*b)*(i1*i1) + B26*(a*a)*(j1*j1))/(a*b*(j1 - l1)*(j1 + l1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += 0.5*l1*((-1)**(j1 + l1) - 1)*((pi*pi)*B66*(b*b)*(i1*i1)*r + (a*a)*(-A44*(b*b) + (pi*pi)*B22*(j1*j1)*r))/(pi*a*b*r*((j1*j1) - (l1*l1)))

                    elif k1 != i1 and l1 == j1:
                        # k0_11 cond_3
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += A12*b*i1*k1*((-1)**(i1 + k1) - 1)/(r*(2.0*(i1*i1) - 2.0*(k1*k1)))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += -0.5*pi*i1*((-1)**(i1 + k1) - 1)*(B11*(b*b)*(k1*k1) + B66*(a*a)*(j1*j1))/(a*b*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += -pi*i1*j1*k1*((-1)**(i1 + k1) - 1)*(B12 + B66)/(2.0*(i1*i1) - 2.0*(k1*k1))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += b*i1*k1*((-1)**(i1 + k1) - 1)*(A26 + A45)/(r*(2.0*(i1*i1) - 2.0*(k1*k1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += -0.5*i1*((-1)**(i1 + k1) - 1)*((pi*pi)*B16*(b*b)*(k1*k1)*r + (a*a)*(-A45*(b*b) + (pi*pi)*B26*(j1*j1)*r))/(pi*a*b*r*((i1*i1) - (k1*k1)))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += -pi*B26*i1*j1*k1*((-1)**(i1 + k1) - 1)/((i1*i1) - (k1*k1))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += -A12*b*i1*k1*((-1)**(i1 + k1) - 1)/(r*(2.0*(i1*i1) - 2.0*(k1*k1)))
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += -b*i1*k1*((-1)**(i1 + k1) - 1)*(A26 + A45)/(r*(2.0*(i1*i1) - 2.0*(k1*k1)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += -0.5*pi*k1*((-1)**(i1 + k1) - 1)*(B11*(b*b)*(i1*i1) + B66*(a*a)*(j1*j1))/(a*b*(-(i1*i1) + (k1*k1)))
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += -0.5*k1*((-1)**(i1 + k1) - 1)*((pi*pi)*B16*(b*b)*(i1*i1)*r + (a*a)*(-A45*(b*b) + (pi*pi)*B26*(j1*j1)*r))/(pi*a*b*r*(-(i1*i1) + (k1*k1)))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += pi*i1*j1*k1*((-1)**(i1 + k1) - 1)*(B12 + B66)/(2.0*(i1*i1) - 2.0*(k1*k1))
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += pi*B26*i1*j1*k1*((-1)**(i1 + k1) - 1)/((i1*i1) - (k1*k1))

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
                        k0v[c] += 0.25*(pi*pi)*A22*a*(j1*j1)/b + 0.25*A44*a*b/(r*r) + 0.25*(pi*pi)*A66*b*(i1*i1)/a
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += 0.25*A22*a*b/(r*r) + 0.25*(pi*pi)*A44*a*(j1*j1)/b + 0.25*(pi*pi)*A55*b*(i1*i1)/a
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += 0.25*pi*b*i1*(A55*r - B12)/r
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += 0.25*pi*a*j1*(A44*r - B22)/r
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += 0.25*pi*b*i1*(A55*r - B12)/r
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += 0.25*(pi*pi)*D11*b*(i1*i1)/a + 0.25*a*(A55*(b*b) + (pi*pi)*D66*(j1*j1))/b
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += 0.25*(pi*pi)*i1*j1*(D12 + D66)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += 0.25*pi*a*j1*(A44*r - B22)/r
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += 0.25*(pi*pi)*i1*j1*(D12 + D66)
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += 0.25*(pi*pi)*D66*b*(i1*i1)/a + 0.25*a*(A44*(b*b) + (pi*pi)*D22*(j1*j1))/b

    size = num0 + num1*m1*n1

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fk0edges(int m1, int n1, double a, double b,
             double kphixBot, double kphixTop,
             double kphiyBot, double kphiyTop,
             double kphixLeft, double kphixRight,
             double kphiyLeft, double kphiyRight):
    cdef int i1, j1, k1, l1, row, col, c, cbkp
    cdef np.ndarray[cINT, ndim=1] k0edgesr, k0edgesc
    cdef np.ndarray[cDOUBLE, ndim=1] k0edgesv

    fdim = 2*m1*n1*m1*n1 + 2*m1*n1*m1*n1

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
    cdef int i1, j1, k1, l1, row, col, c, cbkp
    cdef np.ndarray[cINT, ndim=1] k0stiffr, k0stiffc
    cdef np.ndarray[cDOUBLE, ndim=1] k0stiffv

    fdim = 5*m1*n1*m1*n1

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


def fkA(double beta, double gamma, double a, double b, int m1, int n1):
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
                        c += 1
                        kAr[c] = row+2
                        kAc[c] = col+2
                        kAv[c] += -a*beta*j1*l1*((-1)**(j1 + l1) - 1)/(2.0*(j1*j1) - 2.0*(l1*l1))

                    elif k1 != i1 and l1 == j1:
                        # kA_11 cond_3
                        pass

                    elif k1 == i1 and l1 == j1:
                        # kA_11 cond_4
                        c += 1
                        kAr[c] = row+2
                        kAc[c] = col+2
                        kAv[c] += 0.25*a*b*gamma

    size = num0 + num1*m1*n1

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA


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


def fkM(double mu, double h, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMr, kMc
    cdef np.ndarray[cDOUBLE, ndim=1] kMv

    fdim = 5*m1*n1*m1*n1

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
                        c += 1
                        kMr[c] = row+3
                        kMc[c] = col+3
                        kMv[c] += 0.25*a*b*h*mu
                        c += 1
                        kMr[c] = row+4
                        kMc[c] = col+4
                        kMv[c] += 0.25*a*b*h*mu

    size = num0 + num1*m1*n1

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkMstiff(double mustiff, double ystiff, double Astiff, double a, double b, int m1, int n1):
    cdef int i1, k1, j1, l1, c, row, col
    cdef np.ndarray[cINT, ndim=1] kMstiffr, kMstiffc
    cdef np.ndarray[cDOUBLE, ndim=1] kMstiffv

    fdim = 5*m1*n1*m1*n1

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
                        c += 1
                        kMstiffr[c] = row+3
                        kMstiffc[c] = col+3
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*sin(pi*j1*ystiff/b)*sin(pi*l1*ystiff/b)
                        c += 1
                        kMstiffr[c] = row+4
                        kMstiffc[c] = col+4
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*cos(pi*j1*ystiff/b)*cos(pi*l1*ystiff/b)

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
                        c += 1
                        kMstiffr[c] = row+3
                        kMstiffc[c] = col+3
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*sin(pi*j1*ystiff/b)**2
                        c += 1
                        kMstiffr[c] = row+4
                        kMstiffc[c] = col+4
                        kMstiffv[c] += 0.5*Astiff*a*mustiff*cos(pi*j1*ystiff/b)**2

    size = num0 + num1*m1*n1

    kMstiff = coo_matrix((kMstiffv, (kMstiffr, kMstiffc)), shape=(size, size))

    return kMstiff
