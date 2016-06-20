cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange

from compmech.conecyl.imperfections.mgi cimport cfw0x, cfw0t

DOUBLE = np.float64
INT = np.int64
ctypedef np.double_t cDOUBLE
ctypedef np.int64_t cINT

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

ctypedef void *cfstraintype(double *c, double sina, double cosa, double tLA,
                            double *xs, double *ts, int size,
                            double r2, double L,
                            int m1, int m2, int n2,
                            double *c0, int m0, int n0, int funcnum,
                            double *es) nogil

cdef int i0
cdef int j0
cdef int num0
cdef int num1
cdef int num2
cdef int e_num
cdef int castro
cdef double pi

i0 = 0
j0 = 1
num0 = 3
num1 = 3
num2 = 6
e_num = 6
castro = 0
pi=3.141592653589793
