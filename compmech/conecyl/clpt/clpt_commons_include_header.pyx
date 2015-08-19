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

cdef int i0 = 0
cdef int j0 = 1
cdef int num0 = 3
cdef int num1 = 3
cdef int num2 = 6
cdef int e_num = 6
cdef int castro = 0
cdef double pi=3.141592653589793
