import numpy as np
cimport numpy as np
from scipy.sparse import coo_matrix
from libc.stdlib cimport malloc, free

from compmech.conecyl.imperfections.mgi cimport cfw0x, cfw0t
from compmech.integrate.integratev cimport integratev

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

cdef int i0 = 0
cdef int j0 = 1
cdef int num0 = 3
cdef int num1 = 5
cdef int num2 = 10
cdef int e_num = 8
cdef double pi = 3.141592653589793
cdef int NL_kinematics=0 # to use cfstrain_donnell in cfN
cdef int funcnum = 2 # to use in the cfw0x and cfw0t functions
cdef int castro = 0

cdef struct cc_attributes:
    double *sina
    double *cosa
    double *tLA
    double *r2
    double *L
    double *F
    int *m1
    int *m2
    int *n2
    double *coeffs
    double *c0
    int *m0
    int *n0

