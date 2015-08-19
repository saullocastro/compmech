cimport numpy as np

cdef void cfw0x(double *xs, double *ts, int size, double *c0, double L,
                int m, int n, double *outw0x, int funcnum) nogil

cdef void cfw0t(double *xs, double *ts, int size, double *c0, double L,
                int m, int n, double *outw0x, int funcnum) nogil
