#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False

#from libc.stdlib cimport malloc, free

cdef void invJ2(int nel, int npts, double *J, double *invJ) nogil:
    cdef double a, b, c, d, f
    cdef int i, j, pos
    cdef int num = 4
    for i in range(nel):
        for j in range(npts):
            pos = npts*num*i + num*j
            a = J[pos+0]
            b = J[pos+1]
            c = J[pos+2]
            d = J[pos+3]
            f = 1./(a*d - b*c)
            invJ[pos+0] = f*d
            invJ[pos+1] = -f*b
            invJ[pos+2] = -f*c
            invJ[pos+3] = f*a

cdef void detJ2(int nel, int npts, double *J, double *detJ) nogil:
    cdef double a, b, c, d, f
    cdef int i, j, pos
    cdef int num = 4
    for i in range(nel):
        for j in range(npts):
            pos = npts*num*i + num*j
            a = J[pos+0]
            b = J[pos+1]
            c = J[pos+2]
            d = J[pos+3]
            pos = npts*i + j
            detJ[pos] = a*d-b*c

def test3():
    pass

def test4():
    pass
