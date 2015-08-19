cdef void cfuvw(double *c, int m1, int n1, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws) nogil

cdef void cfwx(double *c, int m1, int n1, double *xs, double *ys, int size,
        double a, double b, double *outwx) nogil

cdef void cfwy(double *c, int m1, int n1, double *xs, double *ys, int size,
        double a, double b, double *outwy) nogil
