cdef void cfuvw(double *c, int m1, int n1, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws) nogil

cdef void cfwx(double *c, int m1, int n1, double *xs, double *ys, int size,
        double a, double b, double *outwx) nogil

cdef void cfwy(double *c, int m1, int n1, double *xs, double *ys, int size,
        double a, double b, double *outwy) nogil

ctypedef void *cfstraintype(double *c, double *xs, double *ys, int size, double
        a, double b, int m1, int n1, double *c0, int m0, int n0, int funcnum,
        double *es) nogil

cdef void cfN(double *c, double *xs, double *ys, int size, double a, double b,
        double *F, int m1, int n1, double *c0, int m0, int n0, int funcnum,
        double *N, int NL_kinematics) nogil

