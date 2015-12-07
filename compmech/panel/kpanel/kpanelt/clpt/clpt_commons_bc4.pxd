cdef void cfuvw(double *c, int m1, int n1, double L,
                double tmin, double tmax, double *xs, double *ts, int size,
                double *us, double *vs, double *ws) nogil

cdef void cfwx(double *c, int m1, int n1, double *xs,
        double *ts, int size, double L, double tmin, double tmax,
        double *outwx) nogil

cdef void cfwt(double *c, int m1, int n1, double *xs,
        double *ts, int size, double L, double tmin, double tmax,
        double *outwx) nogil

ctypedef void *cfstraintype(double *c, double sina, double cosa, double *xs,
        double *ts, int size, double r1, double L, double tmin, double tmax,
        int m1, int n1,
        double *c0, int m0, int n0, int funcnum, double *es) nogil

cdef void cfN(double *c, double sina, double cosa, double *xs, double *ts, int
        size, double r1, double L, double tmin, double tmax, double *F,
        int m1, int n1, double *c0, int m0, int n0,
        int funcnum, double *N, int NL_kinematics) nogil

