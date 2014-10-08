ctypedef void *cfstraintype(double *c, double sina, double cosa, double tLA,
                            double *xs, double *ts, int size,
                            double r1, double L,
                            int m1, int m2, int n2,
                            double *c0, int m0, int n0, int funcnum,
                            double *es) nogil

cdef void cfN(double *c, double sina, double cosa, double tLA,
              double *xs, double *ts, int size,
              double r1, double L, double *F,
              int m1, int m2, int n2,
              double *c0, int m0, int n0, int funcnum,
              double *Ns, int NL_kinematics) nogil

cdef void cfuvw(double *c, int m2, int n3, int m4, int n4,
        double L, double tmin, double tmax,
        double *xs, double *ts, int size, double *us, double *vs, double *ws,
        double *phixs, double *phits) nogil

cdef void cfwx(double *c, int m2, int n3, int m4, int n4, double *xs, double
    *ts, int size, double L, double tmin, double tmax, double *outwx) nogil

cdef void cfwt(double *c, int m2, int n3, int m4, int n4, double *xs, double
    *ts, int size, double L, double tmin, double tmax, double *outwt) nogil

