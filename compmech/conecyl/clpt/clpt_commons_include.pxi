cdef void cfuvw(double *c, int m1, int m2, int n2, double r2, double L,
                double *xs, double *ts, int size,
                double cosa, double tLA,
                double *us, double *vs, double *ws) nogil

cdef void cfN(double *c, double sina, double cosa, double tLA,
              double *xs, double *ts, int size, double r2, double L,
              double *F, int m1, int m2, int n2,
              double *c0, int m0, int n0, int funcnum,
              double *N, int NL_kinematics) nogil

cdef void cfv(double *c, int m1, int m2, int n2, double *xs, double *ts,
              int size, double r2, double L, double *vs) nogil

cdef void cfwx(double *c, int m1, int m2, int n2, double *xs, double *ts,
               int size, double L, double *outwx) nogil

cdef void cfwt(double *c, int m1, int m2, int n2, double *xs, double *ts,
               int size, double L, double *outwt) nogil
