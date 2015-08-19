cdef void cfN(double *c, double sina, double cosa, double tLA,
              double *xs, double *ts, int size,
              double r2, double L, double *F,
              int m1, int m2, int n2,
              double *c0, int m0, int n0, int funcnum,
              double *Ns, int NL_kinematics) nogil

cdef void cfuvw(double *c, int m1, int m2, int n2, double r2,
                double L, double *xs, double *ts, int size,
                double cosa, double tLA,
                double *us, double *vs, double *ws,
                double *phixs, double *phits) nogil

cdef void cfwx(double *c, int m1, int m2, int n2, double L,
               double *xs, double *ts, int size, double *wxs) nogil

cdef void cfwt(double *c, int m1, int m2, int n2, double L,
               double *xs, double *ts, int size, double *wts) nogil

