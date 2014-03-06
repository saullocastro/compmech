ctypedef void *cfstraintype(double *c, double sina, double cosa, double tLA,
                            double x, double t, double r, double r2, double L,
                            int m1, int m2, int n2, double *e) nogil

cdef void cfuvw(double *c, int m1, int m2, int n2, double r2, double L,
                double x, double t,
                double cosa, double tLA, double *uvw) nogil

cdef void cfuvw_x(double *c, int m1, int m2, int n2, double r2, double L,
                  double x, double t,
                  double cosa, double tLA, double *uvw) nogil

cdef void cfuvw_t(double *c, int m1, int m2, int n2, double L,
                  double x, double t,
                  double cosa, double tLA, double *uvw) nogil

cdef void cfN(double *c, double sina, double cosa, double tLA,
              double x, double t, double r, double r2, double L, double *F,
              int m1, int m2, int n2, double *N, int NL_kinematics) nogil

cdef void cfv(double *c, int m1, int m2, int n2, double r2,
              double x, double t, double L, double *v) nogil

cdef void cfvx(double *c, int m1, int m2, int n2, double r2,
               double x, double t, double L, double *vx) nogil

cdef void cfvt(double *c, int m1, int m2, int n2,
               double x, double t, double L, double *vt) nogil

cdef void cfw(double *c, int m1, int m2, int n2,
              double x, double t, double L, double *w) nogil

cdef void cfwx(double *c, int m1, int m2, int n2,
               double x, double t, double L, double *wx) nogil

cdef void cfwt(double *c, int m1, int m2, int n2,
               double x, double t, double L, double *wt) nogil

