ctypedef void *cfstraintype(double *c, double sina, double cosa, double tLA,
                            double x, double t, double r, double r2, double L,
                            int m1, int m2, int n2, double *e) nogil

cdef void cfN(double *c, double sina, double cosa, double tLA,
              double x, double t, double r, double r2, double L, double *F,
              int m1, int m2, int n2, double *N, int NL_kinematics) nogil

cdef void cfuvw(double *c, int m1, int m2, int n2, double r2, double L,
                double x, double t,
                double cosa, double tLA, double *uvw) nogil

cdef void cfphix(double *c, int m1, int m2, int n2, double L,
                 double x, double t, double *refphix) nogil

cdef void cfphit(double *c, int m1, int m2, int n2, double L,
                 double x, double t, double *refphit) nogil

