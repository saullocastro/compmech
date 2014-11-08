cimport numpy as np

ctypedef np.float_t cDOUBLE

ctypedef void (*f_type)(int npts, double *xs, double *ts, double *out,
                        double *alphas, double *betas, void *args) nogil

cdef int trapz2d(void *f, int fdim, np.ndarray[cDOUBLE, ndim=1] final_out,
                 double xmin, double xmax, int m,
                 double ymin, double ymax, int n,
                 void *args, int num_cores)

cdef int simps2d(void *f, int fdim, np.ndarray[cDOUBLE, ndim=1] final_out,
                 double xmin, double xmax, int m,
                 double ymin, double ymax, int n,
                 void *args, int num_cores)

cdef int trapz_wp(int npts, double xa, double xb, double *weights,
                  double *pts) nogil
