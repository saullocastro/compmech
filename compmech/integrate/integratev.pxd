cimport numpy as np

ctypedef np.float_t cDOUBLE

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
