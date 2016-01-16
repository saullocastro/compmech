cimport numpy as np

ctypedef np.float_t cDOUBLE

cdef int integratev(void *f, int fdim, np.ndarray[cDOUBLE, ndim=1] final_out,
                    double xmin, double xmax, int m,
                    double ymin, double ymax, int n,
                    void *args, int num_cores, str method)

cdef int trapz_wp(int npts, double xa, double xb, double *weights,
                  double *pts) nogil
