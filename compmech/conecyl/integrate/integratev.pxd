cimport numpy as np

ctypedef np.float_t cDOUBLE

ctypedef int (*f_type) (double x, double t, double *out,
                        double alpha, double beta, void *args)

cdef int trapz2d(f_type f, int fdim, np.ndarray[cDOUBLE, ndim=1] final_out,
                 double xmin, double xmax, int m,
                 double ymin, double ymax, int n,
                 void *args, int num_cores)

cdef int simps2d(f_type f, int fdim, np.ndarray[cDOUBLE, ndim=1] final_out,
                 double xmin, double xmax, int m,
                 double ymin, double ymax, int n,
                 void *args, int num_cores)
