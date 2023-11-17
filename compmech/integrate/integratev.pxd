cdef int integratev(void *f, int fdim, double *final_out,
                    double xmin, double xmax, int m,
                    double ymin, double ymax, int n,
                    void *args, int num_cores, str method)

cdef int trapz_wp(int npts, double xa, double xb, double *weights,
                  double *pts) nogil
