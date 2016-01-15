#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
import numpy as np
cimport numpy as np
from cython.parallel import prange

from compmech.integrate.integrate import trapz2d_points, simps2d_points


DOUBLE = np.float64

ctypedef void (*f_type)(int npts, double *xs, double *ts, double *out,
                        double *alphas, double *betas, void *args) nogil

cdef int integratev(void *fin, int fdim, np.ndarray[cDOUBLE, ndim=1] out,
                    double xmin, double xmax, int nx,
                    double ymin, double ymax, int ny,
                    void *args, int num_cores, method='trapz'):
    """Integration of vecto-valued functions
    """
    cdef int i, npts, k, rest
    cdef np.ndarray[cDOUBLE, ndim=1] xs2, ys2, alphas, betas
    cdef np.ndarray[cDOUBLE, ndim=2] outs
    cdef f_type f
    f = <f_type>fin

    outs = np.zeros((num_cores, out.shape[0]), DOUBLE)
    if method == 'trapz':
        xs2, ys2, alphas, betas = trapz2d_points(xmin, xmax, nx, ymin, ymax, ny)
    elif method == 'simps':
        xs2, ys2, alphas, betas = simps2d_points(xmin, xmax, nx, ymin, ymax, ny)
    else:
        raise ValueError('Method not recognized!')

    npts = xs2.shape[0]
    k = npts/num_cores
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
            schedule='static'):
        f(k, &xs2[k*i], &ys2[k*i], &outs[i, 0], &alphas[k*i], &betas[k*i],
          args=args)

    rest = npts - k*num_cores
    assert rest >= 0, 'ERROR rest < 0!'
    if rest>0:
        f(rest, &xs2[k*num_cores], &ys2[k*num_cores], &outs[0, 0],
          &alphas[k*num_cores], &betas[k*num_cores], args=args)

    np.sum(outs, axis=0, out=out)

    return 0


cdef int trapz_wp(int npts, double xa, double xb, double *weights,
                    double *pts) nogil:
    cdef int i
    cdef double factor
    factor = (xb - xa)/(2*npts)
    for i in range(1, npts-1):
        weights[i] = 2*factor
        pts[i] = xa + i*(xb - xa)/(npts-1)
    weights[0] = factor
    weights[npts-1] = factor
    pts[0] = xa
    pts[npts-1] = xb
