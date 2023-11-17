#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
import numpy as np
from cython.parallel import prange

from compmech.integrate.integrate import trapz2d_points, simps2d_points


DOUBLE = np.float64

ctypedef void (*f_type)(int npts, double *xs, double *ys, double *out,
                        double *alphas, double *betas, void *args) nogil

cdef extern from "math.h":
    double sin(double x) nogil
    double atan(double x) nogil


cdef int integratev(void *fin, int fdim, double *out,
                    double xmin, double xmax, int nx,
                    double ymin, double ymax, int ny,
                    void *args, int num_cores, str method):
    """Integration of vector-valued functions
    """
    cdef int i, npts, k, rest
    cdef double [:] xs2, ys2, alphas, betas, out_tmp
    cdef double [:, ::1] outs
    cdef f_type f
    f = <f_type>fin

    outs = np.zeros((num_cores, fdim), DOUBLE)
    if method == 'trapz2d':
        xs2, ys2, alphas, betas = trapz2d_points(xmin, xmax, nx, ymin, ymax, ny)
    elif method == 'simps2d':
        xs2, ys2, alphas, betas = simps2d_points(xmin, xmax, nx, ymin, ymax, ny)
    else:
        print('RuntimeError: Method {0} not recognized!'.format(method))
        raise

    npts = xs2.shape[0]
    k = npts/num_cores
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
            schedule='static'):
        f(k, &xs2[k*i], &ys2[k*i], &outs[i, 0], &alphas[k*i], &betas[k*i],
          args=args)

    rest = npts - k*num_cores
    assert rest >= 0, 'ERROR rest < 0!'
    if rest > 0:
        f(rest, &xs2[k*num_cores], &ys2[k*num_cores], &outs[0, 0],
          &alphas[k*num_cores], &betas[k*num_cores], args=args)

    out_tmp = np.sum(outs, axis=0)

    for i in range(fdim):
        out[i] = out_tmp[i]

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


cdef void _fsinsin(int npts, double *xs, double *ys, double *out,
                   double *alphas, double *betas, void *args) nogil:
    cdef int i
    cdef double pi, x, y

    with gil:
        pi = 4*atan(1.)

        for i in range(npts):
            x = xs[i]
            y = ys[i]
            out[0] = out[0]*betas[i] + alphas[i]*(sin(x*pi)*sin(y*pi))
            out[1] = out[1]*betas[i] + alphas[i]*(sin(3*x*pi)*sin(3*y*pi))
            out[2] = out[2]*betas[i] + alphas[i]*(sin(5*x*pi)*sin(5*y*pi))


def _test_integratev(nx, ny, method):
    cdef double args
    cdef double [:] out

    args = 0.

    out = np.zeros((3,), dtype=DOUBLE)

    integratev(<void *>_fsinsin, 3, &out[0], 0., 1., nx, 0., 1., ny,
               <void *>&args, 1, method)

    return np.asarray(out)
