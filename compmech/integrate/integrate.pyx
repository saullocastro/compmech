#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
import numpy as np
cimport numpy as np


ctypedef np.float_t cDOUBLE
DOUBLE = np.float64


cdef void trapz_quad(int nx, double *xis, double *weights) nogil:
    cdef int i
    cdef double hxi
    hxi = 2./(nx-1.)

    for i in range(nx):
        xis[i] = -1. + 2.*i/(nx-1)
        if i == 0 or i == (nx-1):
            weights[i] = hxi/2.
        else:
            weights[i] = hxi


def trapz2d_points(double xmin, double xmax, int nx,
                   double ymin, double ymax, int ny):
    cdef int i, j, npts, k
    cdef double c, hx, hy, x, y, alpha, beta
    cdef np.ndarray[cDOUBLE, ndim=1] xs, ys, xs2, ys2, alphas, betas

    nx -= 1
    ny -= 1

    xs = np.linspace(xmin, xmax, nx+1).astype(DOUBLE)
    ys = np.linspace(ymin, ymax, ny+1).astype(DOUBLE)

    npts = (nx+1)*(ny+1)
    xs2 = np.zeros(npts, DOUBLE)
    ys2 = np.zeros(npts, DOUBLE)
    alphas = np.zeros(npts, DOUBLE)
    betas = np.zeros(npts, DOUBLE)

    hx = (xmax-xmin)/nx
    hy = (ymax-ymin)/ny
    c = 1/4.*hx*hy

    # building integration points
    k = -1
    for i, j in ((0, 0), (nx, 0), (0, ny), (nx, ny)):
        k += 1
        xs2[k] = xs[i]
        ys2[k] = ys[j]
        alphas[k] = 1*c
        betas[k] = 1
    for i in range(1, nx): # i from 1 to nx-1
        for j in (0, ny):
            k += 1
            xs2[k] = xs[i]
            ys2[k] = ys[j]
            alphas[k] = 2*c
            betas[k] = 1
    for i in (0, nx):
        for j in range(1, ny): # j from 1 to ny-1
            k += 1
            xs2[k] = xs[i]
            ys2[k] = ys[j]
            alphas[k] = 2*c
            betas[k] = 1
    for i in range(1, nx): # i from 1 to nx-1
        for j in range(1, ny): # j from 1 to ny-1
            k += 1
            xs2[k] = xs[i]
            ys2[k] = ys[j]
            alphas[k] = 4*c
            betas[k] = 1

    return xs2, ys2, alphas, betas


def simps2d_points(double xmin, double xmax, int nx,
                   double ymin, double ymax, int ny):
    cdef int i, j, npts, k
    cdef double c, hx, hy, x, y, alpha, beta
    cdef np.ndarray[cDOUBLE, ndim=1] xs, ys, xs2, ys2, alphas, betas

    if nx % 2 != 0:
        nx += 1
    if ny % 2 != 0:
        ny += 1

    nx /= 2
    ny /= 2

    xs = np.linspace(xmin, xmax, (2*nx+1)).astype(DOUBLE)
    ys = np.linspace(ymin, ymax, (2*ny+1)).astype(DOUBLE)

    npts = (2*nx + 1)*(2*ny + 1)
    xs2 = np.zeros(npts, DOUBLE)
    ys2 = np.zeros(npts, DOUBLE)
    alphas = np.zeros(npts, DOUBLE)
    betas = np.zeros(npts, DOUBLE)

    hx = (xmax-xmin)/(2*nx)
    hy = (ymax-ymin)/(2*ny)
    c = 1/9.*hx*hy

    # building integration points
    k = -1
    for i,j in ((0, 0), (2*nx, 0), (0, 2*ny), (2*nx, 2*ny)):
        k += 1
        xs2[k] = xs[i]
        ys2[k] = ys[j]
        alphas[k] = 1*c
        betas[k] = 1
    for i in (0, 2*nx):
        for j in range(1, ny+1): # from 1 to ny
            k += 1
            xs2[k] = xs[i]
            ys2[k] = ys[2*j-1]
            alphas[k] = 4*c
            betas[k] = 1
    for i in (0, 2*nx):
        for j in range(1, ny): # from 1 to ny-1
            k += 1
            xs2[k] = xs[i]
            ys2[k] = ys[2*j]
            alphas[k] = 2*c
            betas[k] = 1
    for i in range(1, nx+1): # from 1 to nx
        for j in (0, 2*ny):
            k += 1
            xs2[k] = xs[2*i-1]
            ys2[k] = ys[j]
            alphas[k] = 4*c
            betas[k] = 1
    for i in range(1, nx): # from 1 to nx-1
        for j in (0, 2*ny):
            k += 1
            xs2[k] = xs[2*i]
            ys2[k] = ys[j]
            alphas[k] = 2*c
            betas[k] = 1
    for i in range(1, nx+1): # from 1 to nx
        for j in range(1, ny+1): # from 1 to ny
            k += 1
            xs2[k] = xs[2*i-1]
            ys2[k] = ys[2*j-1]
            alphas[k] = 16*c
            betas[k] = 1
    for i in range(1, nx+1):
        for j in range(1, ny):
            k += 1
            xs2[k] = xs[2*i-1]
            ys2[k] = ys[2*j]
            alphas[k] = 8*c
            betas[k] = 1
    for i in range(1, nx):
        for j in range(1, ny+1):
            k += 1
            xs2[k] = xs[2*i]
            ys2[k] = ys[2*j-1]
            alphas[k] = 8*c
            betas[k] = 1
    for i in range(1, nx):
        for j in range(1, ny):
            k += 1
            xs2[k] = xs[2*i]
            ys2[k] = ys[2*j]
            alphas[k] = 4*c
            betas[k] = 1

    return xs2, ys2, alphas, betas


def python_trapz_quad(int n, np.ndarray[cDOUBLE, ndim=1] xis,
                     np.ndarray[cDOUBLE, ndim=1] weights):
    trapz_quad(n, &xis[0], &weights[0])

