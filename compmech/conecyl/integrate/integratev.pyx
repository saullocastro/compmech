#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
import numpy as np
cimport numpy as np
from cython.parallel import prange
from scipy.linalg import inv

DOUBLE = np.float64

cdef int trapz2d(f_type f, int fdim, np.ndarray[cDOUBLE, ndim=1] final_out,
                 double xmin, double xmax, int m,
                 double ymin, double ymax, int n,
                 void *args, int num_cores):
    '''Integrate `f` for two variables
    '''
    cdef int i, j
    cdef double c, hx, hy, x, y, alpha, beta
    cdef np.ndarray[cDOUBLE, ndim=1] xs, ys
    cdef np.ndarray[cDOUBLE, ndim=2] outs
    cdef np.ndarray[cDOUBLE, ndim=1] o1, o2, o3, o4, o5, o6, o7, o8
    o1 = np.zeros(fdim, DOUBLE)
    o2 = np.zeros(fdim, DOUBLE)
    o3 = np.zeros(fdim, DOUBLE)
    o4 = np.zeros(fdim, DOUBLE)
    o5 = np.zeros(fdim, DOUBLE)
    o6 = np.zeros(fdim, DOUBLE)
    o7 = np.zeros(fdim, DOUBLE)
    o8 = np.zeros(fdim, DOUBLE)

    xs = np.linspace(xmin, xmax, m+1)
    ys = np.linspace(ymin, ymax, n+1)

    hx = (xmax-xmin)/m
    hy = (ymax-ymin)/n
    c = 1/4.*hx*hy

    # building integration points
    pts = []
    for i,j in ( (0, 0), (m, 0), (0, n), (m, n) ):
        x = xs[i]
        y = ys[j]
        alpha = 1*c
        beta = 1
        pts.append([x, y, alpha, beta])
    for i in range(1, m): # i from 1 to m-1
        for j in (0, n):
            x = xs[i]
            y = ys[j]
            alpha = 2*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in (0, m):
        for j in range(1, n): # j from 1 to n-1
            x = xs[i]
            y = ys[j]
            alpha = 2*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in range(1, m): # i from 1 to m-1
        for j in range(1, n): # j from 1 to n-1
            x = xs[i]
            y = ys[j]
            alpha = 4*c
            beta = 1
            pts.append([x, y, alpha, beta])

    outs = np.array([o1, o2, o3, o4, o5, o6, o7, o8])
    distributed = [pts[i::num_cores] for i in range(num_cores)]

    #for i in prange(num_cores, nogil=True, num_threads=num_cores,
    #                           schedule='static'):
    for i in range(num_cores):
        for pt in distributed[i]:
            x = pt[0]
            y = pt[1]
            alpha = pt[2]
            beta = pt[3]
            f(x, y, out=&outs[i, 0], alpha=alpha, beta=beta,
                      args=&args)

    outs.sum(axis=0, out=final_out)

    return 0

cdef int simps2d(f_type f, int fdim, np.ndarray[cDOUBLE, ndim=1] final_out,
                 double xmin, double xmax, int m,
                 double ymin, double ymax, int n,
                 void *args, int num_cores):
    '''Integrate `f` for two variables
    This function must return a np.ndarray.

    '''
    cdef int i, j
    cdef double c, hx, hy, x, y, alpha, beta
    cdef np.ndarray[cDOUBLE, ndim=1] xs, ys
    cdef np.ndarray[cDOUBLE, ndim=2] outs
    cdef np.ndarray[cDOUBLE, ndim=1] o1, o2, o3, o4, o5, o6, o7, o8
    o1 = np.zeros(fdim, DOUBLE)
    o2 = np.zeros(fdim, DOUBLE)
    o3 = np.zeros(fdim, DOUBLE)
    o4 = np.zeros(fdim, DOUBLE)
    o5 = np.zeros(fdim, DOUBLE)
    o6 = np.zeros(fdim, DOUBLE)
    o7 = np.zeros(fdim, DOUBLE)
    o8 = np.zeros(fdim, DOUBLE)

    try:
        assert m%2==0
    except AssertionError:
        print 'WARNING - incrementing m+=1'
        m += 1
    try:
        assert n%2==0
    except AssertionError:
        print 'WARNING - incrementing n+=1'
        n += 1

    m /= 2
    n /= 2

    xs = np.linspace(xmin, xmax, (2*m+1))
    ys = np.linspace(ymin, ymax, (2*n+1))

    hx = (xmax-xmin)/(2*m)
    hy = (ymax-ymin)/(2*n)
    c = 1/9.*hx*hy

    # building integration points
    pts = []
    for i,j in ((0,0), (2*m,0), (0,2*n), (2*m,2*n)):
        x = xs[i]
        y = ys[j]
        alpha = 1*c
        beta = 1
        pts.append([x, y, alpha, beta])
    for i in (0, 2*m):
        for j in range(1, n+1):
            x = xs[i]
            y = ys[2*j-1]
            alpha = 4*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in range(1, m+1):
        for j in (0, 2*n):
            x = xs[2*i-1]
            y = ys[j]
            alpha = 4*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in (0, 2*m):
        for j in range(1, n):
            x = xs[i]
            y = ys[2*j]
            alpha = 2*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in range(1, m):
        for j in (0, 2*n):
            x = xs[2*i]
            y = ys[j]
            alpha = 2*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in range(1, m+1):
        for j in range(1, n+1):
            x = xs[2*i-1]
            y = ys[2*j-1]
            alpha = 16*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in range(1, m+1):
        for j in range(1, n):
            x = xs[2*i-1]
            y = ys[2*j]
            alpha = 8*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in range(1, m):
        for j in range(1, n+1):
            x = xs[2*i]
            y = ys[2*j-1]
            alpha = 8*c
            beta = 1
            pts.append([x, y, alpha, beta])
    for i in range(1, m):
        for j in range(1, n):
            x = xs[2*i]
            y = ys[2*j]
            alpha = 4*c
            beta = 1
            pts.append([x, y, alpha, beta])

    outs = np.array([o1, o2, o3, o4, o5, o6, o7, o8])

    distributed = [pts[i::num_cores] for i in range(num_cores)]

    #for i in prange(num_cores, nogil=True, num_threads=num_cores,
                               #schedule='static'):
    for i in range(num_cores):
        for pt in distributed[i]:
            x = pt[0]
            y = pt[1]
            alpha = pt[2]
            beta = pt[3]
            f(x, y, out=&outs[i, 0], alpha=alpha, beta=beta,
                      args=&args)

    outs.sum(axis=0, out=final_out)

    return 0

