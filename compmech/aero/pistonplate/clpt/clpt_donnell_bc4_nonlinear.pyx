#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
cimport numpy as np
import numpy as np
from scipy.sparse import coo_matrix
from libc.stdlib cimport malloc, free

#TODO not implemented
#from compmech.panels.imperfections.mgi cimport cfw0x, cfw0t
from compmech.integrate.integratev cimport trapz2d, simps2d

from compmech.panels.kpanels.kpanelt.clpt.clpt_commons_bc4 cimport (cfwx,
        cfwt, cfN)

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

cdef int num0 = 0
cdef int num1 = 3
cdef int e_num = 6
cdef double pi = 3.141592653589793
#TODO
cdef int funcnum = 2 # to use in the cfw0x and cfw0t functions

cdef struct cc_attributes:
    double *sina
    double *cosa
    double *tmin
    double *tmax
    double *r1
    double *L
    double *F
    int *m1
    int *n1
    double *coeffs
    double *c0
    int *m0
    int *n0

cdef int NL_kinematics=0 # to use cfstrain_donnell in cfN


def calc_k0L(np.ndarray[cDOUBLE, ndim=1] coeffs,
             double alpharad, double r1, double L, double tmin, double tmax,
             np.ndarray[cDOUBLE, ndim=2] F,
             int m1, int n1,
             int nx, int nt, int num_cores, str method,
             np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, j1, k1, l1

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] k0Lv

    cdef int fdim
    cdef cc_attributes args

    fdim = 3*m1*n1*m1*n1

    k0Lv = np.zeros((fdim,), dtype=DOUBLE)
    rows = np.zeros((fdim,), dtype=INT)
    cols = np.zeros((fdim,), dtype=INT)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    args.sina = &sina
    args.cosa = &cosa
    args.tmin = &tmin
    args.tmax = &tmax
    args.r1 = &r1
    args.L = &L
    args.F = &F[0,0]
    args.m1 = &m1
    args.n1 = &n1
    args.coeffs = &coeffs[0]
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = -L/2.
    xb = +L/2.
    ta = tmin
    tb = tmax

    # numerical integration
    if method=='trapz2d':
        trapz2d(<void *>cfk0L, fdim, k0Lv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<void *>cfk0L, fdim, k0Lv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    c = -1

    # k0L_11
    for i1 in range(m1):
        for j1 in range(n1):
            row = num0 + num1*((i1) + (j1)*m1)
            for k1 in range(m1):
                for l1 in range(n1):
                    col = num0 + num1*((k1) + (l1)*m1)
                    c += 1
                    rows[c] = row+0
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+1
                    cols[c] = col+2
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+2

    size = num0 + num1*m1*n1

    k0L = coo_matrix((k0Lv, (rows, cols)), shape=(size, size))

    return k0L


cdef void cfk0L(int npts, double *xs, double *ts, double *out,
                double *alphas, double *betas, void *args) nogil:
    cdef int i1, j1, k1, l1
    cdef int c, i, pos

    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66

    cdef double r, x, t, bx, bt, alpha, beta

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double sina, cosa, tmin, tmax, r1, L
    cdef int m0, n0, m1, n1
    cdef double wx, wt, w0x, w0t

    cdef cc_attributes *args_in=<cc_attributes *>args

    sina = args_in.sina[0]
    cosa = args_in.cosa[0]
    tmin = args_in.tmin[0]
    tmax = args_in.tmax[0]
    r1 = args_in.r1[0]
    L = args_in.L[0]
    F = args_in.F
    m1 = args_in.m1[0]
    n1 = args_in.n1[0]
    coeffs = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

    A11 = F[0]
    A12 = F[1]
    A16 = F[2]
    A22 = F[7]
    A26 = F[8]
    A66 = F[14]
    B11 = F[3]
    B12 = F[4]
    B16 = F[5]
    B22 = F[10]
    B26 = F[11]
    B66 = F[17]

    cdef double p00, p01, p02, p10, p11, p12, p20, p21, p22
    cdef double q02, q12, q22
    cdef double sini1bx, cosi1bx, sink1bx, cosk1bx
    cdef double sinj1bt, sinj1_bt, cosj1_bt, sinl1bt, cosl1_bt
    cdef double *vsini1bx = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1bx = <double *>malloc(m1 * sizeof(double))
    cdef double *vsinj1bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vsinj1_bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vcosj1_bt = <double *>malloc(n1 * sizeof(double))
    cdef double *k0Lq_1_q02 = <double *>malloc(m1*n1 * sizeof(double))
    cdef double *k0Lq_1_q12 = <double *>malloc(m1*n1 * sizeof(double))
    cdef double *k0Lq_1_q22 = <double *>malloc(m1*n1 * sizeof(double))
    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    #TODO
    #cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    #cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfwx(coeffs, m1, n1, xs, ts, npts, L, tmin, tmax, wxs)
    cfwt(coeffs, m1, n1, xs, ts, npts, L, tmin, tmax, wts)
    #TODO
    #cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    #cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    for i in range(npts):
        x = xs[i]
        t = ts[i]

        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        wx = wxs[i]
        wt = wts[i]
        #TODO
        #w0x = w0xs[i]
        #w0t = w0ts[i]
        w0x = 0.
        w0t = 0.
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(m1):
            vsini1bx[i1] = sin(pi*i1*bx)
            vcosi1bx[i1] = cos(pi*i1*bx)

        for j1 in range(n1):
            vsinj1bt[j1] = sin(pi*j1*bt)
            vsinj1_bt[j1] = sin(-pi*j1*bt)
            vcosj1_bt[j1] = cos(-pi*j1*bt)

        r = r1 - sina*(x + L/2.)

        c = -1

        # creating buffer q_1
        for k1 in range(m1):
            cosk1bx = vcosi1bx[k1]
            sink1bx = vsini1bx[k1]
            for l1 in range(n1):
                sinl1bt = vsinj1bt[l1]
                cosl1_bt = vcosj1_bt[l1]

                q02 = pi*cosk1bx*k1*sinl1bt*(w0x + wx)/L
                q12 = pi*cosl1_bt*l1*sink1bx*(w0t + wt)/((r*r)*(tmax - tmin))
                q22 = pi*(cosl1_bt*l1*sink1bx*(w0x + wx)/(tmax - tmin) + cosk1bx*k1*sinl1bt*(w0t + wt)/L)/r

                # create buffer
                pos = (k1)*n1 + (l1)
                k0Lq_1_q02[pos] = q02
                k0Lq_1_q12[pos] = q12
                k0Lq_1_q22[pos] = q22

        # k0L_11
        for i1 in range(m1):
            sini1bx = vsini1bx[i1]
            cosi1bx = vcosi1bx[i1]
            for j1 in range(n1):
                sinj1bt = vsinj1bt[j1]
                sinj1_bt = vsinj1_bt[j1]
                cosj1_bt = vcosj1_bt[j1]

                # p_1
                p00 = pi*A16*cosi1bx*j1*sinj1_bt/(tmax - tmin) + cosj1_bt*(-pi*A11*i1*r*sini1bx/L + A12*cosi1bx*sina)
                p01 = pi*A26*cosi1bx*j1*sinj1_bt/(tmax - tmin) + cosj1_bt*(-pi*A12*i1*r*sini1bx/L + A22*cosi1bx*sina)
                p02 = pi*A66*cosi1bx*j1*sinj1_bt/(tmax - tmin) + cosj1_bt*(-pi*A16*i1*r*sini1bx/L + A26*cosi1bx*sina)
                p10 = pi*A12*cosi1bx*j1*sinj1_bt/(tmax - tmin) - A16*cosj1_bt*(L*cosi1bx*sina + pi*i1*r*sini1bx)/L
                p11 = pi*A22*cosi1bx*j1*sinj1_bt/(tmax - tmin) - A26*cosj1_bt*(L*cosi1bx*sina + pi*i1*r*sini1bx)/L
                p12 = pi*A26*cosi1bx*j1*sinj1_bt/(tmax - tmin) - A66*cosj1_bt*(L*cosi1bx*sina + pi*i1*r*sini1bx)/L
                p20 = pi*B16*cosj1_bt*j1*(L*sina*sini1bx - 2*pi*cosi1bx*i1*r)/(L*r*(tmax - tmin)) + sinj1bt*(-pi*B12*cosi1bx*i1*sina/L + sini1bx*(A12*cosa + (pi*pi)*B11*(i1*i1)*r/(L*L) + (pi*pi)*B12*(j1*j1)/(r*(tmax - tmin)*(tmax - tmin))))
                p21 = pi*B26*cosj1_bt*j1*(L*sina*sini1bx - 2*pi*cosi1bx*i1*r)/(L*r*(tmax - tmin)) + sinj1bt*(-pi*B22*cosi1bx*i1*sina/L + sini1bx*(A22*cosa + (pi*pi)*B12*(i1*i1)*r/(L*L) + (pi*pi)*B22*(j1*j1)/(r*(tmax - tmin)*(tmax - tmin))))
                p22 = pi*B66*cosj1_bt*j1*(L*sina*sini1bx - 2*pi*cosi1bx*i1*r)/(L*r*(tmax - tmin)) + sinj1bt*(-pi*B26*cosi1bx*i1*sina/L + sini1bx*(A26*cosa + (pi*pi)*B16*(i1*i1)*r/(L*L) + (pi*pi)*B26*(j1*j1)/(r*(tmax - tmin)*(tmax - tmin))))

                for k1 in range(m1):
                    for l1 in range(n1):
                        # access buffer q_1
                        pos = (k1)*n1 + (l1)
                        q02 = k0Lq_1_q02[pos]
                        q12 = k0Lq_1_q12[pos]
                        q22 = k0Lq_1_q22[pos]

                        # k0L_11
                        c += 1
                        out[c] = beta*out[c] + alpha*(p00*q02 + p01*q12 + p02*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p10*q02 + p11*q12 + p12*q22)
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12 + p22*q22)

    free(wxs)
    free(wts)
    #TODO
    #free(w0xs)
    #free(w0ts)
    free(vsini1bx)
    free(vcosi1bx)
    free(vsinj1bt)
    free(vsinj1_bt)
    free(vcosj1_bt)
    free(k0Lq_1_q02)
    free(k0Lq_1_q12)
    free(k0Lq_1_q22)


def calc_kG(np.ndarray[cDOUBLE, ndim=1] coeffs,
            double alpharad, double r1, double L, double tmin, double tmax,
            np.ndarray[cDOUBLE, ndim=2] F,
            int m1, int n1,
            int nx, int nt, int num_cores, str method,
            np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, j1, k1, l1
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kGv

    cdef unsigned fdim
    cdef cc_attributes args

    fdim = 1*m1*n1*m1*n1

    rows = np.zeros((fdim,), dtype=INT)
    cols = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    args.sina = &sina
    args.cosa = &cosa
    args.tmin = &tmin
    args.tmax = &tmax
    args.r1 = &r1
    args.L = &L
    args.F = &F[0,0]
    args.m1 = &m1
    args.n1 = &n1
    args.coeffs = &coeffs[0]
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = -L/2.
    xb = +L/2.
    ta = tmin
    tb = tmax

    # numerical integration
    if method=='trapz2d':
        trapz2d(<void *>cfkG, fdim, kGv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<void *>cfkG, fdim, kGv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    c = -1

    # kG_11
    for i1 in range(m1):
        for j1 in range(n1):
            row = num0 + num1*((i1) + (j1)*m1)
            for k1 in range(m1):
                for l1 in range(n1):
                    col = num0 + num1*((k1) + (l1)*m1)
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+2

    size = num0 + num1*m1*n1

    kG = coo_matrix((kGv, (rows, cols)), shape=(size, size))

    return kG


cdef void cfkG(int npts, double *xs, double *ts, double *out,
               double *alphas, double *betas, void *args) nogil:
    cdef int i1, j1, k1, l1

    cdef double r, x, t, bx, bt, alpha, beta
    cdef int c, i, pos

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double r1, L, sina, cosa, tmin, tmax
    cdef int m0, n0, m1, n1
    cdef double Nxx, Ntt, Nxt

    cdef cc_attributes *args_in=<cc_attributes *>args

    sina = args_in.sina[0]
    cosa = args_in.cosa[0]
    tmin = args_in.tmin[0]
    tmax = args_in.tmax[0]
    r1 = args_in.r1[0]
    L = args_in.L[0]
    F = args_in.F
    m1 = args_in.m1[0]
    n1 = args_in.n1[0]
    coeffs = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]


    cdef double p20, p21
    cdef double q02, q12
    cdef double sini1bx, cosi1bx, sink1bx, cosk1bx
    cdef double sinj1bt, sinl1bt, cosl1bt, cosj1_bt
    cdef double *vsini1bx = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1bx = <double *>malloc(m1 * sizeof(double))
    cdef double *vsinj1bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vcosj1bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vcosj1_bt = <double *>malloc(n1 * sizeof(double))
    cdef double *kGq_1_q02 = <double *>malloc(m1*n1 * sizeof(double))
    cdef double *kGq_1_q12 = <double *>malloc(m1*n1 * sizeof(double))
    cdef double *Ns = <double *>malloc(e_num*npts * sizeof(double))

    cfN(coeffs, sina, cosa, xs, ts, npts, r1, L, tmin, tmax, F, m1, n1,
        c0, m0, n0, funcnum, Ns, NL_kinematics)

    for i in range(npts):
        x = xs[i]
        t = ts[i]

        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        Nxx = Ns[e_num*i + 0]
        Ntt = Ns[e_num*i + 1]
        Nxt = Ns[e_num*i + 2]
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(m1):
            vsini1bx[i1] = sin(pi*i1*bx)
            vcosi1bx[i1] = cos(pi*i1*bx)

        for j1 in range(n1):
            vsinj1bt[j1] = sin(pi*j1*bt)
            vcosj1bt[j1] = cos(pi*j1*bt)
            vcosj1_bt[j1] = cos(-pi*j1*bt)

        r = r1 - sina*(x + L/2.)

        c = -1

        # creating buffer q_1
        for k1 in range(m1):
            sink1bx = vsini1bx[k1]
            cosk1bx = vcosi1bx[i1]
            for l1 in range(n1):
                sinl1bt = vsinj1bt[l1]
                cosl1bt = vcosj1bt[l1]

                # q_1
                q02 = pi*cosk1bx*k1*sinl1bt/L
                q12 = pi*cosl1bt*l1*sink1bx/(r*tmax - r*tmin)

                # create buffer q_1
                pos = (k1)*n1 + (l1)
                kGq_1_q02[pos] = q02
                kGq_1_q12[pos] = q12

        # kG_11
        for i1 in range(m1):
            sini1bx = vsini1bx[i1]
            cosi1bx = vcosi1bx[i1]
            for j1 in range(n1):
                sinj1bt = vsinj1bt[j1]
                cosj1_bt = vcosj1_bt[j1]

                # p_1
                p20 = pi*Nxt*cosj1_bt*j1*sini1bx/(tmax - tmin) + pi*Nxx*cosi1bx*i1*r*sinj1bt/L
                p21 = pi*Ntt*cosj1_bt*j1*sini1bx/(tmax - tmin) + pi*Nxt*cosi1bx*i1*r*sinj1bt/L

                for k1 in range(m1):
                    for l1 in range(n1):
                        # access buffer q_1
                        pos = (k1)*n1 + (l1)
                        q02 = kGq_1_q02[pos]
                        q12 = kGq_1_q12[pos]

                        # kG_11
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12)

    free(Ns)

    free(vsini1bx)
    free(vcosi1bx)
    free(vsinj1bt)
    free(vcosj1bt)
    free(vcosj1_bt)
    free(kGq_1_q02)
    free(kGq_1_q12)


def calc_kLL(np.ndarray[cDOUBLE, ndim=1] coeffs,
             double alpharad, double r1, double L, double tmin, double tmax,
             np.ndarray[cDOUBLE, ndim=2] F,
             int m1, int n1,
             int nx, int nt, int num_cores, str method,
             np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef double sina, cosa, xa, xb, ta, tb
    cdef int c, row, col
    cdef int i1, j1, k1, l1
    cdef int size

    cdef np.ndarray[cINT, ndim=1] rows, cols
    cdef np.ndarray[cDOUBLE, ndim=1] kLLv

    cdef int fdim
    cdef cc_attributes args

    fdim = 1*m1*n1*m1*n1

    rows = np.zeros((fdim,), dtype=INT)
    cols = np.zeros((fdim,), dtype=INT)
    kLLv = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    args.sina = &sina
    args.cosa = &cosa
    args.tmin = &tmin
    args.tmax = &tmax
    args.r1 = &r1
    args.L = &L
    args.F = &F[0,0]
    args.m1 = &m1
    args.n1 = &n1
    args.coeffs = &coeffs[0]
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = -L/2.
    xb = +L/2.
    ta = tmin
    tb = tmax

    # numerical integration
    if method=='trapz2d':
        trapz2d(<void *>cfkLL, fdim, kLLv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<void *>cfkLL, fdim, kLLv, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    c = -1

    # kLL_11
    for i1 in range(m1):
        for j1 in range(n1):
            row = num0 + num1*((i1) + (j1)*m1)
            for k1 in range(m1):
                for l1 in range(n1):
                    col = num0 + num1*((k1) + (l1)*m1)
                    c += 1
                    rows[c] = row+2
                    cols[c] = col+2

    size = num0 + num1*m1*n1

    kLL = coo_matrix((kLLv, (rows, cols)), shape=(size, size))

    return kLL


cdef void cfkLL(int npts, double *xs, double *ts, double *out,
                double *alphas, double *betas, void *args) nogil:
    cdef int i1, j1, k1, l1
    cdef int c, i, pos

    cdef double A11, A12, A16, A22, A26, A66

    cdef double r, x, t, bx, bt, alpha, beta

    cdef double *F
    cdef double *coeffs
    cdef double *c0
    cdef double sina, r1, L, tmin, tmax
    cdef int m0, n0, m1, n1
    cdef double wx, wt, w0x, w0t

    cdef cc_attributes *args_in=<cc_attributes *>args
    sina = args_in.sina[0]
    r1 = args_in.r1[0]
    L = args_in.L[0]
    tmin = args_in.tmin[0]
    tmax = args_in.tmax[0]
    F = args_in.F
    m1 = args_in.m1[0]
    n1 = args_in.n1[0]
    coeffs = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

    A11 = F[0]
    A12 = F[1]
    A16 = F[2]
    A22 = F[7]
    A26 = F[8]
    A66 = F[14]

    cdef double p20, p21, p22
    cdef double q02, q12, q22
    cdef double sini1bx, cosi1bx, sink1bx, cosk1bx
    cdef double sinl1bt, sinj1bt, cosj1_bt, cosl1_bt
    cdef double *vsini1bx = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1bx = <double *>malloc(m1 * sizeof(double))
    cdef double *vsinj1bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vcosj1bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vcosj1_bt = <double *>malloc(n1 * sizeof(double))
    cdef double *kLLq_1_q02 = <double *>malloc(m1*n1 * sizeof(double))
    cdef double *kLLq_1_q12 = <double *>malloc(m1*n1 * sizeof(double))
    cdef double *kLLq_1_q22 = <double *>malloc(m1*n1 * sizeof(double))
    cdef double *wxs = <double *>malloc(npts * sizeof(double))
    cdef double *wts = <double *>malloc(npts * sizeof(double))
    #TODO
    #cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    #cdef double *w0ts = <double *>malloc(npts * sizeof(double))

    cfwx(coeffs, m1, n1, xs, ts, npts, L, tmin, tmax, wxs)
    cfwt(coeffs, m1, n1, xs, ts, npts, L, tmin, tmax, wts)
    #TODO
    #cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    #cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    for i in range(npts):
        x = xs[i]
        t = ts[i]

        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        wx = wxs[i]
        wt = wts[i]
        #TODO
        #w0x = w0xs[i]
        #w0t = w0ts[i]
        w0x = 0
        w0t = 0
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(m1):
            vsini1bx[i1] = sin(pi*i1*bx)
            vcosi1bx[i1] = cos(pi*i1*bx)

        for j1 in range(n1):
            vsinj1bt[j1] = sin(pi*j1*bt)
            vcosj1bt[j1] = cos(pi*j1*bt)
            vcosj1_bt[j1] = cos(-pi*j1*bt)

        r = r1 - sina*(x + L/2.)

        c = -1

        # creating buffer q_1
        for k1 in range(m1):
            sink1bx = vsini1bx[k1]
            cosk1bx = vcosi1bx[k1]
            for l1 in range(n1):
                sinl1bt = vsinj1bt[l1]
                cosl1_bt = vcosj1_bt[l1]

                q02 = pi*cosk1bx*k1*sinl1bt*(w0x + wx)/L
                q12 = pi*cosl1_bt*l1*sink1bx*(w0t + wt)/((r*r)*(tmax - tmin))
                q22 = pi*(cosl1_bt*l1*sink1bx*(w0x + wx)/(tmax - tmin) + cosk1bx*k1*sinl1bt*(w0t + wt)/L)/r

                # access buffer q_1
                pos = (k1)*n1 + (l1)
                kLLq_1_q02[pos] = q02
                kLLq_1_q12[pos] = q12
                kLLq_1_q22[pos] = q22

        for i1 in range(m1):
            sini1bx = vsini1bx[i1]
            cosi1bx = vcosi1bx[i1]
            for j1 in range(n1):
                sinj1bt = vsinj1bt[j1]
                cosj1_bt = vcosj1_bt[j1]

                # p_1
                p20 = pi*(cosj1_bt*j1*sini1bx*(A12*(w0t + wt) + A16*r*(w0x + wx))/(tmax - tmin) + cosi1bx*i1*r*sinj1bt*(A11*r*(w0x + wx) + A16*(w0t + wt))/L)/r
                p21 = pi*(cosj1_bt*j1*sini1bx*(A22*(w0t + wt) + A26*r*(w0x + wx))/(tmax - tmin) + cosi1bx*i1*r*sinj1bt*(A12*r*(w0x + wx) + A26*(w0t + wt))/L)/r
                p22 = pi*(cosj1_bt*j1*sini1bx*(A26*(w0t + wt) + A66*r*(w0x + wx))/(tmax - tmin) + cosi1bx*i1*r*sinj1bt*(A16*r*(w0x + wx) + A66*(w0t + wt))/L)/r

                for k1 in range(m1):
                    for l1 in range(n1):
                        # access buffer q_1
                        pos = (k1)*n1 + (l1)
                        q02 = kLLq_1_q02[pos]
                        q12 = kLLq_1_q12[pos]
                        q22 = kLLq_1_q22[pos]

                        # kLL_11
                        c += 1
                        out[c] = beta*out[c] + alpha*(p20*q02 + p21*q12 + p22*q22)

    free(wxs)
    free(wts)
    #TODO
    #free(w0xs)
    #free(w0ts)

    free(vsini1bx)
    free(vcosi1bx)
    free(vsinj1bt)
    free(vcosj1bt)
    free(vcosj1_bt)
    free(kLLq_1_q02)
    free(kLLq_1_q12)
    free(kLLq_1_q22)


def calc_fint_0L_L0_LL(np.ndarray[cDOUBLE, ndim=1] coeffs,
              double alpharad, double r1, double L, double tmin, double tmax,
              np.ndarray[cDOUBLE, ndim=2] F,
              int m1, int n1,
              int nx, int nt, int num_cores, str method,
              np.ndarray[cDOUBLE, ndim=1] c0, int m0, int n0):
    cdef cc_attributes args
    cdef double sina, cosa

    fdim = num0 + num1*m1*n1
    fint = np.zeros((fdim,), dtype=DOUBLE)

    sina = sin(alpharad)
    cosa = cos(alpharad)

    args.sina = &sina
    args.cosa = &cosa
    args.tmin = &tmin
    args.tmax = &tmax
    args.r1 = &r1
    args.L = &L
    args.F = &F[0,0]
    args.m1 = &m1
    args.n1 = &n1
    args.coeffs = &coeffs[0]
    args.c0 = &c0[0]
    args.m0 = &m0
    args.n0 = &n0

    xa = -L/2.
    xb = +L/2.
    ta = tmin
    tb = tmax

    # numerical integration
    if method=='trapz2d':
        trapz2d(<void *>cffint, fdim, fint, xa, xb, nx, ta, tb, nt,
                &args, num_cores)
    elif method=='simps2d':
        simps2d(<void *>cffint, fdim, fint, xa, xb, nx, ta, tb, nt,
                &args, num_cores)

    return fint


cdef void cffint(int npts, double *xs, double *ts, double *fint,
                 double *alphas, double *betas, void *args) nogil:
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double Nxx0, Ntt0, Nxt0, Mxx0, Mtt0, Mxt0
    cdef double NxxL, NttL, NxtL, MxxL, MttL, MxtL
    cdef double exx0, ett0, gxt0, kxx0, ktt0, kxt0
    cdef double exxL, ettL, gxtL, kxxL, kttL, kxtL
    cdef double x, t, bx, bt, wx, wt, w0x, w0t

    cdef double alpha, beta

    cdef double *F
    cdef double *c
    cdef double *c0
    cdef double sina, cosa, tmin, tmax, r, r1, L
    cdef int m0, n0, m1, n1
    cdef int i1, j1, i, col

    cdef cc_attributes *args_in=<cc_attributes *>args

    sina = args_in.sina[0]
    cosa = args_in.cosa[0]
    tmin = args_in.tmin[0]
    tmax = args_in.tmax[0]
    r1 = args_in.r1[0]
    L = args_in.L[0]
    F = args_in.F
    m1 = args_in.m1[0]
    n1 = args_in.n1[0]
    c = args_in.coeffs
    c0 = args_in.c0
    m0 = args_in.m0[0]
    n0 = args_in.n0[0]

    A11 = F[0]
    A12 = F[1]
    A16 = F[2]
    A22 = F[7]
    A26 = F[8]
    A66 = F[14]
    B11 = F[3]
    B12 = F[4]
    B16 = F[5]
    B22 = F[10]
    B26 = F[11]
    B66 = F[17]
    D11 = F[18]
    D12 = F[19]
    D16 = F[20]
    D22 = F[25]
    D26 = F[26]
    D66 = F[32]

    #TODO
    #cdef double *w0xs = <double *>malloc(npts * sizeof(double))
    #cdef double *w0ts = <double *>malloc(npts * sizeof(double))
    cdef double sini1bx, cosi1bx
    cdef double sinj1bt, cosj1bt, sinj1_bt, cosj1_bt
    cdef double dsini1bx, dsinj1bt
    cdef double *vsini1bx = <double *>malloc(m1 * sizeof(double))
    cdef double *vcosi1bx = <double *>malloc(m1 * sizeof(double))
    cdef double *vsinj1_bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vsinj1bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vcosj1_bt = <double *>malloc(n1 * sizeof(double))
    cdef double *vcosj1bt = <double *>malloc(n1 * sizeof(double))

    #TODO
    #cfw0x(xs, ts, npts, c0, L, m0, n0, w0xs, funcnum)
    #cfw0t(xs, ts, npts, c0, L, m0, n0, w0ts, funcnum)

    for i in range(npts):
        x = xs[i]
        t = ts[i]

        bx = (x + L/2.)/L
        bt = (t - tmin)/(tmax - tmin)

        r = r1 - sina*(x + L/2.)

        #TODO
        #w0x = w0xs[i]
        #w0t = w0ts[i]
        w0x = 0
        w0t = 0
        alpha = alphas[i]
        beta = betas[i]

        for i1 in range(m1):
            vsini1bx[i1] = sin(pi*i1*bx)
            vcosi1bx[i1] = cos(pi*i1*bx)

        for j1 in range(n1):
            vsinj1bt[j1] = sin(pi*j1*bt)
            vcosj1bt[j1] = cos(pi*j1*bt)
            vsinj1_bt[j1] = sin(-pi*j1*bt)
            vcosj1_bt[j1] = cos(-pi*j1*bt)

        # wx and wt
        wx = 0.
        wt = 0.

        for j1 in range(n1):
            sinj1bt = sin(j1*pi*bt)
            dsinj1bt = j1*pi/(tmax-tmin)*cos(j1*pi*bt)
            for i1 in range(m1):
                col = num0 + num1*((i1) + (j1)*m1)
                dsini1bx = i1*pi/L*cos(i1*pi*bx)
                sini1bx = sin(i1*pi*bx)

                wx += c[col+2]*dsini1bx*sinj1bt

                wt += c[col+2]*sini1bx*dsinj1bt

        # strains
        exx0 = 0.
        ett0 = 0.
        gxt0 = 0.
        kxx0 = 0.
        ktt0 = 0.
        kxt0 = 0.

        exxL = 0.
        ettL = 0.
        gxtL = 0.
        kxxL = 0.
        kttL = 0.
        kxtL = 0.

        for j1 in range(n1):
            sinj1bt = vsinj1bt[j1]
            cosj1bt = vcosj1bt[j1]
            cosj1_bt = vcosj1_bt[j1]
            for i1 in range(m1):
                sini1bx = vsini1bx[i1]
                cosi1bx = vcosi1bx[i1]
                col = num0 + num1*((i1) + (j1)*m1)

                exx0 += -pi*c[col+0]*cosj1_bt*i1*sini1bx/L
                exxL += c[col+2]*(pi*cosi1bx*i1*sinj1bt*w0x/L + 0.5*pi*cosi1bx*i1*sinj1bt*wx/L)

                ett0 += (c[col+0]*cosi1bx*cosj1_bt*sina/r
                         - pi*c[col+1]*cosi1bx*j1*sinj1bt/(r*tmax - r*tmin)
                         + c[col+2]*cosa*sini1bx*sinj1bt/r)
                ettL += c[col+2]*(pi*cosj1bt*j1*sini1bx*w0t/((r*r)*(tmax - tmin)) + 0.5*pi*cosj1bt*j1*sini1bx*wt/((r*r)*(tmax - tmin)))

                gxt0 += (- pi*c[col+0]*cosi1bx*j1*sinj1bt/(r*tmax - r*tmin)
                         - c[col+1]*cosj1_bt*(L*cosi1bx*sina + pi*i1*r*sini1bx)/(L*r))
                gxtL += c[col+2]*(pi*cosj1bt*j1*sini1bx*w0x/(r*(tmax - tmin)) + 0.5*pi*cosj1bt*j1*sini1bx*wx/(r*(tmax - tmin)) + pi*cosi1bx*i1*sinj1bt*w0t/(L*r) + 0.5*pi*cosi1bx*i1*sinj1bt*wt/(L*r))

                kxx0 += (pi*pi)*c[col+2]*(i1*i1)*sini1bx*sinj1bt/(L*L)

                ktt0 += pi*c[col+2]*sinj1bt*(pi*L*(j1*j1)*sini1bx - cosi1bx*i1*r*sina*(tmax - tmin)*(tmax - tmin))/(L*(r*r)*(tmax - tmin)*(tmax - tmin))

                kxt0 += pi*c[col+2]*cosj1_bt*j1*(L*sina*sini1bx - 2*pi*cosi1bx*i1*r)/(L*(r*r)*(tmax - tmin))


        # stresses
        Nxx0 = A11*exx0 + A12*ett0 + A16*gxt0 + B11*kxx0 + B12*ktt0 + B16*kxt0
        Ntt0 = A12*exx0 + A22*ett0 + A26*gxt0 + B12*kxx0 + B22*ktt0 + B26*kxt0
        Nxt0 = A16*exx0 + A26*ett0 + A66*gxt0 + B16*kxx0 + B26*ktt0 + B66*kxt0
        Mxx0 = B11*exx0 + B12*ett0 + B16*gxt0 + D11*kxx0 + D12*ktt0 + D16*kxt0
        Mtt0 = B12*exx0 + B22*ett0 + B26*gxt0 + D12*kxx0 + D22*ktt0 + D26*kxt0
        Mxt0 = B16*exx0 + B26*ett0 + B66*gxt0 + D16*kxx0 + D26*ktt0 + D66*kxt0

        NxxL = A11*exxL + A12*ettL + A16*gxtL + B11*kxxL + B12*kttL + B16*kxtL
        NttL = A12*exxL + A22*ettL + A26*gxtL + B12*kxxL + B22*kttL + B26*kxtL
        NxtL = A16*exxL + A26*ettL + A66*gxtL + B16*kxxL + B26*kttL + B66*kxtL
        MxxL = B11*exxL + B12*ettL + B16*gxtL + D11*kxxL + D12*kttL + D16*kxtL
        MttL = B12*exxL + B22*ettL + B26*gxtL + D12*kxxL + D22*kttL + D26*kxtL
        MxtL = B16*exxL + B26*ettL + B66*gxtL + D16*kxxL + D26*kttL + D66*kxtL

        for j1 in range(n1):
            sinj1_bt = vsinj1_bt[j1]
            sinj1bt = vsinj1bt[j1]
            cosj1_bt = vcosj1_bt[j1]
            for i1 in range(m1):
                sini1bx = vsini1bx[i1]
                cosi1bx = vcosi1bx[i1]
                col = num0 + num1*((i1) + (j1)*m1)

                fint[col+0] = beta*(fint[col+0]) + alpha*((pi*L*NxtL*cosi1bx*j1*sinj1_bt + cosj1_bt*(tmax - tmin)*(L*NttL*cosi1bx*sina - pi*NxxL*i1*r*sini1bx))/(L*(tmax - tmin)))
                fint[col+1] = beta*(fint[col+1]) + alpha*((pi*L*NttL*cosi1bx*j1*sinj1_bt + NxtL*cosj1_bt*(-tmax + tmin)*(L*cosi1bx*sina + pi*i1*r*sini1bx))/(L*(tmax - tmin)))
                fint[col+2] = beta*(fint[col+2]) + alpha*(sini1bx*(pi*cosj1_bt*j1*(Nxt0 + NxtL)*(w0x + wx)/(tmax - tmin) + sinj1bt*((pi*pi)*MttL*(j1*j1)/(r*(tmax - tmin)*(tmax - tmin)) + NttL*cosa + (pi*pi)*MxxL*(i1*i1)*r/(L*L))) - pi*cosi1bx*i1*sinj1bt*(MttL*sina - Nxt0*(w0t + wt) - NxtL*w0t - NxtL*wt - Nxx0*r*w0x - Nxx0*r*wx - NxxL*r*w0x - NxxL*r*wx)/L - pi*cosj1_bt*j1*(-L*sini1bx*(MxtL*sina + (Ntt0 + NttL)*(w0t + wt)) + 2*pi*MxtL*cosi1bx*i1*r)/(L*r*(tmax - tmin)))

    #TODO
    #free(w0xs)
    #free(w0ts)
    free(vsini1bx)
    free(vcosi1bx)
    free(vsinj1_bt)
    free(vsinj1bt)
    free(vcosj1_bt)
    free(vcosj1bt)
