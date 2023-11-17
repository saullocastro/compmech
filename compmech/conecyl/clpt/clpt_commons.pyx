#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange

from compmech.conecyl.imperfections.mgi cimport cfw0x, cfw0t


DOUBLE = np.float64

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil

cdef int i0 = 0
cdef int j0 = 1
cdef int num0 = 3
cdef int num1 = 3
cdef int num2 = 6
cdef int e_num = 6
cdef int castro = 0
cdef int num_cores = 4
cdef double pi=3.141592653589793

def fstrain(str model, double [:] c,
            double sina, double cosa, double tLA,
            double [:] xs,
            double [:] ts,
            double r2, double L, int m1, int m2, int n2,
            double [:] c0, int m0, int n0, int funcnum,
            int NL_kinematics):
    model = model.lower()
    if not 'clpt' in model:
        raise ValueError('{0} is not a valid CLPT model!'.format(model))
    try:
        if model=='clpt_donnell_bc1' or model=='clpt_sanders_bc1':
        from clpt_commons_bc1 cimport cfuvw, cfwx, cfwt
    except:
        raise ImportError('Could not cimport {0}'.format(model))
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef double [:] es
    cdef cfstraintype *cfstrain
    if NL_kinematics==0:
        cfstrain = &cfstrain_donnell
    elif NL_kinematics==1:
        cfstrain = &cfstrain_sanders
    size = xs.shape[0]
    es = np.zeros((size*e_num), dtype=DOUBLE)
    cfstrain(&c[0], sina, cosa, tLA, &xs[0], &ts[0], size, r2, L, m1, m2, n2,
              &c0[0], m0, n0, funcnum, &es[0])
    return es

def fuvw(str model, double [:] c, int m1, int m2, int n2,
         double alpharad, double r2, double L, double tLA,
         double [:] xs, double [:] ts):
    model = model.lower()
    if not 'clpt' in model:
        raise ValueError('{0} is not a valid CLPT model!'.format(model))
    try:
        if model=='clpt_donnell_bc1' or model=='clpt_sanders_bc1':
            from clpt_commons_bc1 cimport cfuvw, cfwx, cfwt
        elif model==
    except:
        raise ImportError('Could not cimport {0}'.format(model))
    cdef int i, core_size
    cdef double sina, cosa
    cdef double[:, ::1] us, vs, ws, phixs, phits
    cdef double[:, ::1] xs_r, ts_r

    sina = sin(alpharad)
    cosa = cos(alpharad)

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_r = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ts_r = np.hstack((ts, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_r = xs.reshape(num_cores, -1)
        ts_r = ts.reshape(num_cores, -1)

    core_size = xs_r.shape[1]

    us = np.zeros((num_cores, core_size), dtype=DOUBLE)
    vs = np.zeros((num_cores, core_size), dtype=DOUBLE)
    ws = np.zeros((num_cores, core_size), dtype=DOUBLE)
    phixs = np.zeros((num_cores, core_size), dtype=DOUBLE)
    phits = np.zeros((num_cores, core_size), dtype=DOUBLE)

    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m1, m2, n2, r2, L, &xs_r[i,0], &ts_r[i,0],
              core_size, cosa, tLA, &us[i,0], &vs[i,0], &ws[i,0])
        cfwx(&c[0], m1, m2, n2, &xs_r[i,0], &ts_r[i,0], core_size, L,
             &phixs[i,0])
        cfwt(&c[0], m1, m2, n2, &xs_r[i,0], &ts_r[i,0], core_size, L,
             &phits[i,0])
    phixs *= -1
    r = r2 + xs_r*sina
    phits *= -1/r
    return (np.ravel(us)[:size], np.ravel(vs)[:size], np.ravel(ws)[:size],
            np.ravel(phixs)[:size], np.ravel(phits)[:size])

cdef void cfN(double *c, double sina, double cosa, double tLA,
              double *xs, double *ts, int size,
              double r2, double L, double *F,
              int m1, int m2, int n2,
              double *c0, int m0, int n0, int funcnum,
              double *Ns, int NL_kinematics) nogil:
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int i
    cdef double exx, ett, gxt, kxx, ktt, kxt
    cdef double *es = <double *>malloc(size*e_num * sizeof(double))
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef cfstraintype *cfstrain
    if NL_kinematics==0:
        cfstrain = &cfstrain_donnell
    elif NL_kinematics==1:
        cfstrain = &cfstrain_sanders
    cfstrain(c, sina, cosa, tLA, xs, ts, size, r2, L, m1, m2, n2,
             c0, m0, n0, funcnum, es)

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

    for i in range(size):
        exx = es[e_num*i + 0]
        ett = es[e_num*i + 1]
        gxt = es[e_num*i + 2]
        kxx = es[e_num*i + 3]
        ktt = es[e_num*i + 4]
        kxt = es[e_num*i + 5]
        Ns[e_num*i + 0] = A11*exx + A12*ett + A16*gxt + B11*kxx + B12*ktt + B16*kxt
        Ns[e_num*i + 1] = A12*exx + A22*ett + A26*gxt + B12*kxx + B22*ktt + B26*kxt
        Ns[e_num*i + 2] = A16*exx + A26*ett + A66*gxt + B16*kxx + B26*ktt + B66*kxt
        Ns[e_num*i + 3] = B11*exx + B12*ett + B16*gxt + D11*kxx + D12*ktt + D16*kxt
        Ns[e_num*i + 4] = B12*exx + B22*ett + B26*gxt + D12*kxx + D22*ktt + D26*kxt
        Ns[e_num*i + 5] = B16*exx + B26*ett + B66*gxt + D16*kxx + D26*ktt + D66*kxt
    free(es)
