#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange


DOUBLE = np.float64
INT = np.int64
ctypedef np.double_t cDOUBLE
ctypedef np.int64_t cINT


cdef int nmax = 30
cdef int num1 = 3
cdef double pi=3.141592653589793


def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m1, int n1, double a, double b,
        np.ndarray[cDOUBLE, ndim=1] xs, np.ndarray[cDOUBLE, ndim=1] ys,
        int num_cores=4):
    cdef int size_core, i
    cdef np.ndarray[cDOUBLE, ndim=2] us, vs, ws, phixs, phiys
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ys_core = np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ys_core = ys.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phiys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m1, n1, a, b, &xs_core[i,0],
              &ys_core[i,0], size_core, &us[i,0], &vs[i,0], &ws[i,0])

        #cfwx(&c[0], m1, n1, &xs_core[i,0], &ys_core[i,0],
             #size_core, a, b, &phixs[i,0])

        #cfwy(&c[0], m1, n1, &xs_core[i,0], &ys_core[i,0],
             #size_core, a, b, &phiys[i,0])

    phixs *= -1.
    phiys *= -1.
    return (us.ravel()[:size], vs.ravel()[:size], ws.ravel()[:size],
            phixs.ravel()[:size], phiys.ravel()[:size])


cdef void cfuvw(double *c, int m1, int n1, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws) nogil:
    cdef int i1, j1, col, i
    cdef double x, y, u, v, w, xi, eta
    cdef double *fxi
    cdef double *feta

    fxi = <double *>malloc(nmax * sizeof(double *))
    feta = <double *>malloc(nmax * sizeof(double *))

    for i in range(size):
        x = xs[i]
        y = ys[i]

        xi = (2*x - a)/a
        eta = (2*y - b)/b

        calc_fxi(fxi, xi)
        calc_fxi(feta, eta)

        u = 0
        v = 0
        w = 0

        for i1 in range(m1):
            for j1 in range(n1):
                col = num1*(j1*m1 + i1)
                u += c[col+0]*fxi[i1]*feta[j1]
                v += c[col+1]*fxi[i1]*feta[j1]
                w += c[col+2]*fxi[i1]*feta[j1]

        us[i] = u
        vs[i] = v
        ws[i] = w

    free(fxi)
    free(feta)


def fg(double[:,::1] g, int m1, int n1,
       double x, double y, double a, double b):
    cfg(g, m1, n1, x, y, a, b)


cdef void cfg(double[:,::1] g, int m1, int n1,
              double x, double y, double a, double b) nogil:
    cdef int i1, j1, col
    cdef double xi, eta
    cdef double *fxi
    cdef double *feta

    fxi = <double *>malloc(nmax * sizeof(double *))
    feta = <double *>malloc(nmax * sizeof(double *))

    xi = (2*x - a)/a
    eta = (2*y - b)/b

    calc_fxi(fxi, xi)
    calc_fxi(feta, eta)

    for i1 in range(m1):
        for j1 in range(n1):
            col = num1*(j1*m1 + i1)
            g[0, col+0] = fxi[i1]*feta[j1]
            g[1, col+1] = fxi[i1]*feta[j1]
            g[2, col+2] = fxi[i1]*feta[j1]

    free(fxi)
    free(feta)


cdef void calc_fxi(double *fxi, double xi) nogil:
    fxi[0] = 0.25*xi**3 - 0.75*xi + 0.5
    fxi[1] = 0.125*xi**3 - 0.125*xi**2 - 0.125*xi + 0.125
    fxi[2] = -0.25*xi**3 + 0.75*xi + 0.5
    fxi[3] = 0.125*xi**3 + 0.125*xi**2 - 0.125*xi - 0.125
    fxi[4] = 0.125*xi**4 - 0.25*xi**2 + 0.125
    fxi[5] = 0.125*xi**5 - 0.25*xi**3 + 0.125*xi
    fxi[6] = 0.145833333333333*xi**6 - 0.3125*xi**4 + 0.1875*xi**2 - 0.0208333333333333
    fxi[7] = 0.1875*xi**7 - 0.4375*xi**5 + 0.3125*xi**3 - 0.0625*xi
    fxi[8] = 0.2578125*xi**8 - 0.65625*xi**6 + 0.546875*xi**4 - 0.15625*xi**2 + 0.0078125
    fxi[9] = 0.372395833333333*xi**9 - 1.03125*xi**7 + 0.984375*xi**5 - 0.364583333333333*xi**3 + 0.0390625*xi
    fxi[10] = 0.55859375*xi**10 - 1.67578125*xi**8 + 1.8046875*xi**6 - 0.8203125*xi**4 + 0.13671875*xi**2 - 0.00390625
    fxi[11] = 0.86328125*xi**11 - 2.79296875*xi**9 + 3.3515625*xi**7 - 1.8046875*xi**5 + 0.41015625*xi**3 - 0.02734375*xi
    fxi[12] = 1.36686197916667*xi**12 - 4.748046875*xi**10 + 6.2841796875*xi**8 - 3.91015625*xi**6 + 1.1279296875*xi**4 - 0.123046875*xi**2 + 0.00227864583333333
    fxi[13] = 2.2080078125*xi**13 - 8.201171875*xi**11 + 11.8701171875*xi**9 - 8.37890625*xi**7 + 2.9326171875*xi**5 - 0.451171875*xi**3 + 0.0205078125*xi
    fxi[14] = 3.62744140625*xi**14 - 14.35205078125*xi**12 + 22.55322265625*xi**10 - 17.80517578125*xi**8 + 7.33154296875*xi**6 - 1.46630859375*xi**4 + 0.11279296875*xi**2 - 0.00146484375
    fxi[15] = 6.04573567708333*xi**15 - 25.39208984375*xi**13 + 43.05615234375*xi**11 - 37.5887044270833*xi**9 + 17.80517578125*xi**7 - 4.39892578125*xi**5 + 0.48876953125*xi**3 - 0.01611328125*xi
    fxi[16] = 10.2021789550781*xi**16 - 45.343017578125*xi**14 + 82.5242919921875*xi**12 - 78.936279296875*xi**10 + 42.2872924804688*xi**8 - 12.463623046875*xi**6 + 1.8328857421875*xi**4 - 0.104736328125*xi**2 + 0.001007080078125
    fxi[17] = 17.4037170410156*xi**17 - 81.617431640625*xi**15 + 158.700561523438*xi**13 - 165.048583984375*xi**11 + 98.6703491210938*xi**9 - 33.829833984375*xi**7 + 6.2318115234375*xi**5 - 0.523681640625*xi**3 + 0.013092041015625*xi
    fxi[18] = 29.9730682373047*xi**18 - 147.931594848633*xi**16 + 306.065368652344*xi**14 - 343.851216634115*xi**12 + 226.941802978516*xi**10 - 88.8033142089844*xi**8 + 19.7340698242188*xi**6 - 2.22564697265625*xi**4 + 0.0981903076171875*xi**2 - 0.000727335611979167
    fxi[19] = 52.0584869384766*xi**19 - 269.757614135742*xi**17 + 591.726379394531*xi**15 - 714.152526855469*xi**13 + 515.776824951172*xi**11 - 226.941802978516*xi**9 + 59.2022094726562*xi**7 - 8.45745849609375*xi**5 + 0.556411743164063*xi**3 - 0.0109100341796875*xi
    fxi[20] = 91.102352142334*xi**20 - 494.555625915527*xi**18 + 1146.4698600769*xi**16 - 1479.31594848633*xi**14 + 1160.49785614014*xi**12 - 567.354507446289*xi**10 + 170.206352233887*xi**8 - 29.6011047363281*xi**6 + 2.6429557800293*xi**4 - 0.0927352905273438*xi**2 + 0.000545501708984375
    fxi[21] = 160.513668060303*xi**21 - 911.02352142334*xi**19 + 2225.50031661987*xi**17 - 3057.25296020508*xi**15 + 2588.80290985107*xi**13 - 1392.59742736816*xi**11 + 472.795422871908*xi**9 - 97.2607727050781*xi**7 + 11.100414276123*xi**5 - 0.587323506673177*xi**3 + 0.00927352905273438*xi
    fxi[22] = 284.546957015991*xi**22 - 1685.39351463318*xi**20 + 4327.36172676086*xi**18 - 6305.58423042297*xi**16 + 5732.34930038452*xi**14 - 3365.4437828064*xi**12 + 1276.54764175415*xi**10 - 303.939914703369*xi**8 + 42.5515880584717*xi**6 - 3.08344841003418*xi**4 + 0.0880985260009765*xi**2 - 0.000421524047851563
    fxi[23] = 507.235879898071*xi**23 - 3130.0165271759*xi**21 + 8426.96757316589*xi**19 - 12982.0851802826*xi**17 + 12611.1684608459*xi**15 - 8025.28902053833*xi**13 + 3365.4437828064*xi**11 - 911.819744110107*xi**9 + 151.969957351685*xi**7 - 14.1838626861572*xi**5 + 0.616689682006836*xi**3 - 0.00800895690917969*xi
    fxi[24] = 908.797618150711*xi**24 - 5833.21261882782*xi**22 + 16432.5867676735*xi**20 - 26685.3973150253*xi**18 + 27586.9310081005*xi**16 - 18916.7526912689*xi**14 + 8694.06310558319*xi**12 - 2644.27725791931*xi**10 + 512.898606061935*xi**8 - 59.0994278589884*xi**6 + 3.54596567153931*xi**4 - 0.0840940475463867*xi**2 + 0.000333706537882487
    fxi[25] = 1635.83571267128*xi**25 - 10905.5714178085*xi**23 + 32082.669403553*xi**21 - 54775.2892255783*xi**19 + 60042.143958807*xi**17 - 44139.0896129608*xi**15 + 22069.5448064804*xi**13 - 7452.05409049988*xi**11 + 1652.67328619957*xi**9 - 227.954936027527*xi**7 + 17.7298283576965*xi**5 - 0.644721031188965*xi**3 + 0.00700783729553223*xi
    fxi[26] = 2957.08763444424*xi**26 - 20447.946408391*xi**24 + 62707.0356523991*xi**22 - 112289.342912436*xi**20 + 130091.311910748*xi**18 - 102071.644729972*xi**16 + 55173.862016201*xi**14 - 20493.1487488747*xi**12 + 5123.28718721867*xi**10 - 826.336643099785*xi**8 + 79.7842276096344*xi**6 - 4.02950644493103*xi**4 + 0.0805901288986206*xi**2 - 0.000269532203674316
    fxi[27] = 5366.5664476951*xi**27 - 38442.1392477751*xi**25 + 122687.678450346*xi**23 - 229925.79739213*xi**21 + 280723.357281089*xi**19 - 234164.361439347*xi**17 + 136095.526306629*xi**15 - 55173.862016201*xi**13 + 15369.861561656*xi**11 - 2846.27065956593*xi**9 + 330.534657239914*xi**7 - 21.7593348026276*xi**5 + 0.671584407488505*xi**3 - 0.00619924068450928*xi
    fxi[28] = 9774.81745830178*xi**28 - 72448.6470438838*xi**26 + 240263.370298594*xi**24 - 470302.767392993*xi**22 + 603555.218154341*xi**20 - 533374.378834069*xi**18 + 331732.845372409*xi**16 - 145816.635328531*xi**14 + 44828.7628881633*xi**12 - 9392.69317656755*xi**10 + 1280.82179680467*xi**8 - 105.1701182127*xi**6 + 4.53319475054741*xi**4 - 0.077490508556366*xi**2 + 0.000221401453018188
    fxi[29] = 17864.3215617239*xi**29 - 136847.444416225*xi**27 + 470916.205785245*xi**25 - 961053.481194377*xi**23 + 1293332.61033073*xi**21 - 1207110.43630868*xi**19 + 800061.568251103*xi**17 - 379123.251854181*xi**15 + 127589.555912465*xi**13 - 29885.8419254422*xi**11 + 4696.34658828378*xi**9 - 465.753380656242*xi**7 + 26.292529553175*xi**5 - 0.697414577007294*xi**3 + 0.00553503632545471*xi
