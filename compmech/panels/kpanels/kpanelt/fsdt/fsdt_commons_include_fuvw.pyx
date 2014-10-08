def fuvw(np.ndarray[cDOUBLE, ndim=1] c, int m2, int n3, int m4, int n4,
         double L, double tmin, double tmax,
         np.ndarray[cDOUBLE, ndim=1] xs,
         np.ndarray[cDOUBLE, ndim=1] ts, double alpharad, int num_cores=4):
    cdef int i, size_core
    cdef np.ndarray[cDOUBLE, ndim=2] us, vs, ws, phixs, phits
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ts_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1)
        ts_core = np.hstack((ts, np.zeros(add_size))).reshape(num_cores, -1)
    else:
        xs_core = xs.reshape(num_cores, -1)
        ts_core = ts.reshape(num_cores, -1)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phits = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m2, n3, m4, n4, L, tmin, tmax, &xs_core[i,0],
                &ts_core[i,0], size_core, &us[i,0], &vs[i,0], &ws[i,0],
                &phixs[i,0], &phits[i,0])

    return (us.ravel()[:size],
            vs.ravel()[:size],
            ws.ravel()[:size],
            phixs.ravel()[:size],
            phits.ravel()[:size])
