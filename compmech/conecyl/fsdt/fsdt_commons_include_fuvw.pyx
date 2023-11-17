def fuvw(double [:] c, int m1, int m2, int n2,
         double alpharad, double r2, double L, double tLA,
         double [:] xs,
         double [:] ts, int num_cores=4):
    cdef int i, size_core
    cdef double sina, cosa
    cdef double [:, ::1] us, vs, ws, phixs, phits
    cdef double [:, ::1] xs_core, ts_core

    sina = sin(alpharad)
    cosa = cos(alpharad)

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size==num_cores:
        add_size=0
    new_size = size + add_size

    if (size % num_cores)!=0:
        xs_core = np.ascontiguousarray(np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
        ts_core = np.ascontiguousarray(np.hstack((ts, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
    else:                              
        xs_core = np.ascontiguousarray(np.reshape(xs, (num_cores, -1)), dtype=DOUBLE)
        ts_core = np.ascontiguousarray(np.reshape(ts, (num_cores, -1)), dtype=DOUBLE)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phits = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m1, m2, n2, r2, L, &xs_core[i,0], &ts_core[i,0],
              size_core, cosa, tLA, &us[i,0], &vs[i,0], &ws[i,0],
              &phixs[i,0], &phits[i,0])

    return (np.ravel(us)[:size], np.ravel(vs)[:size], np.ravel(ws)[:size],
            np.ravel(phixs)[:size], np.ravel(phits)[:size])
