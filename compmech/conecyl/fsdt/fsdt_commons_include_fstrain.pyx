def fstrain(double [:] c,
            double sina, double cosa, double tLA,
            double [:] xs,
            double [:] ts,
            double r2, double L, int m1, int m2, int n2,
            double [:] c0, int m0, int n0, int funcnum,
            int NL_kinematics, int num_cores=4):
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int size_core, i
    cdef double [:, ::1] es
    cdef double [:, ::1] xs_core, ts_core
    cdef cfstraintype *cfstrain

    if NL_kinematics==0:
        cfstrain = &cfstrain_donnell
    elif NL_kinematics==1:
        raise NotImplementedError

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

    es = np.zeros((num_cores, size_core*e_num), dtype=DOUBLE)
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfstrain(&c[0], sina, cosa, tLA, &xs_core[i,0], &ts_core[i,0],
                 size_core, r2, L, m1, m2, n2,
                 &c0[0], m0, n0, funcnum, &es[i,0])
    return np.ravel(es)[:size*e_num]
