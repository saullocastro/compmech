def fstress(double [:] c,
       double [:, ::1] F,
       double sina, double cosa, double tLA,
       double [:] xs,
       double [:] ts,
       double r2, double L, int m1, int m2, int n2,
       double [:] c0, int m0, int n0, int funcnum,
       int NL_kinematics, int num_cores=4):
    # NL_kinematics = 0 donnell
    # NL_kinematics = 1 sanders
    cdef int size_core, i
    cdef double [:, ::1] Ns
    cdef double [:, ::1] xs_core, ts_core

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

    Ns = np.zeros((num_cores, size_core*e_num), dtype=DOUBLE)
    for i in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfN(&c[0], sina, cosa, tLA, &xs_core[i,0], &ts_core[i,0],
            size_core, r2, L, &F[0,0], m1, m2, n2,
            &c0[0], m0, n0, funcnum, &Ns[i,0], NL_kinematics)
    return np.ravel(Ns)[:size*e_num]

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
