ctypedef void *scfstraintype(float *c, float sina, float cosa,
        float x, float t, float r, float L,
        int m1, int m2, int n2, int pdoff, float c00, float *e) nogil

cdef void scfN(float *c, float sina, float cosa,
        float x, float t, float r, float L,
        float *F, int m1, int m2, int n2, int pdoff, float c00,
        float *N, int NL_kinematics) nogil

cdef void scfwx(float *c, int m1, int m2, int n2,
          float x, float t, float L, int pdoff, float *wx) nogil

cdef void scfwt(float *c, int m1, int m2, int n2,
            float x, float t, float L, int pdoff, float *wt) nogil
