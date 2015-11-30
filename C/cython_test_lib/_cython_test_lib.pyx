#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False

cdef extern from 'bardell.h':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_ffxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_ffxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxifxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)
    double integral_fxixifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r)

cdef extern from 'bardell_functions.h':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r)
    double calc_fxi(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r)

def test():
    print integral_ff(1, 1, 1., 1., 1., 1., 1., 1., 1., 1.)
    print integral_ffxi(1, 1, 1., 1., 1., 1., 1., 1., 1., 1.)
    print integral_fxifxi(1, 1, 1., 1., 1., 1., 1., 1., 1., 1.)
    print integral_ffxixi(1, 1, 1., 1., 1., 1., 1., 1., 1., 1.)
    print integral_fxifxixi(1, 1, 1., 1., 1., 1., 1., 1., 1., 1.)
    print integral_fxixifxixi(1, 1, 1., 1., 1., 1., 1., 1., 1., 1.)

    print calc_f(1, 0.5, 1., 1., 1., 1.)
    print calc_fxi(1, 0.5, 1., 1., 1., 1.)
