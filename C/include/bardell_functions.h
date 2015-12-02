#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif

#ifndef BARDELL_FUNCTIONS_H
#define BARDELL_FUNCTIONS_H

IMPORTIT void calc_vec_f(double *f, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

IMPORTIT void calc_vec_fxi(double *fxi, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

IMPORTIT double calc_f(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

IMPORTIT double calc_fxi(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

#endif /** BARDELL_FUNCTIONS_H */