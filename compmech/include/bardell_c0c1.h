
#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif

#ifndef BARDELL_FFXI_C0C1_H
#define BARDELL_FFXI_C0C1_H
IMPORTIT double integral_ffxi_c0c1(double c0, double c1, int i, int j,
                   double x1t, double x1r, double x2t, double x2r,
                   double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FFXI_C0C1_H */


#ifndef BARDELL_FF_C0C1_H
#define BARDELL_FF_C0C1_H
IMPORTIT double integral_ff_c0c1(double c0, double c1, int i, int j,
                   double x1t, double x1r, double x2t, double x2r,
                   double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FF_C0C1_H */

