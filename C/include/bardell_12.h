
#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif

#ifndef BARDELL_FFXIXI_12_H
#define BARDELL_FFXIXI_12_H
IMPORTIT double integral_ffxixi_12(double xi1, double xi2, int i, int j,
                   double x1t, double x1r, double x2t, double x2r,
                   double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FFXIXI_12_H */


#ifndef BARDELL_FFXI_12_H
#define BARDELL_FFXI_12_H
IMPORTIT double integral_ffxi_12(double xi1, double xi2, int i, int j,
                   double x1t, double x1r, double x2t, double x2r,
                   double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FFXI_12_H */


#ifndef BARDELL_FF_12_H
#define BARDELL_FF_12_H
IMPORTIT double integral_ff_12(double xi1, double xi2, int i, int j,
                   double x1t, double x1r, double x2t, double x2r,
                   double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FF_12_H */


#ifndef BARDELL_FXIFXIXI_12_H
#define BARDELL_FXIFXIXI_12_H
IMPORTIT double integral_fxifxixi_12(double xi1, double xi2, int i, int j,
                   double x1t, double x1r, double x2t, double x2r,
                   double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FXIFXIXI_12_H */


#ifndef BARDELL_FXIFXI_12_H
#define BARDELL_FXIFXI_12_H
IMPORTIT double integral_fxifxi_12(double xi1, double xi2, int i, int j,
                   double x1t, double x1r, double x2t, double x2r,
                   double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FXIFXI_12_H */


#ifndef BARDELL_FXIXIFXIXI_12_H
#define BARDELL_FXIXIFXIXI_12_H
IMPORTIT double integral_fxixifxixi_12(double xi1, double xi2, int i, int j,
                   double x1t, double x1r, double x2t, double x2r,
                   double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FXIXIFXIXI_12_H */

