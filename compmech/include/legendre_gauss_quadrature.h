#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif

#ifndef LEGENDRE_GAUSS_QUADRATURE_H
#define LEGENDRE_GAUSS_QUADRATURE_H
IMPORTIT void leggauss_quad(int n, double *points, double *weights);
#endif /** LEGENDRE_GAUSS_QUADRATURE_H */