#include  <stdlib.h>
#include  <math.h>

struct cc_attributes {
  double *sina;
  double *cosa;
  double *tLA;
  double *r2;
  double *L;
  double *F;
  int *m1;
  int *m2;
  int *n2;
  double *coeffs;
  double *c0;
  int *m0;
  int *n0;
};

void cfk0L_clean(double *wxs, double *wts, double *w0xs, double *w0ts, 
                 int npts, double *xs, double *ts, double *out,
                 double *alphas, double *betas, void *args) {
  int i1;
  int k1;
  int i2;
  int j2;
  int k2;
  int l2;
  int c;
  int i;
  int pos;
  double A11;
  double A12;
  double A16;
  double A22;
  double A26;
  double A66;
  double B11;
  double B12;
  double B16;
  double B22;
  double B26;
  double B66;
  double p00;
  double p01;
  double p02;
  double p10;
  double p11;
  double p12;
  double p20;
  double p21;
  double p22;
  double p30;
  double p31;
  double p32;
  double p40;
  double p41;
  double p42;
  double p50;
  double p51;
  double p52;
  double q02;
  double q04;
  double q05;
  double q14;
  double q15;
  double q22;
  double q24;
  double q25;
  double r;
  double x;
  double t;
  double alpha;
  double beta;
  double *F;
  double *coeffs;
  double *c0;
  double sina;
  double cosa;
  double tLA;
  double r2;
  double L;
  int m0;
  int n0;
  int m1;
  int m2;
  int n2;
  double wx;
  double wt;
  double w0x;
  double w0t;
  struct cc_attributes *args_in;
  double sini1x;
  double cosi1x;
  double cosk1x;
  double sini2x;
  double cosi2x;
  double sink2x;
  double cosk2x;
  double sinl2t;
  double cosl2t;
  double sinj2t;
  double cosj2t;
  double *vsini1x;
  double *vcosi1x;
  double *vsini2x;
  double *vcosi2x;
  double *vsinj2t;
  double *vcosj2t;
  double *k0Lq_1_q02;
  double *k0Lq_1_q22;
  double *k0Lq_2_q04;
  double *k0Lq_2_q05;
  double *k0Lq_2_q14;
  double *k0Lq_2_q15;
  double *k0Lq_2_q24;
  double *k0Lq_2_q25;
  double pi = 3.141592653589793;
  int i0 = 0;
  int j0 = 1;
  printf("DEBUG C code running...");
  args_in = ((struct cc_attributes *)args);
  sina = (args_in->sina[0]);
  cosa = (args_in->cosa[0]);
  tLA = (args_in->tLA[0]);
  r2 = (args_in->r2[0]);
  L = (args_in->L[0]);
  F = args_in->F;
  m1 = (args_in->m1[0]);
  m2 = (args_in->m2[0]);
  n2 = (args_in->n2[0]);
  coeffs = args_in->coeffs;
  c0 = args_in->c0;
  m0 = (args_in->m0[0]);
  n0 = (args_in->n0[0]);
  A11 = (F[0]);
  A12 = (F[1]);
  A16 = (F[2]);
  A22 = (F[7]);
  A26 = (F[8]);
  A66 = (F[14]);
  B11 = (F[3]);
  B12 = (F[4]);
  B16 = (F[5]);
  B22 = (F[10]);
  B26 = (F[11]);
  B66 = (F[17]);
  vsini1x = ((double *)malloc((m1 * (sizeof(double)))));
  vcosi1x = ((double *)malloc((m1 * (sizeof(double)))));
  vsini2x = ((double *)malloc((m2 * (sizeof(double)))));
  vcosi2x = ((double *)malloc((m2 * (sizeof(double)))));
  vsinj2t = ((double *)malloc((n2 * (sizeof(double)))));
  vcosj2t = ((double *)malloc((n2 * (sizeof(double)))));
  k0Lq_1_q02 = ((double *)malloc((m1 * (sizeof(double)))));
  k0Lq_1_q22 = ((double *)malloc((m1 * (sizeof(double)))));
  k0Lq_2_q04 = ((double *)malloc(((m2 * n2) * (sizeof(double)))));
  k0Lq_2_q05 = ((double *)malloc(((m2 * n2) * (sizeof(double)))));
  k0Lq_2_q14 = ((double *)malloc(((m2 * n2) * (sizeof(double)))));
  k0Lq_2_q15 = ((double *)malloc(((m2 * n2) * (sizeof(double)))));
  k0Lq_2_q24 = ((double *)malloc(((m2 * n2) * (sizeof(double)))));
  k0Lq_2_q25 = ((double *)malloc(((m2 * n2) * (sizeof(double)))));
  for (i = 0; i < npts; i++) {
    x = (xs[i]);
    t = (ts[i]);
    wx = (wxs[i]);
    wt = (wts[i]);
    w0x = (w0xs[i]);
    w0t = (w0ts[i]);
    alpha = (alphas[i]);
    beta = (betas[i]);
    for (i1 = i0; i1 < (m1 + i0); i1++) {
      (vsini1x[(i1 - i0)]) = sin((((pi * i1) * x) / L));
      (vcosi1x[(i1 - i0)]) = cos((((pi * i1) * x) / L));
    }
    for (i2 = i0; i2 < (m2 + i0); i2++) {
      (vsini2x[(i2 - i0)]) = sin((((pi * i2) * x) / L));
      (vcosi2x[(i2 - i0)]) = cos((((pi * i2) * x) / L));
    }
    for (j2 = j0; j2 < (n2 + j0); j2++) {
      (vsinj2t[(j2 - j0)]) = sin((j2 * t));
      (vcosj2t[(j2 - j0)]) = cos((j2 * t));
    }
    r = (r2 + (sina * x));
    c = -1;
    p00 = ((((-A11) * r) + ((A12 * sina) * (L - x))) / (L * cosa));
    p01 = ((((-A12) * r) + ((A22 * sina) * (L - x))) / (L * cosa));
    p02 = ((((-A16) * r) + ((A26 * sina) * (L - x))) / (L * cosa));
    p10 = ((((-A16) * r2) * (r + (sina * (L - x)))) / L);
    p11 = ((((-A26) * r2) * (r + (sina * (L - x)))) / L);
    p12 = ((((-A66) * r2) * (r + (sina * (L - x)))) / L);
    p20 = ((((A16 * ((-L) + x)) * sin((t - tLA))) - (((A11 * r) + ((A12 * sina) * ((-L) + x))) * cos((t - tLA)))) / (L * cosa));
    p21 = ((((A26 * ((-L) + x)) * sin((t - tLA))) - (((A12 * r) + ((A22 * sina) * ((-L) + x))) * cos((t - tLA)))) / (L * cosa));
    p22 = ((((A66 * ((-L) + x)) * sin((t - tLA))) - (((A16 * r) + ((A26 * sina) * ((-L) + x))) * cos((t - tLA)))) / (L * cosa));
    for (k1 = i0; k1 < (m1 + i0); k1++) {
      cosk1x = (vcosi1x[(k1 - i0)]);
      q02 = ((((pi * cosk1x) * k1) * (w0x + wx)) / L);
      q22 = ((((pi * cosk1x) * k1) * (w0t + wt)) / (L * r));
      out[c++] += (alpha * ((p00 * q02) + (p02 * q22)));
      out[c++] += (alpha * ((p10 * q02) + (p12 * q22)));
      out[c++] += (alpha * ((p20 * q02) + (p22 * q22)));
      (k0Lq_1_q02[(k1 - i0)]) = q02;
      (k0Lq_1_q22[(k1 - i0)]) = q22;
    }
    for (k2 = i0; k2 < (m2 + i0); k2++) {
      sink2x = (vsini2x[(k2 - i0)]);
      cosk2x = (vcosi2x[(k2 - i0)]);
      for (l2 = j0; l2 < (n2 + j0); l2++) {
        sinl2t = (vsinj2t[(l2 - j0)]);
        cosl2t = (vcosj2t[(l2 - j0)]);
        q04 = (((((pi * cosk2x) * k2) * sinl2t) * (w0x + wx)) / L);
        q05 = (((((pi * cosk2x) * cosl2t) * k2) * (w0x + wx)) / L);
        q14 = ((((cosl2t * l2) * sink2x) * (w0t + wt)) / (r * r));
        q15 = (((((-l2) * sink2x) * sinl2t) * (w0t + wt)) / (r * r));
        q24 = ((((((L * cosl2t) * l2) * sink2x) * (w0x + wx)) + ((((pi * cosk2x) * k2) * sinl2t) * (w0t + wt))) / (L * r));
        q25 = (((((((-L) * l2) * sink2x) * sinl2t) * (w0x + wx)) + ((((pi * cosk2x) * cosl2t) * k2) * (w0t + wt))) / (L * r));
        out[c++] += (alpha * (((p00 * q04) + (p01 * q14)) + (p02 * q24)));
        out[c++] += (alpha * (((p00 * q05) + (p01 * q15)) + (p02 * q25)));
        out[c++] += (alpha * (((p10 * q04) + (p11 * q14)) + (p12 * q24)));
        out[c++] += (alpha * (((p10 * q05) + (p11 * q15)) + (p12 * q25)));
        out[c++] += (alpha * (((p20 * q04) + (p21 * q14)) + (p22 * q24)));
        out[c++] += (alpha * (((p20 * q05) + (p21 * q15)) + (p22 * q25)));
        pos = (((k2 - i0) * n2) + (l2 - j0));
        (k0Lq_2_q04[pos]) = q04;
        (k0Lq_2_q05[pos]) = q05;
        (k0Lq_2_q14[pos]) = q14;
        (k0Lq_2_q15[pos]) = q15;
        (k0Lq_2_q24[pos]) = q24;
        (k0Lq_2_q25[pos]) = q25;
      }
    }
    for (i1 = i0; i1 < (m1 + i0); i1++) {
      sini1x = (vsini1x[(i1 - i0)]);
      cosi1x = (vcosi1x[(i1 - i0)]);
      p00 = ((((((pi * A11) * i1) * r) * cosi1x) / L) + ((A12 * sina) * sini1x));
      p01 = ((((((pi * A12) * i1) * r) * cosi1x) / L) + ((A22 * sina) * sini1x));
      p02 = ((((((pi * A16) * i1) * r) * cosi1x) / L) + ((A26 * sina) * sini1x));
      p10 = ((((-A16) * sina) * sini1x) + (((((pi * A16) * i1) * r) * cosi1x) / L));
      p11 = ((((-A26) * sina) * sini1x) + (((((pi * A26) * i1) * r) * cosi1x) / L));
      p12 = ((((-A66) * sina) * sini1x) + (((((pi * A66) * i1) * r) * cosi1x) / L));
      p20 = ((((((((-pi) * B12) * L) * i1) * sina) * cosi1x) + ((((A12 * (L * L)) * cosa) + ((((pi * pi) * B11) * (i1 * i1)) * r)) * sini1x)) / (L * L));
      p21 = ((((((((-pi) * B22) * L) * i1) * sina) * cosi1x) + ((((A22 * (L * L)) * cosa) + ((((pi * pi) * B12) * (i1 * i1)) * r)) * sini1x)) / (L * L));
      p22 = ((((((((-pi) * B26) * L) * i1) * sina) * cosi1x) + ((((A26 * (L * L)) * cosa) + ((((pi * pi) * B16) * (i1 * i1)) * r)) * sini1x)) / (L * L));
      for (k1 = i0; k1 < (m1 + i0); k1++) {
        q02 = (k0Lq_1_q02[(k1 - i0)]);
        q22 = (k0Lq_1_q22[(k1 - i0)]);
        out[c++] += (alpha * ((p00 * q02) + (p02 * q22)));
        out[c++] += (alpha * ((p10 * q02) + (p12 * q22)));
        out[c++] += (alpha * ((p20 * q02) + (p22 * q22)));
      }
      for (k2 = i0; k2 < (m2 + i0); k2++) {
        for (l2 = j0; l2 < (n2 + j0); l2++) {
          pos = (((k2 - i0) * n2) + (l2 - j0));
          q04 = (k0Lq_2_q04[pos]);
          q05 = (k0Lq_2_q05[pos]);
          q14 = (k0Lq_2_q14[pos]);
          q15 = (k0Lq_2_q15[pos]);
          q24 = (k0Lq_2_q24[pos]);
          q25 = (k0Lq_2_q25[pos]);
          out[c++] += (alpha * (((p00 * q04) + (p01 * q14)) + (p02 * q24)));
          out[c++] += (alpha * (((p00 * q05) + (p01 * q15)) + (p02 * q25)));
          out[c++] += (alpha * (((p10 * q04) + (p11 * q14)) + (p12 * q24)));
          out[c++] += (alpha * (((p10 * q05) + (p11 * q15)) + (p12 * q25)));
          out[c++] += (alpha * (((p20 * q04) + (p21 * q14)) + (p22 * q24)));
          out[c++] += (alpha * (((p20 * q05) + (p21 * q15)) + (p22 * q25)));
        }
      }
    }
    for (i2 = i0; i2 < (m2 + i0); i2++) {
      sini2x = (vsini2x[(i2 - i0)]);
      cosi2x = (vcosi2x[(i2 - i0)]);
      for (j2 = j0; j2 < (n2 + j0); j2++) {
        sinj2t = (vsinj2t[(j2 - j0)]);
        cosj2t = (vcosj2t[(j2 - j0)]);
        p00 = (((((((pi * A11) * i2) * r) * sinj2t) * cosi2x) / L) + ((((A12 * sina) * sinj2t) + ((A16 * j2) * cosj2t)) * sini2x));
        p01 = (((((((pi * A12) * i2) * r) * sinj2t) * cosi2x) / L) + ((((A22 * sina) * sinj2t) + ((A26 * j2) * cosj2t)) * sini2x));
        p02 = (((((((pi * A16) * i2) * r) * sinj2t) * cosi2x) / L) + ((((A26 * sina) * sinj2t) + ((A66 * j2) * cosj2t)) * sini2x));
        p10 = (((((((pi * A11) * i2) * r) * cosj2t) * cosi2x) / L) + ((((A12 * sina) * cosj2t) - ((A16 * j2) * sinj2t)) * sini2x));
        p11 = (((((((pi * A12) * i2) * r) * cosj2t) * cosi2x) / L) + ((((A22 * sina) * cosj2t) - ((A26 * j2) * sinj2t)) * sini2x));
        p12 = (((((((pi * A16) * i2) * r) * cosj2t) * cosi2x) / L) + ((((A26 * sina) * cosj2t) - ((A66 * j2) * sinj2t)) * sini2x));
        p20 = (((((((pi * A16) * i2) * r) * sinj2t) * cosi2x) / L) + ((((A12 * j2) * cosj2t) - ((A16 * sina) * sinj2t)) * sini2x));
        p21 = (((((((pi * A26) * i2) * r) * sinj2t) * cosi2x) / L) + ((((A22 * j2) * cosj2t) - ((A26 * sina) * sinj2t)) * sini2x));
        p22 = (((((((pi * A66) * i2) * r) * sinj2t) * cosi2x) / L) + ((((A26 * j2) * cosj2t) - ((A66 * sina) * sinj2t)) * sini2x));
        p30 = (((((((pi * A16) * i2) * r) * cosj2t) * cosi2x) / L) - ((((A12 * j2) * sinj2t) + ((A16 * sina) * cosj2t)) * sini2x));
        p31 = (((((((pi * A26) * i2) * r) * cosj2t) * cosi2x) / L) - ((((A22 * j2) * sinj2t) + ((A26 * sina) * cosj2t)) * sini2x));
        p32 = (((((((pi * A66) * i2) * r) * cosj2t) * cosi2x) / L) - ((((A26 * j2) * sinj2t) + ((A66 * sina) * cosj2t)) * sini2x));
        p40 = ((((((((-pi) * L) * i2) * r) * (((B12 * sina) * sinj2t) + (((2.0 * B16) * j2) * cosj2t))) * cosi2x) + ((((((B16 * (L * L)) * j2) * sina) * cosj2t) + ((((B12 * (L * L)) * (j2 * j2)) + (r * (((A12 * (L * L)) * cosa) + ((((pi * pi) * B11) * (i2 * i2)) * r)))) * sinj2t)) * sini2x)) / ((L * L) * r));
        p41 = ((((((((-pi) * L) * i2) * r) * (((B22 * sina) * sinj2t) + (((2.0 * B26) * j2) * cosj2t))) * cosi2x) + ((((((B26 * (L * L)) * j2) * sina) * cosj2t) + ((((B22 * (L * L)) * (j2 * j2)) + (r * (((A22 * (L * L)) * cosa) + ((((pi * pi) * B12) * (i2 * i2)) * r)))) * sinj2t)) * sini2x)) / ((L * L) * r));
        p42 = ((((((((-pi) * L) * i2) * r) * (((B26 * sina) * sinj2t) + (((2.0 * B66) * j2) * cosj2t))) * cosi2x) + ((((((B66 * (L * L)) * j2) * sina) * cosj2t) + ((((B26 * (L * L)) * (j2 * j2)) + (r * (((A26 * (L * L)) * cosa) + ((((pi * pi) * B16) * (i2 * i2)) * r)))) * sinj2t)) * sini2x)) / ((L * L) * r));
        p50 = (((((((pi * L) * i2) * r) * ((((-B12) * sina) * cosj2t) + (((2.0 * B16) * j2) * sinj2t))) * cosi2x) + (((((((-B16) * (L * L)) * j2) * sina) * sinj2t) + ((((B12 * (L * L)) * (j2 * j2)) + (r * (((A12 * (L * L)) * cosa) + ((((pi * pi) * B11) * (i2 * i2)) * r)))) * cosj2t)) * sini2x)) / ((L * L) * r));
        p51 = (((((((pi * L) * i2) * r) * ((((-B22) * sina) * cosj2t) + (((2.0 * B26) * j2) * sinj2t))) * cosi2x) + (((((((-B26) * (L * L)) * j2) * sina) * sinj2t) + ((((B22 * (L * L)) * (j2 * j2)) + (r * (((A22 * (L * L)) * cosa) + ((((pi * pi) * B12) * (i2 * i2)) * r)))) * cosj2t)) * sini2x)) / ((L * L) * r));
        p52 = (((((((pi * L) * i2) * r) * ((((-B26) * sina) * cosj2t) + (((2.0 * B66) * j2) * sinj2t))) * cosi2x) + (((((((-B66) * (L * L)) * j2) * sina) * sinj2t) + ((((B26 * (L * L)) * (j2 * j2)) + (r * (((A26 * (L * L)) * cosa) + ((((pi * pi) * B16) * (i2 * i2)) * r)))) * cosj2t)) * sini2x)) / ((L * L) * r));
        for (k1 = i0; k1 < (m1 + i0); k1++) {
          q02 = (k0Lq_1_q02[(k1 - i0)]);
          q22 = (k0Lq_1_q22[(k1 - i0)]);
          out[c++] += (alpha * ((p00 * q02) + (p02 * q22)));
          out[c++] += (alpha * ((p10 * q02) + (p12 * q22)));
          out[c++] += (alpha * ((p20 * q02) + (p22 * q22)));
          out[c++] += (alpha * ((p30 * q02) + (p32 * q22)));
          out[c++] += (alpha * ((p40 * q02) + (p42 * q22)));
          out[c++] += (alpha * ((p50 * q02) + (p52 * q22)));
        }
        for (k2 = i0; k2 < (m2 + i0); k2++) {
          for (l2 = j0; l2 < (n2 + j0); l2++) {
            pos = (((k2 - i0) * n2) + (l2 - j0));
            q04 = (k0Lq_2_q04[pos]);
            q05 = (k0Lq_2_q05[pos]);
            q14 = (k0Lq_2_q14[pos]);
            q15 = (k0Lq_2_q15[pos]);
            q24 = (k0Lq_2_q24[pos]);
            q25 = (k0Lq_2_q25[pos]);
            out[c++] += (alpha * (((p00 * q04) + (p01 * q14)) + (p02 * q24)));
            out[c++] += (alpha * (((p00 * q05) + (p01 * q15)) + (p02 * q25)));
            out[c++] += (alpha * (((p10 * q04) + (p11 * q14)) + (p12 * q24)));
            out[c++] += (alpha * (((p10 * q05) + (p11 * q15)) + (p12 * q25)));
            out[c++] += (alpha * (((p20 * q04) + (p21 * q14)) + (p22 * q24)));
            out[c++] += (alpha * (((p20 * q05) + (p21 * q15)) + (p22 * q25)));
            out[c++] += (alpha * (((p30 * q04) + (p31 * q14)) + (p32 * q24)));
            out[c++] += (alpha * (((p30 * q05) + (p31 * q15)) + (p32 * q25)));
            out[c++] += (alpha * (((p40 * q04) + (p41 * q14)) + (p42 * q24)));
            out[c++] += (alpha * (((p40 * q05) + (p41 * q15)) + (p42 * q25)));
            out[c++] += (alpha * (((p50 * q04) + (p51 * q14)) + (p52 * q24)));
            out[c++] += (alpha * (((p50 * q05) + (p51 * q15)) + (p52 * q25)));
          }
        }
      }
    }
  }
  free(vsini1x);
  free(vcosi1x);
  free(vsini2x);
  free(vcosi2x);
  free(vsinj2t);
  free(vcosj2t);
  free(k0Lq_1_q02);
  free(k0Lq_1_q22);
  free(k0Lq_2_q04);
  free(k0Lq_2_q05);
  free(k0Lq_2_q14);
  free(k0Lq_2_q15);
  free(k0Lq_2_q24);
  free(k0Lq_2_q25);
};
