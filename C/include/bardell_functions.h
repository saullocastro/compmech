__declspec(dllimport) void calc_vec_f(double *f, double xi, double xi1t, double xi1r,
                double xi2t, double xi2r);

__declspec(dllimport) void calc_vec_fxi(double *fxi, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r);

__declspec(dllimport) double calc_f(int i, double xi, double xi1t, double xi1r,
              double xi2t, double xi2r);

__declspec(dllimport) double calc_fxi(int i, double xi, double xi1t, double xi1r,
                double xi2t, double xi2r);
