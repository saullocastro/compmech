#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
cdef int nmax() nogil:
    return 30

cdef void calc_fxi(double *fxi, double xi) nogil:
    fxi[0] = 0.25*xi**3 - 0.75*xi + 0.5
    fxi[1] = 0.125*xi**3 - 0.125*xi**2 - 0.125*xi + 0.125
    fxi[2] = -0.25*xi**3 + 0.75*xi + 0.5
    fxi[3] = 0.125*xi**3 + 0.125*xi**2 - 0.125*xi - 0.125
    fxi[4] = 0.125*xi**4 - 0.25*xi**2 + 0.125
    fxi[5] = 0.125*xi**5 - 0.25*xi**3 + 0.125*xi
    fxi[6] = 0.145833333333333*xi**6 - 0.3125*xi**4 + 0.1875*xi**2 - 0.0208333333333333
    fxi[7] = 0.1875*xi**7 - 0.4375*xi**5 + 0.3125*xi**3 - 0.0625*xi
    fxi[8] = 0.2578125*xi**8 - 0.65625*xi**6 + 0.546875*xi**4 - 0.15625*xi**2 + 0.0078125
    fxi[9] = 0.372395833333333*xi**9 - 1.03125*xi**7 + 0.984375*xi**5 - 0.364583333333333*xi**3 + 0.0390625*xi
    fxi[10] = 0.55859375*xi**10 - 1.67578125*xi**8 + 1.8046875*xi**6 - 0.8203125*xi**4 + 0.13671875*xi**2 - 0.00390625
    fxi[11] = 0.86328125*xi**11 - 2.79296875*xi**9 + 3.3515625*xi**7 - 1.8046875*xi**5 + 0.41015625*xi**3 - 0.02734375*xi
    fxi[12] = 1.36686197916667*xi**12 - 4.748046875*xi**10 + 6.2841796875*xi**8 - 3.91015625*xi**6 + 1.1279296875*xi**4 - 0.123046875*xi**2 + 0.00227864583333333
    fxi[13] = 2.2080078125*xi**13 - 8.201171875*xi**11 + 11.8701171875*xi**9 - 8.37890625*xi**7 + 2.9326171875*xi**5 - 0.451171875*xi**3 + 0.0205078125*xi
    fxi[14] = 3.62744140625*xi**14 - 14.35205078125*xi**12 + 22.55322265625*xi**10 - 17.80517578125*xi**8 + 7.33154296875*xi**6 - 1.46630859375*xi**4 + 0.11279296875*xi**2 - 0.00146484375
    fxi[15] = 6.04573567708333*xi**15 - 25.39208984375*xi**13 + 43.05615234375*xi**11 - 37.5887044270833*xi**9 + 17.80517578125*xi**7 - 4.39892578125*xi**5 + 0.48876953125*xi**3 - 0.01611328125*xi
    fxi[16] = 10.2021789550781*xi**16 - 45.343017578125*xi**14 + 82.5242919921875*xi**12 - 78.936279296875*xi**10 + 42.2872924804688*xi**8 - 12.463623046875*xi**6 + 1.8328857421875*xi**4 - 0.104736328125*xi**2 + 0.001007080078125
    fxi[17] = 17.4037170410156*xi**17 - 81.617431640625*xi**15 + 158.700561523438*xi**13 - 165.048583984375*xi**11 + 98.6703491210938*xi**9 - 33.829833984375*xi**7 + 6.2318115234375*xi**5 - 0.523681640625*xi**3 + 0.013092041015625*xi
    fxi[18] = 29.9730682373047*xi**18 - 147.931594848633*xi**16 + 306.065368652344*xi**14 - 343.851216634115*xi**12 + 226.941802978516*xi**10 - 88.8033142089844*xi**8 + 19.7340698242188*xi**6 - 2.22564697265625*xi**4 + 0.0981903076171875*xi**2 - 0.000727335611979167
    fxi[19] = 52.0584869384766*xi**19 - 269.757614135742*xi**17 + 591.726379394531*xi**15 - 714.152526855469*xi**13 + 515.776824951172*xi**11 - 226.941802978516*xi**9 + 59.2022094726562*xi**7 - 8.45745849609375*xi**5 + 0.556411743164063*xi**3 - 0.0109100341796875*xi
    fxi[20] = 91.102352142334*xi**20 - 494.555625915527*xi**18 + 1146.4698600769*xi**16 - 1479.31594848633*xi**14 + 1160.49785614014*xi**12 - 567.354507446289*xi**10 + 170.206352233887*xi**8 - 29.6011047363281*xi**6 + 2.6429557800293*xi**4 - 0.0927352905273438*xi**2 + 0.000545501708984375
    fxi[21] = 160.513668060303*xi**21 - 911.02352142334*xi**19 + 2225.50031661987*xi**17 - 3057.25296020508*xi**15 + 2588.80290985107*xi**13 - 1392.59742736816*xi**11 + 472.795422871908*xi**9 - 97.2607727050781*xi**7 + 11.100414276123*xi**5 - 0.587323506673177*xi**3 + 0.00927352905273438*xi
    fxi[22] = 284.546957015991*xi**22 - 1685.39351463318*xi**20 + 4327.36172676086*xi**18 - 6305.58423042297*xi**16 + 5732.34930038452*xi**14 - 3365.4437828064*xi**12 + 1276.54764175415*xi**10 - 303.939914703369*xi**8 + 42.5515880584717*xi**6 - 3.08344841003418*xi**4 + 0.0880985260009765*xi**2 - 0.000421524047851563
    fxi[23] = 507.235879898071*xi**23 - 3130.0165271759*xi**21 + 8426.96757316589*xi**19 - 12982.0851802826*xi**17 + 12611.1684608459*xi**15 - 8025.28902053833*xi**13 + 3365.4437828064*xi**11 - 911.819744110107*xi**9 + 151.969957351685*xi**7 - 14.1838626861572*xi**5 + 0.616689682006836*xi**3 - 0.00800895690917969*xi
    fxi[24] = 908.797618150711*xi**24 - 5833.21261882782*xi**22 + 16432.5867676735*xi**20 - 26685.3973150253*xi**18 + 27586.9310081005*xi**16 - 18916.7526912689*xi**14 + 8694.06310558319*xi**12 - 2644.27725791931*xi**10 + 512.898606061935*xi**8 - 59.0994278589884*xi**6 + 3.54596567153931*xi**4 - 0.0840940475463867*xi**2 + 0.000333706537882487
    fxi[25] = 1635.83571267128*xi**25 - 10905.5714178085*xi**23 + 32082.669403553*xi**21 - 54775.2892255783*xi**19 + 60042.143958807*xi**17 - 44139.0896129608*xi**15 + 22069.5448064804*xi**13 - 7452.05409049988*xi**11 + 1652.67328619957*xi**9 - 227.954936027527*xi**7 + 17.7298283576965*xi**5 - 0.644721031188965*xi**3 + 0.00700783729553223*xi
    fxi[26] = 2957.08763444424*xi**26 - 20447.946408391*xi**24 + 62707.0356523991*xi**22 - 112289.342912436*xi**20 + 130091.311910748*xi**18 - 102071.644729972*xi**16 + 55173.862016201*xi**14 - 20493.1487488747*xi**12 + 5123.28718721867*xi**10 - 826.336643099785*xi**8 + 79.7842276096344*xi**6 - 4.02950644493103*xi**4 + 0.0805901288986206*xi**2 - 0.000269532203674316
    fxi[27] = 5366.5664476951*xi**27 - 38442.1392477751*xi**25 + 122687.678450346*xi**23 - 229925.79739213*xi**21 + 280723.357281089*xi**19 - 234164.361439347*xi**17 + 136095.526306629*xi**15 - 55173.862016201*xi**13 + 15369.861561656*xi**11 - 2846.27065956593*xi**9 + 330.534657239914*xi**7 - 21.7593348026276*xi**5 + 0.671584407488505*xi**3 - 0.00619924068450928*xi
    fxi[28] = 9774.81745830178*xi**28 - 72448.6470438838*xi**26 + 240263.370298594*xi**24 - 470302.767392993*xi**22 + 603555.218154341*xi**20 - 533374.378834069*xi**18 + 331732.845372409*xi**16 - 145816.635328531*xi**14 + 44828.7628881633*xi**12 - 9392.69317656755*xi**10 + 1280.82179680467*xi**8 - 105.1701182127*xi**6 + 4.53319475054741*xi**4 - 0.077490508556366*xi**2 + 0.000221401453018188
    fxi[29] = 17864.3215617239*xi**29 - 136847.444416225*xi**27 + 470916.205785245*xi**25 - 961053.481194377*xi**23 + 1293332.61033073*xi**21 - 1207110.43630868*xi**19 + 800061.568251103*xi**17 - 379123.251854181*xi**15 + 127589.555912465*xi**13 - 29885.8419254422*xi**11 + 4696.34658828378*xi**9 - 465.753380656242*xi**7 + 26.292529553175*xi**5 - 0.697414577007294*xi**3 + 0.00553503632545471*xi


cdef double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return 0.742857142857143*x1t*y1t
        elif j == 1:
            return 0.104761904761905*x1t*y1r
        elif j == 2:
            return 0.257142857142857*x1t*y2t
        elif j == 3:
            return -0.0619047619047619*x1t*y2r
        elif j == 4:
            return 0.0666666666666667*x1t
        elif j == 5:
            return -0.0126984126984127*x1t
        elif j == 7:
            return 0.000288600288600289*x1t
    elif i == 1:
        if j == 0:
            return 0.104761904761905*x1r*y1t
        elif j == 1:
            return 0.019047619047619*x1r*y1r
        elif j == 2:
            return 0.0619047619047619*x1r*y2t
        elif j == 3:
            return -0.0142857142857143*x1r*y2r
        elif j == 4:
            return 0.0142857142857143*x1r
        elif j == 5:
            return -0.00158730158730159*x1r
        elif j == 6:
            return -0.000529100529100529*x1r
        elif j == 7:
            return 0.000144300144300144*x1r
    elif i == 2:
        if j == 0:
            return 0.257142857142857*x2t*y1t
        elif j == 1:
            return 0.0619047619047619*x2t*y1r
        elif j == 2:
            return 0.742857142857143*x2t*y2t
        elif j == 3:
            return -0.104761904761905*x2t*y2r
        elif j == 4:
            return 0.0666666666666667*x2t
        elif j == 5:
            return 0.0126984126984127*x2t
        elif j == 7:
            return -0.000288600288600289*x2t
    elif i == 3:
        if j == 0:
            return -0.0619047619047619*x2r*y1t
        elif j == 1:
            return -0.0142857142857143*x2r*y1r
        elif j == 2:
            return -0.104761904761905*x2r*y2t
        elif j == 3:
            return 0.019047619047619*x2r*y2r
        elif j == 4:
            return -0.0142857142857143*x2r
        elif j == 5:
            return -0.00158730158730159*x2r
        elif j == 6:
            return 0.000529100529100529*x2r
        elif j == 7:
            return 0.000144300144300144*x2r
    elif i == 4:
        if j == 0:
            return 0.0666666666666667*y1t
        elif j == 1:
            return 0.0142857142857143*y1r
        elif j == 2:
            return 0.0666666666666667*y2t
        elif j == 3:
            return -0.0142857142857143*y2r
        elif j == 4:
            return 0.0126984126984127
        elif j == 6:
            return -0.000769600769600770
        elif j == 8:
            return 4.44000444000444e-5
    elif i == 5:
        if j == 0:
            return -0.0126984126984127*y1t
        elif j == 1:
            return -0.00158730158730159*y1r
        elif j == 2:
            return 0.0126984126984127*y2t
        elif j == 3:
            return -0.00158730158730159*y2r
        elif j == 5:
            return 0.00115440115440115
        elif j == 7:
            return -0.000177600177600178
        elif j == 9:
            return 1.48000148000148e-5
    elif i == 6:
        if j == 1:
            return -0.000529100529100529*y1r
        elif j == 3:
            return 0.000529100529100529*y2r
        elif j == 4:
            return -0.000769600769600770
        elif j == 6:
            return 0.000266400266400266
        elif j == 8:
            return -5.92000592000592e-5
        elif j == 10:
            return 6.09412374118256e-6
    elif i == 7:
        if j == 0:
            return 0.000288600288600289*y1t
        elif j == 1:
            return 0.000144300144300144*y1r
        elif j == 2:
            return -0.000288600288600289*y2t
        elif j == 3:
            return 0.000144300144300144*y2r
        elif j == 5:
            return -0.000177600177600178
        elif j == 7:
            return 8.88000888000888e-5
        elif j == 9:
            return -2.43764949647303e-5
        elif j == 11:
            return 2.88669019319174e-6
    elif i == 8:
        if j == 4:
            return 4.44000444000444e-5
        elif j == 6:
            return -5.92000592000592e-5
        elif j == 8:
            return 3.65647424470954e-5
        elif j == 10:
            return -1.15467607727670e-5
        elif j == 12:
            return 1.51207581548139e-6
    elif i == 9:
        if j == 5:
            return 1.48000148000148e-5
        elif j == 7:
            return -2.43764949647303e-5
        elif j == 9:
            return 1.73201411591504e-5
        elif j == 11:
            return -6.04830326192555e-6
        elif j == 13:
            return 8.54651547880785e-7
    elif i == 10:
        if j == 6:
            return 6.09412374118256e-6
        elif j == 8:
            return -1.15467607727670e-5
        elif j == 10:
            return 9.07245489288833e-6
        elif j == 12:
            return -3.41860619152314e-6
        elif j == 14:
            return 5.12790928728471e-7
    elif i == 11:
        if j == 7:
            return 2.88669019319174e-6
        elif j == 9:
            return -6.04830326192555e-6
        elif j == 11:
            return 5.12790928728471e-6
        elif j == 13:
            return -2.05116371491388e-6
        elif j == 15:
            return 3.22868362532741e-7
    elif i == 12:
        if j == 8:
            return 1.51207581548139e-6
        elif j == 10:
            return -3.41860619152314e-6
        elif j == 12:
            return 3.07674557237082e-6
        elif j == 14:
            return -1.29147345013096e-6
        elif j == 16:
            return 2.11534444418003e-7
    elif i == 13:
        if j == 9:
            return 8.54651547880785e-7
        elif j == 11:
            return -2.05116371491388e-6
        elif j == 13:
            return 1.93721017519645e-6
        elif j == 15:
            return -8.46137777672011e-7
        elif j == 17:
            return 1.43297526863808e-7
    elif i == 14:
        if j == 10:
            return 5.12790928728471e-7
        elif j == 12:
            return -1.29147345013096e-6
        elif j == 14:
            return 1.26920666650802e-6
        elif j == 16:
            return -5.73190107455233e-7
        elif j == 18:
            return 9.98740338747754e-8
    elif i == 15:
        if j == 11:
            return 3.22868362532741e-7
        elif j == 13:
            return -8.46137777672011e-7
        elif j == 15:
            return 8.59785161182849e-7
        elif j == 17:
            return -3.99496135499102e-7
        elif j == 19:
            return 7.13385956248396e-8
    elif i == 16:
        if j == 12:
            return 2.11534444418003e-7
        elif j == 14:
            return -5.73190107455233e-7
        elif j == 16:
            return 5.99244203248653e-7
        elif j == 18:
            return -2.85354382499358e-7
        elif j == 20:
            return 5.20578941046127e-8
    elif i == 17:
        if j == 13:
            return 1.43297526863808e-7
        elif j == 15:
            return -3.99496135499102e-7
        elif j == 17:
            return 4.28031573749038e-7
        elif j == 19:
            return -2.08231576418451e-7
        elif j == 21:
            return 3.87097161290710e-8
    elif i == 18:
        if j == 14:
            return 9.98740338747754e-8
        elif j == 16:
            return -2.85354382499358e-7
        elif j == 18:
            return 3.12347364627676e-7
        elif j == 20:
            return -1.54838864516284e-7
        elif j == 22:
            return 2.92683219512488e-8
    elif i == 19:
        if j == 15:
            return 7.13385956248396e-8
        elif j == 17:
            return -2.08231576418451e-7
        elif j == 19:
            return 2.32258296774426e-7
        elif j == 21:
            return -1.17073287804995e-7
        elif j == 23:
            return 2.24617354509584e-8
    elif i == 20:
        if j == 16:
            return 5.20578941046127e-8
        elif j == 18:
            return -1.54838864516284e-7
        elif j == 20:
            return 1.75609931707493e-7
        elif j == 22:
            return -8.98469418038335e-8
        elif j == 24:
            return 1.74702386840787e-8
    elif i == 21:
        if j == 17:
            return 3.87097161290710e-8
        elif j == 19:
            return -1.17073287804995e-7
        elif j == 21:
            return 1.34770412705750e-7
        elif j == 23:
            return -6.98809547363149e-8
        elif j == 25:
            return 1.37531666236364e-8
    elif i == 22:
        if j == 18:
            return 2.92683219512488e-8
        elif j == 20:
            return -8.98469418038335e-8
        elif j == 22:
            return 1.04821432104472e-7
        elif j == 24:
            return -5.50126664945458e-8
        elif j == 26:
            return 1.09463979249351e-8
    elif i == 23:
        if j == 19:
            return 2.24617354509584e-8
        elif j == 21:
            return -6.98809547363149e-8
        elif j == 23:
            return 8.25189997418187e-8
        elif j == 25:
            return -4.37855916997405e-8
        elif j == 27:
            return 8.80004539063412e-9
    elif i == 24:
        if j == 20:
            return 1.74702386840787e-8
        elif j == 22:
            return -5.50126664945458e-8
        elif j == 24:
            return 6.56783875496108e-8
        elif j == 26:
            return -3.52001815625365e-8
        elif j == 28:
            return 7.13965946787297e-9
    elif i == 25:
        if j == 21:
            return 1.37531666236364e-8
        elif j == 23:
            return -4.37855916997405e-8
        elif j == 25:
            return 5.28002723438047e-8
        elif j == 27:
            return -2.85586378714919e-8
        elif j == 29:
            return 5.84153956462334e-9
    elif i == 26:
        if j == 22:
            return 1.09463979249351e-8
        elif j == 24:
            return -3.52001815625365e-8
        elif j == 26:
            return 4.28379568072378e-8
        elif j == 28:
            return -2.33661582584934e-8
    elif i == 27:
        if j == 23:
            return 8.80004539063412e-9
        elif j == 25:
            return -2.85586378714919e-8
        elif j == 27:
            return 3.50492373877400e-8
        elif j == 29:
            return -1.92668322482314e-8
    elif i == 28:
        if j == 24:
            return 7.13965946787297e-9
        elif j == 26:
            return -2.33661582584934e-8
        elif j == 28:
            return 2.89002483723470e-8
    elif i == 29:
        if j == 25:
            return 5.84153956462334e-9
        elif j == 27:
            return -1.92668322482314e-8
        elif j == 29:
            return 2.40019011905933e-8
    return 0


cdef double integral_ffxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return -0.5*x1t*y1t
        elif j == 1:
            return 0.1*x1t*y1r
        elif j == 2:
            return 0.5*x1t*y2t
        elif j == 3:
            return -0.1*x1t*y2r
        elif j == 4:
            return 0.0857142857142857*x1t
        elif j == 6:
            return -0.00317460317460317*x1t
    elif i == 1:
        if j == 0:
            return -0.1*x1r*y1t
        elif j == 2:
            return 0.1*x1r*y2t
        elif j == 3:
            return -0.0166666666666667*x1r*y2r
        elif j == 4:
            return 0.00952380952380952*x1r
        elif j == 5:
            return 0.00476190476190476*x1r
        elif j == 6:
            return -0.00158730158730159*x1r
    elif i == 2:
        if j == 0:
            return -0.5*x2t*y1t
        elif j == 1:
            return -0.1*x2t*y1r
        elif j == 2:
            return 0.5*x2t*y2t
        elif j == 3:
            return 0.1*x2t*y2r
        elif j == 4:
            return -0.0857142857142857*x2t
        elif j == 6:
            return 0.00317460317460317*x2t
    elif i == 3:
        if j == 0:
            return 0.1*x2r*y1t
        elif j == 1:
            return 0.0166666666666667*x2r*y1r
        elif j == 2:
            return -0.1*x2r*y2t
        elif j == 4:
            return 0.00952380952380952*x2r
        elif j == 5:
            return -0.00476190476190476*x2r
        elif j == 6:
            return -0.00158730158730159*x2r
    elif i == 4:
        if j == 0:
            return -0.0857142857142857*y1t
        elif j == 1:
            return -0.00952380952380952*y1r
        elif j == 2:
            return 0.0857142857142857*y2t
        elif j == 3:
            return -0.00952380952380952*y2r
        elif j == 5:
            return 0.00634920634920635
        elif j == 7:
            return -0.000577200577200577
    elif i == 5:
        if j == 1:
            return -0.00476190476190476*y1r
        elif j == 3:
            return 0.00476190476190476*y2r
        elif j == 4:
            return -0.00634920634920635
        elif j == 6:
            return 0.00173160173160173
        elif j == 8:
            return -0.000222000222000222
    elif i == 6:
        if j == 0:
            return 0.00317460317460317*y1t
        elif j == 1:
            return 0.00158730158730159*y1r
        elif j == 2:
            return -0.00317460317460317*y2t
        elif j == 3:
            return 0.00158730158730159*y2r
        elif j == 5:
            return -0.00173160173160173
        elif j == 7:
            return 0.000666000666000666
        elif j == 9:
            return -0.000103600103600104
    elif i == 7:
        if j == 4:
            return 0.000577200577200577
        elif j == 6:
            return -0.000666000666000666
        elif j == 8:
            return 0.000310800310800311
        elif j == 10:
            return -5.48471136706431e-5
    elif i == 8:
        if j == 5:
            return 0.000222000222000222
        elif j == 7:
            return -0.000310800310800311
        elif j == 9:
            return 0.000164541341011929
        elif j == 11:
            return -3.17535921251092e-5
    elif i == 9:
        if j == 6:
            return 0.000103600103600104
        elif j == 8:
            return -0.000164541341011929
        elif j == 10:
            return 9.52607763753275e-5
        elif j == 12:
            return -1.96569856012580e-5
    elif i == 10:
        if j == 7:
            return 5.48471136706431e-5
        elif j == 9:
            return -9.52607763753275e-5
        elif j == 11:
            return 5.89709568037741e-5
        elif j == 13:
            return -1.28197732182118e-5
    elif i == 11:
        if j == 8:
            return 3.17535921251092e-5
        elif j == 10:
            return -5.89709568037741e-5
        elif j == 12:
            return 3.84593196546353e-5
        elif j == 14:
            return -8.71744578838400e-6
    elif i == 12:
        if j == 9:
            return 1.96569856012580e-5
        elif j == 11:
            return -3.84593196546353e-5
        elif j == 13:
            return 2.61523373651520e-5
        elif j == 15:
            return -6.13449888812208e-6
    elif i == 13:
        if j == 10:
            return 1.28197732182118e-5
        elif j == 12:
            return -2.61523373651520e-5
        elif j == 14:
            return 1.84034966643662e-5
        elif j == 16:
            return -4.44222333277806e-6
    elif i == 14:
        if j == 11:
            return 8.71744578838400e-6
        elif j == 13:
            return -1.84034966643662e-5
        elif j == 15:
            return 1.33266699983342e-5
        elif j == 17:
            return -3.29584311786759e-6
    elif i == 15:
        if j == 12:
            return 6.13449888812208e-6
        elif j == 14:
            return -1.33266699983342e-5
        elif j == 16:
            return 9.88752935360277e-6
        elif j == 18:
            return -2.49685084686939e-6
    elif i == 16:
        if j == 13:
            return 4.44222333277806e-6
        elif j == 15:
            return -9.88752935360277e-6
        elif j == 17:
            return 7.49055254060816e-6
        elif j == 19:
            return -1.92614208187067e-6
    elif i == 17:
        if j == 14:
            return 3.29584311786759e-6
        elif j == 16:
            return -7.49055254060816e-6
        elif j == 18:
            return 5.77842624561201e-6
        elif j == 20:
            return -1.50967892903377e-6
    elif i == 18:
        if j == 15:
            return 2.49685084686939e-6
        elif j == 17:
            return -5.77842624561201e-6
        elif j == 19:
            return 4.52903678710130e-6
        elif j == 21:
            return -1.20000120000120e-6
    elif i == 19:
        if j == 16:
            return 1.92614208187067e-6
        elif j == 18:
            return -4.52903678710130e-6
        elif j == 20:
            return 3.60000360000360e-6
        elif j == 22:
            return -9.65854624391210e-7
    elif i == 20:
        if j == 17:
            return 1.50967892903377e-6
        elif j == 19:
            return -3.60000360000360e-6
        elif j == 21:
            return 2.89756387317363e-6
        elif j == 23:
            return -7.86160740783543e-7
    elif i == 21:
        if j == 18:
            return 1.20000120000120e-6
        elif j == 20:
            return -2.89756387317363e-6
        elif j == 22:
            return 2.35848222235063e-6
        elif j == 24:
            return -6.46398831310913e-7
    elif i == 22:
        if j == 19:
            return 9.65854624391210e-7
        elif j == 21:
            return -2.35848222235063e-6
        elif j == 23:
            return 1.93919649393274e-6
        elif j == 25:
            return -5.36373498321821e-7
    elif i == 23:
        if j == 20:
            return 7.86160740783543e-7
        elif j == 22:
            return -1.93919649393274e-6
        elif j == 24:
            return 1.60912049496546e-6
        elif j == 26:
            return -4.48802314922340e-7
    elif i == 24:
        if j == 21:
            return 6.46398831310913e-7
        elif j == 23:
            return -1.60912049496546e-6
        elif j == 25:
            return 1.34640694476702e-6
        elif j == 27:
            return -3.78401951797267e-7
    elif i == 25:
        if j == 22:
            return 5.36373498321821e-7
        elif j == 24:
            return -1.34640694476702e-6
        elif j == 26:
            return 1.13520585539180e-6
        elif j == 28:
            return -3.21284676054284e-7
    elif i == 26:
        if j == 23:
            return 4.48802314922340e-7
        elif j == 25:
            return -1.13520585539180e-6
        elif j == 27:
            return 9.63854028162851e-7
        elif j == 29:
            return -2.74552359537297e-7
    elif i == 27:
        if j == 24:
            return 3.78401951797267e-7
        elif j == 26:
            return -9.63854028162851e-7
        elif j == 28:
            return 8.23657078611891e-7
    elif i == 28:
        if j == 25:
            return 3.21284676054284e-7
        elif j == 27:
            return -8.23657078611891e-7
        elif j == 29:
            return 7.08056085122503e-7
    elif i == 29:
        if j == 26:
            return 2.74552359537297e-7
        elif j == 28:
            return -7.08056085122503e-7
    return 0


cdef double integral_ffxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return -0.6*x1t*y1t
        elif j == 1:
            return -0.55*x1t*y1r
        elif j == 2:
            return 0.6*x1t*y2t
        elif j == 3:
            return -0.05*x1t*y2r
        elif j == 5:
            return 0.0285714285714286*x1t
    elif i == 1:
        if j == 0:
            return -0.05*x1r*y1t
        elif j == 1:
            return -0.0666666666666667*x1r*y1r
        elif j == 2:
            return 0.05*x1r*y2t
        elif j == 3:
            return 0.0166666666666667*x1r*y2r
        elif j == 4:
            return -0.0333333333333333*x1r
        elif j == 5:
            return 0.0142857142857143*x1r
    elif i == 2:
        if j == 0:
            return 0.6*x2t*y1t
        elif j == 1:
            return 0.05*x2t*y1r
        elif j == 2:
            return -0.6*x2t*y2t
        elif j == 3:
            return 0.55*x2t*y2r
        elif j == 5:
            return -0.0285714285714286*x2t
    elif i == 3:
        if j == 0:
            return -0.05*x2r*y1t
        elif j == 1:
            return 0.0166666666666667*x2r*y1r
        elif j == 2:
            return 0.05*x2r*y2t
        elif j == 3:
            return -0.0666666666666667*x2r*y2r
        elif j == 4:
            return 0.0333333333333333*x2r
        elif j == 5:
            return 0.0142857142857143*x2r
    elif i == 4:
        if j == 1:
            return -0.0333333333333333*y1r
        elif j == 3:
            return 0.0333333333333333*y2r
        elif j == 4:
            return -0.0380952380952381
        elif j == 6:
            return 0.00634920634920635
    elif i == 5:
        if j == 0:
            return 0.0285714285714286*y1t
        elif j == 1:
            return 0.0142857142857143*y1r
        elif j == 2:
            return -0.0285714285714286*y2t
        elif j == 3:
            return 0.0142857142857143*y2r
        elif j == 5:
            return -0.0126984126984127
        elif j == 7:
            return 0.00288600288600289
    elif i == 6:
        if j == 4:
            return 0.00634920634920635
        elif j == 6:
            return -0.00577200577200577
        elif j == 8:
            return 0.00155400155400155
    elif i == 7:
        if j == 5:
            return 0.00288600288600289
        elif j == 7:
            return -0.00310800310800311
        elif j == 9:
            return 0.000932400932400932
    elif i == 8:
        if j == 6:
            return 0.00155400155400155
        elif j == 8:
            return -0.00186480186480186
        elif j == 10:
            return 0.000603318250377074
    elif i == 9:
        if j == 7:
            return 0.000932400932400932
        elif j == 9:
            return -0.00120663650075415
        elif j == 11:
            return 0.000412796697626419
    elif i == 10:
        if j == 8:
            return 0.000603318250377074
        elif j == 10:
            return -0.000825593395252838
        elif j == 12:
            return 0.000294854784018871
    elif i == 11:
        if j == 9:
            return 0.000412796697626419
        elif j == 11:
            return -0.000589709568037741
        elif j == 13:
            return 0.000217936144709600
    elif i == 12:
        if j == 10:
            return 0.000294854784018871
        elif j == 12:
            return -0.000435872289419200
        elif j == 14:
            return 0.000165631469979296
    elif i == 13:
        if j == 11:
            return 0.000217936144709600
        elif j == 13:
            return -0.000331262939958592
        elif j == 15:
            return 0.000128824476650564
    elif i == 14:
        if j == 12:
            return 0.000165631469979296
        elif j == 14:
            return -0.000257648953301127
        elif j == 16:
            return 0.000102171136653895
    elif i == 15:
        if j == 13:
            return 0.000128824476650564
        elif j == 15:
            return -0.000204342273307791
        elif j == 17:
            return 8.23960779466897e-5
    elif i == 16:
        if j == 14:
            return 0.000102171136653895
        elif j == 16:
            return -0.000164792155893379
        elif j == 18:
            return 6.74149728654734e-5
    elif i == 17:
        if j == 15:
            return 8.23960779466897e-5
        elif j == 17:
            return -0.000134829945730947
        elif j == 19:
            return 5.58581203742494e-5
    elif i == 18:
        if j == 16:
            return 6.74149728654734e-5
        elif j == 18:
            return -0.000111716240748499
        elif j == 20:
            return 4.68000468000468e-5
    elif i == 19:
        if j == 17:
            return 5.58581203742494e-5
        elif j == 19:
            return -9.36000936000936e-5
        elif j == 21:
            return 3.96000396000396e-5
    elif i == 20:
        if j == 18:
            return 4.68000468000468e-5
        elif j == 20:
            return -7.92000792000792e-5
        elif j == 22:
            return 3.38049118536923e-5
    elif i == 21:
        if j == 19:
            return 3.96000396000396e-5
        elif j == 21:
            return -6.76098237073847e-5
        elif j == 23:
            return 2.90879474089911e-5
    elif i == 22:
        if j == 20:
            return 3.38049118536923e-5
        elif j == 22:
            return -5.81758948179822e-5
        elif j == 24:
            return 2.52095544211256e-5
    elif i == 23:
        if j == 21:
            return 2.90879474089911e-5
        elif j == 23:
            return -5.04191088422512e-5
        elif j == 25:
            return 2.19913134311947e-5
    elif i == 24:
        if j == 22:
            return 2.52095544211256e-5
        elif j == 24:
            return -4.39826268623894e-5
        elif j == 26:
            return 1.92984995416606e-5
    elif i == 25:
        if j == 23:
            return 2.19913134311947e-5
        elif j == 25:
            return -3.85969990833213e-5
        elif j == 27:
            return 1.70280878308770e-5
    elif i == 26:
        if j == 24:
            return 1.92984995416606e-5
        elif j == 26:
            return -3.40561756617541e-5
        elif j == 28:
            return 1.51003797745513e-5
    elif i == 27:
        if j == 25:
            return 1.70280878308770e-5
        elif j == 27:
            return -3.02007595491027e-5
        elif j == 29:
            return 1.34530656173275e-5
    elif i == 28:
        if j == 26:
            return 1.51003797745513e-5
        elif j == 28:
            return -2.69061312346551e-5
    elif i == 29:
        if j == 27:
            return 1.34530656173275e-5
        elif j == 29:
            return -2.40739068941651e-5
    return 0


cdef double integral_fxifxi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return 0.6*x1t*y1t
        elif j == 1:
            return 0.05*x1t*y1r
        elif j == 2:
            return -0.6*x1t*y2t
        elif j == 3:
            return 0.05*x1t*y2r
        elif j == 5:
            return -0.0285714285714286*x1t
    elif i == 1:
        if j == 0:
            return 0.05*x1r*y1t
        elif j == 1:
            return 0.0666666666666667*x1r*y1r
        elif j == 2:
            return -0.05*x1r*y2t
        elif j == 3:
            return -0.0166666666666667*x1r*y2r
        elif j == 4:
            return 0.0333333333333333*x1r
        elif j == 5:
            return -0.0142857142857143*x1r
    elif i == 2:
        if j == 0:
            return -0.6*x2t*y1t
        elif j == 1:
            return -0.05*x2t*y1r
        elif j == 2:
            return 0.6*x2t*y2t
        elif j == 3:
            return -0.05*x2t*y2r
        elif j == 5:
            return 0.0285714285714286*x2t
    elif i == 3:
        if j == 0:
            return 0.05*x2r*y1t
        elif j == 1:
            return -0.0166666666666667*x2r*y1r
        elif j == 2:
            return -0.05*x2r*y2t
        elif j == 3:
            return 0.0666666666666667*x2r*y2r
        elif j == 4:
            return -0.0333333333333333*x2r
        elif j == 5:
            return -0.0142857142857143*x2r
    elif i == 4:
        if j == 1:
            return 0.0333333333333333*y1r
        elif j == 3:
            return -0.0333333333333333*y2r
        elif j == 4:
            return 0.0380952380952381
        elif j == 6:
            return -0.00634920634920635
    elif i == 5:
        if j == 0:
            return -0.0285714285714286*y1t
        elif j == 1:
            return -0.0142857142857143*y1r
        elif j == 2:
            return 0.0285714285714286*y2t
        elif j == 3:
            return -0.0142857142857143*y2r
        elif j == 5:
            return 0.0126984126984127
        elif j == 7:
            return -0.00288600288600289
    elif i == 6:
        if j == 4:
            return -0.00634920634920635
        elif j == 6:
            return 0.00577200577200577
        elif j == 8:
            return -0.00155400155400155
    elif i == 7:
        if j == 5:
            return -0.00288600288600289
        elif j == 7:
            return 0.00310800310800311
        elif j == 9:
            return -0.000932400932400932
    elif i == 8:
        if j == 6:
            return -0.00155400155400155
        elif j == 8:
            return 0.00186480186480186
        elif j == 10:
            return -0.000603318250377074
    elif i == 9:
        if j == 7:
            return -0.000932400932400932
        elif j == 9:
            return 0.00120663650075415
        elif j == 11:
            return -0.000412796697626419
    elif i == 10:
        if j == 8:
            return -0.000603318250377074
        elif j == 10:
            return 0.000825593395252838
        elif j == 12:
            return -0.000294854784018871
    elif i == 11:
        if j == 9:
            return -0.000412796697626419
        elif j == 11:
            return 0.000589709568037741
        elif j == 13:
            return -0.000217936144709600
    elif i == 12:
        if j == 10:
            return -0.000294854784018871
        elif j == 12:
            return 0.000435872289419200
        elif j == 14:
            return -0.000165631469979296
    elif i == 13:
        if j == 11:
            return -0.000217936144709600
        elif j == 13:
            return 0.000331262939958592
        elif j == 15:
            return -0.000128824476650564
    elif i == 14:
        if j == 12:
            return -0.000165631469979296
        elif j == 14:
            return 0.000257648953301127
        elif j == 16:
            return -0.000102171136653895
    elif i == 15:
        if j == 13:
            return -0.000128824476650564
        elif j == 15:
            return 0.000204342273307791
        elif j == 17:
            return -8.23960779466897e-5
    elif i == 16:
        if j == 14:
            return -0.000102171136653895
        elif j == 16:
            return 0.000164792155893379
        elif j == 18:
            return -6.74149728654734e-5
    elif i == 17:
        if j == 15:
            return -8.23960779466897e-5
        elif j == 17:
            return 0.000134829945730947
        elif j == 19:
            return -5.58581203742494e-5
    elif i == 18:
        if j == 16:
            return -6.74149728654734e-5
        elif j == 18:
            return 0.000111716240748499
        elif j == 20:
            return -4.68000468000468e-5
    elif i == 19:
        if j == 17:
            return -5.58581203742494e-5
        elif j == 19:
            return 9.36000936000936e-5
        elif j == 21:
            return -3.96000396000396e-5
    elif i == 20:
        if j == 18:
            return -4.68000468000468e-5
        elif j == 20:
            return 7.92000792000792e-5
        elif j == 22:
            return -3.38049118536923e-5
    elif i == 21:
        if j == 19:
            return -3.96000396000396e-5
        elif j == 21:
            return 6.76098237073847e-5
        elif j == 23:
            return -2.90879474089911e-5
    elif i == 22:
        if j == 20:
            return -3.38049118536923e-5
        elif j == 22:
            return 5.81758948179822e-5
        elif j == 24:
            return -2.52095544211256e-5
    elif i == 23:
        if j == 21:
            return -2.90879474089911e-5
        elif j == 23:
            return 5.04191088422512e-5
        elif j == 25:
            return -2.19913134311947e-5
    elif i == 24:
        if j == 22:
            return -2.52095544211256e-5
        elif j == 24:
            return 4.39826268623894e-5
        elif j == 26:
            return -1.92984995416606e-5
    elif i == 25:
        if j == 23:
            return -2.19913134311947e-5
        elif j == 25:
            return 3.85969990833213e-5
        elif j == 27:
            return -1.70280878308770e-5
    elif i == 26:
        if j == 24:
            return -1.92984995416606e-5
        elif j == 26:
            return 3.40561756617541e-5
        elif j == 28:
            return -1.51003797745513e-5
    elif i == 27:
        if j == 25:
            return -1.70280878308770e-5
        elif j == 27:
            return 3.02007595491027e-5
        elif j == 29:
            return -1.34530656173275e-5
    elif i == 28:
        if j == 26:
            return -1.51003797745513e-5
        elif j == 28:
            return 2.69061312346551e-5
    elif i == 29:
        if j == 27:
            return -1.34530656173275e-5
        elif j == 29:
            return 2.40739068941651e-5
    return 0


cdef double integral_fxifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 1:
            return 0.25*x1t*y1r
        elif j == 3:
            return -0.25*x1t*y2r
        elif j == 4:
            return 0.2*x1t
    elif i == 1:
        if j == 0:
            return -0.25*x1r*y1t
        elif j == 1:
            return -0.125*x1r*y1r
        elif j == 2:
            return 0.25*x1r*y2t
        elif j == 3:
            return -0.125*x1r*y2r
        elif j == 4:
            return 0.1*x1r
    elif i == 2:
        if j == 1:
            return -0.25*x2t*y1r
        elif j == 3:
            return 0.25*x2t*y2r
        elif j == 4:
            return -0.2*x2t
    elif i == 3:
        if j == 0:
            return 0.25*x2r*y1t
        elif j == 1:
            return 0.125*x2r*y1r
        elif j == 2:
            return -0.25*x2r*y2t
        elif j == 3:
            return 0.125*x2r*y2r
        elif j == 4:
            return 0.1*x2r
    elif i == 4:
        if j == 0:
            return -0.2*y1t
        elif j == 1:
            return -0.1*y1r
        elif j == 2:
            return 0.2*y2t
        elif j == 3:
            return -0.1*y2r
        elif j == 5:
            return 0.0571428571428571
    elif i == 5:
        if j == 4:
            return -0.0571428571428571
        elif j == 6:
            return 0.0317460317460317
    elif i == 6:
        if j == 5:
            return -0.0317460317460317
        elif j == 7:
            return 0.0202020202020202
    elif i == 7:
        if j == 6:
            return -0.0202020202020202
        elif j == 8:
            return 0.0139860139860140
    elif i == 8:
        if j == 7:
            return -0.0139860139860140
        elif j == 9:
            return 0.0102564102564103
    elif i == 9:
        if j == 8:
            return -0.0102564102564103
        elif j == 10:
            return 0.00784313725490196
    elif i == 10:
        if j == 9:
            return -0.00784313725490196
        elif j == 11:
            return 0.00619195046439629
    elif i == 11:
        if j == 10:
            return -0.00619195046439629
        elif j == 12:
            return 0.00501253132832080
    elif i == 12:
        if j == 11:
            return -0.00501253132832080
        elif j == 13:
            return 0.00414078674948240
    elif i == 13:
        if j == 12:
            return -0.00414078674948240
        elif j == 14:
            return 0.00347826086956522
    elif i == 14:
        if j == 13:
            return -0.00347826086956522
        elif j == 15:
            return 0.00296296296296296
    elif i == 15:
        if j == 14:
            return -0.00296296296296296
        elif j == 16:
            return 0.00255427841634738
    elif i == 16:
        if j == 15:
            return -0.00255427841634738
        elif j == 17:
            return 0.00222469410456062
    elif i == 17:
        if j == 16:
            return -0.00222469410456062
        elif j == 18:
            return 0.00195503421309873
    elif i == 18:
        if j == 17:
            return -0.00195503421309873
        elif j == 19:
            return 0.00173160173160173
    elif i == 19:
        if j == 18:
            return -0.00173160173160173
        elif j == 20:
            return 0.00154440154440154
    elif i == 20:
        if j == 19:
            return -0.00154440154440154
        elif j == 21:
            return 0.00138600138600139
    elif i == 21:
        if j == 20:
            return -0.00138600138600139
        elif j == 22:
            return 0.00125078173858662
    elif i == 22:
        if j == 21:
            return -0.00125078173858662
        elif j == 23:
            return 0.00113442994895065
    elif i == 23:
        if j == 22:
            return -0.00113442994895065
        elif j == 24:
            return 0.00103359173126615
    elif i == 24:
        if j == 23:
            return -0.00103359173126615
        elif j == 25:
            return 0.000945626477541371
    elif i == 25:
        if j == 24:
            return -0.000945626477541371
        elif j == 26:
            return 0.000868432479374729
    elif i == 26:
        if j == 25:
            return -0.000868432479374729
        elif j == 27:
            return 0.000800320128051221
    elif i == 27:
        if j == 26:
            return -0.000800320128051221
        elif j == 28:
            return 0.000739918608953015
    elif i == 28:
        if j == 27:
            return -0.000739918608953015
        elif j == 29:
            return 0.000686106346483705
    elif i == 29:
        if j == 28:
            return -0.000686106346483705
    return 0


cdef double integral_fxixifxixi(int i, int j, double x1t, double x1r, double x2t, double x2r,
                        double y1t, double y1r, double y2t, double y2r) nogil:
    if i == 0:
        if j == 0:
            return 1.5*x1t*y1t
        elif j == 1:
            return 0.75*x1t*y1r
        elif j == 2:
            return -1.5*x1t*y2t
        elif j == 3:
            return 0.75*x1t*y2r
    elif i == 1:
        if j == 0:
            return 0.75*x1r*y1t
        elif j == 1:
            return 0.5*x1r*y1r
        elif j == 2:
            return -0.75*x1r*y2t
        elif j == 3:
            return 0.25*x1r*y2r
    elif i == 2:
        if j == 0:
            return -1.5*x2t*y1t
        elif j == 1:
            return -0.75*x2t*y1r
        elif j == 2:
            return 1.5*x2t*y2t
        elif j == 3:
            return -0.75*x2t*y2r
    elif i == 3:
        if j == 0:
            return 0.75*x2r*y1t
        elif j == 1:
            return 0.25*x2r*y1r
        elif j == 2:
            return -0.75*x2r*y2t
        elif j == 3:
            return 0.5*x2r*y2r
    elif i == 4:
        if j == 4:
            return 0.400000000000000
    elif i == 5:
        if j == 5:
            return 0.285714285714286
    elif i == 6:
        if j == 6:
            return 0.222222222222222
    elif i == 7:
        if j == 7:
            return 0.181818181818182
    elif i == 8:
        if j == 8:
            return 0.153846153846154
    elif i == 9:
        if j == 9:
            return 0.133333333333333
    elif i == 10:
        if j == 10:
            return 0.117647058823529
    elif i == 11:
        if j == 11:
            return 0.105263157894737
    elif i == 12:
        if j == 12:
            return 0.0952380952380952
    elif i == 13:
        if j == 13:
            return 0.0869565217391304
    elif i == 14:
        if j == 14:
            return 0.0800000000000000
    elif i == 15:
        if j == 15:
            return 0.0740740740740741
    elif i == 16:
        if j == 16:
            return 0.0689655172413793
    elif i == 17:
        if j == 17:
            return 0.0645161290322581
    elif i == 18:
        if j == 18:
            return 0.0606060606060606
    elif i == 19:
        if j == 19:
            return 0.0571428571428571
    elif i == 20:
        if j == 20:
            return 0.0540540540540541
    elif i == 21:
        if j == 21:
            return 0.0512820512820513
    elif i == 22:
        if j == 22:
            return 0.0487804878048781
    elif i == 23:
        if j == 23:
            return 0.0465116279069767
    elif i == 24:
        if j == 24:
            return 0.0444444444444444
    elif i == 25:
        if j == 25:
            return 0.0425531914893617
    elif i == 26:
        if j == 26:
            return 0.0408163265306122
    elif i == 27:
        if j == 27:
            return 0.0392156862745098
    elif i == 28:
        if j == 28:
            return 0.0377358490566038
    elif i == 29:
        if j == 29:
            return 0.0363636363636364
    return 0
