// Bardell's hierarchical functions

// Number of terms: 30

#include <stdlib.h>
#include <math.h>

__declspec(dllexport) void calc_vec_f(double *f, double xi, double xi1t, double xi1r,
                double xi2t, double xi2r) {
    f[0] = xi1t*(0.25*pow(xi, 3) - 0.75*xi + 0.5);
    f[1] = xi1r*(0.125*pow(xi, 3) - 0.125*pow(xi, 2) - 0.125*xi + 0.125);
    f[2] = xi2t*(-0.25*pow(xi, 3) + 0.75*xi + 0.5);
    f[3] = xi2r*(0.125*pow(xi, 3) + 0.125*pow(xi, 2) - 0.125*xi - 0.125);
    f[4] = 0.125*pow(xi, 4) - 0.25*pow(xi, 2) + 0.125;
    f[5] = 0.125*pow(xi, 5) - 0.25*pow(xi, 3) + 0.125*xi;
    f[6] = 0.145833333333333*pow(xi, 6) - 0.3125*pow(xi, 4) + 0.1875*pow(xi, 2) - 0.0208333333333333;
    f[7] = 0.1875*pow(xi, 7) - 0.4375*pow(xi, 5) + 0.3125*pow(xi, 3) - 0.0625*xi;
    f[8] = 0.2578125*pow(xi, 8) - 0.65625*pow(xi, 6) + 0.546875*pow(xi, 4) - 0.15625*pow(xi, 2) + 0.0078125;
    f[9] = 0.372395833333333*pow(xi, 9) - 1.03125*pow(xi, 7) + 0.984375*pow(xi, 5) - 0.364583333333333*pow(xi, 3) + 0.0390625*xi;
    f[10] = 0.55859375*pow(xi, 10) - 1.67578125*pow(xi, 8) + 1.8046875*pow(xi, 6) - 0.8203125*pow(xi, 4) + 0.13671875*pow(xi, 2) - 0.00390625;
    f[11] = 0.86328125*pow(xi, 11) - 2.79296875*pow(xi, 9) + 3.3515625*pow(xi, 7) - 1.8046875*pow(xi, 5) + 0.41015625*pow(xi, 3) - 0.02734375*xi;
    f[12] = 1.36686197916667*pow(xi, 12) - 4.748046875*pow(xi, 10) + 6.2841796875*pow(xi, 8) - 3.91015625*pow(xi, 6) + 1.1279296875*pow(xi, 4) - 0.123046875*pow(xi, 2) + 0.00227864583333333;
    f[13] = 2.2080078125*pow(xi, 13) - 8.201171875*pow(xi, 11) + 11.8701171875*pow(xi, 9) - 8.37890625*pow(xi, 7) + 2.9326171875*pow(xi, 5) - 0.451171875*pow(xi, 3) + 0.0205078125*xi;
    f[14] = 3.62744140625*pow(xi, 14) - 14.35205078125*pow(xi, 12) + 22.55322265625*pow(xi, 10) - 17.80517578125*pow(xi, 8) + 7.33154296875*pow(xi, 6) - 1.46630859375*pow(xi, 4) + 0.11279296875*pow(xi, 2) - 0.00146484375;
    f[15] = 6.04573567708333*pow(xi, 15) - 25.39208984375*pow(xi, 13) + 43.05615234375*pow(xi, 11) - 37.5887044270833*pow(xi, 9) + 17.80517578125*pow(xi, 7) - 4.39892578125*pow(xi, 5) + 0.48876953125*pow(xi, 3) - 0.01611328125*xi;
    f[16] = 10.2021789550781*pow(xi, 16) - 45.343017578125*pow(xi, 14) + 82.5242919921875*pow(xi, 12) - 78.936279296875*pow(xi, 10) + 42.2872924804688*pow(xi, 8) - 12.463623046875*pow(xi, 6) + 1.8328857421875*pow(xi, 4) - 0.104736328125*pow(xi, 2) + 0.001007080078125;
    f[17] = 17.4037170410156*pow(xi, 17) - 81.617431640625*pow(xi, 15) + 158.700561523438*pow(xi, 13) - 165.048583984375*pow(xi, 11) + 98.6703491210938*pow(xi, 9) - 33.829833984375*pow(xi, 7) + 6.2318115234375*pow(xi, 5) - 0.523681640625*pow(xi, 3) + 0.013092041015625*xi;
    f[18] = 29.9730682373047*pow(xi, 18) - 147.931594848633*pow(xi, 16) + 306.065368652344*pow(xi, 14) - 343.851216634115*pow(xi, 12) + 226.941802978516*pow(xi, 10) - 88.8033142089844*pow(xi, 8) + 19.7340698242188*pow(xi, 6) - 2.22564697265625*pow(xi, 4) + 0.0981903076171875*pow(xi, 2) - 0.000727335611979167;
    f[19] = 52.0584869384766*pow(xi, 19) - 269.757614135742*pow(xi, 17) + 591.726379394531*pow(xi, 15) - 714.152526855469*pow(xi, 13) + 515.776824951172*pow(xi, 11) - 226.941802978516*pow(xi, 9) + 59.2022094726563*pow(xi, 7) - 8.45745849609375*pow(xi, 5) + 0.556411743164063*pow(xi, 3) - 0.0109100341796875*xi;
    f[20] = 91.102352142334*pow(xi, 20) - 494.555625915527*pow(xi, 18) + 1146.4698600769*pow(xi, 16) - 1479.31594848633*pow(xi, 14) + 1160.49785614014*pow(xi, 12) - 567.354507446289*pow(xi, 10) + 170.206352233887*pow(xi, 8) - 29.6011047363281*pow(xi, 6) + 2.6429557800293*pow(xi, 4) - 0.0927352905273438*pow(xi, 2) + 0.000545501708984375;
    f[21] = 160.513668060303*pow(xi, 21) - 911.02352142334*pow(xi, 19) + 2225.50031661987*pow(xi, 17) - 3057.25296020508*pow(xi, 15) + 2588.80290985107*pow(xi, 13) - 1392.59742736816*pow(xi, 11) + 472.795422871908*pow(xi, 9) - 97.2607727050781*pow(xi, 7) + 11.100414276123*pow(xi, 5) - 0.587323506673177*pow(xi, 3) + 0.00927352905273438*xi;
    f[22] = 284.546957015991*pow(xi, 22) - 1685.39351463318*pow(xi, 20) + 4327.36172676086*pow(xi, 18) - 6305.58423042297*pow(xi, 16) + 5732.34930038452*pow(xi, 14) - 3365.4437828064*pow(xi, 12) + 1276.54764175415*pow(xi, 10) - 303.939914703369*pow(xi, 8) + 42.5515880584717*pow(xi, 6) - 3.08344841003418*pow(xi, 4) + 0.0880985260009766*pow(xi, 2) - 0.000421524047851563;
    f[23] = 507.235879898071*pow(xi, 23) - 3130.0165271759*pow(xi, 21) + 8426.96757316589*pow(xi, 19) - 12982.0851802826*pow(xi, 17) + 12611.1684608459*pow(xi, 15) - 8025.28902053833*pow(xi, 13) + 3365.4437828064*pow(xi, 11) - 911.819744110107*pow(xi, 9) + 151.969957351685*pow(xi, 7) - 14.1838626861572*pow(xi, 5) + 0.616689682006836*pow(xi, 3) - 0.00800895690917969*xi;
    f[24] = 908.797618150711*pow(xi, 24) - 5833.21261882782*pow(xi, 22) + 16432.5867676735*pow(xi, 20) - 26685.3973150253*pow(xi, 18) + 27586.9310081005*pow(xi, 16) - 18916.7526912689*pow(xi, 14) + 8694.06310558319*pow(xi, 12) - 2644.27725791931*pow(xi, 10) + 512.898606061935*pow(xi, 8) - 59.0994278589884*pow(xi, 6) + 3.54596567153931*pow(xi, 4) - 0.0840940475463867*pow(xi, 2) + 0.000333706537882487;
    f[25] = 1635.83571267128*pow(xi, 25) - 10905.5714178085*pow(xi, 23) + 32082.669403553*pow(xi, 21) - 54775.2892255783*pow(xi, 19) + 60042.143958807*pow(xi, 17) - 44139.0896129608*pow(xi, 15) + 22069.5448064804*pow(xi, 13) - 7452.05409049988*pow(xi, 11) + 1652.67328619957*pow(xi, 9) - 227.954936027527*pow(xi, 7) + 17.7298283576965*pow(xi, 5) - 0.644721031188965*pow(xi, 3) + 0.00700783729553223*xi;
    f[26] = 2957.08763444424*pow(xi, 26) - 20447.946408391*pow(xi, 24) + 62707.0356523991*pow(xi, 22) - 112289.342912436*pow(xi, 20) + 130091.311910748*pow(xi, 18) - 102071.644729972*pow(xi, 16) + 55173.862016201*pow(xi, 14) - 20493.1487488747*pow(xi, 12) + 5123.28718721867*pow(xi, 10) - 826.336643099785*pow(xi, 8) + 79.7842276096344*pow(xi, 6) - 4.02950644493103*pow(xi, 4) + 0.0805901288986206*pow(xi, 2) - 0.000269532203674316;
    f[27] = 5366.5664476951*pow(xi, 27) - 38442.1392477751*pow(xi, 25) + 122687.678450346*pow(xi, 23) - 229925.79739213*pow(xi, 21) + 280723.357281089*pow(xi, 19) - 234164.361439347*pow(xi, 17) + 136095.526306629*pow(xi, 15) - 55173.862016201*pow(xi, 13) + 15369.861561656*pow(xi, 11) - 2846.27065956593*pow(xi, 9) + 330.534657239914*pow(xi, 7) - 21.7593348026276*pow(xi, 5) + 0.671584407488505*pow(xi, 3) - 0.00619924068450928*xi;
    f[28] = 9774.81745830178*pow(xi, 28) - 72448.6470438838*pow(xi, 26) + 240263.370298594*pow(xi, 24) - 470302.767392993*pow(xi, 22) + 603555.218154341*pow(xi, 20) - 533374.378834069*pow(xi, 18) + 331732.845372409*pow(xi, 16) - 145816.635328531*pow(xi, 14) + 44828.7628881633*pow(xi, 12) - 9392.69317656755*pow(xi, 10) + 1280.82179680467*pow(xi, 8) - 105.1701182127*pow(xi, 6) + 4.53319475054741*pow(xi, 4) - 0.077490508556366*pow(xi, 2) + 0.000221401453018188;
    f[29] = 17864.3215617239*pow(xi, 29) - 136847.444416225*pow(xi, 27) + 470916.205785245*pow(xi, 25) - 961053.481194377*pow(xi, 23) + 1293332.61033073*pow(xi, 21) - 1207110.43630868*pow(xi, 19) + 800061.568251103*pow(xi, 17) - 379123.251854181*pow(xi, 15) + 127589.555912465*pow(xi, 13) - 29885.8419254422*pow(xi, 11) + 4696.34658828378*pow(xi, 9) - 465.753380656242*pow(xi, 7) + 26.292529553175*pow(xi, 5) - 0.697414577007294*pow(xi, 3) + 0.00553503632545471*xi;
}


__declspec(dllexport) void calc_vec_fxi(double *fxi, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) {
    fxi[0] = xi1t*(0.75*pow(xi, 2) - 0.75);
    fxi[1] = xi1r*(0.375*pow(xi, 2) - 0.25*xi - 0.125);
    fxi[2] = xi2t*(-0.75*pow(xi, 2) + 0.75);
    fxi[3] = xi2r*(0.375*pow(xi, 2) + 0.25*xi - 0.125);
    fxi[4] = 0.5*pow(xi, 3) - 0.5*xi;
    fxi[5] = 0.625*pow(xi, 4) - 0.75*pow(xi, 2) + 0.125;
    fxi[6] = 0.875*pow(xi, 5) - 1.25*pow(xi, 3) + 0.375*xi;
    fxi[7] = 1.3125*pow(xi, 6) - 2.1875*pow(xi, 4) + 0.9375*pow(xi, 2) - 0.0625;
    fxi[8] = 2.0625*pow(xi, 7) - 3.9375*pow(xi, 5) + 2.1875*pow(xi, 3) - 0.3125*xi;
    fxi[9] = 3.3515625*pow(xi, 8) - 7.21875*pow(xi, 6) + 4.921875*pow(xi, 4) - 1.09375*pow(xi, 2) + 0.0390625;
    fxi[10] = 5.5859375*pow(xi, 9) - 13.40625*pow(xi, 7) + 10.828125*pow(xi, 5) - 3.28125*pow(xi, 3) + 0.2734375*xi;
    fxi[11] = 9.49609375*pow(xi, 10) - 25.13671875*pow(xi, 8) + 23.4609375*pow(xi, 6) - 9.0234375*pow(xi, 4) + 1.23046875*pow(xi, 2) - 0.02734375;
    fxi[12] = 16.40234375*pow(xi, 11) - 47.48046875*pow(xi, 9) + 50.2734375*pow(xi, 7) - 23.4609375*pow(xi, 5) + 4.51171875*pow(xi, 3) - 0.24609375*xi;
    fxi[13] = 28.7041015625*pow(xi, 12) - 90.212890625*pow(xi, 10) + 106.8310546875*pow(xi, 8) - 58.65234375*pow(xi, 6) + 14.6630859375*pow(xi, 4) - 1.353515625*pow(xi, 2) + 0.0205078125;
    fxi[14] = 50.7841796875*pow(xi, 13) - 172.224609375*pow(xi, 11) + 225.5322265625*pow(xi, 9) - 142.44140625*pow(xi, 7) + 43.9892578125*pow(xi, 5) - 5.865234375*pow(xi, 3) + 0.2255859375*xi;
    fxi[15] = 90.68603515625*pow(xi, 14) - 330.09716796875*pow(xi, 12) + 473.61767578125*pow(xi, 10) - 338.29833984375*pow(xi, 8) + 124.63623046875*pow(xi, 6) - 21.99462890625*pow(xi, 4) + 1.46630859375*pow(xi, 2) - 0.01611328125;
    fxi[16] = 163.23486328125*pow(xi, 15) - 634.80224609375*pow(xi, 13) + 990.29150390625*pow(xi, 11) - 789.36279296875*pow(xi, 9) + 338.29833984375*pow(xi, 7) - 74.78173828125*pow(xi, 5) + 7.33154296875*pow(xi, 3) - 0.20947265625*xi;
    fxi[17] = 295.863189697266*pow(xi, 16) - 1224.26147460938*pow(xi, 14) + 2063.10729980469*pow(xi, 12) - 1815.53442382813*pow(xi, 10) + 888.033142089844*pow(xi, 8) - 236.808837890625*pow(xi, 6) + 31.1590576171875*pow(xi, 4) - 1.571044921875*pow(xi, 2) + 0.013092041015625;
    fxi[18] = 539.515228271484*pow(xi, 17) - 2366.90551757813*pow(xi, 15) + 4284.91516113281*pow(xi, 13) - 4126.21459960938*pow(xi, 11) + 2269.41802978516*pow(xi, 9) - 710.426513671875*pow(xi, 7) + 118.404418945313*pow(xi, 5) - 8.902587890625*pow(xi, 3) + 0.196380615234375*xi;
    fxi[19] = 989.111251831055*pow(xi, 18) - 4585.87944030762*pow(xi, 16) + 8875.89569091797*pow(xi, 14) - 9283.98284912109*pow(xi, 12) + 5673.54507446289*pow(xi, 10) - 2042.47622680664*pow(xi, 8) + 414.415466308594*pow(xi, 6) - 42.2872924804688*pow(xi, 4) + 1.66923522949219*pow(xi, 2) - 0.0109100341796875;
    fxi[20] = 1822.04704284668*pow(xi, 19) - 8902.00126647949*pow(xi, 17) + 18343.5177612305*pow(xi, 15) - 20710.4232788086*pow(xi, 13) + 13925.9742736816*pow(xi, 11) - 5673.54507446289*pow(xi, 9) + 1361.65081787109*pow(xi, 7) - 177.606628417969*pow(xi, 5) + 10.5718231201172*pow(xi, 3) - 0.185470581054688*xi;
    fxi[21] = 3370.78702926636*pow(xi, 20) - 17309.4469070435*pow(xi, 18) + 37833.5053825378*pow(xi, 16) - 45858.7944030762*pow(xi, 14) + 33654.437828064*pow(xi, 12) - 15318.5717010498*pow(xi, 10) + 4255.15880584717*pow(xi, 8) - 680.825408935547*pow(xi, 6) + 55.5020713806152*pow(xi, 4) - 1.76197052001953*pow(xi, 2) + 0.00927352905273438;
    fxi[22] = 6260.03305435181*pow(xi, 21) - 33707.8702926636*pow(xi, 19) + 77892.5110816956*pow(xi, 17) - 100889.347686768*pow(xi, 15) + 80252.8902053833*pow(xi, 13) - 40385.3253936768*pow(xi, 11) + 12765.4764175415*pow(xi, 9) - 2431.51931762695*pow(xi, 7) + 255.30952835083*pow(xi, 5) - 12.3337936401367*pow(xi, 3) + 0.176197052001953*xi;
    fxi[23] = 11666.4252376556*pow(xi, 22) - 65730.347070694*pow(xi, 20) + 160112.383890152*pow(xi, 18) - 220695.448064804*pow(xi, 16) + 189167.526912689*pow(xi, 14) - 104328.757266998*pow(xi, 12) + 37019.8816108704*pow(xi, 10) - 8206.37769699097*pow(xi, 8) + 1063.78970146179*pow(xi, 6) - 70.9193134307861*pow(xi, 4) + 1.85006904602051*pow(xi, 2) - 0.00800895690917969;
    fxi[24] = 21811.1428356171*pow(xi, 23) - 128330.677614212*pow(xi, 21) + 328651.73535347*pow(xi, 19) - 480337.151670456*pow(xi, 17) + 441390.896129608*pow(xi, 15) - 264834.537677765*pow(xi, 13) + 104328.757266998*pow(xi, 11) - 26442.7725791931*pow(xi, 9) + 4103.18884849548*pow(xi, 7) - 354.596567153931*pow(xi, 5) + 14.1838626861572*pow(xi, 3) - 0.168188095092773*xi;
    fxi[25] = 40895.892816782*pow(xi, 24) - 250828.142609596*pow(xi, 22) + 673736.057474613*pow(xi, 20) - 1040730.49528599*pow(xi, 18) + 1020716.44729972*pow(xi, 16) - 662086.344194412*pow(xi, 14) + 286904.082484245*pow(xi, 12) - 81972.5949954987*pow(xi, 10) + 14874.0595757961*pow(xi, 8) - 1595.68455219269*pow(xi, 6) + 88.6491417884827*pow(xi, 4) - 1.93416309356689*pow(xi, 2) + 0.00700783729553223;
    fxi[26] = 76884.2784955502*pow(xi, 25) - 490750.713801384*pow(xi, 23) + 1379554.78435278*pow(xi, 21) - 2245786.85824871*pow(xi, 19) + 2341643.61439347*pow(xi, 17) - 1633146.31567955*pow(xi, 15) + 772434.068226814*pow(xi, 13) - 245917.784986496*pow(xi, 11) + 51232.8718721867*pow(xi, 9) - 6610.69314479828*pow(xi, 7) + 478.705365657806*pow(xi, 5) - 16.1180257797241*pow(xi, 3) + 0.161180257797241*xi;
    fxi[27] = 144897.294087768*pow(xi, 26) - 961053.481194377*pow(xi, 24) + 2821816.60435796*pow(xi, 22) - 4828441.74523473*pow(xi, 20) + 5333743.78834069*pow(xi, 18) - 3980794.1444689*pow(xi, 16) + 2041432.89459944*pow(xi, 14) - 717260.206210613*pow(xi, 12) + 169068.477178216*pow(xi, 10) - 25616.4359360933*pow(xi, 8) + 2313.7426006794*pow(xi, 6) - 108.796674013138*pow(xi, 4) + 2.01475322246552*pow(xi, 2) - 0.00619924068450928;
    fxi[28] = 273694.88883245*pow(xi, 27) - 1883664.82314098*pow(xi, 25) + 5766320.88716626*pow(xi, 23) - 10346660.8826458*pow(xi, 21) + 12071104.3630868*pow(xi, 19) - 9600738.81901324*pow(xi, 17) + 5307725.52595854*pow(xi, 15) - 2041432.89459944*pow(xi, 13) + 537945.15465796*pow(xi, 11) - 93926.9317656755*pow(xi, 9) + 10246.5743744373*pow(xi, 7) - 631.020709276199*pow(xi, 5) + 18.1327790021896*pow(xi, 3) - 0.154981017112732*xi;
    fxi[29] = 518065.325289994*pow(xi, 28) - 3694880.99923807*pow(xi, 26) + 11772905.1446311*pow(xi, 24) - 22104230.0674707*pow(xi, 22) + 27159984.8169453*pow(xi, 20) - 22935098.289865*pow(xi, 18) + 13601046.6602688*pow(xi, 16) - 5686848.77781272*pow(xi, 14) + 1658664.22686204*pow(xi, 12) - 328744.261179864*pow(xi, 10) + 42267.119294554*pow(xi, 8) - 3260.2736645937*pow(xi, 6) + 131.462647765875*pow(xi, 4) - 2.09224373102188*pow(xi, 2) + 0.00553503632545471;
}


__declspec(dllexport) double calc_f(int i, double xi, double xi1t, double xi1r,
              double xi2t, double xi2r) {
    switch(i) {
    case 0:
        return xi1t*(0.25*pow(xi, 3) - 0.75*xi + 0.5);
    case 1:
        return xi1r*(0.125*pow(xi, 3) - 0.125*pow(xi, 2) - 0.125*xi + 0.125);
    case 2:
        return xi2t*(-0.25*pow(xi, 3) + 0.75*xi + 0.5);
    case 3:
        return xi2r*(0.125*pow(xi, 3) + 0.125*pow(xi, 2) - 0.125*xi - 0.125);
    case 4:
        return 0.125*pow(xi, 4) - 0.25*pow(xi, 2) + 0.125;
    case 5:
        return 0.125*pow(xi, 5) - 0.25*pow(xi, 3) + 0.125*xi;
    case 6:
        return 0.145833333333333*pow(xi, 6) - 0.3125*pow(xi, 4) + 0.1875*pow(xi, 2) - 0.0208333333333333;
    case 7:
        return 0.1875*pow(xi, 7) - 0.4375*pow(xi, 5) + 0.3125*pow(xi, 3) - 0.0625*xi;
    case 8:
        return 0.2578125*pow(xi, 8) - 0.65625*pow(xi, 6) + 0.546875*pow(xi, 4) - 0.15625*pow(xi, 2) + 0.0078125;
    case 9:
        return 0.372395833333333*pow(xi, 9) - 1.03125*pow(xi, 7) + 0.984375*pow(xi, 5) - 0.364583333333333*pow(xi, 3) + 0.0390625*xi;
    case 10:
        return 0.55859375*pow(xi, 10) - 1.67578125*pow(xi, 8) + 1.8046875*pow(xi, 6) - 0.8203125*pow(xi, 4) + 0.13671875*pow(xi, 2) - 0.00390625;
    case 11:
        return 0.86328125*pow(xi, 11) - 2.79296875*pow(xi, 9) + 3.3515625*pow(xi, 7) - 1.8046875*pow(xi, 5) + 0.41015625*pow(xi, 3) - 0.02734375*xi;
    case 12:
        return 1.36686197916667*pow(xi, 12) - 4.748046875*pow(xi, 10) + 6.2841796875*pow(xi, 8) - 3.91015625*pow(xi, 6) + 1.1279296875*pow(xi, 4) - 0.123046875*pow(xi, 2) + 0.00227864583333333;
    case 13:
        return 2.2080078125*pow(xi, 13) - 8.201171875*pow(xi, 11) + 11.8701171875*pow(xi, 9) - 8.37890625*pow(xi, 7) + 2.9326171875*pow(xi, 5) - 0.451171875*pow(xi, 3) + 0.0205078125*xi;
    case 14:
        return 3.62744140625*pow(xi, 14) - 14.35205078125*pow(xi, 12) + 22.55322265625*pow(xi, 10) - 17.80517578125*pow(xi, 8) + 7.33154296875*pow(xi, 6) - 1.46630859375*pow(xi, 4) + 0.11279296875*pow(xi, 2) - 0.00146484375;
    case 15:
        return 6.04573567708333*pow(xi, 15) - 25.39208984375*pow(xi, 13) + 43.05615234375*pow(xi, 11) - 37.5887044270833*pow(xi, 9) + 17.80517578125*pow(xi, 7) - 4.39892578125*pow(xi, 5) + 0.48876953125*pow(xi, 3) - 0.01611328125*xi;
    case 16:
        return 10.2021789550781*pow(xi, 16) - 45.343017578125*pow(xi, 14) + 82.5242919921875*pow(xi, 12) - 78.936279296875*pow(xi, 10) + 42.2872924804688*pow(xi, 8) - 12.463623046875*pow(xi, 6) + 1.8328857421875*pow(xi, 4) - 0.104736328125*pow(xi, 2) + 0.001007080078125;
    case 17:
        return 17.4037170410156*pow(xi, 17) - 81.617431640625*pow(xi, 15) + 158.700561523438*pow(xi, 13) - 165.048583984375*pow(xi, 11) + 98.6703491210938*pow(xi, 9) - 33.829833984375*pow(xi, 7) + 6.2318115234375*pow(xi, 5) - 0.523681640625*pow(xi, 3) + 0.013092041015625*xi;
    case 18:
        return 29.9730682373047*pow(xi, 18) - 147.931594848633*pow(xi, 16) + 306.065368652344*pow(xi, 14) - 343.851216634115*pow(xi, 12) + 226.941802978516*pow(xi, 10) - 88.8033142089844*pow(xi, 8) + 19.7340698242188*pow(xi, 6) - 2.22564697265625*pow(xi, 4) + 0.0981903076171875*pow(xi, 2) - 0.000727335611979167;
    case 19:
        return 52.0584869384766*pow(xi, 19) - 269.757614135742*pow(xi, 17) + 591.726379394531*pow(xi, 15) - 714.152526855469*pow(xi, 13) + 515.776824951172*pow(xi, 11) - 226.941802978516*pow(xi, 9) + 59.2022094726563*pow(xi, 7) - 8.45745849609375*pow(xi, 5) + 0.556411743164063*pow(xi, 3) - 0.0109100341796875*xi;
    case 20:
        return 91.102352142334*pow(xi, 20) - 494.555625915527*pow(xi, 18) + 1146.4698600769*pow(xi, 16) - 1479.31594848633*pow(xi, 14) + 1160.49785614014*pow(xi, 12) - 567.354507446289*pow(xi, 10) + 170.206352233887*pow(xi, 8) - 29.6011047363281*pow(xi, 6) + 2.6429557800293*pow(xi, 4) - 0.0927352905273438*pow(xi, 2) + 0.000545501708984375;
    case 21:
        return 160.513668060303*pow(xi, 21) - 911.02352142334*pow(xi, 19) + 2225.50031661987*pow(xi, 17) - 3057.25296020508*pow(xi, 15) + 2588.80290985107*pow(xi, 13) - 1392.59742736816*pow(xi, 11) + 472.795422871908*pow(xi, 9) - 97.2607727050781*pow(xi, 7) + 11.100414276123*pow(xi, 5) - 0.587323506673177*pow(xi, 3) + 0.00927352905273438*xi;
    case 22:
        return 284.546957015991*pow(xi, 22) - 1685.39351463318*pow(xi, 20) + 4327.36172676086*pow(xi, 18) - 6305.58423042297*pow(xi, 16) + 5732.34930038452*pow(xi, 14) - 3365.4437828064*pow(xi, 12) + 1276.54764175415*pow(xi, 10) - 303.939914703369*pow(xi, 8) + 42.5515880584717*pow(xi, 6) - 3.08344841003418*pow(xi, 4) + 0.0880985260009766*pow(xi, 2) - 0.000421524047851563;
    case 23:
        return 507.235879898071*pow(xi, 23) - 3130.0165271759*pow(xi, 21) + 8426.96757316589*pow(xi, 19) - 12982.0851802826*pow(xi, 17) + 12611.1684608459*pow(xi, 15) - 8025.28902053833*pow(xi, 13) + 3365.4437828064*pow(xi, 11) - 911.819744110107*pow(xi, 9) + 151.969957351685*pow(xi, 7) - 14.1838626861572*pow(xi, 5) + 0.616689682006836*pow(xi, 3) - 0.00800895690917969*xi;
    case 24:
        return 908.797618150711*pow(xi, 24) - 5833.21261882782*pow(xi, 22) + 16432.5867676735*pow(xi, 20) - 26685.3973150253*pow(xi, 18) + 27586.9310081005*pow(xi, 16) - 18916.7526912689*pow(xi, 14) + 8694.06310558319*pow(xi, 12) - 2644.27725791931*pow(xi, 10) + 512.898606061935*pow(xi, 8) - 59.0994278589884*pow(xi, 6) + 3.54596567153931*pow(xi, 4) - 0.0840940475463867*pow(xi, 2) + 0.000333706537882487;
    case 25:
        return 1635.83571267128*pow(xi, 25) - 10905.5714178085*pow(xi, 23) + 32082.669403553*pow(xi, 21) - 54775.2892255783*pow(xi, 19) + 60042.143958807*pow(xi, 17) - 44139.0896129608*pow(xi, 15) + 22069.5448064804*pow(xi, 13) - 7452.05409049988*pow(xi, 11) + 1652.67328619957*pow(xi, 9) - 227.954936027527*pow(xi, 7) + 17.7298283576965*pow(xi, 5) - 0.644721031188965*pow(xi, 3) + 0.00700783729553223*xi;
    case 26:
        return 2957.08763444424*pow(xi, 26) - 20447.946408391*pow(xi, 24) + 62707.0356523991*pow(xi, 22) - 112289.342912436*pow(xi, 20) + 130091.311910748*pow(xi, 18) - 102071.644729972*pow(xi, 16) + 55173.862016201*pow(xi, 14) - 20493.1487488747*pow(xi, 12) + 5123.28718721867*pow(xi, 10) - 826.336643099785*pow(xi, 8) + 79.7842276096344*pow(xi, 6) - 4.02950644493103*pow(xi, 4) + 0.0805901288986206*pow(xi, 2) - 0.000269532203674316;
    case 27:
        return 5366.5664476951*pow(xi, 27) - 38442.1392477751*pow(xi, 25) + 122687.678450346*pow(xi, 23) - 229925.79739213*pow(xi, 21) + 280723.357281089*pow(xi, 19) - 234164.361439347*pow(xi, 17) + 136095.526306629*pow(xi, 15) - 55173.862016201*pow(xi, 13) + 15369.861561656*pow(xi, 11) - 2846.27065956593*pow(xi, 9) + 330.534657239914*pow(xi, 7) - 21.7593348026276*pow(xi, 5) + 0.671584407488505*pow(xi, 3) - 0.00619924068450928*xi;
    case 28:
        return 9774.81745830178*pow(xi, 28) - 72448.6470438838*pow(xi, 26) + 240263.370298594*pow(xi, 24) - 470302.767392993*pow(xi, 22) + 603555.218154341*pow(xi, 20) - 533374.378834069*pow(xi, 18) + 331732.845372409*pow(xi, 16) - 145816.635328531*pow(xi, 14) + 44828.7628881633*pow(xi, 12) - 9392.69317656755*pow(xi, 10) + 1280.82179680467*pow(xi, 8) - 105.1701182127*pow(xi, 6) + 4.53319475054741*pow(xi, 4) - 0.077490508556366*pow(xi, 2) + 0.000221401453018188;
    case 29:
        return 17864.3215617239*pow(xi, 29) - 136847.444416225*pow(xi, 27) + 470916.205785245*pow(xi, 25) - 961053.481194377*pow(xi, 23) + 1293332.61033073*pow(xi, 21) - 1207110.43630868*pow(xi, 19) + 800061.568251103*pow(xi, 17) - 379123.251854181*pow(xi, 15) + 127589.555912465*pow(xi, 13) - 29885.8419254422*pow(xi, 11) + 4696.34658828378*pow(xi, 9) - 465.753380656242*pow(xi, 7) + 26.292529553175*pow(xi, 5) - 0.697414577007294*pow(xi, 3) + 0.00553503632545471*xi;
    }
}


__declspec(dllexport) double calc_fxi(int i, double xi, double xi1t, double xi1r,
                double xi2t, double xi2r) {
    switch(i) {
    case 0:
        return xi1t*(0.75*pow(xi, 2) - 0.75);
    case 1:
        return xi1r*(0.375*pow(xi, 2) - 0.25*xi - 0.125);
    case 2:
        return xi2t*(-0.75*pow(xi, 2) + 0.75);
    case 3:
        return xi2r*(0.375*pow(xi, 2) + 0.25*xi - 0.125);
    case 4:
        return 0.5*pow(xi, 3) - 0.5*xi;
    case 5:
        return 0.625*pow(xi, 4) - 0.75*pow(xi, 2) + 0.125;
    case 6:
        return 0.875*pow(xi, 5) - 1.25*pow(xi, 3) + 0.375*xi;
    case 7:
        return 1.3125*pow(xi, 6) - 2.1875*pow(xi, 4) + 0.9375*pow(xi, 2) - 0.0625;
    case 8:
        return 2.0625*pow(xi, 7) - 3.9375*pow(xi, 5) + 2.1875*pow(xi, 3) - 0.3125*xi;
    case 9:
        return 3.3515625*pow(xi, 8) - 7.21875*pow(xi, 6) + 4.921875*pow(xi, 4) - 1.09375*pow(xi, 2) + 0.0390625;
    case 10:
        return 5.5859375*pow(xi, 9) - 13.40625*pow(xi, 7) + 10.828125*pow(xi, 5) - 3.28125*pow(xi, 3) + 0.2734375*xi;
    case 11:
        return 9.49609375*pow(xi, 10) - 25.13671875*pow(xi, 8) + 23.4609375*pow(xi, 6) - 9.0234375*pow(xi, 4) + 1.23046875*pow(xi, 2) - 0.02734375;
    case 12:
        return 16.40234375*pow(xi, 11) - 47.48046875*pow(xi, 9) + 50.2734375*pow(xi, 7) - 23.4609375*pow(xi, 5) + 4.51171875*pow(xi, 3) - 0.24609375*xi;
    case 13:
        return 28.7041015625*pow(xi, 12) - 90.212890625*pow(xi, 10) + 106.8310546875*pow(xi, 8) - 58.65234375*pow(xi, 6) + 14.6630859375*pow(xi, 4) - 1.353515625*pow(xi, 2) + 0.0205078125;
    case 14:
        return 50.7841796875*pow(xi, 13) - 172.224609375*pow(xi, 11) + 225.5322265625*pow(xi, 9) - 142.44140625*pow(xi, 7) + 43.9892578125*pow(xi, 5) - 5.865234375*pow(xi, 3) + 0.2255859375*xi;
    case 15:
        return 90.68603515625*pow(xi, 14) - 330.09716796875*pow(xi, 12) + 473.61767578125*pow(xi, 10) - 338.29833984375*pow(xi, 8) + 124.63623046875*pow(xi, 6) - 21.99462890625*pow(xi, 4) + 1.46630859375*pow(xi, 2) - 0.01611328125;
    case 16:
        return 163.23486328125*pow(xi, 15) - 634.80224609375*pow(xi, 13) + 990.29150390625*pow(xi, 11) - 789.36279296875*pow(xi, 9) + 338.29833984375*pow(xi, 7) - 74.78173828125*pow(xi, 5) + 7.33154296875*pow(xi, 3) - 0.20947265625*xi;
    case 17:
        return 295.863189697266*pow(xi, 16) - 1224.26147460938*pow(xi, 14) + 2063.10729980469*pow(xi, 12) - 1815.53442382813*pow(xi, 10) + 888.033142089844*pow(xi, 8) - 236.808837890625*pow(xi, 6) + 31.1590576171875*pow(xi, 4) - 1.571044921875*pow(xi, 2) + 0.013092041015625;
    case 18:
        return 539.515228271484*pow(xi, 17) - 2366.90551757813*pow(xi, 15) + 4284.91516113281*pow(xi, 13) - 4126.21459960938*pow(xi, 11) + 2269.41802978516*pow(xi, 9) - 710.426513671875*pow(xi, 7) + 118.404418945313*pow(xi, 5) - 8.902587890625*pow(xi, 3) + 0.196380615234375*xi;
    case 19:
        return 989.111251831055*pow(xi, 18) - 4585.87944030762*pow(xi, 16) + 8875.89569091797*pow(xi, 14) - 9283.98284912109*pow(xi, 12) + 5673.54507446289*pow(xi, 10) - 2042.47622680664*pow(xi, 8) + 414.415466308594*pow(xi, 6) - 42.2872924804688*pow(xi, 4) + 1.66923522949219*pow(xi, 2) - 0.0109100341796875;
    case 20:
        return 1822.04704284668*pow(xi, 19) - 8902.00126647949*pow(xi, 17) + 18343.5177612305*pow(xi, 15) - 20710.4232788086*pow(xi, 13) + 13925.9742736816*pow(xi, 11) - 5673.54507446289*pow(xi, 9) + 1361.65081787109*pow(xi, 7) - 177.606628417969*pow(xi, 5) + 10.5718231201172*pow(xi, 3) - 0.185470581054688*xi;
    case 21:
        return 3370.78702926636*pow(xi, 20) - 17309.4469070435*pow(xi, 18) + 37833.5053825378*pow(xi, 16) - 45858.7944030762*pow(xi, 14) + 33654.437828064*pow(xi, 12) - 15318.5717010498*pow(xi, 10) + 4255.15880584717*pow(xi, 8) - 680.825408935547*pow(xi, 6) + 55.5020713806152*pow(xi, 4) - 1.76197052001953*pow(xi, 2) + 0.00927352905273438;
    case 22:
        return 6260.03305435181*pow(xi, 21) - 33707.8702926636*pow(xi, 19) + 77892.5110816956*pow(xi, 17) - 100889.347686768*pow(xi, 15) + 80252.8902053833*pow(xi, 13) - 40385.3253936768*pow(xi, 11) + 12765.4764175415*pow(xi, 9) - 2431.51931762695*pow(xi, 7) + 255.30952835083*pow(xi, 5) - 12.3337936401367*pow(xi, 3) + 0.176197052001953*xi;
    case 23:
        return 11666.4252376556*pow(xi, 22) - 65730.347070694*pow(xi, 20) + 160112.383890152*pow(xi, 18) - 220695.448064804*pow(xi, 16) + 189167.526912689*pow(xi, 14) - 104328.757266998*pow(xi, 12) + 37019.8816108704*pow(xi, 10) - 8206.37769699097*pow(xi, 8) + 1063.78970146179*pow(xi, 6) - 70.9193134307861*pow(xi, 4) + 1.85006904602051*pow(xi, 2) - 0.00800895690917969;
    case 24:
        return 21811.1428356171*pow(xi, 23) - 128330.677614212*pow(xi, 21) + 328651.73535347*pow(xi, 19) - 480337.151670456*pow(xi, 17) + 441390.896129608*pow(xi, 15) - 264834.537677765*pow(xi, 13) + 104328.757266998*pow(xi, 11) - 26442.7725791931*pow(xi, 9) + 4103.18884849548*pow(xi, 7) - 354.596567153931*pow(xi, 5) + 14.1838626861572*pow(xi, 3) - 0.168188095092773*xi;
    case 25:
        return 40895.892816782*pow(xi, 24) - 250828.142609596*pow(xi, 22) + 673736.057474613*pow(xi, 20) - 1040730.49528599*pow(xi, 18) + 1020716.44729972*pow(xi, 16) - 662086.344194412*pow(xi, 14) + 286904.082484245*pow(xi, 12) - 81972.5949954987*pow(xi, 10) + 14874.0595757961*pow(xi, 8) - 1595.68455219269*pow(xi, 6) + 88.6491417884827*pow(xi, 4) - 1.93416309356689*pow(xi, 2) + 0.00700783729553223;
    case 26:
        return 76884.2784955502*pow(xi, 25) - 490750.713801384*pow(xi, 23) + 1379554.78435278*pow(xi, 21) - 2245786.85824871*pow(xi, 19) + 2341643.61439347*pow(xi, 17) - 1633146.31567955*pow(xi, 15) + 772434.068226814*pow(xi, 13) - 245917.784986496*pow(xi, 11) + 51232.8718721867*pow(xi, 9) - 6610.69314479828*pow(xi, 7) + 478.705365657806*pow(xi, 5) - 16.1180257797241*pow(xi, 3) + 0.161180257797241*xi;
    case 27:
        return 144897.294087768*pow(xi, 26) - 961053.481194377*pow(xi, 24) + 2821816.60435796*pow(xi, 22) - 4828441.74523473*pow(xi, 20) + 5333743.78834069*pow(xi, 18) - 3980794.1444689*pow(xi, 16) + 2041432.89459944*pow(xi, 14) - 717260.206210613*pow(xi, 12) + 169068.477178216*pow(xi, 10) - 25616.4359360933*pow(xi, 8) + 2313.7426006794*pow(xi, 6) - 108.796674013138*pow(xi, 4) + 2.01475322246552*pow(xi, 2) - 0.00619924068450928;
    case 28:
        return 273694.88883245*pow(xi, 27) - 1883664.82314098*pow(xi, 25) + 5766320.88716626*pow(xi, 23) - 10346660.8826458*pow(xi, 21) + 12071104.3630868*pow(xi, 19) - 9600738.81901324*pow(xi, 17) + 5307725.52595854*pow(xi, 15) - 2041432.89459944*pow(xi, 13) + 537945.15465796*pow(xi, 11) - 93926.9317656755*pow(xi, 9) + 10246.5743744373*pow(xi, 7) - 631.020709276199*pow(xi, 5) + 18.1327790021896*pow(xi, 3) - 0.154981017112732*xi;
    case 29:
        return 518065.325289994*pow(xi, 28) - 3694880.99923807*pow(xi, 26) + 11772905.1446311*pow(xi, 24) - 22104230.0674707*pow(xi, 22) + 27159984.8169453*pow(xi, 20) - 22935098.289865*pow(xi, 18) + 13601046.6602688*pow(xi, 16) - 5686848.77781272*pow(xi, 14) + 1658664.22686204*pow(xi, 12) - 328744.261179864*pow(xi, 10) + 42267.119294554*pow(xi, 8) - 3260.2736645937*pow(xi, 6) + 131.462647765875*pow(xi, 4) - 2.09224373102188*pow(xi, 2) + 0.00553503632545471;
    }
}
