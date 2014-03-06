import cProfile
import pstats

import numpy as np

def regions_conditions(cc, name='k0L'):
    if cc.NL_linear_kinematics=='sanders_numerical':
        from clpt_NL_sanders_numerical import calc_k0L, calc_kLL, calc_kGNL
    elif cc.NL_linear_kinematics=='donnell_numerical':
        from clpt_NL_donnell_numerical import calc_k0L, calc_kLL, calc_kGNL
    else:
        raise
    which = {'k0L':calc_k0L, 'kGNL':calc_kGNL, 'kLL':calc_kLL}
    m = which[name](cc.cs[0], cc.alpharad, cc.r2, cc.L, cc.F,
            cc.m1, cc.m2, cc.n2, cc.pdoff, c00=cc.uTM,
            nx=cc.nx, nt=cc.nt, num_cores=cc.ni_num_cores,
            method=cc.ni_method)
    m = m.toarray()
    if 'clpt' in cc.linear_kinematics:
        print('CLPT')
        g1p = 3
        g2p = 6
    elif 'fsdt' in cc.linear_kinematics:
        print('FSDT')
        g1p = 5
        g2p = 10
    print('REPORT k 00')
    print('cond_1', m[0, 0])
    print('REPORT k 01')
    terms1 = 0
    cond_1 = 0
    for k1 in range(1, cc.m1+1):
        c = cc.pdoff + (k1-1)*g1p
        terms1 += 1
        cond_1 += np.abs(m[0, c:c+g1p]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)

    print('REPORT k 02')
    terms1 = 0
    cond_1 = 0
    for k2 in range(1, cc.m2+1):
        for l2 in range(1, cc.n2+1):
            c = cc.pdoff + cc.m1*g1p + (k2-1)*g2p + (l2-1)*cc.m2*g2p
            terms1 += 1
            cond_1 += np.abs(m[0, c:c+g2p]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)

    print('REPORT k 10')
    terms1 = 0
    cond_1 = 0
    for i1 in range(1, cc.m1+1):
        r = cc.pdoff + (i1-1)*g1p
        terms1 += 1
        cond_1 += np.abs(m[r:r+g1p, 0]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)

    print('REPORT k 11')
    terms1 = 0
    terms2 = 0
    cond_1 = 0
    cond_2 = 0
    for i1 in range(1, cc.m1+1):
        r = cc.pdoff + (i1-1)*g1p
        for k1 in range(1, cc.m1+1):
            c = cc.pdoff + (k1-1)*g1p
            if i1==k1:
                terms1 += 1
                cond_1 += np.abs(m[r:r+g1p, c:c+g1p]).sum()
            elif k1!=i1:
                terms2 += 1
                cond_2 += np.abs(m[r:r+g1p, c:c+g1p]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)
    print('cond_2', cond_2, terms2, cond_2/terms2)

    print('REPORT k 12')
    termsall = 0
    condsall = 0
    for i1 in range(1, cc.m1+1):
        r = cc.pdoff + (i1-1)*g1p
        for k2 in range(1, cc.m2+1):
            for l2 in range(1, cc.n2+1):
                c = cc.pdoff + cc.m1*g1p + (k2-1)*g2p + (l2-1)*cc.m2*g2p
                termsall += 1
                condsall += np.abs(m[r:r+g1p, c:c+g2p]).sum()
    print('conds_all', condsall, termsall, condsall/termsall)

    print('REPORT k 20')
    terms1 = 0
    cond_1 = 0
    for i2 in range(1, cc.m2+1):
        for j2 in range(1, cc.n2+1):
            r = cc.pdoff + cc.m1*g1p + (i2-1)*g2p + (j2-1)*cc.m2*g2p
            terms1 += 1
            cond_1 += np.abs(m[r:r+g2p, 0]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)

    print('REPORT k 21')
    termsall = 0
    condsall = 0
    for i2 in range(1, cc.m2+1):
        for j2 in range(1, cc.n2+1):
            r = cc.pdoff + cc.m1*g1p + (i2-1)*g2p + (j2-1)*cc.m2*g2p
            for k1 in range(1, cc.m1+1):
                c = cc.pdoff + (k1-1)*g1p
                termsall += 1
                condsall += np.abs(m[r:r+g2p, c:c+g1p]).sum()
    print('condsall', condsall, termsall, condsall/termsall)

    print('REPORT k 22')
    terms1 = 0
    terms2 = 0
    terms3 = 0
    terms4 = 0
    cond_1 = 0
    cond_2 = 0
    cond_3 = 0
    cond_4 = 0
    for i2 in range(1, cc.m2+1):
        for j2 in range(1, cc.n2+1):
            r = cc.pdoff + cc.m1*g1p + (i2-1)*g2p + (j2-1)*cc.m2*g2p
            for k2 in range(1, cc.m2+1):
                for l2 in range(1, cc.n2+1):
                    c = cc.pdoff + cc.m1*g1p + (k2-1)*g2p + (l2-1)*cc.m2*g2p
                    if k2==i2 and l2==j2:
                        terms1 += 1
                        cond_1 += np.abs(m[r:r+g2p, c:c+g2p]).sum()
                    elif k2!=i2 and l2==j2:
                        terms2 += 1
                        cond_2 += np.abs(m[r:r+g2p, c:c+g2p]).sum()
                    elif k2!=i2 and l2!=j2:
                        terms3 += 1
                        cond_3 += np.abs(m[r:r+g2p, c:c+g2p]).sum()
                    elif k2==i2 and l2!=j2:
                        terms4 += 1
                        cond_4 += np.abs(m[r:r+g2p, c:c+g2p]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)
    print('cond_2', cond_2, terms2, cond_2/terms2)
    print('cond_3', cond_3, terms3, cond_3/terms3)
    print('cond_4', cond_4, terms4, cond_4/terms4)

def convergence_nx_nt():
    '''It was selected a hypothetical cone with the laminate of cylinder
    Z33 and a semi-vertex angle of 35.

    The data generated from this function is used to define the dictionary
    `nx_nt_table` inside `ConeCyl.rebuild()`.

    For simplification, it is assumed `m1=m2=n2`.
    '''
    from desicos.conecyl import ConeCyl
    cc = ConeCyl()
    cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
    cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
    cc.plyt = 0.125
    cc.r2 = 250.
    cc.H = 510.
    cc.alphadeg = 35.
    cc.linear_kinematics='clpt_donnell'
    cc.NL_kinematics='donnell_numerical'
    cc.Fc = 200000
    cc.initialInc = 0.2
    cc.minInc = 1.e-3
    cc.modified_NR = False
    cc.compute_every_n = 10
    cc.pd = False
    PLs = [5, 20, 30, 45, 60, 70, 80]
    PLs = [90]
    for n in [20, 25, 30, 35]:
        cc.m1 = cc.m2 = cc.n2 = n
        for nt in [60, 80, 100, 120]:
            cc.nx = cc.nt = nt
            print cProfile.runctx('curves = cc.SPLA(PLs, NLgeom=True)',
                              globals(), locals(), 'tester.prof')


def main():
    from numpy import pi, sin, cos
    import matplotlib.pyplot as plt
    def f(x, i2, wx, wt):
        k2 = i2
        j2 = 1.
        l2 = 1.
        sina = 1.
        r = 1.
        L = 1.
        A11 = 1.
        A12 = 1.
        A16 = 1.
        A22 = 1.
        A26 = 1.
        A66 = 1.
        sinj2t = 1.
        cosj2t = 1.
        sinl2t = 1.
        cosl2t = 1.
        sini2x = sin(pi*i2*x/L)
        cosi2x = cos(pi*i2*x/L)
        sink2x = sin(pi*k2*x/L)
        cosk2x = cos(pi*k2*x/L)
        p00 = pi*A11*i2*r*sinj2t*cosi2x/L + (A12*sina*sinj2t + A16*j2*cosj2t)*sini2x
        p01 = pi*A12*i2*r*sinj2t*cosi2x/L + (A22*sina*sinj2t + A26*j2*cosj2t)*sini2x
        p02 = pi*A16*i2*r*sinj2t*cosi2x/L + (A26*sina*sinj2t + A66*j2*cosj2t)*sini2x
        q04 = pi*k2*wx*sinl2t*cosk2x/L
        q14 = l2*wt*sink2x*cosl2t/(r*r)
        q24 = (L*l2*wx*sink2x*cosl2t + pi*k2*wt*sinl2t*cosk2x)/(L*r)
        return p00*q04 + p01*q14 + p02*q24
    L = 1.
    m2 = 20
    #x2 = np.linspace(0, 2*L, 4*i2+1)
    x2 = np.linspace(0, 2*L, 100)
    wx = np.random.random(x2.shape[0])
    wt = np.random.random(x2.shape[0])
    for i2 in range(1, m2+1):
        fig = plt.figure(figsize=(12, 6))
        x1 = np.linspace(0, 2*L, 1000)
        #plt.plot(x1, f(x1, i2, wx, wt))
        plt.plot(x2, f(x2, i2, wx, wt), lw=1.)
    plt.show()

def test():
    from desicos.conecyl import ConeCyl
    cc = ConeCyl()
    cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
    cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
    cc.plyt = 0.125
    cc.r2 = 250.
    cc.H = 510.
    cc.alphadeg = 0.

    cc.m1 = 25
    cc.m2 = 30
    cc.n2 = 35

    cc.thetaT = None
    cc.Fc = 0.

    cc.add_SPL(1.)


    cases = [
             ('clpt_sanders', 'sanders_numerical'),
             ('clpt_donnell', 'donnell_numerical'),
            ]

    for case in cases:
        cc.linear_kinematics = case[0]
        cc.NL_kinematics = case[1]
        cc.Ft = 1.
        cc.lb()
        cc.plot(cc.eigvecs[:,0])
        cc.Ft = 100.
        cc.static()
        cc.plot(cc.cs[0], vec='v')
        cc.plot(cc.cs[0], vec='w')

    return cc

if __name__=='__main__':
    cc = test()
