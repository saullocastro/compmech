import cProfile

import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt

from compmech.conecyl import ConeCyl
from compmech.conecyl.modelDB import db

def regions_conditions(cc, name='k0L'):
    nlmodule = db[cc.model]['non-linear']
    if nlmodule:
        calc_k0L = nlmodule.calc_k0L
        calc_kG = nlmodule.calc_kG
        calc_kLL = nlmodule.calc_kLL
    else:
        raise ValueError(
        'Non-Linear analysis not implemented for model {}'.format(
            self.model))
    which = {'k0L':calc_k0L, 'kG':calc_kG, 'kLL':calc_kLL}
    cc.static()
    m = which[name](cc.cs[0], cc.alpharad, cc.r2, cc.L, cc.tLArad, cc.F,
            cc.m1, cc.m2, cc.n2, nx=cc.nx, nt=cc.nt, num_cores=cc.ni_num_cores,
            method=cc.ni_method, c0=cc.c0, m0=cc.m0, n0=cc.n0)
    m = m.toarray()
    i0 = db[cc.model]['i0']
    j0 = db[cc.model]['j0']
    num0 = db[cc.model]['num0']
    num1 = db[cc.model]['num1']
    num2 = db[cc.model]['num2']
    m1 = cc.m1
    m2 = cc.m2
    n2 = cc.n2
    print('REPORT k 00')
    print('cond_1', m[:num0, :num0])
    print('REPORT k 01')
    terms1 = 0
    cond_1 = 0
    for k1 in range(i0, m1+i0):
        c = num0 + (k1-i0)*num1
        terms1 += 1
        cond_1 += np.abs(m[:num0, c:c+num1]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)

    print('REPORT k 02')
    terms1 = 0
    cond_1 = 0
    for k2 in range(i0, m2+i0):
        for l2 in range(j0, cc.n2+j0):
            c = num0 + m1*num1 + (k2-i0)*num2 + (l2-j0)*m2*num2
            terms1 += 1
            cond_1 += np.abs(m[:num0, c:c+num2]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)

    print('REPORT k 10')
    terms1 = 0
    cond_1 = 0
    for i1 in range(i0, m1+i0):
        r = num0 + (i1-i0)*num1
        terms1 += 1
        cond_1 += np.abs(m[r:r+num1, :num0]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)

    print('REPORT k 11')
    terms1 = 0
    terms2 = 0
    cond_1 = 0
    cond_2 = 0
    for i1 in range(i0, m1+i0):
        r = num0 + (i1-i0)*num1
        for k1 in range(i0, m1+i0):
            c = num0 + (k1-i0)*num1
            if i1==k1:
                terms1 += 1
                cond_1 += np.abs(m[r:r+num1, c:c+num1]).sum()
            elif k1!=i1:
                terms2 += 1
                cond_2 += np.abs(m[r:r+num1, c:c+num1]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)
    print('cond_2', cond_2, terms2, cond_2/terms2)

    print('REPORT k 12')
    termsall = 0
    condsall = 0
    for i1 in range(i0, m1+i0):
        r = num0 + (i1-i0)*num1
        for k2 in range(i0, m2+i0):
            for l2 in range(j0, cc.n2+j0):
                c = num0 + m1*num1 + (k2-i0)*num2 + (l2-j0)*m2*num2
                termsall += 1
                condsall += np.abs(m[r:r+num1, c:c+num2]).sum()
    print('conds_all', condsall, termsall, condsall/termsall)

    print('REPORT k 20')
    terms1 = 0
    cond_1 = 0
    for i2 in range(i0, m2+i0):
        for j2 in range(j0, cc.n2+j0):
            r = num0 + m1*num1 + (i2-i0)*num2 + (j2-j0)*m2*num2
            terms1 += 1
            cond_1 += np.abs(m[r:r+num2, :num0]).sum()
    print('cond_1', cond_1, terms1, cond_1/terms1)

    print('REPORT k 21')
    termsall = 0
    condsall = 0
    for i2 in range(i0, m2+i0):
        for j2 in range(j0, cc.n2+j0):
            r = num0 + m1*num1 + (i2-i0)*num2 + (j2-j0)*m2*num2
            for k1 in range(i0, m1+i0):
                c = num0 + (k1-i0)*num1
                termsall += 1
                condsall += np.abs(m[r:r+num2, c:c+num1]).sum()
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
    for i2 in range(i0, m2+i0):
        for j2 in range(j0, cc.n2+j0):
            r = num0 + m1*num1 + (i2-i0)*num2 + (j2-j0)*m2*num2
            for k2 in range(i0, m2+i0):
                for l2 in range(j0, cc.n2+j0):
                    c = num0 + m1*num1 + (k2-i0)*num2 + (l2-j0)*m2*num2
                    if k2==i2 and l2==j2:
                        terms1 += 1
                        cond_1 += np.abs(m[r:r+num2, c:c+num2]).sum()
                    elif k2!=i2 and l2==j2:
                        terms2 += 1
                        cond_2 += np.abs(m[r:r+num2, c:c+num2]).sum()
                    elif k2!=i2 and l2!=j2:
                        terms3 += 1
                        cond_3 += np.abs(m[r:r+num2, c:c+num2]).sum()
                    elif k2==i2 and l2!=j2:
                        terms4 += 1
                        cond_4 += np.abs(m[r:r+num2, c:c+num2]).sum()
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
            print(cProfile.runctx('curves = cc.SPLA(PLs, NLgeom=True)',
                              globals(), locals(), 'tester.prof'))


def test0():
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

def test1():
    cc = ConeCyl()
    cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
    cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
    cc.plyt = 0.125
    cc.r2 = 250.
    cc.H = 510.
    cc.alphadeg = 0.

    cc.m1 = 120
    cc.m2 = 15
    cc.n2 = 15

    cc.thetaT = None
    cc.Fc = 1.
    cc.pdC = False

    cc.add_SPL(10.)

    cc.model = 'clpt_donnell_bc1'

    cc.lb()
    cc.Fc = cc.eigvals[0]
    cc.plot(cc.eigvecs[:,0])
    cc.Fc = 100.
    regions_conditions(cc, 'k0L')

if __name__=='__main__':
    cc = test1()
