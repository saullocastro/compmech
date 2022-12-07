from itertools import product

import numpy as np
import sympy
from sympy import Matrix

from compmech.symbolic.doperator import evaluateExpr

from constitutive import LC


def calc_matrices(cs_left, gs_left, cs_right, gs_right,
                  prefix='print_integrands', NL='Donnell',
                  analytical=False,
                  ind_subs={}):
    NL = NL.lower()
    if 'donnell' in NL:
        from kinematics_donnell import d, A, Gaux
    elif NL==None:
        pass
    else:
        raise ValueError(
                'Non-linear kinematics option "{0}" not defined!'.format(NL))
    print('Non-linear equations: {}'.format(NL))
    matrices = {}
    csa = cs_left
    gsa = gs_left
    csb = cs_right
    gsb = gs_right
    na = len(csa)
    nb = len(csb)
    #
    g = Matrix(np.hstack(gsb))
    c = Matrix(np.vstack(csb))
    e0 = evaluateExpr(d*g)*c
    B0_N = evaluateExpr(d*g.subs(ind_subs))
    G = evaluateExpr(Gaux*g)
    BL = A*G
    eL = BL*c/2
    #
    for mi, mj in product(range(na), range(nb)):
        sufix = '{0}{1}'.format(str(mi), str(mj))
        ca = csa[mi]
        ga = gsa[mi]
        cb = csb[mj]
        gb = gsb[mj]
        #
        not_assigned = []
        # creating a nan that will be useful to track if sth goes wrong
        wx = wt = sympy.nan
        for matrix in [ca, ga, cb, gb, d, A, Gaux]:
            for expr in matrix:
                for s in expr.free_symbols:
                    s.__class__.is_commutative = False
                    if str(s)=='x':
                        x = s
                    elif str(s)=='t':
                        t = s
                    elif str(s)=='r':
                        r = s
                    elif str(s)=='ux':
                        ux = s
                    elif str(s)=='ut':
                        ut = s
                    elif str(s)=='v':
                        v = s
                    elif str(s)=='vx':
                        vx = s
                    elif str(s)=='wx':
                        wx = s
                    elif str(s)=='wt':
                        wt = s
                    elif str(s)=='sina':
                        sina = s
                    elif str(s)=='cosa':
                        cosa = s
                    else:
                        not_assigned.append(s)
        print('Not assigned variables:')
        print('\t{}'.format(set(not_assigned)))
        #
        B0a = evaluateExpr(d*ga)
        B0b = evaluateExpr(d*gb)
        Ga = evaluateExpr(Gaux*ga)
        Gb = evaluateExpr(Gaux*gb)
        BLa = A*Ga
        BLb = A*Gb
        #
        #
        if analytical:
            def uvwNL(gs, cs, var):
                # [g] = [g0, g1, g2...]
                # [g][c] = [g0][c0] + [g1][c1] + [g2][c2]
                return sum(
                        [gi.subs(ind_subs).diff(var)*ci for
                            gi,ci in zip(gs,cs)])
            # NON-LINEAR substitutions a
            dummy, dummy, wxa = uvwNL(gsa, csa, x)
            dummy, dummy, wta = uvwNL(gsa, csa, t)
            subs_a = {
                    wx: wxa,
                    wt: wta,
                     }
            BLa = BLa.subs(subs_a)
            # NON-LINEAR substitutions b
            dummy, dummy, wxb = uvwNL(gsb, csb, x)
            dummy, dummy, wtb = uvwNL(gsb, csb, t)
            subs_b = {
                    wx: wxb,
                    wt: wtb,
                     }
            BLb = BLb.subs(subs_b)
            BL = BL.subs(subs_b)
        #
        # kG linear
        Nxx, Ntt, Nxt, Mxx, Mtt, Mxt = sympy.var(
                                       'Nxx, Ntt, Nxt, Mxx, Mtt, Mxt')


        if 'donnell' in NL:
            N = Matrix([[Nxx, Nxt],
                        [Nxt, Ntt]])
        if 'donnellmb' in NL:
            N = Matrix(
                    [[Ntt*cosa**2, -1/(2*r)*Mtt*sina*cosa, 0, -1/r*Mxt*cosa, -1/(2*r)*Mtt*cosa],
                     [-1/(2*r)*Mtt*sina*cosa, Nxx, Nxt + Mxt*cosa/r, 0, 0],
                     [0, Nxt + Mxt*cosa/r, Ntt + Mtt*cosa/r, 0, 0],
                     [-1/r*Mxt*cosa, 0, 0, 0, 0],
                     [-1/(2*r)*Mtt*cosa, 0, 0, 0, 0]])
        if 'xsanders_rev_00' in NL:
            N = Matrix(
                 [[0, 0, -Nxt*sina, 0, 0, 0],
                  [0, 0, -Ntt*sina, 0, 0, 0],
                  [-Nxt*sina, -Ntt*sina, Ntt, 0, -Nxt*cosa, -Ntt*cosa],
                  [0, 0, 0, Nxx, 0, 0 ],
                  [0, 0, -Nxt*cosa, 0, Nxx, Nxt],
                  [0, 0, -Ntt*cosa, 0, Nxt, Ntt]])
        if NL!=None:
            # kG NL
            N_vec = LC*(B0_N + BL/2)*c
            Nxx = N_vec[0]
            Ntt = N_vec[1]
            Nxt = N_vec[2]
            Mxx = N_vec[3]
            Mtt = N_vec[4]
            Mxt = N_vec[5]
            N_NL = Matrix([[Nxx, Nxt],
                           [Nxt, Ntt]])
            if NL=='donnellmb_rev_00':
                N_NL = Matrix(
                        [[Ntt*cosa**2, -1/(2*r)*Mtt*sina*cosa, 0, -1/r*Mxt*cosa, -1/(2*r)*Mtt*cosa],
                         [-1/(2*r)*Mtt*sina*cosa, Nxx, Nxt + Mxt*cosa/r, 0, 0],
                         [0, Nxt + Mxt*cosa/r, Ntt + Mtt*cosa/r, 0, 0],
                         [-1/r*Mxt*cosa, 0, 0, 0, 0],
                         [-1/(2*r)*Mtt*cosa, 0, 0, 0, 0]])

            elif NL=='sanders_rev_00':
                N_NL = Matrix(
                     [[0, 0, -Nxt*sina, 0, 0, 0],
                      [0, 0, -Ntt*sina, 0, 0, 0],
                      [-Nxt*sina, -Ntt*sina, Ntt, 0, -Nxt*cosa, -Ntt*cosa],
                      [0, 0, 0, Nxx, 0, 0 ],
                      [0, 0, -Nxt*cosa, 0, Nxx, Nxt],
                      [0, 0, -Ntt*cosa, 0, Nxt, Ntt]])

        #
        kLL = r*B0a.T*LC*B0b
        kLNL = r*B0a.T*LC*BLb
        kNLL = r*BLa.T*LC*B0b
        kNLNL = r*BLa.T*LC*BLb
        #
        # kG
        kG = r*Ga.T*N*Gb
        kG_NL = r*Ga.T*N_NL*Gb
        #
        ks = [['k0_'+sufix, kLL],
              ['k0L_'+sufix, kLNL],
              ['kLL_'+sufix, kNLNL],
              ['kG0_'+sufix, kG],
              ['kG_'+sufix, kG_NL],
              ['e0_', e0],
              ['eL_', eL],
              ]
        #
        with open('{prefix}_k{sufix}.txt'.format(prefix=prefix,
                                                 sufix=sufix), 'w') as outf:
            def myprint(sth):
                outf.write(str(sth).strip() + '\n')
            for kname, kab in ks:
                myprint('#')
                num = len([kabi for kabi in kab if kabi])
                myprint('# {0} with {1} non-null temrs'.format(kname, num))
                myprint('#')
                myprint(kab)
                myprint('#')
                for (i, j), v in np.ndenumerate(kab):
                    if v:
                        myprint(kname+'[{0},{1}] = {2}'.format(i, j, str(v)))
        matrices['k{0}_{1}'.format(mi, mj)] = ks

    return matrices

