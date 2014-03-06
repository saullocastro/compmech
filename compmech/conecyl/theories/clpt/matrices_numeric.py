from itertools import product

import numpy as np
import sympy
from sympy import Matrix

from mapy.sympytools.doperator import evaluateExpr

from constitutive import LC

def calc_matrices(cs_left, gs_left, cs_right, gs_right,
                  prefix='print_derivations', NL_kinematics='donnell_rev_03'):
    sanders = False
    NL_kinematics = NL_kinematics.lower()
    if NL_kinematics==('donnell_rev_03' or 'donnell_rev_04' or
                       'donnell_rev_05'):
        from kinematics_donnell import d, A_w, Gaux_w
    elif NL_kinematics=='donnellmb_rev_00':
        from kinematics_donnellmb_rev_00 import d, A_w, Gaux_w
    elif NL_kinematics=='sanders_rev_00':
        sanders = True
        from kinematics_sanders_rev_00 import d, A_v, A_w, Gaux_v, Gaux_w
    elif NL_kinematics=='full':
        from kinematics_full import d, dNL
    else:
        raise ValueError(
                'Non-linear kinematics option "{}" not defined!'.format(
                    NL_kinematics))
    print('Non-linear kinematics: {}'.format(NL_kinematics))
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
    G_w = evaluateExpr(Gaux_w*g)
    BL = A_w*G_w
    if sanders:
        G_v = evaluateExpr(Gaux_v*g)
        BL += A_v*G_v
    else:
        A_v = None
        G_v = None
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
        w = wx = w0x = wt = w0t = wxt = wtt = sympy.nan
        for matrix in [ca, ga, cb, gb, d, A_v, A_w, Gaux_v, Gaux_w]:
            if matrix==None:
                continue
            for expr in matrix:
                for s in expr.free_symbols:
                    s.__class__.is_commutative = False
                    if str(s)=='x':
                        x = s
                    elif str(s)=='t':
                        t = s
                    elif str(s)=='r':
                        r = s
                    elif str(s)=='v':
                        v = s
                    elif str(s)=='vx':
                        vx = s
                    elif str(s)=='vt':
                        vt = s
                    elif str(s)=='w':
                        w = s
                    elif str(s)=='wx':
                        wx = s
                    elif str(s)=='wt':
                        wt = s
                    elif str(s)=='wxt':
                        wxt = s
                    elif str(s)=='wtt':
                        wtt = s
                    elif str(s)=='w0x':
                        w0x = s
                    elif str(s)=='w0t':
                        w0t = s
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
        Ga_w = evaluateExpr(Gaux_w*ga)
        Gb_w = evaluateExpr(Gaux_w*gb)
        BLa = A_w*Ga_w
        BLb = A_w*Gb_w
        if sanders:
            Ga_v = evaluateExpr(Gaux_v*ga)
            Gb_v = evaluateExpr(Gaux_v*gb)
            BLa += A_v*Ga_v
            BLb += A_v*Gb_v
        #
        k00 = r*B0a.T*LC*B0b
        k0L = r*B0a.T*LC*BLb
        kNLL = r*BLa.T*LC*B0b
        kLL = r*BLa.T*LC*BLb
        #
        # kG
        Nxx, Ntt, Nxt = sympy.var('Nxx, Ntt, Nxt')
        N = Matrix([[Nxx, Nxt],
                    [Nxt, Ntt]])
        # kG
        if NL_kinematics=='donnellmb':
            Nxx, Ntt, Nxt, Mxx, Mtt, Mxt = sympy.var(
                                           'Nxx, Ntt, Nxt, Mxx, Mtt, Mxt')
            N = Matrix(
                    [[Ntt*cosa**2, -1/(2*r)*Mtt*sina*cosa, 0, -1/r*Mxt*cosa, -1/(2*r)*Mtt*cosa],
                     [-1/(2*r)*Mtt*sina*cosa, Nxx, Nxt + Mxt*cosa/r, 0, 0],
                     [0, Nxt + Mxt*cosa/r, Ntt + Mtt*cosa/r, 0, 0],
                     [-1/r*Mxt*cosa, 0, 0, 0, 0],
                     [-1/(2*r)*Mtt*cosa, 0, 0, 0, 0]])
        #
        kG = r*Ga_w.T*N*Gb_w
        if sanders:
            kG += r*Ga_v.T*N*Gb_v
        #
        ks = [['k00'+sufix, k00],
              ['k0L'+sufix, k0L],
              ['kLL'+sufix, kLL],
              ['kG'+sufix, kG],
              ['e0', e0],
              ['eL', eL],
              ['BL', BL],
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
        matrices['k{0}{1}'.format(mi, mj)] = ks

    with open('all_matrices.txt', 'w') as outf:
        def myprint(sth):
            outf.write(str(sth).strip() + '\n')
        all = []
        m = matrices
        for a,dummy in enumerate(ks):
            full = Matrix(
            np.vstack((np.hstack((m['k00'][a][1], m['k01'][a][1], m['k02'][a][1])),
                       np.hstack((m['k10'][a][1], m['k11'][a][1], m['k12'][a][1])),
                       np.hstack((m['k20'][a][1], m['k21'][a][1], m['k22'][a][1]))))
                        )
            kname = m['k00'][a][0]
            for (i, j), v in np.ndenumerate(full):
                if v:
                    myprint(kname+'[{0},{1}] = {2}'.format(i, j, str(v)))

    return matrices

