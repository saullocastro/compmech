import numpy as np
import sympy
from sympy import Matrix

from mapy.sympytools.doperator import evaluateExpr

from constitutive import LC


def calc_matrices(c, g,
                  prefix='print_derivations', NL_kinematics='donnell',
                  analytical=True):
    NL_kinematics = NL_kinematics.lower()
    if (NL_kinematics=='donnell'):
        from kinematics_donnell import d, A, Gaux
    elif NL_kinematics=='donnellmb_rev_00':
        from kinematics_donnellmb_rev_00 import d, A, Gaux
    elif NL_kinematics=='sanders_rev_00':
        from kinematics_sanders_rev_00 import d, A, Gaux
    else:
        print(NL_kinematics)
        raise ValueError(
                'Non-linear kinematics option "{}" not defined!'.format(
                    NL_kinematics))
    print('Non-linear kinematics: {}'.format(NL_kinematics))
    matrices = {}
    #
    B0 = evaluateExpr(d*g)
    e0 = B0*c
    G = evaluateExpr(Gaux*g)
    BL = A*G
    #
    sufix = 'ALL'
    #
    not_assigned = []
    # creating a nan that will be useful to track if sth goes wrong
    wx = wt = sympy.nan
    for matrix in [c, g, d, A, Gaux]:
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
    if analytical:
        # NON-LINEAR substitutions
        dummy, dummy, wxb = g.diff(x)*c
        dummy, dummy, wtb = g.diff(t)*c
        subs_b = {
                wx: wxb,
                wt: wtb,
                 }
        BL = BL.subs(subs_b)
    #
    eL = (BL/2)*c
    #
    # kG linear
    Nxx, Ntt, Nxt, Mxx, Mtt, Mxt = sympy.var(
                                   'Nxx, Ntt, Nxt, Mxx, Mtt, Mxt')
    N = Matrix([[Nxx, Nxt],
                [Nxt, Ntt]])

    if NL_kinematics=='donnellmb_rev_00':
        N = Matrix(
                [[Ntt*cosa**2, -1/(2*r)*Mtt*sina*cosa, 0, -1/r*Mxt*cosa, -1/(2*r)*Mtt*cosa],
                 [-1/(2*r)*Mtt*sina*cosa, Nxx, Nxt + Mxt*cosa/r, 0, 0],
                 [0, Nxt + Mxt*cosa/r, Ntt + Mtt*cosa/r, 0, 0],
                 [-1/r*Mxt*cosa, 0, 0, 0, 0],
                 [-1/(2*r)*Mtt*cosa, 0, 0, 0, 0]])

    elif NL_kinematics=='sanders_rev_00':
        N = Matrix(
             [[0, 0, -Nxt*sina, 0, 0, 0],
              [0, 0, -Ntt*sina, 0, 0, 0],
              [-Nxt*sina, -Ntt*sina, Ntt, 0, -Nxt*cosa, -Ntt*cosa],
              [0, 0, 0, Nxx, 0, 0 ],
              [0, 0, -Nxt*cosa, 0, Nxx, Nxt],
              [0, 0, -Ntt*cosa, 0, Nxt, Ntt]])

    # kG NL
    N_vec = LC*(e0 + eL)
    Nxx = N_vec[0]
    Ntt = N_vec[1]
    Nxt = N_vec[2]
    Mxx = N_vec[3]
    Mtt = N_vec[4]
    Mxt = N_vec[5]
    N_NL = Matrix([[Nxx, Nxt],
                   [Nxt, Ntt]])
    if NL_kinematics=='donnellmb_rev_00':
        N_NL = Matrix(
                [[Ntt*cosa**2, -1/(2*r)*Mtt*sina*cosa, 0, -1/r*Mxt*cosa, -1/(2*r)*Mtt*cosa],
                 [-1/(2*r)*Mtt*sina*cosa, Nxx, Nxt + Mxt*cosa/r, 0, 0],
                 [0, Nxt + Mxt*cosa/r, Ntt + Mtt*cosa/r, 0, 0],
                 [-1/r*Mxt*cosa, 0, 0, 0, 0],
                 [-1/(2*r)*Mtt*cosa, 0, 0, 0, 0]])

    elif NL_kinematics=='sanders_rev_00':
        N_NL = Matrix(
             [[0, 0, -Nxt*sina, 0, 0, 0],
              [0, 0, -Ntt*sina, 0, 0, 0],
              [-Nxt*sina, -Ntt*sina, Ntt, 0, -Nxt*cosa, -Ntt*cosa],
              [0, 0, 0, Nxx, 0, 0 ],
              [0, 0, -Nxt*cosa, 0, Nxx, Nxt],
              [0, 0, -Ntt*cosa, 0, Nxt, Ntt]])
    #
    kLL = r*B0.T*LC*B0
    kLNL = r*B0.T*LC*BL
    kNLL = r*BL.T*LC*B0
    kNLNL = r*BL.T*LC*BL
    #
    # kG
    kG = r*G.T*N*G
    kG_NL = r*G.T*N_NL*G
    #
    #
    ks = [['k00'+sufix, kLL],
          ['k0L'+sufix, kLNL],
          ['kLL'+sufix, kNLNL],
          ['kG'+sufix, kG],
          ['kG_NL'+sufix, kG_NL],
          ['e0', e0],
          ['eL', eL],
          ]
    #
    #
    with open('{prefix}_k{sufix}.txt'.format(prefix=prefix,
                                             sufix=sufix), 'w') as outf:
        def myprint(sth):
            outf.write(str(sth).strip() + '\n')
        for kname, kab in ks:
            myprint('#')
            myprint('# {0}'.format(kname))
            myprint('#')
            myprint(kab)
            myprint('#')
            for (i, j), v in np.ndenumerate(kab):
                if v:
                    myprint(kname+'[{0},{1}] = {2}'.format(i, j, str(v)))
    #
    #
    matrices['kALL'] = ks

    return matrices

