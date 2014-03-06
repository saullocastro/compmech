import cProfile, pstats
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from desicos.conecyl import ConeCyl

def test_lb(cc):
    print('--------------------------------')
    print('LINEAR BUCKLING ANALYSIS FOR {}'.format(cc.name.upper()))
    print('--------------------------------')
    cc.pdT = True
    cc.thetaTdeg = 0.
    cc.pdLA = True
    cc.betadeg = 0.
    cc.pdC = False
    cc.Fc = 1.

    cc.linear_kinematics = 'clpt_donnell'
    cc.lb()
    cc.plot(cc.eigvecs[:,0], filename='test_lb_{}_donnell.png'.format(cc.name))

    cc.linear_kinematics = 'clpt_sanders'
    cc.lb()
    cc.plot(cc.eigvecs[:,0], filename='test_lb_{}_sanders.png'.format(cc.name))

def test_linear_static(cc):
    print('------------------------------')
    print('LINEAR STATIC ANALYSIS FOR {}'.format(cc.name.upper()))
    print('------------------------------')
    cc.pdC = False
    cc.Fc = 100000

    cc.pdT = False
    cc.T = 10000

    cc.tLAdeg = 90.
    cc.betadeg = 10.

    cc.add_SPL(50.)

    cc.linear_kinematics = 'clpt_donnell'
    cc.static(NLgeom=False)
    cc.plot(cc.cs[-1], vec='u',
            filename='test_static_{}_donnell_u.png'.format(cc.name))
    cc.plot(cc.cs[-1], vec='v',
            filename='test_static_{}_donnell_v.png'.format(cc.name))
    cc.plot(cc.cs[-1], vec='w',
            filename='test_static_{}_donnell_w.png'.format(cc.name))

    cc.linear_kinematics = 'clpt_sanders'
    cc.static(NLgeom=False)
    cc.plot(cc.cs[-1], vec='u',
            filename='test_static_{}_sanders_w.png'.format(cc.name))
    cc.plot(cc.cs[-1], vec='v',
            filename='test_static_{}_sanders_w.png'.format(cc.name))
    cc.plot(cc.cs[-1], vec='w',
            filename='test_static_{}_sanders_w.png'.format(cc.name))

def test_non_linear_static(cc):
    print('----------------------------------')
    print('NON-LINEAR STATIC ANALYSIS FOR {}'.format(cc.name.upper()))
    print('----------------------------------')
    cc.ni_num_cores = multiprocessing.cpu_count() - 1
    cc.pdT = True
    cc.thetaTdeg = 0.
    cc.pdLA = True
    cc.betadeg = 0.
    cc.pdC = False
    cc.Fc = 100000

    cc.add_SPL(50.)

    cc.linear_kinematics = 'clpt_donnell'
    cc.NL_kinematics = 'donnell_numerical'
    cc.static(NLgeom=True)
    cc.plot(cc.cs[-1], vec='u',
            filename='test_NL_static_{}_donnell_u.png'.format(cc.name))
    cc.plot(cc.cs[-1], vec='v',
            filename='test_NL_static_{}_donnell_v.png'.format(cc.name))
    cc.plot(cc.cs[-1], vec='w',
            filename='test_NL_static_{}_donnell_w.png'.format(cc.name))

    cc.linear_kinematics = 'clpt_sanders'
    cc.NL_kinematics = 'sanders_numerical'
    cc.static(NLgeom=True)
    cc.plot(cc.cs[-1], vec='u',
            filename='test_NL_static_{}_sanders_w.png'.format(cc.name))
    cc.plot(cc.cs[-1], vec='v',
            filename='test_NL_static_{}_sanders_w.png'.format(cc.name))
    cc.plot(cc.cs[-1], vec='w',
            filename='test_NL_static_{}_sanders_w.png'.format(cc.name))

def main():
    print('==============')
    print('STARTING TESTS')
    print('==============')
    print('')
    cc = ConeCyl()
    cc.m1 = 15
    cc.m2 = 16
    cc.n2 = 17
    cc.name = 'C02'
    cc.laminaprop = (142.5e3, 8.7e3, 0.28, 5.1e3, 5.1e3, 5.1e3)
    cc.stack = [30,-30,-60,60,0,60,-60,-30,30]
    cc.plyt = 0.125
    cc.alphadeg = 45.
    cc.r2 = 200.
    cc.H = 200
    test_non_linear_static(cc)
    test_lb(cc)
    test_linear_static(cc)

    cc = ConeCyl()
    cc.m1 = 15
    cc.m2 = 16
    cc.n2 = 17
    cc.name = 'Z33'
    cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
    cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
    cc.plyt = 0.125
    cc.r2 = 250.
    cc.H = 510.
    test_non_linear_static(cc)
    test_lb(cc)
    test_linear_static(cc)


if __name__=='__main__':
    main()

if False:
    cc = ConeCyl()
    cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
    cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
    cc.plyt = 0.125
    #cc.alphadeg = 35.
    cc.r2 = 250.
    cc.H = 510.
    cc.linear_kinematics = 'fsdt_general_donnell'
    cc.NL_kinematics = 'fsdt_general_donnell'

    # boundary conditions
    cc.kuBot = 0.
    cc.kvBot = 0.
    cc.kwBot = 0.
    cc.kphixBot = 0.
    cc.kphitBot = 0.

    cc.kuTop = 1.e6
    cc.kvTop = 1.e6
    cc.kwTop = 1.e6
    cc.kphixTop = 1.e6
    cc.kphitTop = 1.e6

    # shape functions
    cc.m1 = 10
    cc.m2 = 11
    cc.n2 = 12

    # applying perturbation load
    cc.Fc = 10000
    cc.pdC = True
    cc.uTM = 0.
    cc.add_SPL(10.)

    cc.static()

    cc.plot(cc.cs[-1], filename='tester.png')

