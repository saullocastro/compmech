import sys
import cProfile, pstats
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from compmech.conecyl import ConeCyl

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

def main(which=['standard']):
    which = [i.lower() for i in which]

    if 'standard' in which:
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

        return cc

    cc = ConeCyl()

    # z33
    cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
    cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
    cc.r2 = 250.
    cc.H = 500.
    cc.plyt = 0.125

    #cc.alphadeg = 0.1
    cc.linear_kinematics = 'fsdt_donnell2'
    #cc.linear_kinematics = 'clpt_donnell'
    cc.NL_kinematics = 'donnell_numerical'

    # shape functions
    cc.m1 = 15
    cc.m2 = 16
    cc.n2 = 17

    cc.kuBot = 0.
    cc.kuTop = 0.

    cc.kphixBot = 1.e10
    cc.kphixTop = 1.e10

    if 'lb' in which:
        # lb
        cc.Fc = 1.
        cc.pdC = False
        cc.lb()
        name = r'C:\Temp\test_' + cc.linear_kinematics.split('_')[0]
        cc.plot(cc.eigvecs[:,0], vec='w', filename=(name + '_1.png'))
        cc.plot(cc.eigvecs[:,1], vec='w', filename=(name + '_2.png'))
        cc.plot(cc.eigvecs[:,2], vec='w', filename=(name + '_3.png'))
        cc.plot(cc.eigvecs[:,3], vec='w', filename=(name + '_4.png'))
        cc.plot(cc.eigvecs[:,4], vec='w', filename=(name + '_5.png'))

    if 'lb_test' in which:
        # lb
        cc.Fc = 1.
        cc.pdC = False
        cc.lb_test()
        cc.plot(cc.eigvecs[:,0], vec='w', filename=r'C:\Temp\test_res.png')
        cc.plot(cc.eigvecs[:,1], vec='w', filename=r'C:\Temp\test_res_2.png')
        cc.plot(cc.eigvecs[:,2], vec='w', filename=r'C:\Temp\test_res_3.png')
        cc.plot(cc.eigvecs[:,3], vec='w', filename=r'C:\Temp\test_res_4.png')
        cc.plot(cc.eigvecs[:,4], vec='w', filename=r'C:\Temp\test_res_5.png')

    if 'static' in which:
        # linear static
        cc.Fc = 100.
        cc.uTM = 0.6
        cc.pdC = False
        cc.add_SPL(10.)

        cc.static(NLgeom=False)

        cc.plot(cc.cs[-1], vec='w', filename=r'C:\Temp\test_res.png',
                colorbar=True)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,6))

        ts = np.linspace(-np.pi, np.pi, 1000)

        xs = np.zeros_like(ts)
        cc.uvw(cc.cs[-1], x=xs, t=ts)
        axes[0].plot(ts, cc.u)
        axes[0].invert_yaxis()

        xs = np.zeros_like(ts) + cc.L
        cc.uvw(cc.cs[-1], x=xs, t=ts)
        axes[1].plot(ts, cc.u)
        axes[1].invert_yaxis()

        plt.gcf().savefig(r'C:\Temp\test_res2.png', bbox_inches='tight')

    if 'NL' in which:
        # NL static
        cc.Fc = 10000
        cc.pdC = False
        cc.add_SPL(10.)

        cc.ni_num_cores = 7
        cc.static(NLgeom=True)
        cc.initialInc = 0.5

        cc.plot(cc.cs[-1], vec='w', filename=r'C:\Temp\test_res.png',
                colorbar=True)

    return cc


if __name__=='__main__':
    which = sys.argv[1:]
    main(which)
