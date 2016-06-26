import os
import time
import __main__

import numpy as np
import matplotlib.pyplot as plt

def go_to_out_dir(cc):
    out_dir = os.path.join('.', cc.name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    os.chdir(out_dir)

def mrs_pl(cc, ns, compare_theories=False):
    mrs(cc, ns, 'pl', compare_theories)

def mrs_lb(cc, ns, compare_theories=False):
    mrs(cc, ns, 'lb', compare_theories)

def mrs(cc, ns, sol, compare_theories):
    cc.num_eigvalues = 5
    if not cc.Fc==None:
        cc.Fc = 0.
    if not cc.PLvalues:
        cc.add_SPL(1.)
    go_to_out_dir(cc)
    figsize = (3.5, 2.)
    bkp_lk = cc.linear_kinematics
    bkp_bc = cc.bc
    if compare_theories:
        cc.bc = 'ss'
        linear_kinematics = ['clpt_donnell', 'clpt_sanders', 'fsdt']
    else:
        linear_kinematics = [cc.linear_kinematics]
    figsize = (3.5, 2.5)
    figmrs = plt.figure(figsize=figsize)
    axmrs = plt.subplot(111)
    figcomp = plt.figure(figsize=figsize)
    axcomp = plt.subplot(111)
    linestyles = ['-', '--', '-.']
    for i, lk in enumerate(linear_kinematics):
        cc.linear_kinematics = lk
        mins = []
        comp_cost = []
        for n in ns:
            m = n
            cc.m2 = m
            cc.n2 = n
            time1 = time.clock()
            if sol == 'pl':
                #TODO required workaround until fsdt, pd=True is implemented
                cc.pd = False
                c = cc.static()
                cc.uvw(c)
                mins.append(1000*cc.w.min())
            elif sol == 'lb':
                #TODO required workaround until fsdt, pd=True is implemented
                cc.pd = True
                cc.lb()
                cc.uvw(cc.eigvecs[:, cc.eigvals.argmin()])
                mins.append(cc.eigvals.min()/1000)
            time2 = time.clock()
            comp_cost.append(time2-time1)
            #
            levels = np.linspace(cc.w.min(), cc.w.max(), 200)
            # NOTE X must be plotted inverted because it starts at the top
            # for the semi-analytical model, but it starts at the bottom
            # for the finite-element model
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            X = cc.X
            T = cc.T
            ax.contourf(cc.r2*T, X[:,::-1], cc.w, levels=levels)
            ax.grid(False)
            ax.set_aspect('equal')
            #ax.set_title('$m=n={}$'.format(n))
            fig.tight_layout()
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.set_frame_on(False)
            fig.savefig(
                'plot_{name}_{sol}_mrs_{lk}_{bc}_contourf_n2_{n2:03d}.png'.\
                    format(sol=sol, name=cc.name, lk=cc.linear_kinematics,
                        bc=cc.bc, n2=cc.n2),
                    transparent=True, bbox_inches='tight', pad_inches=0.05)
            plt.close()

        if compare_theories:
            axmrs.plot(ns, mins, 'k', label=lk.upper(), ls=linestyles[i])
            axcomp.plot(ns, comp_cost, 'k', label=lk.upper(),
                        ls=linestyles[i])
        else:
            print('MRS curve x', ns)
            print('MRS curve y', mins)
            axmrs.plot(ns, mins)
            axcomp.plot(ns, comp_cost)
    #axmrs.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    if compare_theories:
        axmrs.legend()
    axmrs.xaxis.set_ticks_position('bottom')
    axmrs.yaxis.set_ticks_position('left')
    axmrs.set_xlabel('$n_2$')
    if sol=='pl':
        axmrs.set_ylabel(r'$w_{PL} \times 10^{-3}$, $mm$')
    elif sol=='lb':
        axmrs.set_ylabel(r'$P_{{CR}}$, $kN$')
    figmrs.tight_layout()
    ylim = axmrs.get_ylim()
    axmrs.set_ylim(ylim[0], round(ylim[1]*1.1))
    figmrs.savefig('plot_{name}_{sol}_mrs_{lk}_{bc}.png'.format(
        sol=sol, name=cc.name,
        lk=cc.linear_kinematics, bc=cc.bc),
        bbox_inches='tight', pad_inches=0.05)
    plt.close()

    #axcomp.yaxis.set_major_formatter(FormatStrFormatter('%0.f'))
    if compare_theories:
        axcomp.legend(loc='upper left')
    axcomp.xaxis.set_ticks_position('bottom')
    axcomp.yaxis.set_ticks_position('left')
    axcomp.set_xlabel('$n_2$')
    axcomp.set_ylabel(r'Computational cost, $seconds$')
    figcomp.tight_layout()
    figcomp.savefig('plot_{name}_{sol}_mrs_{lk}_{bc}_comp_cost.png'.\
            format( sol=sol, name=cc.name, lk=cc.linear_kinematics, bc=cc.bc),
        bbox_inches='tight', pad_inches=0.05)
    plt.close()
    os.chdir('..')
    cc.linear_kinematics = bkp_lk
    cc.bc = bkp_bc

def mrs_ni(cc, nts):
    from clpt_NL_donnell_numerical import calc_k0L, calc_kLL, calc_kGNL
    print('nx nt k0L.sum()')
    for nt in nts:
        cc.nx = None
        cc.nt = nt
        cc.rebuild()
        k0L = calc_k0L(cc.cs[0], cc.alpharad, cc.r2, cc.L, cc.F,
                cc.m1, cc.m2, cc.n2, cc.pdoff, c00=cc.uTM,
                nx=cc.nx, nt=cc.nt, num_cores=cc.ni_num_cores,
                method=cc.ni_method)
        print('{0} {1} {2}'.format(cc.nx, cc.nt, k0L.sum()))
    print('nx nt kGNL.sum()')
    for nt in nts:
        cc.nx = None
        cc.nt = nt
        cc.rebuild()
        kGNL = calc_kGNL(cc.cs[0], cc.alpharad, cc.r2, cc.L, cc.F,
                cc.m1, cc.m2, cc.n2, cc.pdoff, c00=cc.uTM,
                nx=cc.nx, nt=cc.nt, num_cores=cc.ni_num_cores,
                method=cc.ni_method)
        print('{0} {1} {2}'.format(cc.nx, cc.nt, kGNL.sum()))
    print('nx nt kLL.sum()')
    for nt in nts:
        cc.nx = None
        cc.nt = nt
        cc.rebuild()
        kLL = calc_kLL(cc.cs[0], cc.alpharad, cc.r2, cc.L, cc.F,
                cc.m1, cc.m2, cc.n2, cc.pdoff, c00=cc.uTM,
                nx=cc.nx, nt=cc.nt, num_cores=cc.ni_num_cores,
                method=cc.ni_method)
        print('{0} {1} {2}'.format(cc.nx, cc.nt, kLL.sum()))

