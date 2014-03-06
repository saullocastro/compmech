from __future__ import division

import numpy as np
from numpy import dot
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def NR_Broyden(cc):
    increment_PL = cc.increment_PL
    cen = cc.compute_every_n
    #
    inc = cc.initialInc
    total = inc
    #
    k0 = cc.k0
    kT = k0
    c = np.zeros((cc.pdoff + 6*cc.m2*cc.n2 + 3*cc.m1), float)
    delta = np.zeros_like(c)
    delta += 0.001
    c = c + delta
    while True:
        iteration = 0
        converged = False
        Rmax_old = 1e6
        while True:
            if increment_PL:
                f_ext = cc.calc_f_ext(increment_PL=total)
            else:
                f_ext = cc.calc_f_ext(inc=total)
            if True:#iteration==0:
                R = f_ext - k0*c
            #else:
                #cc.calc_NL_matrices(c)
                #R = f_ext - (cc.k0 + cc.k0L/2 + cc.kL0 + cc.kLL/2)*c
            Rmax = np.abs(R).max()

            if cc.debug:
                cc.uvw(c, x=cc.L/2., t=0.)
                w1 = cc.w[0]
                print '\t\tDEBUG w1 = {0} mm'.format(w1)
                print '\t\tDEBUG Rmax = {0}'.format(np.abs(R).max())

            if abs(Rmax-Rmax_old) <= cc.absTOL:
                converged = True
                print('Converged!')
                break
            else:
                Rmax_old = Rmax
            iteration += 1
            check_divergence = True
            update_kT = True
            if cc.modified_NR:
                test = iteration % cen
                if iteration<=1 or test==0:
                    update_kT = True
                else:
                    update_kT = False
            if update_kT:
                print('Updating kT...')
                kT = kT - csc_matrix(np.outer(R, delta)/np.dot(delta, delta))
                delta = spsolve(kT, f_ext)
            c += delta
        if converged:
            cc.cs.append(c)
            cc.increments.append(inc)
            total += inc

    #
    step_num = 1

    print('Finished Non-Linear Static Analysis')
    print('\tat time {0}'.format(total))

def NR(cc, break_when_diverge=True):
    increment_PL = cc.increment_PL
    cen = cc.compute_every_n

    inc = cc.initialInc
    total = inc
    once_at_total = False

    k0 = cc.k0uu
    kT = cc.k0uu
    if increment_PL:
        f_ext = cc.calc_f_ext(increment_PL=inc)
    else:
        f_ext = cc.calc_f_ext(inc=inc)
    c = spsolve(kT, f_ext)

    kSuk = None

    step_num = 1
    while True:
        print('\tStarted Load Step {} - '.format(step_num)
             + 'Attempting time = {0}'.format(total))

        modified_NR = cc.modified_NR


        if increment_PL:
            f_ext = cc.calc_f_ext(increment_PL=total, kuk=kSuk)
        else:
            f_ext = cc.calc_f_ext(inc=total, kuk=kSuk)

        # TODO maybe for pdC, pdT the f_ext must be calculated with the
        #      last kT available...

        absERR = 1.e6
        relERR = 1.e6
        min_Rmax = 1.e6
        last_min_Rmax = 1.e6
        iteration = 0
        converged = False
        compute_NL_matrices = True
        while True:

            #if increment_PL:
                #f_ext = cc.calc_f_ext(increment_PL=total, kuk=kSuk)
            #else:
                #f_ext = cc.calc_f_ext(inc=total, kuk=kSuk)

            iteration += 1
            print('\t    Iteration: {}'.format(iteration))
            check_divergence = True

            #TODO is this really an advantage?
            if False:
                if not modified_NR and iteration >= 20:
                    min_Rmax = 1.e6
                    modified_NR = True

            if modified_NR:
                test = iteration % cen
                if iteration<=1 or test==0:
                    compute_NL_matrices = True
                else:
                    compute_NL_matrices = False

            if compute_NL_matrices:
                if modified_NR:
                    if min_Rmax > last_min_Rmax:
                        print('\t\tWARNING - Diverged!')
                        if break_when_diverge:
                            break
                    check_divergence = False
                    last_min_Rmax = min_Rmax
                    min_Rmax = 1.e6

                cc.calc_NL_matrices(c, inc=total)

                # secant stiffness matrix
                kSuu = cc.kSuu
                kSuk = cc.kSuk

                # tangent stiffness matrix
                kTuu = cc.kTuu

            f_int = kSuu*c

            for i, dof in enumerate(cc.excluded_dofs):
                kSuk_col = cc.kSuk[:, dof].ravel()
                f_int += total*cc.excluded_dofs_ck[i]*kSuk_col

            R = f_ext - f_int

            # convergence criteria:
            # - maximum residual force Rmax
            Rmax = np.abs(R).max()
            print '\t\t\tRmax = {0}'.format(Rmax)

            if cc.debug:
                cc.uvw(c, x=cc.L/2., t=0.)
                w1 = cc.w[0]
                print '\t\tDEBUG w1 = {0} mm'.format(w1)

            if iteration >= 2 and Rmax < cc.absTOL:
                converged = True
                break

            if (check_divergence and Rmax>min_Rmax and iteration>2):
                print('\t\tWARNING - Diverged!')
                if break_when_diverge:
                    break
            else:
                min_Rmax = min(min_Rmax, Rmax)

            if iteration > cc.maxNumIter:
                print('\t\tWARNING - Maximum number of iterations achieved!')
                break

            print('\t\tSolving... '),
            delta_c = spsolve(kTuu, R)
            c += delta_c
            print('finished!')


        if converged:
            finished = False
            print('\tFinished Load Step {} at'.format(step_num)
                  + ' time = {0}'.format(total))
            cc.increments.append(total)
            if abs(total - 1) < 1e-3:
                finished = True
            else:
                if once_at_total:
                    inc_new = min(1.1*inc, cc.maxInc, (1.-total)/2)
                else:
                    inc_new = min(1.1*inc, cc.maxInc, 1.-total)
                print('\tChanging time increment from {0} to {1}'.format(
                      inc, inc_new))
                inc = inc_new
                total += inc
                total = min(1, total)
                step_num += 1
            cc.cs.append(c.copy()) #NOTE copy required
            if finished:
                break
        else:
            print('\tBisecting time increment from {0} to {1}'.format(
                  inc, inc/2))
            if abs(total -1) <1e-3:
                once_at_total = True
            total -= inc
            inc /= 2
            if inc < cc.minInc:
                print('Minimum step size achieved!')
                break
            total += inc
        #
        if len(cc.cs)>0:
            c = cc.cs[-1].copy() #NOTE copy required
        else:
            # means that a bisection must be done in initialInc
            if increment_PL:
                f_ext = cc.calc_f_ext(increment_PL=inc)
            else:
                f_ext = cc.calc_f_ext(inc=inc)
            c = spsolve(k0, f_ext)
            #cc.cs = [c.copy()]
            #cc.increments = [inc]

    print('Finished Non-Linear Static Analysis')
    print('\tat time {0}'.format(total))

def arc_length(cc):
    lbd = 0.1
    length = 0.5
    beta = 0.2
    cen = cc.compute_every_n
    f_ext = cc.calc_f_ext(inc=1.)
    k0 = cc.k0
    c = spsolve(k0, f_ext)*0
    count = 0
    while count < 20:
        min_Rmax = 1.e6
        converged = False
        iteration = 0
        compute_NL_matrices = True
        dc = np.zeros_like(c)
        while True:
            iteration += 1
            check_divergence = True
            test = iteration % cen
            if cc.modified_NR:
                if iteration<=1 or test==0:
                    compute_NL_matrices = True
                else:
                    compute_NL_matrices = False
            #
            if compute_NL_matrices:
                if cc.modified_NR:
                    min_Rmax = 1.e6
                cc.calc_NL_matrices(c)
                k0L = cc.k0L
                kL0 = cc.kL0
                kGNL = cc.kGNL
                kLL = cc.kLL
                kpd0L = cc.kpd0L
                # secant stiffness matrix
                kS = k0 + k0L/2 + kL0 + kLL/2
                # tangent stiffness matrix
                kT = k0 + k0L + kL0 + kLL + kGNL
            #
            df_int = kS*dc
            #
            dc2 = spsolve(kT, f_ext)
            dc1 = spsolve(kT, (lbd*f_ext - df_int))
            a2 = beta + dot(dc2.T, dc2)
            a1 = 2*beta*lbd + 2*dot(c.T, dc2) + 2*dot(dc1.T, dc2)
            a0 = (beta*lbd**2 + dot(c.T, c) + 2*dot(c.T, dc1)
                  + dot(dc1.T, dc1) - beta*length**2)
            delta = a1**2 - 4*a2*a0
            dlbd1 = (-a1 + np.sqrt(delta))/(2*a2)
            dlbd2 = (-a1 - np.sqrt(delta))/(2*a2)
            dlbd = max(dlbd1, dlbd2)
            lbd += dlbd
            c += (dc1 + dlbd*dc2)
            dc += (dc1 + dlbd*dc2)
            #beta = 10*dot(c.T, c)

            if cc.debug:
                cc.uvw(c, x=cc.L/2., t=0.)
                w1 = cc.w[0]
                print '\t\tDEBUG w1 = {0} mm'.format(w1)
                print '\t\t\tDEBUG c00 = {0} mm'.format(c[0])

            Rmax = np.abs(lbd*f_ext - df_int).max()
            #print '\tDEBUG Rmax', Rmax
            #print '\tDEBUG lbd', lbd
            if Rmax <= cc.absTOL:
                converged = True
                break
            if (Rmax>min_Rmax and iteration>2):
                print('\t\tWARNING - Diverged!')
                break
            min_Rmax = min(min_Rmax, Rmax)
        if converged:
            length *= 1.2
            cc.increments.append(lbd)
            cc.cs.append(c.copy())
            count += 1
        else:
            c = cc.cs[-1].copy()
            length *= 0.9

    print 'lambda', lbd



