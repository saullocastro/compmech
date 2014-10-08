r"""
Non-linear algorithms used with the ``ConeCyl`` object.

.. currentmodule:: compmech.conecyl.non_linear


"""
from __future__ import division

import numpy as np
from numpy import dot
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from compmech.constants import DOUBLE
from compmech.logger import *
from compmech.sparse import solve

TOO_SLOW = 0.01

def NR_Broyden(cc):
    #
    inc = cc.initialInc
    total = inc
    #
    k0 = cc.k0
    kT = k0
    c = np.zeros((cc.pdoff + 6*cc.m2*cc.n2 + 3*cc.m1), DOUBLE)
    delta = np.zeros_like(c)
    delta += 0.001
    c = c + delta
    while True:
        iteration = 0
        converged = False
        Rmax_old = 1e6
        while True:
            fext = cc.calc_fext(inc=total)
            if True:#iteration==0:
                R = fext - k0*c
            #else:
                #cc._calc_NL_matrices(c)
                #R = fext - (cc.k0 + cc.k0L/2 + cc.kL0 + cc.kLL/2)*c
            Rmax = np.abs(R).max()

            if abs(Rmax-Rmax_old) <= cc.absTOL:
                converged = True
                log('Converged!')
                break
            else:
                Rmax_old = Rmax
            iteration += 1
            update_kT = True
            if cc.modified_NR:
                if iteration<=5 or test==0:
                    update_kT = True
                else:
                    update_kT = False
            if update_kT:
                print('Updating kT...')
                kT = kT - csc_matrix(np.outer(R, delta)/np.dot(delta, delta))
                delta = spsolve(kT, fext)
            c += delta
        if converged:
            cc.cs.append(c)
            cc.increments.append(inc)
            total += inc

    #
    log('Finished Non-Linear Static Analysis')
    log('at time {0}'.format(total), level=1)

def NR(cc):
    r"""Newton-Raphson algorithm for non-linear analysis.

    Parameters
    ----------
    cc : compmech.conecyl.ConeCyl
        The ``ConeCyl`` object.

    """
    log('Initialization...', level=1)

    line_search = cc.line_search
    compute_every_n = cc.compute_every_n
    modified_NR = cc.modified_NR
    inc = cc.initialInc
    total = inc
    once_at_total = False
    max_total = 0.

    fext = cc.calc_fext(inc=inc)
    c = solve(cc.k0uu, fext)

    if modified_NR:
        if cc.c0!=None:
            log('Updating kT for initial imperfections...', level=1)
            cc._calc_NL_matrices(c*0, inc=0.)
            log('kT updated!', level=1)
            kSuu_last = cc.kSuu
            kSuk_last = cc.kSuk
            kTuu_last = cc.kTuu
        else:
            kSuu_last = cc.k0uu
            kSuk_last = cc.k0uk
            kTuu_last = cc.k0uu
        compute_NL_matrices = False
    else:
        compute_NL_matrices = True
        kSuu_last = cc.k0uu
        kSuk_last = cc.k0uk
        kTuu_last = cc.k0uu

    step_num = 1
    while True:
        log('Started Load Step {} - '.format(step_num)
            + 'Attempting time = {0}'.format(total), level=1)

        # TODO maybe for pdC=True, pdT the fext must be calculated with the
        #      last kT available...

        absERR = 1.e6
        relERR = 1.e6
        min_Rmax = 1.e6
        prev_Rmax = 1.e6
        last_min_Rmax = 1.e6
        iteration = 0
        converged = False

        kSuu = kSuu_last
        kSuk = kSuk_last
        kTuu = kTuu_last

        fext = cc.calc_fext(inc=total, kuk=kSuk)

        iter_NR = 0
        while True:
            iteration += 1
            log('Iteration: {}'.format(iteration), level=2)
            if iteration > cc.maxNumIter:
                warn('Maximum number of iterations achieved!', level=2)
                break

            if compute_NL_matrices or (cc.c0==None and step_num==1 and
                    iteration==1) or iter_NR==(compute_every_n-1):
                iter_NR = 0
                cc._calc_NL_matrices(c, inc=total)
                kSuu = cc.kSuu
                kSuk = cc.kSuk
                kTuu = cc.kTuu
            else:
                iter_NR += 1
                if not modified_NR:
                    compute_NL_matrices = True

            fint = cc.calc_fint(c, inc=total, m=1)

            for i, dof in enumerate(cc.excluded_dofs):
                kSuk_col = kSuk[:, dof].ravel()
                fint += total*cc.excluded_dofs_ck[i]*kSuk_col

            R = fext - fint

            # convergence criteria:
            # - maximum residual force Rmax
            Rmax = np.abs(R).max()
            log('Rmax = {0}'.format(Rmax), level=3)

            if iteration >= 2 and Rmax < cc.absTOL:
                converged = True
                break
            if (Rmax > prev_Rmax and Rmax > min_Rmax and iteration > 2):
                warn('Diverged!', level=2)
                break
            else:
                min_Rmax = min(min_Rmax, Rmax)
            change_rate_Rmax = abs(prev_Rmax-Rmax)/abs(prev_Rmax)
            if (iteration > 2 and change_rate_Rmax < TOO_SLOW):
                warn('Diverged! (convergence too slow)', level=2)
                break
            prev_Rmax = Rmax

            log('Solving... ', level=2)
            delta_c = solve(kTuu, R)
            log('finished!', level=2)

            eta1 = 0.
            eta2 = 1.
            if line_search:
                while True:
                    c1 = c + eta1*delta_c
                    c2 = c + eta2*delta_c
                    fint1 = cc.calc_fint(c1, inc=total, m=1)
                    fint2 = cc.calc_fint(c2, inc=total, m=1)
                    R1 = fext - fint1
                    R2 = fext - fint2
                    s1 = delta_c.dot(R1)
                    s2 = delta_c.dot(R2)
                    eta_new = (eta2-eta1)*(-s1/(s2-s1)) + eta1
                    eta1 = eta2
                    eta2 = eta_new
                    eta2 = min(max(eta2, 0.2), 10.)
                    if abs(eta2-eta1) < 0.01:
                        break
            c = c + eta2*delta_c


        if converged:
            log('Finished Load Step {} at'.format(step_num)
                + ' time = {0}'.format(total), level=1)
            cc.increments.append(total)
            cc.cs.append(c.copy()) #NOTE copy required
            finished = False
            if abs(total - 1) < 1e-3:
                finished = True
            else:
                factor = 1.1
                if once_at_total:
                    inc_new = min(factor*inc, cc.maxInc, (1.-total)/2)
                else:
                    inc_new = min(factor*inc, cc.maxInc, 1.-total)
                log('Changing time increment from {0} to {1}'.format(
                    inc, inc_new), level=1)
                inc = inc_new
                total += inc
                total = min(1, total)
                step_num += 1
            if finished:
                break
            if modified_NR:
                log('Updating kT...', level=1)
                cc._calc_NL_matrices(c, inc=total)
                log('kT updated!', level=1)
                kSuu = cc.kSuu
                kSuk = cc.kSuk
                kTuu = cc.kTuu
            compute_NL_matrices = False
            kSuu_last = kSuu
            kSuk_last = kSuk
            kTuu_last = kTuu
        else:
            max_total = max(max_total, total)
            while True:
                factor = 0.3
                log('Bisecting time increment from {0} to {1}'.format(
                    inc, inc*factor), level=1)
                if abs(total -1) < 1e-3:
                    once_at_total = True
                total -= inc
                inc *= factor
                if inc < cc.minInc:
                    log('Minimum step size achieved!', level=1)
                    break
                total += inc
                if total >= max_total:
                    continue
                else:
                    break
            if inc < cc.minInc:
                log('Stopping solver: minimum step size achieved!', level=1)
                break

        if len(cc.cs)>0:
            c = cc.cs[-1].copy() #NOTE copy required
        else:
            # means that a bisection must be done in initialInc
            fext = cc.calc_fext(inc=inc)
            c = solve(cc.k0uu, fext)

            #cc.cs = [c.copy()]
            #cc.increments = [inc]

    log('Finished Non-Linear Static Analysis')
    log('at time {0}'.format(total), level=1)

def arc_length(cc):
    r"""Arc-Length algorithm for non-linear analysis.

    Parameters
    ----------
    cc : compmech.conecyl.ConeCyl
        The ``ConeCyl`` object.

    """
    log('Initialization...', level=1)
    lbd_init = cc.initialInc
    lbd = lbd_init
    last_lbd = 0.

    length_inc = 1.
    length = length_inc
    max_length = length
    modified_NR = cc.modified_NR
    fext = cc.calc_fext(inc=1.)
    kTuu = cc.k0uu
    c = solve(kTuu, lbd*fext)
    beta = c.dot(c)
    fint = kTuu*c
    last_fint = fint
    step_num = 1
    total_length = 0

    if modified_NR:
        if cc.c0!=None:
            log('Updating kT for initial imperfections...', level=1)
            cc._calc_NL_matrices(c*0, inc=0.)
            log('kT updated!', level=1)
            kSuu_last = cc.kSuu
            kSuk_last = cc.kSuk
            kTuu_last = cc.kTuu
        else:
            kSuu_last = cc.k0uu
            kSuk_last = cc.k0uk
            kTuu_last = cc.k0uu
        compute_NL_matrices = False
    else:
        compute_NL_matrices = True
        kSuu_last = cc.k0uu
        kSuk_last = cc.k0uk
        kTuu_last = cc.k0uu

    while step_num < 100:
        log('Attempting arc-length = {0}'.format(length), level=1)
        min_Rmax = 1.e6
        prev_Rmax = 1.e6
        converged = False
        iteration = 0

        kSuu = kSuu_last
        kSuk = kSuk_last
        kTuu = kTuu_last

        while True:
            iteration += 1
            log('Iteration: {}'.format(iteration), level=2)
            if iteration > cc.maxNumIter:
                warn('Maximum number of iterations achieved!', level=2)
                break

            # applying the arc-length constraint to find the new lbd and the
            # new c
            dc2 = solve(kTuu, fext)
            dc1 = solve(kTuu, (lbd*fext - fint))
            a2 = beta + dot(dc2.T, dc2)
            a1 = 2*beta*lbd + 2*dot(c.T, dc2) + 2*dot(dc1.T, dc2)
            a0 = (beta*lbd**2 + dot(c.T, c) + 2*dot(c.T, dc1)
                  + dot(dc1.T, dc1) - beta*length**2)
            delta = a1**2 - 4*a2*a0
            if delta<0:
                warn('Diverged! (negative delta)', level=2)
                break

            eta = 1.
            dlbd1 = (-a1 + np.sqrt(delta))/(2*a2)
            dc_1 = (dc1 + dlbd1*dc2)
            dlbd2 = (-a1 - np.sqrt(delta))/(2*a2)
            dc_2 = (dc1 + dlbd2*dc2)

            lbd = lbd + eta*dlbd1
            c = c + eta*dc_1

            if lbd > 1.:
                warn('Diverged! (lbd > 1.)', level=2)
                break

            # computing the Non-Linear matrices
            if compute_NL_matrices:
                cc._calc_NL_matrices(c, inc=lbd)
                kTuu = cc.kTuu
                kSuu = cc.kSuu
            else:
                if not modified_NR:
                    compute_NL_matrices = True
                #NOTE attempt to calculate fint more often than kT

            fint = cc.calc_fint(c, inc=lbd, m=1)

            # calculating the residual
            Rmax = np.abs(lbd*fext - fint).max()
            log('Rmax = {0}'.format(Rmax), level=3)
            log('lbd = {0}'.format(lbd), level=3)
            if Rmax <= cc.absTOL:
                converged = True
                break
            if (Rmax > min_Rmax and Rmax > prev_Rmax and iteration > 2):
                warn('Diverged!', level=2)
                break
            else:
                min_Rmax = min(min_Rmax, Rmax)
            change_rate_Rmax = abs(prev_Rmax-Rmax)/abs(prev_Rmax)
            if (iteration > 2 and change_rate_Rmax < TOO_SLOW):
                warn('Diverged! (convergence too slow)', level=2)
                break
            prev_Rmax = Rmax

        if converged:
            log('Finished increment with total length = {0}'.format(
                length), level=1)
            cc.increments.append(lbd)
            cc.cs.append(c.copy())
            if abs(lbd - 1.) < 0.001:
                log('Condition abs(lbd - 1.) < 0.001 achieved!', level=1)
                break

            if lbd < last_lbd and len(cc.cs) > 0:
                log('Drop of reaction load achieved!', level=1)
                #TODO maybe when we need to detect the local snap-through
                #     this if should be extended...
                #break
            last_fint = fint.copy()
            last_lbd = lbd
            step_num += 1
            factor = 0.95
            new_length = length + length_inc*factor
            #TODO
            #while new_length >= max_length:
            #    new_length *= 0.9
            log('(lambda of this step = {0})'.format(lbd), level=1)
            log('Changing arc-length from {0} to {1}'.format(length,
                new_length), level=1)
            length = new_length
            if modified_NR:
                log('Updating kT...', level=1)
                cc._calc_NL_matrices(c, inc=lbd)
                log('kT updated!', level=1)
                kSuu = cc.kSuu
                kSuk = cc.kSuk
                kTuu = cc.kTuu
            compute_NL_matrices = False
            kSuu_last = kSuu
            kSuk_last = kSuk
            kTuu_last = kTuu
        else:
            if len(cc.cs) > 0:
                c = cc.cs[-1].copy()
            else:
                lbd = lbd_init
                kTuu = cc.k0uu
                c = solve(kTuu, lbd*fext)
            factor = 0.3 # keep in the range 0 < factor < 1.
            fint = last_fint
            lbd = last_lbd
            max_length = max(length, max_length)
            old_length = length
            length -= length_inc
            length_inc *= factor
            length += length_inc
            log('Diverged - reducing arc-length from {0} to {1}'.format(
                old_length, length), level=1)
            if length_inc < cc.minInc:
                log('Minimum arc-length achieved!', level=1)
                break

    log('Finished Non-Linear Static Analysis')
    log('with a total arc-length {0}'.format(length), level=1)

def NR_lebofsky(cc):
    r"""Newton-Raphson algorithm using the Lebofsky method to
    calculate the tangent stiffness matrix.

    Parameters
    ----------
    cc : compmech.conecyl.ConeCyl
        The ``ConeCyl`` object.

    """
    log('Initialization...', level=1)

    inc = cc.initialInc
    total = inc
    once_at_total = False
    max_total = 0.

    k0 = cc.k0uu
    fext = cc.calc_fext(inc=inc)

    c = solve(k0, fext)

    kSuk = None

    step_num = 1
    while True:
        log('Started Load Step {} - '.format(step_num)
            + 'Attempting time = {0}'.format(total), level=1)

        modified_NR = cc.modified_NR

        fext = cc.calc_fext(inc=total, kuk=kSuk)

        # TODO maybe for pdC, pdT the fext must be calculated with the
        #      last kT available...

        absERR = 1.e6
        relERR = 1.e6
        min_Rmax = 1.e6
        prev_Rmax = 1.e6
        last_min_Rmax = 1.e6
        iteration = 0
        converged = False
        compute_NL_matrices = True
        while True:
            iteration += 1
            log('Iteration: {}'.format(iteration), level=2)
            if iteration > cc.maxNumIter:
                warn('Maximum number of iterations achieved!', level=2)
                break

            # secant stiffness matrix
            kSuu = cc.k0uu
            kSuk = cc.k0uk
            cc.kSuk = kSuk
            cc.kSuu = kSuu

            # tangent stiffness matrix
            kT = cc._calc_kT(c)

            fint = cc.calc_fint(c, inc=total, m=1)

            for i, dof in enumerate(cc.excluded_dofs):
                kSuk_col = cc.kSuk[:, dof].ravel()
                fint += total*cc.excluded_dofs_ck[i]*kSuk_col

            R = fext - fint

            # convergence criteria:
            # - maximum residual force Rmax
            Rmax = np.abs(R).max()
            log('Rmax = {0}'.format(Rmax), level=3)

            if iteration >= 2 and Rmax < cc.absTOL:
                converged = True
                break

            if (Rmax>prev_Rmax and Rmax>min_Rmax and iteration>2):
                warn('Diverged!', level=2)
                break
            min_Rmax = min(min_Rmax, Rmax)
            prev_Rmax = Rmax

            log('Solving... ', level=2)
            delta_c = solve(kTuu, R)
            c += delta_c
            log('finished!', level=2)


        if converged:
            finished = False
            log('Finished Load Step {} at'.format(step_num)
                + ' time = {0}'.format(total), level=1)
            cc.increments.append(total)
            if abs(total - 1) < 1e-3:
                finished = True
            else:
                if once_at_total:
                    inc_new = min(1.1*inc, cc.maxInc, (1.-total)/2)
                else:
                    inc_new = min(1.1*inc, cc.maxInc, 1.-total)
                log('Changing time increment from {0} to {1}'.format(
                    inc, inc_new), level=1)
                inc = inc_new
                total += inc
                total = min(1, total)
                step_num += 1
            cc.cs.append(c.copy()) #NOTE copy required
            if finished:
                break
        else:
            max_total = max(max_total, total)
            while True:
                log('Bisecting time increment from {0} to {1}'.format(
                    inc, inc/4), level=1)
                if abs(total -1) < 1e-3:
                    once_at_total = True
                total -= inc
                inc /= 4
                if inc < cc.minInc:
                    log('Minimum step size achieved!', level=1)
                    break
                total += inc
                if total >= max_total:
                    continue
                else:
                    break
            if inc < cc.minInc:
                log('Stopping solver: minimum step size achieved!', level=1)
                break


        if len(cc.cs)>0:
            c = cc.cs[-1].copy() #NOTE copy required
        else:
            # means that a bisection must be done in initialInc
            fext = cc.calc_fext(inc=inc)
            c = solve(k0, fext)

            #cc.cs = [c.copy()]
            #cc.increments = [inc]

    log('Finished Non-Linear Static Analysis')
    log('at time {0}'.format(total), level=1)
