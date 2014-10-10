def _solver_arc_length(a):
    r"""Arc-Length solver

    """
    log('Initialization...', level=1)
    lbd_init = a.initialInc
    lbd = lbd_init
    last_lbd = 0.

    length_inc = 1.
    length = length_inc
    max_length = length
    modified_NR = a.modified_NR
    fext = a.calc_fext(inc=1.)
    k0 = a.calc_k0()
    c = solve(k0, lbd*fext)
    beta = c.dot(c)
    fint = k0*c
    last_fint = fint
    step_num = 1
    total_length = 0

    if modified_NR:
        if a.kT_initial_state:
            log('Updating kT for initial imperfections...', level=1)
            kT_last = a.calc_kT(c*0)
            log('kT updated!', level=1)
        else:
            kT_last = k0
        compute_NL_matrices = False
    else:
        compute_NL_matrices = True
        kT_last = k0

    while step_num < 100:
        log('Attempting arc-length = {0}'.format(length), level=1)
        min_Rmax = 1.e6
        prev_Rmax = 1.e6
        converged = False
        iteration = 0

        kT = kT_last

        while True:
            iteration += 1
            log('Iteration: {}'.format(iteration), level=2)
            if iteration > a.maxNumIter:
                warn('Maximum number of iterations achieved!', level=2)
                break

            # applying the arc-length constraint to find the new lbd and the
            # new c
            dc2 = solve(kT, fext)
            dc1 = solve(kT, (lbd*fext - fint))
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
                kT = a.calc_kT(c)
            else:
                if not modified_NR:
                    compute_NL_matrices = True
                #NOTE attempt to calculate fint more often than kT

            fint = a.calc_fint(c)

            # calculating the residual
            Rmax = np.abs(lbd*fext - fint).max()
            log('Rmax = {0}'.format(Rmax), level=3)
            log('lbd = {0}'.format(lbd), level=3)
            if Rmax <= a.absTOL:
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
            a.increments.append(lbd)
            a.cs.append(c.copy())
            if abs(lbd - 1.) < 0.001:
                log('Condition abs(lbd - 1.) < 0.001 achieved!', level=1)
                break

            if lbd < last_lbd and len(a.cs) > 0:
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
                kT = a.calc_kT(c)
                log('kT updated!', level=1)
            compute_NL_matrices = False
            kT_last = kT
        else:
            if len(a.cs) > 0:
                c = a.cs[-1].copy()
            else:
                lbd = lbd_init
                kT = k0
                c = solve(kT, lbd*fext)
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
            if length_inc < a.minInc:
                log('Minimum arc-length achieved!', level=1)
                break

    log('Finished Non-Linear Static Analysis')
    log('with a total arc-length {0}'.format(length), level=1)
