def conecyl(**kwargs):
    '''
    Calculates the displacement field for a cone/cylinder under `NPL`
    perturbation loads `PLs` located at positions `PL_thetas` and under
    a prescribed axial compression `u` **OR** a prescribed axial load
    `FC` at the bottom

              ||
              || u OR FC
              \/
            ______     / PL
           /      \   / (normal to
          /        \ v   surface)
         /          \
        /            \
       /______________\

    The required paramters are:
        - geometric parameters: ``r,alpha,H``
        - material properties (isotropic): ``E, nu, thickness``
        - material properties (composite): ``stack, plyts, laminaprops``
        - p-refinement of the shape functions: ``m, n``
        - shear correction factor: ``K``
        - perturbation loads: ``PLs, PL_thetas``
        - axial compression ``u or FC``
        - load asymmetry ``omega`` and (``beta when displ. controlled`` **OR**
                         ``xi when load controlled``)
        - anylysis type: ``lb, NL``

    Parameters
    ----------
    r: float
        Cone/cylinder radius at the bottom.
    H: float
        Cone/cylinder height measured along the axial axis.
    alpha: float
        Cone semi-vertex angle. When ``alpha=0.`` it becomes a cylinder.
    stack: list or tuple
        Stacking sequence, given as ``[0,-45,45,-30,30]``.
    plyts: list or tuple
        Ply thicknesses for each ply in ``stack``.
        The data from the first lamina will be used for all the others
        in case: ``len(laminaprops) < len(stack)``.
    laminaprops: list or tuple
        Material properties for each ply in ``stack``.
        The data from the first lamina will be used for all the others
        in case: ``len(laminaprops) < len(stack)``.
        The material constants must be given in the order:
        ``E11, E22, nu12, G12, G13, G23``.
        Example:
        ``laminaprops = ((142.5e3, 8.7e3, 0.28, 5.1e3, 5.1e3,  5.1e3),)``
    E: float
        Elastic Young Modulus of the isotropic material.
    nu: float
        Poisson's coefficient of the isotropic material.
    thickness: float
        Shell thickness.
    K: float
        Shear correction factor
        (default = 5/6.)
    F:  2D numpy.ndarray
        Material constitutive matrix, when passed will overwrite
        `stack`,`plyts`,`laminaprops`,`K`,`E` and `nu`.
    m: integer
        Number of terms for the functions along the cone/cylinder `x` axis.
        (default = 40)
    n: integer
        Number of terms for the functions along the cone/cylinder `theta` axis.
        (default = 40)
    PLs: list or tuple
        Perturbation loads to be applied.
    PL_xs: list or tuple
        ``x`` values in radians for each perturbation load in ``PLs``.
    PL_thetas: list or tuple
       ``theta`` values in radians for each perturbation load in ``PLs``.
    pd: boolean
        Optional flag to force or not prescribed displacement. Useful when
        this function is called with both `u` and `FC`.
        (default = None)
    u: float
        Prescribed displacement axial compression.
        (default = 0.)
    beta: float
        Kink angle of the testing machine about the top edge of
        the cone/cylinder. This causes a load asymmetry.
        (default = 0.)
    FC: float
        Prescribed load axial compression.
    xi: float
        Misalignment between load ``FC`` and the cone/cylinder axis.
        This causes a load asymmetry due to a moment component ``MC=xi*FC``.
        (default = 0.)
    omega: float
        Angular position of the load asymmetry. This works like rotating the
        cone/cylinder in the testing machine in order to produce a
        load asymmetry at different positions.
    bc: string
        Boundary conditions. Can be `cc` for clamped or `ss` for simply
        supported.
        (default = 'ss')
    lb: boolean
        Linear-Buckling analysis.
        Uses the formula presented in PhD thesis from Shadmehri 2012.
        (default = False)
    NL: boolean
        Non-Linear analysis.
        Considers only geometric non-linearity, finding
        convergence for the non-linear kinematic equations.
        The iterative solution will proceed until the element-wise
        absolute difference between two sets of Ritz constants from two
        consecutive iterations is smaller than `NL_tol` **OR** until
        `maxNumIncr` is reached.
        (default = False)
    NL_tol: float
        Non-Linear tolerance that will dictate when the iteration stops.
        (default = 1.e-3)
    maxNumIncr: integer
        Maximum number of increments along the non-linear analysis.
        (default = 10)
    int_order: integer
        Integration order. 1 is equivalent than the trapezoidal rule, 2 uses a
        series of parabolas for the integration, 3 cubic and so forth...
        (default = 1)
    xs: 1D numpy.ndarray
        Meridional positions for the numerical integration.
        This can be optimized to add more points where the integration is
        needed the most.
        (default = 4000 points from ``0`` to ``L=H/cos(alpha)``)
    ts: 1D numpy.ndarray
        Circumferential positions for the numerical integration.
        This can be optimized to add more points where the integration is
        needed the most.
        If ``int_order`` is > 1, the number of integration points must be a
        multiple of ``(int_order + 1)``.
        (default = 4000 points from ``0`` to ``2*pi``)

    Returns
    -------
    displ: function
        displ(xs, thetas) to calculate the displacement field
        ``[u, v, w, phi_x, phi_theta]``.
        - xs and thetas must be an iterable containing the angles in radians
        - the function returns a ``(5,N)`` 2D array where ``N=len(thetas)``
    '''
    # reading kwargs and setting defaults when applicable
    r   = kwargs.get('r')
    H   = kwargs.get('H')
    a = kwargs.get('alpha')
    L = H/cos(a)
    r2 = r - tan(a) * H
    stack = kwargs.get('stack')
    plyts = kwargs.get('plyts')
    laminaprops = kwargs.get('laminaprops')
    if len(stack) != len(laminaprops):
        laminaprops = [laminaprops[0] for i in stack]
    E   = kwargs.get('E')
    nu  = kwargs.get('nu')
    h = kwargs.get('thickness')
    K   = kwargs.get('K', 5/6.)
    F   = kwargs.get('F')
    m   = kwargs.get('m', 40)
    n   = kwargs.get('n', 40)
    PLs = kwargs.get('PLs')
    PL_xs = kwargs.get('PL_xs')
    PL_thetas = kwargs.get('PL_thetas')
    pd = kwargs.get('pd')
    u   = kwargs.get('u')
    b = kwargs.get('beta', 0.)
    sina = sin(a)
    cosa = cos(a)
    r2tanb = r2*tan(b)
    FC  = kwargs.get('FC')
    xi = kwargs.get('xi',0.)
    omega = kwargs.get('omega',0.)
    bc  = str(kwargs.get('bc', 'ss'))
    lb = kwargs.get('lb', False)
    NL  = kwargs.get('NL', False)
    NL_tol  = kwargs.get('NL_tol', 1.e-3)
    maxNumIncr = kwargs.get('maxNumIncr', 10)
    int_order = kwargs.get('int_order', 2)
    xs = kwargs.get('xs')
    ts = kwargs.get('ts')
    #
    if F==None and not stack:
        G = E/(2*(1+nu))
        A11 =    E*h/(1-nu**2)
        A12 = nu*E*h/(1-nu**2)
        A22 =    E*h/(1-nu**2)
        A66 =    G*h
        D11 =    E*h**3/(12*(1-nu**2))
        D12 = nu*E*h**3/(12*(1-nu**2))
        D22 =    E*h**3/(12*(1-nu**2))
        D66 =    G*h**3/12
        F = np.array( [[ A11, A12,   0,   0,   0,   0 ],
                       [ A12, A22,   0,   0,   0,   0 ],
                       [   0,   0, A66,   0,   0,   0 ],
                       [   0,   0,   0, D11, D12,   0 ],
                       [   0,   0,   0, D12, D22,   0 ],
                       [   0,   0,   0,   0,   0, D66 ]])
    elif stack:
        lam = composite.read_stack( stack, plyts, laminaprops=laminaprops )
        A11,A12,A16 = lam.A[0]
        A12,A22,A26 = lam.A[1]
        A16,A26,A66 = lam.A[2]
        B11,B12,B16 = lam.B[0]
        B12,B22,B26 = lam.B[1]
        B16,B26,B66 = lam.B[2]
        D11,D12,D16 = lam.D[0]
        D12,D22,D26 = lam.D[1]
        D16,D26,D66 = lam.D[2]
        A44, A45 = K*lam.E[0]
        A45, A55 = K*lam.E[1]
        F = np.array([[A11, A12, A16, B11, B12, B16,   0,   0],
                      [A12, A22, A26, B12, B22, B26,   0,   0],
                      [A16, A26, A66, B16, B26, B66,   0,   0],
                      [B11, B12, B16, D11, D12, D16,   0,   0],
                      [B12, B22, B26, D12, D22, D26,   0,   0],
                      [B16, B26, B66, D16, D26, D66,   0,   0],
                      [  0,   0,   0,   0,   0,   0, A44, A45],
                      [  0,   0,   0,   0,   0,   0, A45, A55]])
    # needed for the BLAS routines
    F = np.array(F, order='F')
    #
    if xs == None:
        xs = linspace(0, L, 4000)
    if ts == None:
        ts = linspace(0, 2*pi, 4000)
    # setting other variables
    if pd == None:
        if u==None:
            if FC==None:
                u=0.
                pd=True
            else:
                u=0.
                pd=False
        else:
            pd=True
    k0, k1 = trapz(a,b,omega,r,r2,L,bc,m,n,F,xs,ts,u,NL=False,pd=pd,lb=lb)
    #else:
        #k1 = k1_poly(b,r,phi,bc,n,F,xs,int_order,NL=False,pd=pd)
    #
    if not lb:
        if pd:
            k2 = np.zeros((1,5*m*n), dtype=float)
        else:
            k2 = np.zeros((1,5*m*n+1), dtype=float)
        if bc=='cc':
            for PL, PL_x, PL_theta in zip(PLs, PL_xs, PL_thetas):
                g = g_cc(sina,cosa,r2tanb,omega,m,n,L,PL_x,PL_theta,pd)
                k2 += -g[2,:]*PL
            if not pd:
                pass
                #TODO integrate fC ???
                #k2 += -g_cc(m,n,L,0,0,pd)[0,:]*FC*cos(phi)
                #k2 += -g_cc(m,n,L,phi,pd)[1,:]*FC*sin(phi)
        elif bc=='ss':
            for PL, PL_x, PL_theta in zip(PLs, PL_xs, PL_thetas):
                g = g_ss(sina,cosa,r2tanb,omega,m,n,L,PL_x,PL_theta,pd)
                k2 += -g[2,:]*PL
            if not pd:
                pass
                #TODO integrate fC ???
                #k2 += -g_ss(m,n,L,0,0,pd)[0,:]*FC*cos(phi)
                #k2 += -g_ss(m,n,L,phi,pd)[1,:]*FC*sin(phi)
        if pd:
            c = solve(k1,(k2-k0).T)
        else:
            c = solve(k1,k2.T)
    else:
        c = eig(k1)
    if NL and False:
        print('Starting non-linear analysis...')
        for i in range(1,maxNumIncr+1):
            print('iteration number %d' % i)
            if int_order == 1:
                k1 = k1_trapz(b,r,phi,bc,n,F,xs,u,c,NL=NL,pd=pd)
            else:
                k1 = k1_poly(b,r,phi,bc,n,F,xs,int_order,u,c,NL=NL,pd=pd)
            if pd:
                if int_order == 1:
                    k0 = k0_trapz(u,b,r,phi,bc,n,F,xs,c,NL=NL)
                else:
                    k0 = k0_poly(u,b,r,phi,bc,n,F,xs,int_order,c,NL=NL)
                c2 = solve(k1,(k2-k0).T)
            else:
                c2 = solve(k1,k2.T)
            if np.allclose(c,c2,atol=NL_tol):
                print('MESSAGE: convergence found!')
                c=c2
                break
            c=c2
        if i==maxNumIncr:
            print('WARNING: maxNumIncr reached!')
    def displ(xs, ts):
        '''
        Calculates the displacement field: ``[v, w, phi_theta]``.
        - ``thetas`` must be an iterable containing the angles in radians.
        - the function returns a ``(3,N)`` 2D array where ``N=len(thetas)``
        '''
        xs = np.array(xs, order='F')
        ts = np.array(ts, order='F')
        args = displ.seq1 + [xs,ts] + displ.seq2
        return u_vec(*args)
    displ.seq1 = [c,a,b,omega,r2,L,m,n]
    displ.seq2 = [u,bc,pd]
    return displ, c
