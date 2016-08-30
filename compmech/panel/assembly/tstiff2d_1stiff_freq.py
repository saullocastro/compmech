from __future__ import division, absolute_import

from compmech.panel import Panel
from compmech.panel.assembly import PanelAssembly
from compmech.analysis import freq


def tstiff2d_1stiff_freq(a, b, ys, bb, bf, defect_a, mu, plyt, laminaprop,
        stack_skin, stack_base, stack_flange,
        r=None, m=8, n=8, mb=None, nb=None, mf=None, nf=None):
    r"""Frequency T-Stiffened Panel with possible defect at middle

    For more details about each parameter and the aerodynamic formulation see
    Ref. [castro2016FlutterPanel]_ .

    For more details about the theory involved on the assembly of panels, see
    [castro2017AssemblyModels]_.

    The panel assembly looks like::

        skin
         _________ _____ _________
        |         |     |         |
        |         |     |         |
        |   p01   | p02 |   p03   |
        |         |     |         |
        |_________|_____|_________|
        |   p04   | p05 |   p06   |
        |_________|_____|_________|
        |         |     |         |
        |         |     |         |
        |   p07   | p08 |   p09   |
        |         |     |         |
        |         |     |         |
        |_________|_____|_________|

                  base            flange
                   _____           _____
                  |     |         |     |
                  |     |         |     |
                  | p10 |         | p11 |
                  |     |         |     |
                  |_____|         |_____|
                  | p12 |         | p13 |
                  |_____|         |_____|
                  |     |         |     |
                  |     |         |     |
                  | p14 |         | p15 |
                  |     |         |     |
                  |     |         |     |
                  |_____|         |_____|

    Parameters
    ----------

    a : float
        Total length of the assembly (along `x`).
    b : float
        Total width of the assembly (along `y`).
    ys : float
        Position of the stiffener along `y`.
    bb : float
        Stiffener's base width.
    bf : float
        Stiffener's flange width.
    defect_a : float
        Debonding defect/assembly length ratio.
    mu : float
        Material density.
    plyt : float
        Ply thickness.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`.
    stack_skin : list or tuple
        Stacking sequence for the skin.
    stack_base : list or tuple
        Stacking sequence for the stiffener's base.
    stack_flange : list or tuple
        Stacking sequence for the stiffener's flange.
    r : float or None, optional
        Radius of the stiffened panel.
    m, n : int, optional
        Number of terms of the approximation function for the skin.
    mb, nb : int, optional
        Number of terms of the approximation function for the stiffener's base.
    mf, nf : int, optional
        Number of terms of the approximation function for the stiffener's
        flange.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../../../compmech/panel/assembly/tests/test_tstiff2d_assembly.py
        :pyobject: test_tstiff2d_1stiff_freq


    """
    defect = defect_a * a
    has_defect = True if defect > 0 else False
    defect = 0.33*a if defect == 0 else defect # to avoid weird domains
    aup = (a - defect)/2.
    alow = (a - defect)/2.
    bleft = b - ys - bb/2.
    bright = ys - bb/2.
    mb = m if mb is None else mb
    nb = n if nb is None else nb
    mf = m if mf is None else mf
    nf = n if nf is None else nf
    # skin panels
    p01 = Panel(group='skin', x0=alow+defect, y0=ys+bb/2., a=aup, b=bleft, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)
    p02 = Panel(group='skin', x0=alow+defect, y0=ys-bb/2., a=aup, b=bb, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)
    p03 = Panel(group='skin', x0=alow+defect, y0=0,        a=aup, b=bright, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)
    # defect
    p04 = Panel(group='skin', x0=alow, y0=ys+bb/2., a=defect, b=bleft, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)
    p05 = Panel(group='skin', x0=alow, y0=ys-bb/2., a=defect, b=bb, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)
    p06 = Panel(group='skin', x0=alow, y0=0,        a=defect, b=bright, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)
    #
    p07 = Panel(group='skin', x0=0, y0=ys+bb/2., a=alow, b=bleft, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)
    p08 = Panel(group='skin', x0=0, y0=ys-bb/2., a=alow, b=bb, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)
    p09 = Panel(group='skin', x0=0, y0=0,        a=alow, b=bright, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, mu=mu)

    # stiffeners
    p10 = Panel(group='base', x0=alow+defect, y0=ys-bb/2., a=aup, b=bb, r=r, m=mb, n=nb, plyt=plyt, stack=stack_base, laminaprop=laminaprop, mu=mu)
    p11 = Panel(group='flange', x0=alow+defect, y0=0,        a=aup, b=bf, m=mf, n=nf, plyt=plyt, stack=stack_flange, laminaprop=laminaprop, mu=mu)
    # defect
    p12 = Panel(group='base', x0=alow, y0=ys-bb/2., a=defect, b=bb, r=r, m=mb, n=nb, plyt=plyt, stack=stack_base, laminaprop=laminaprop, mu=mu)
    p13 = Panel(group='flange', x0=alow, y0=0,        a=defect, b=bf, m=mf, n=nf, plyt=plyt, stack=stack_flange, laminaprop=laminaprop, mu=mu)
    #
    p14 = Panel(group='base', x0=0, y0=ys-bb/2., a=alow, b=bb, r=r, m=mb, n=nb, plyt=plyt, stack=stack_base, laminaprop=laminaprop, mu=mu)
    p15 = Panel(group='flange', x0=0, y0=0,        a=alow, b=bf, m=mf, n=nf, plyt=plyt, stack=stack_flange, laminaprop=laminaprop, mu=mu)

    # boundary conditions
    p01.u1tx = 1 ; p01.u1rx = 1 ; p01.u2tx = 0 ; p01.u2rx = 1
    p01.v1tx = 1 ; p01.v1rx = 1 ; p01.v2tx = 0 ; p01.v2rx = 1
    p01.w1tx = 1 ; p01.w1rx = 1 ; p01.w2tx = 0 ; p01.w2rx = 1
    p01.u1ty = 1 ; p01.u1ry = 1 ; p01.u2ty = 0 ; p01.u2ry = 1
    p01.v1ty = 1 ; p01.v1ry = 1 ; p01.v2ty = 0 ; p01.v2ry = 1
    p01.w1ty = 1 ; p01.w1ry = 1 ; p01.w2ty = 0 ; p01.w2ry = 1

    p02.u1tx = 1 ; p02.u1rx = 1 ; p02.u2tx = 0 ; p02.u2rx = 1
    p02.v1tx = 1 ; p02.v1rx = 1 ; p02.v2tx = 0 ; p02.v2rx = 1
    p02.w1tx = 1 ; p02.w1rx = 1 ; p02.w2tx = 0 ; p02.w2rx = 1
    p02.u1ty = 1 ; p02.u1ry = 1 ; p02.u2ty = 1 ; p02.u2ry = 1
    p02.v1ty = 1 ; p02.v1ry = 1 ; p02.v2ty = 1 ; p02.v2ry = 1
    p02.w1ty = 1 ; p02.w1ry = 1 ; p02.w2ty = 1 ; p02.w2ry = 1

    p03.u1tx = 1 ; p03.u1rx = 1 ; p03.u2tx = 0 ; p03.u2rx = 1
    p03.v1tx = 1 ; p03.v1rx = 1 ; p03.v2tx = 0 ; p03.v2rx = 1
    p03.w1tx = 1 ; p03.w1rx = 1 ; p03.w2tx = 0 ; p03.w2rx = 1
    p03.u1ty = 0 ; p03.u1ry = 1 ; p03.u2ty = 1 ; p03.u2ry = 1
    p03.v1ty = 0 ; p03.v1ry = 1 ; p03.v2ty = 1 ; p03.v2ry = 1
    p03.w1ty = 0 ; p03.w1ry = 1 ; p03.w2ty = 1 ; p03.w2ry = 1

    p04.u1tx = 1 ; p04.u1rx = 1 ; p04.u2tx = 1 ; p04.u2rx = 1
    p04.v1tx = 1 ; p04.v1rx = 1 ; p04.v2tx = 1 ; p04.v2rx = 1
    p04.w1tx = 1 ; p04.w1rx = 1 ; p04.w2tx = 1 ; p04.w2rx = 1
    p04.u1ty = 1 ; p04.u1ry = 1 ; p04.u2ty = 0 ; p04.u2ry = 1
    p04.v1ty = 1 ; p04.v1ry = 1 ; p04.v2ty = 0 ; p04.v2ry = 1
    p04.w1ty = 1 ; p04.w1ry = 1 ; p04.w2ty = 0 ; p04.w2ry = 1

    p05.u1tx = 1 ; p05.u1rx = 1 ; p05.u2tx = 1 ; p05.u2rx = 1
    p05.v1tx = 1 ; p05.v1rx = 1 ; p05.v2tx = 1 ; p05.v2rx = 1
    p05.w1tx = 1 ; p05.w1rx = 1 ; p05.w2tx = 1 ; p05.w2rx = 1
    p05.u1ty = 1 ; p05.u1ry = 1 ; p05.u2ty = 1 ; p05.u2ry = 1
    p05.v1ty = 1 ; p05.v1ry = 1 ; p05.v2ty = 1 ; p05.v2ry = 1
    p05.w1ty = 1 ; p05.w1ry = 1 ; p05.w2ty = 1 ; p05.w2ry = 1

    p06.u1tx = 1 ; p06.u1rx = 1 ; p06.u2tx = 1 ; p06.u2rx = 1
    p06.v1tx = 1 ; p06.v1rx = 1 ; p06.v2tx = 1 ; p06.v2rx = 1
    p06.w1tx = 1 ; p06.w1rx = 1 ; p06.w2tx = 1 ; p06.w2rx = 1
    p06.u1ty = 0 ; p06.u1ry = 1 ; p06.u2ty = 1 ; p06.u2ry = 1
    p06.v1ty = 0 ; p06.v1ry = 1 ; p06.v2ty = 1 ; p06.v2ry = 1
    p06.w1ty = 0 ; p06.w1ry = 1 ; p06.w2ty = 1 ; p06.w2ry = 1

    p07.u1tx = 0 ; p07.u1rx = 1 ; p07.u2tx = 1 ; p07.u2rx = 1
    p07.v1tx = 0 ; p07.v1rx = 1 ; p07.v2tx = 1 ; p07.v2rx = 1
    p07.w1tx = 0 ; p07.w1rx = 1 ; p07.w2tx = 1 ; p07.w2rx = 1
    p07.u1ty = 1 ; p07.u1ry = 1 ; p07.u2ty = 0 ; p07.u2ry = 1
    p07.v1ty = 1 ; p07.v1ry = 1 ; p07.v2ty = 0 ; p07.v2ry = 1
    p07.w1ty = 1 ; p07.w1ry = 1 ; p07.w2ty = 0 ; p07.w2ry = 1

    p08.u1tx = 0 ; p08.u1rx = 1 ; p08.u2tx = 1 ; p08.u2rx = 1
    p08.v1tx = 0 ; p08.v1rx = 1 ; p08.v2tx = 1 ; p08.v2rx = 1
    p08.w1tx = 0 ; p08.w1rx = 1 ; p08.w2tx = 1 ; p08.w2rx = 1
    p08.u1ty = 1 ; p08.u1ry = 1 ; p08.u2ty = 1 ; p08.u2ry = 1
    p08.v1ty = 1 ; p08.v1ry = 1 ; p08.v2ty = 1 ; p08.v2ry = 1
    p08.w1ty = 1 ; p08.w1ry = 1 ; p08.w2ty = 1 ; p08.w2ry = 1

    p09.u1tx = 0 ; p09.u1rx = 1 ; p09.u2tx = 1 ; p09.u2rx = 1
    p09.v1tx = 0 ; p09.v1rx = 1 ; p09.v2tx = 1 ; p09.v2rx = 1
    p09.w1tx = 0 ; p09.w1rx = 1 ; p09.w2tx = 1 ; p09.w2rx = 1
    p09.u1ty = 0 ; p09.u1ry = 1 ; p09.u2ty = 1 ; p09.u2ry = 1
    p09.v1ty = 0 ; p09.v1ry = 1 ; p09.v2ty = 1 ; p09.v2ry = 1
    p09.w1ty = 0 ; p09.w1ry = 1 ; p09.w2ty = 1 ; p09.w2ry = 1

    # base up
    p10.u1tx = 1 ; p10.u1rx = 1 ; p10.u2tx = 1 ; p10.u2rx = 1
    p10.v1tx = 1 ; p10.v1rx = 1 ; p10.v2tx = 1 ; p10.v2rx = 1
    p10.w1tx = 1 ; p10.w1rx = 1 ; p10.w2tx = 1 ; p10.w2rx = 1
    p10.u1ty = 1 ; p10.u1ry = 1 ; p10.u2ty = 1 ; p10.u2ry = 1
    p10.v1ty = 1 ; p10.v1ry = 1 ; p10.v2ty = 1 ; p10.v2ry = 1
    p10.w1ty = 1 ; p10.w1ry = 1 ; p10.w2ty = 1 ; p10.w2ry = 1

    # flange up
    p11.u1tx = 1 ; p11.u1rx = 1 ; p11.u2tx = 0 ; p11.u2rx = 1
    p11.v1tx = 1 ; p11.v1rx = 1 ; p11.v2tx = 0 ; p11.v2rx = 1
    p11.w1tx = 1 ; p11.w1rx = 1 ; p11.w2tx = 0 ; p11.w2rx = 1
    p11.u1ty = 1 ; p11.u1ry = 1 ; p11.u2ty = 1 ; p11.u2ry = 1
    p11.v1ty = 1 ; p11.v1ry = 1 ; p11.v2ty = 1 ; p11.v2ry = 1
    p11.w1ty = 1 ; p11.w1ry = 1 ; p11.w2ty = 1 ; p11.w2ry = 1

    # base mid
    p12.u1tx = 1 ; p12.u1rx = 1 ; p12.u2tx = 1 ; p12.u2rx = 1
    p12.v1tx = 1 ; p12.v1rx = 1 ; p12.v2tx = 1 ; p12.v2rx = 1
    p12.w1tx = 1 ; p12.w1rx = 1 ; p12.w2tx = 1 ; p12.w2rx = 1
    p12.u1ty = 1 ; p12.u1ry = 1 ; p12.u2ty = 1 ; p12.u2ry = 1
    p12.v1ty = 1 ; p12.v1ry = 1 ; p12.v2ty = 1 ; p12.v2ry = 1
    p12.w1ty = 1 ; p12.w1ry = 1 ; p12.w2ty = 1 ; p12.w2ry = 1

    # flange mid
    p13.u1tx = 1 ; p13.u1rx = 1 ; p13.u2tx = 1 ; p13.u2rx = 1
    p13.v1tx = 1 ; p13.v1rx = 1 ; p13.v2tx = 1 ; p13.v2rx = 1
    p13.w1tx = 1 ; p13.w1rx = 1 ; p13.w2tx = 1 ; p13.w2rx = 1
    p13.u1ty = 1 ; p13.u1ry = 1 ; p13.u2ty = 1 ; p13.u2ry = 1
    p13.v1ty = 1 ; p13.v1ry = 1 ; p13.v2ty = 1 ; p13.v2ry = 1
    p13.w1ty = 1 ; p13.w1ry = 1 ; p13.w2ty = 1 ; p13.w2ry = 1

    # base low
    p14.u1tx = 1 ; p14.u1rx = 1 ; p14.u2tx = 1 ; p14.u2rx = 1
    p14.v1tx = 1 ; p14.v1rx = 1 ; p14.v2tx = 1 ; p14.v2rx = 1
    p14.w1tx = 1 ; p14.w1rx = 1 ; p14.w2tx = 1 ; p14.w2rx = 1
    p14.u1ty = 1 ; p14.u1ry = 1 ; p14.u2ty = 1 ; p14.u2ry = 1
    p14.v1ty = 1 ; p14.v1ry = 1 ; p14.v2ty = 1 ; p14.v2ry = 1
    p14.w1ty = 1 ; p14.w1ry = 1 ; p14.w2ty = 1 ; p14.w2ry = 1

    # flange low
    p15.u1tx = 0 ; p15.u1rx = 1 ; p15.u2tx = 1 ; p15.u2rx = 1
    p15.v1tx = 0 ; p15.v1rx = 1 ; p15.v2tx = 1 ; p15.v2rx = 1
    p15.w1tx = 0 ; p15.w1rx = 1 ; p15.w2tx = 1 ; p15.w2rx = 1
    p15.u1ty = 1 ; p15.u1ry = 1 ; p15.u2ty = 1 ; p15.u2ry = 1
    p15.v1ty = 1 ; p15.v1ry = 1 ; p15.v2ty = 1 ; p15.v2ry = 1
    p15.w1ty = 1 ; p15.w1ry = 1 ; p15.w2ty = 1 ; p15.w2ry = 1

    conn = [
        # skin-skin
        dict(p1=p01, p2=p02, func='SSycte', ycte1=0, ycte2=p02.b),
        dict(p1=p01, p2=p04, func='SSxcte', xcte1=0, xcte2=p04.a),
        dict(p1=p02, p2=p03, func='SSycte', ycte1=0, ycte2=p03.b),
        dict(p1=p02, p2=p05, func='SSxcte', xcte1=0, xcte2=p05.a),
        dict(p1=p03, p2=p06, func='SSxcte', xcte1=0, xcte2=p06.a),
        dict(p1=p04, p2=p05, func='SSycte', ycte1=0, ycte2=p05.b),
        dict(p1=p04, p2=p07, func='SSxcte', xcte1=0, xcte2=p07.a),
        dict(p1=p05, p2=p06, func='SSycte', ycte1=0, ycte2=p06.b),
        dict(p1=p05, p2=p08, func='SSxcte', xcte1=0, xcte2=p08.a),
        dict(p1=p06, p2=p09, func='SSxcte', xcte1=0, xcte2=p09.a),
        dict(p1=p07, p2=p08, func='SSycte', ycte1=0, ycte2=p08.b),
        dict(p1=p08, p2=p09, func='SSycte', ycte1=0, ycte2=p09.b),

        # skin-base
        dict(p1=p02, p2=p10, func='SB'),
        dict(p1=p05, p2=p12, func='SB', has_defect=has_defect), # defect
        dict(p1=p08, p2=p14, func='SB'),

        # base-base
        dict(p1=p10, p2=p12, func='SSxcte', xcte1=0, xcte2=p12.a),
        dict(p1=p12, p2=p14, func='SSxcte', xcte1=0, xcte2=p14.a),

        # base-flange
        dict(p1=p10, p2=p11, func='BFycte', ycte1=p10.b/2., ycte2=0),
        dict(p1=p12, p2=p13, func='BFycte', ycte1=p12.b/2., ycte2=0),
        dict(p1=p14, p2=p15, func='BFycte', ycte1=p14.b/2., ycte2=0),

        # flange-flange
        dict(p1=p11, p2=p13, func='SSxcte', xcte1=0, xcte2=p13.a),
        dict(p1=p13, p2=p15, func='SSxcte', xcte1=0, xcte2=p15.a),
        ]

    panels = [p01, p02, p03, p04, p05, p06, p07, p08, p09,
            p10, p11, p12, p13, p14, p15]
    skin = [p01, p02, p03, p04, p05, p06, p07, p08, p09]
    base = [p10, p12, p14]
    flange = [p11, p13, p15]

    assy = PanelAssembly(panels)

    size = assy.get_size()

    row0 = 0
    col0 = 0
    for p in panels:
        p.row_start = row0
        p.col_start = col0
        row0 += 3*p.m*p.n
        col0 += 3*p.m*p.n
        p.row_end = row0
        p.col_end = col0

    valid_conn = []
    for connecti in conn:
        if connecti.get('has_defect'):
            continue
        valid_conn.append(connecti)

    k0 = assy.calc_k0(conn=valid_conn)
    kM = assy.calc_kM()

    eigvals, eigvecs = freq(k0, kM, tol=0, sparse_solver=True, silent=True,
             sort=True, reduced_dof=False,
             num_eigvalues=25, num_eigvalues_print=5)

    return assy, eigvals, eigvecs
