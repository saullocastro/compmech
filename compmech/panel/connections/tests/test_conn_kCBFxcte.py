'''
Test description
================

This test verifies the connection kCBFxcte.
It is built two identical panels using different connections.

panel_BFycte: 
    Closed section using the existing BFycte connection.

panel_BFxcte: 
    Closed section using the BFxcte connection. 
    This model is equivalent the "panel_BFycte", except it used loads, connections and BCs
    in the xcte direction.

The first eigen value of each model are compared in this test and a reference NASTRAN model
"reference_model_BFxcte.dat" is used to double check the results. Compmech models presents an error 
less than 1% compared with NASTRAN.
'''

import pytest
import numpy as np

from compmech.panel import Panel
from compmech.panel.assembly import PanelAssembly
from compmech.analysis import lb, static


@pytest.fixture
def eig_value_panel_BFycte():
    '''
    Closed section using BFYcte connections

    returns
    -------
        First eigenvalue of the assembly.   
    '''

    # Properties
    E1 = 127560 # MPa
    E2 = 13030. # MPa
    G12 = 6410. # MPa
    nu12 = 0.3
    ply_thickness = 0.127 # mm

    # Plate dimensions
    aB = 1181.1
    bB = 746.74

    # Spar L
    aL = 1181.1
    bL = 381.0

    #others
    m = 10
    n = 10

    simple_layup = [+45, -45]*20 + [0, 90]*20
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)

    # skin panels
    B1 = Panel(group='B1', a=aB, b=bB,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    B2 = Panel(group='B2', a=aB, b=bB,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # spar
    L1 = Panel(group='L1', a=aL, b=bL, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    L2 = Panel(group='L2', a=aL, b=bL, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # boundary conditions
    B1.u1tx = 1 ; B1.u1rx = 1 ; B1.u2tx = 1 ; B1.u2rx = 1
    B1.v1tx = 1 ; B1.v1rx = 1 ; B1.v2tx = 1 ; B1.v2rx = 1
    B1.w1tx = 1 ; B1.w1rx = 1 ; B1.w2tx = 1 ; B1.w2rx = 1
    B1.u1ty = 1 ; B1.u1ry = 1 ; B1.u2ty = 1 ; B1.u2ry = 1
    B1.v1ty = 1 ; B1.v1ry = 1 ; B1.v2ty = 1 ; B1.v2ry = 1
    B1.w1ty = 1 ; B1.w1ry = 1 ; B1.w2ty = 1 ; B1.w2ry = 1

    B2.u1tx = 1 ; B2.u1rx = 1 ; B2.u2tx = 1 ; B2.u2rx = 1
    B2.v1tx = 1 ; B2.v1rx = 1 ; B2.v2tx = 1 ; B2.v2rx = 1
    B2.w1tx = 1 ; B2.w1rx = 1 ; B2.w2tx = 1 ; B2.w2rx = 1
    B2.u1ty = 1 ; B2.u1ry = 1 ; B2.u2ty = 1 ; B2.u2ry = 1
    B2.v1ty = 1 ; B2.v1ry = 1 ; B2.v2ty = 1 ; B2.v2ry = 1
    B2.w1ty = 1 ; B2.w1ry = 1 ; B2.w2ty = 1 ; B2.w2ry = 1

    L1.u1tx = 0 ; L1.u1rx = 0 ; L1.u2tx = 1 ; L1.u2rx = 1
    L1.v1tx = 0 ; L1.v1rx = 0 ; L1.v2tx = 1 ; L1.v2rx = 1
    L1.w1tx = 0 ; L1.w1rx = 0 ; L1.w2tx = 1 ; L1.w2rx = 1
    L1.u1ty = 1 ; L1.u1ry = 1 ; L1.u2ty = 1 ; L1.u2ry = 1
    L1.v1ty = 1 ; L1.v1ry = 1 ; L1.v2ty = 1 ; L1.v2ry = 1
    L1.w1ty = 1 ; L1.w1ry = 1 ; L1.w2ty = 1 ; L1.w2ry = 1

    L2.u1tx = 0 ; L2.u1rx = 0 ; L2.u2tx = 1 ; L2.u2rx = 1
    L2.v1tx = 0 ; L2.v1rx = 0 ; L2.v2tx = 1 ; L2.v2rx = 1
    L2.w1tx = 0 ; L2.w1rx = 0 ; L2.w2tx = 1 ; L2.w2rx = 1
    L2.u1ty = 1 ; L2.u1ry = 1 ; L2.u2ty = 1 ; L2.u2ry = 1
    L2.v1ty = 1 ; L2.v1ry = 1 ; L2.v2ty = 1 ; L2.v2ry = 1
    L2.w1ty = 1 ; L2.w1ry = 1 ; L2.w2ty = 1 ; L2.w2ry = 1

    # Assembly
    conn = [   
        dict(p1=B1, p2=L1, func='BFycte', ycte1=0, ycte2=L1.b), #LB
        dict(p1=B1, p2=L2, func='BFycte', ycte1=B1.b, ycte2=L2.b),
        dict(p1=B2, p2=L1, func='BFycte', ycte1=0, ycte2=0), #LB
        dict(p1=B2, p2=L2, func='BFycte', ycte1=B1.b, ycte2=0),
    ]

    panels = [B1, B2, L1, L2]
    assy = PanelAssembly(panels, conn)
    k0 = assy.calc_k0()

    #Static load case
    size = sum([3*p.m*p.n for p in panels])
    fext = np.zeros(size)

    c = None

    L1.add_force(L1.a, L1.b/2, 0, 90010, 0)
    L2.add_force(L2.a, L1.b/2, 0, 187888, 0)
    fext[L1.col_start: L1.col_end] = L1.calc_fext(silent=True)
    fext[L2.col_start: L2.col_end] = L2.calc_fext(silent=True)

    incs, cs = static(k0, fext, silent=True)

    kG = assy.calc_kG0(c=cs[0])

    #Buckling load case
    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
         num_eigvalues=25, num_eigvalues_print=5)
    
    return eigvals[0]


@pytest.fixture
def eig_value_panel_BFxcte():
    '''
    Closed section using BFxcte connections

    returns
    -------
        First eigenvalue of the assembly.   
    '''

    # Properties
    E1 = 127560 # MPa
    E2 = 13030. # MPa
    G12 = 6410. # MPa
    nu12 = 0.3
    ply_thickness = 0.127 # mm

    # Plate dimensions
    bB = 1181.1 # inverted with relation to the BFycte panel
    aB = 746.74

    # Spar L
    bT = 1181.1 # inverted with relation to the BFycte panel
    aT = 381.0

    #others
    m = 10
    n = 10

    simple_layup = [+45, -45]*20 + [90, 0]*20 # 90 and 0 inverted
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)

    # skin panels
    B1 = Panel(group='B1', a=aB, b=bB,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    B2 = Panel(group='B2', a=aB, b=bB,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # spar
    T1 = Panel(group='T1', a=aT, b=bT, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    T2 = Panel(group='T2', a=aT, b=bT, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # boundary conditions
    B1.u1tx = 1 ; B1.u1rx = 1 ; B1.u2tx = 1 ; B1.u2rx = 1
    B1.v1tx = 1 ; B1.v1rx = 1 ; B1.v2tx = 1 ; B1.v2rx = 1
    B1.w1tx = 1 ; B1.w1rx = 1 ; B1.w2tx = 1 ; B1.w2rx = 1
    B1.u1ty = 1 ; B1.u1ry = 1 ; B1.u2ty = 1 ; B1.u2ry = 1
    B1.v1ty = 1 ; B1.v1ry = 1 ; B1.v2ty = 1 ; B1.v2ry = 1
    B1.w1ty = 1 ; B1.w1ry = 1 ; B1.w2ty = 1 ; B1.w2ry = 1

    B2.u1tx = 1 ; B2.u1rx = 1 ; B2.u2tx = 1 ; B2.u2rx = 1
    B2.v1tx = 1 ; B2.v1rx = 1 ; B2.v2tx = 1 ; B2.v2rx = 1
    B2.w1tx = 1 ; B2.w1rx = 1 ; B2.w2tx = 1 ; B2.w2rx = 1
    B2.u1ty = 1 ; B2.u1ry = 1 ; B2.u2ty = 1 ; B2.u2ry = 1
    B2.v1ty = 1 ; B2.v1ry = 1 ; B2.v2ty = 1 ; B2.v2ry = 1
    B2.w1ty = 1 ; B2.w1ry = 1 ; B2.w2ty = 1 ; B2.w2ry = 1

    T1.u1tx = 1 ; T1.u1rx = 1 ; T1.u2tx = 1 ; T1.u2rx = 1
    T1.v1tx = 1 ; T1.v1rx = 1 ; T1.v2tx = 1 ; T1.v2rx = 1
    T1.w1tx = 1 ; T1.w1rx = 1 ; T1.w2tx = 1 ; T1.w2rx = 1
    T1.u1ty = 0 ; T1.u1ry = 0 ; T1.u2ty = 1 ; T1.u2ry = 1
    T1.v1ty = 0 ; T1.v1ry = 0 ; T1.v2ty = 1 ; T1.v2ry = 1
    T1.w1ty = 0 ; T1.w1ry = 0 ; T1.w2ty = 1 ; T1.w2ry = 1

    T2.u1tx = 1 ; T2.u1rx = 1 ; T2.u2tx = 1 ; T2.u2rx = 1
    T2.v1tx = 1 ; T2.v1rx = 1 ; T2.v2tx = 1 ; T2.v2rx = 1
    T2.w1tx = 1 ; T2.w1rx = 1 ; T2.w2tx = 1 ; T2.w2rx = 1
    T2.u1ty = 0 ; T2.u1ry = 0 ; T2.u2ty = 1 ; T2.u2ry = 1
    T2.v1ty = 0 ; T2.v1ry = 0 ; T2.v2ty = 1 ; T2.v2ry = 1
    T2.w1ty = 0 ; T2.w1ry = 0 ; T2.w2ty = 1 ; T2.w2ry = 1

    # Assembly
    conn = [   
        dict(p1=B1, p2=T1, func='BFxcte', xcte1=0, xcte2=T1.a), #TB
        dict(p1=B1, p2=T2, func='BFxcte', xcte1=B1.a, xcte2=T2.a),
        dict(p1=B2, p2=T1, func='BFxcte', xcte1=0, xcte2=0),
        dict(p1=B2, p2=T2, func='BFxcte', xcte1=B2.a, xcte2=0),
    ]
    panels = [B1, B2, T1, T2]
    assy = PanelAssembly(panels, conn)
    k0 = assy.calc_k0()

    #Static load case
    size = sum([3*p.m*p.n for p in panels])
    fext = np.zeros(size)

    c = None

    T1.add_force(T1.a/2, T1.b, -90010, 0, 0)
    T2.add_force(T2.a/2, T1.b, -187888, 0, 0)
    fext[T1.col_start: T1.col_end] = T1.calc_fext(silent=True)
    fext[T2.col_start: T2.col_end] = T2.calc_fext(silent=True)

    incs, cs = static(k0, fext, silent=True)

    kG = assy.calc_kG0(c=cs[0])

    #Buckling load case
    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
         num_eigvalues=25, num_eigvalues_print=5)
    
    return eigvals[0]


def test_kCBFxte(eig_value_panel_BFycte, eig_value_panel_BFxcte):
    '''
    This test compare the first eigenvalue of the assemblies.
    They cannot present an error higher than 1%.
    '''
    assert np.isclose(eig_value_panel_BFycte, eig_value_panel_BFycte, atol=0.01, rtol=0.01)
    


