from __future__ import division, absolute_import

import numpy as np
from scipy.sparse import csr_matrix

from compmech.panel import Panel
from compmech.panel.assembly import PanelAssembly
from compmech.sparse import make_symmetric
from compmech.analysis import lb, static


def create_cylinder_assy(height, r, stack, plyt, laminaprop,
        npanels, m=8, n=8):
    skin = []
    perimiter = 2*np.pi*r
    b_skin = perimiter / npanels
    for i in range(npanels):
        y0 = i*b_skin
        panel = Panel(group='skin', x0=0, y0=y0, a=height, b=b_skin,
            r=r, m=m, n=n, plyt=plyt, stack=stack, laminaprop=laminaprop,
            u1tx=0, u1rx=1, u2tx=0, u2rx=1,
            v1tx=0, v1rx=1, v2tx=0, v2rx=1,
            w1tx=0, w1rx=1, w2tx=0, w2rx=1,
            u1ty=1, u1ry=1, u2ty=1, u2ry=1,
            v1ty=1, v1ry=1, v2ty=1, v2ry=1,
            w1ty=1, w1ry=1, w2ty=1, w2ry=1)
        skin.append(panel)
    conn_dict = []
    skin_loop = skin + [skin[0]]
    for i in range(len(skin)):
        if i != len(skin) - 1:
            p01 = skin_loop[i]
            p02 = skin_loop[i+1]
            conn_dict.append(dict(p1=p01, p2=p02, func='SSycte', ycte1=p01.b, ycte2=0))
        else:
            p01 = skin_loop[i+1]
            p02 = skin_loop[i]
            conn_dict.append(dict(p1=p01, p2=p02, func='SSycte', ycte1=0, ycte2=p02.b))

    assy = PanelAssembly(skin)

    row0 = 0
    col0 = 0
    for p in skin:
        p.row_start = row0
        p.col_start = col0
        row0 += 3*p.m*p.n
        col0 += 3*p.m*p.n
        p.row_end = row0
        p.col_end = col0

    return assy, conn_dict


def cylinder_compression_lb_Nxx_cte(height, r, stack, plyt, laminaprop,
        npanels, Nxxs, m=8, n=8):
    assy, conn_dict = create_cylinder_assy(height=height, r=r, stack=stack, plyt=plyt,
            laminaprop=laminaprop, npanels=npanels, m=m, n=n)
    for i, p in enumerate(assy.panels):
        p.Nxx = Nxxs[i]

    k0 = assy.calc_k0(conn_dict)
    kG = assy.calc_kG0()
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=20, num_eigvalues_print=5)
    return assy, eigvals, eigvecs


def cylinder_compression_lb_Nxx_from_static(height, r, stack, plyt, laminaprop,
        npanels, Nxxs, m=8, n=8):
    assy, conn_dict = create_cylinder_assy(height=height, r=r, stack=stack, plyt=plyt,
            laminaprop=laminaprop, npanels=npanels, m=m, n=n)
    for i, p in enumerate(assy.panels):
        p.Nxx = Nxxs[i]
        p.u2tx = 1
    fext = np.zeros(assy.get_size())
    for p in assy.panels:
        Nforces = 1000
        fx = p.Nxx*p.b/(Nforces-1.)
        for i in range(Nforces):
            y = i*p.b/(Nforces-1.)
            p.add_force(p.a, y, fx, 0, 0)
        fext[p.col_start: p.col_end] = p.calc_fext(silent=True)

    k0 = assy.calc_k0(conn_dict)
    incs, cs = static(k0, fext, silent=True)
    c = cs[0]
    kG = assy.calc_kG0(c=c)

    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=20, num_eigvalues_print=5)

    return assy, c, eigvals, eigvecs


def TBD():
    #skin[1].forces.append([p.a/2, p.b/2, 0, 0, -10])
    fext = np.zeros(size)
    for p in skin:
        Nforces = 1000
        fx = p.Nxx*p.b/(Nforces-1.)
        for i in range(Nforces):
            y = i*p.b/(Nforces-1.)
            p.add_force(p.a, y, fx, 0, 0)
        fext[p.col_start: p.col_end] = p.calc_fext(silent=True)

    incs, cs = static(k0, fext, silent=True)
    c = cs[0]
    kG = assy.calc_kG0(c=c)

    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=20, num_eigvalues_print=5)

    return assy, c, eigvals, eigvecs
