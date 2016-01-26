from compmech.stiffpanelbay import StiffPanelBay

spb = StiffPanelBay()
spb.a = 2.
spb.b = 1.
spb.r = 2.
spb.stack = [0]
spb.plyt = 1e-3*1.6
E = 71.e9
nu = 0.33
G = E/(2.*(1. + nu))
spb.laminaprop = (E, E, nu, G, G, G)
spb.model = 'cpanel_clt_donnell_bardell'
spb.m = 22
spb.n = 22

spb.u1tx = 0.
spb.u1rx = 1.
spb.u2tx = 0.
spb.u2rx = 1.
spb.u1ty = 0.
spb.u1ry = 1.
spb.u2ty = 0.
spb.u2ry = 1.
spb.v1tx = 0.
spb.v1rx = 1.
spb.v2tx = 0.
spb.v2rx = 1.
spb.v1ty = 0.
spb.v1ry = 1.
spb.v2ty = 0.
spb.v2ry = 1.
spb.w1tx = 0.
spb.w1rx = 1.
spb.w2tx = 0.
spb.w2rx = 1.
spb.w1ty = 0.
spb.w1ry = 1.
spb.w2ty = 0.
spb.w2ry = 1.

spb.add_panel(y1=0, y2=spb.b, Nxx=-1.)

spb.lb(silent=False)
