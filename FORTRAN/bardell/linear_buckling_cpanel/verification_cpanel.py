from compmech.stiffpanelbay import StiffPanelBay

spb = StiffPanelBay()
spb.a = 2.
spb.b = 1.
spb.r = 2.
spb.stack = [0, 90, 90, 0, -45, +45]
spb.plyt = 1e-3*0.125
spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
spb.model = 'cpanel_clt_donnell_bardell'
spb.m = 15
spb.n = 16

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
