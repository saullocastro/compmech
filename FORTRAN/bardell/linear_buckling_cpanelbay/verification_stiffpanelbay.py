from compmech.stiffpanelbay import StiffPanelBay

spb = StiffPanelBay()
spb.a = 1.
spb.b = 0.5
spb.stack = [0, 90, 90, 0, -45, +45]
spb.plyt = 1e-3*0.125
spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
spb.model = 'plate_clt_donnell_bardell'
spb.mu = 1.3e3
spb.m = 15
spb.n = 16

spb.add_panel(y1=0, y2=spb.b/2., plyt=spb.plyt, Nxx=-1.)
spb.add_panel(y1=spb.b/2., y2=spb.b, plyt=spb.plyt, Nxx_cte=1000.)

spb.add_bladestiff1d(ys=spb.b/2., Fx=-10., bf=0.05, fstack=[0, 90, 90, 0],
        fplyt=spb.plyt, flaminaprop=spb.laminaprop)

spb.lb(silent=False)
