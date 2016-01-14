from compmech.stiffpanelbay import StiffPanelBay

spb = StiffPanelBay()
spb.a = 2.
spb.b = 1.
spb.r = 10.
spb.stack = [0, 90, 90, 0, -45, +45]
spb.plyt = 1e-3*0.125
spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
spb.model = 'cpanel_clt_donnell_bardell'
spb.mu = 1.3e3
spb.m = 13
spb.n = 14

spb.add_panel(y1=0, y2=spb.b/3., plyt=spb.plyt, Nxx=-100.)
spb.add_panel(y1=spb.b/3., y2=2*spb.b/3., plyt=spb.plyt, Nxx=-100.)
spb.add_panel(y1=2*spb.b/3., y2=spb.b, plyt=spb.plyt, Nxx=-100.)

spb.add_bladestiff1d(ys=spb.b/3., Fx=-100., bf=0.05,
                     fstack=[0, 90, 90, 0]*4, fplyt=spb.plyt,
                     flaminaprop=spb.laminaprop)
spb.add_bladestiff1d(ys=2*spb.b/3., Fx=-100., bf=0.05,
                     fstack=[0, 90, 90, 0]*4, fplyt=spb.plyt,
                     flaminaprop=spb.laminaprop)

spb.lb(silent=False)

print 'Fx', spb.bladestiff1ds[0].Fx
print 'ys', spb.bladestiff1ds[0].ys
print 'bf', spb.bladestiff1ds[0].bf
print 'df', spb.bladestiff1ds[0].df
print 'E1', spb.bladestiff1ds[0].E1
print 'F1', spb.bladestiff1ds[0].F1
print 'S1', spb.bladestiff1ds[0].S1
print 'Jxx', spb.bladestiff1ds[0].Jxx

print spb.k0.sum()
print spb.kG0.sum()

