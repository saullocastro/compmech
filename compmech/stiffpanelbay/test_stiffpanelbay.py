from compmech.stiffpanelbay import StiffPanelBay

sp = StiffPanelBay()
sp.a = 1.
sp.b = 0.5
sp.stack = [0, 90, -45, +45]
sp.plyt = 1e-3*0.125
sp.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
sp.model = 'cpanel_clt_donnell_bardell'

sp.add_panel(0, sp.b/2., Nxx=-1)
sp.add_panel(sp.b/2., sp.b, plyt=sp.plyt/2., Nxx=-1)

sp.lb()



