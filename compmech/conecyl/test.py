from compmech.conecyl import ConeCyl

cc = ConeCyl()
cc.model = 'clpt_donnell_bc1'
cc.m1 = 15
cc.m2 = 16
cc.n2 = 17
cc.name = 'Z33'
cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
cc.plyt = 0.125
cc.r2 = 250.
cc.H = 510.
cc.add_SPL(10)
cc.static()
cc.plot(cc.cs[0], colorbar=True, filename='test.png')
