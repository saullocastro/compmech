from numpy import deg2rad

from compmech.conecyl import ConeCyl

if __name__ == '__main__':
    cc = ConeCyl()
    cc.name = 'z33'
    cc.laminaprop = (123.55e3 , 8.708e3,  0.319, 5.695e3, 5.695e3, 5.695e3)
    cc.stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
    cc.plyt = 0.125
    cc.r2 = 250.
    cc.H = 510.

    cc.alphadeg = 0

    cc.m1 = cc.m2 = cc.n2 = 40
    cc.pdC = False
    cc.Fc = 2000.

    cc.add_SPL(0.4, pt=0.25, theta=deg2rad(-30))
    cc.add_SPL(0.4, pt=0.25, theta=deg2rad(-60))
    cc.add_SPL(0.4, pt=0.25, theta=deg2rad(-90))
    cc.add_SPL(0.3, pt=0.25, theta=deg2rad(-120))
    cc.add_SPL(0.3, pt=0.5, theta=deg2rad(-120))
    cc.add_SPL(0.3, pt=0.75, theta=deg2rad(-120))
    cc.add_SPL(0.4, pt=0.75, theta=deg2rad(-90))
    cc.add_SPL(0.4, pt=0.75, theta=deg2rad(-60))
    cc.add_SPL(0.4, pt=0.75, theta=deg2rad(-30))

    cc.add_SPL(0.5, pt=0.75, theta=deg2rad(30))
    cc.add_SPL(0.4, pt=0.50, theta=deg2rad(30))
    cc.add_SPL(0.6, pt=0.25, theta=deg2rad(30))
    cc.add_SPL(0.7, pt=0.35, theta=deg2rad(50))
    cc.add_SPL(0.4, pt=0.50, theta=deg2rad(75))
    cc.add_SPL(0.7, pt=0.35, theta=deg2rad(100))
    cc.add_SPL(0.5, pt=0.25, theta=deg2rad(120))
    cc.add_SPL(0.3, pt=0.50, theta=deg2rad(120))
    cc.add_SPL(0.3, pt=0.75, theta=deg2rad(120))

    cc.static()

    cc.plot(cc.cs[-1], filename='logo.png')
