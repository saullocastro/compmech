import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt

L = 1000.
xs = np.linspace(-L/2., +L/2., 1000)
bx = (xs+L/2)/L
ys = np.zeros_like(xs)
for i in range(1, 21):
    ys += (-1 + 2*np.random.random())*cos(i*pi*bx)
plt.plot(xs, ys)
plt.savefig('test2.png')




