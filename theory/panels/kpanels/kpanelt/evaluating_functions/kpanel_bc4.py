import numpy as np
from numpy import pi, sin, cos
from numpy.random import random
import matplotlib.pyplot as plt

bx = np.linspace(0, 1, 1000)
bt = np.linspace(0, 1, 1000)
bx, bt = np.meshgrid(bx, bt, copy=False)

m = 10
n = 10
vs = np.zeros((1000, 1000))
for i in range(1, m+1):
    for j in range(1, n+1):
        vs += (-1 + 2*random())*cos(i*pi*bx)*cos(j*pi*bt)

levels = np.linspace(vs.min(), vs.max(), 1000)
plt.contour(bx, bt, vs, levels=levels)
plt.savefig('kpanel_bc4.png')
