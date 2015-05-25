import numpy as np
from numpy import pi, sin, cos
from numpy.random import random
import matplotlib.pyplot as plt

num = 500
bx = np.linspace(0, 1, num)
by = np.linspace(0, 1, num)
bx, by = np.meshgrid(bx, by, copy=False)

m = 10
n = 10
vs = np.zeros((num, num))
for i in range(1, m+1):
    for j in range(1, n+1):
        vs += (-1 + 2*random())*sin(i*pi*bx)*cos(j*pi*by)

levels = np.linspace(vs.min(), vs.max(), 200)
plt.contourf(bx, by, vs, levels=levels)
plt.savefig('plate_bc4.png')
