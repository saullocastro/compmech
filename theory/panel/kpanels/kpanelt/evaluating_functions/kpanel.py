import numpy as np
from numpy import sin
from numpy.linalg import lstsq
import matplotlib.pyplot as plt


xdata = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14.])
ydata = np.array([0, 0, 0, -1, -2, -4, -8, -16, -8, -4, -2, -1, 0, 0, 0.])

m = 10
a = np.array([[1*sin(i*np.pi*(xdata-0)/(14-0))] for i in range(1, m+1)])
a = a.swapaxes(0,2).swapaxes(1,2).reshape(xdata.shape[0],-1)
c0, residues, rank, s = lstsq(a, ydata)

xs = np.linspace(0, 14, 1000)
a = np.array([[1*sin(i*np.pi*(xs-0)/(14-0))] for i in range(1, m+1)])
a = a.swapaxes(0,2).swapaxes(1,2).reshape(xs.shape[0],-1)
print plt.plot(xdata, ydata, label='data')
print plt.plot(xs, a.dot(c0), label='adjusted')
plt.savefig('test.png')

print a.shape



