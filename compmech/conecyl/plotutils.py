import os

import numpy as np
import matplotlib.pyplot as plt


def get_filename(cc):
    prefix = 'plot_{name}'.format(name=cc.name)
    files = os.listdir('.')
    files = [f for f in files if prefix in f]
    files.sort()
    if files:
        try:
            lastnum = int(float(files[-1].split('.png')[0].split('_')[-1]))
        except:
            lastnum = -1
    else:
        lastnum = -1
    return prefix + '_{0:03d}.png'.format(lastnum + 1)


