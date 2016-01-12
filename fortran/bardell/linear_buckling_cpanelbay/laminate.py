import sys
sys.path.append(r'C:\repos\compmech')

import numpy as np

from compmech.composite.laminate import read_stack
from compmech.plates import Plate

stack = [-45, +45, 0, 0, -30, +30, +30, -30, 0, 0, +45, -45]
# prop format (E1, E2, nu12, G12, G13, G23)
prop = (152.4e3, 8.8e3, 0.31, 4.9e3, 4.9e3, 4.9e3)
lam = read_stack(stack, plyts=[0.125 for i in stack],
                 laminaprops=[prop for i in stack])


np.savetxt('laminate_ABD_matrix.txt', lam.ABD, fmt='%1.8f')
