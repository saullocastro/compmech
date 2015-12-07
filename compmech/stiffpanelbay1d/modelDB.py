r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.stiffpanelbay1d.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *

db = {
    'clpt_donnell_bc1': {
                    'linear static': False,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_bc1,
                    'linear': clpt_donnell_bc1_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 0,
                    'num1': 3,
                    },
    }
