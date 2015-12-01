r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.aero.pistonstiffpanel.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *

db = {
    'clpt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_bardell,
                    'linear': clpt_donnell_bardell_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    'num1': 3,
                    },
    }
