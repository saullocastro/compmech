r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.panel.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *
#from fsdt import *

db = {
    'plate_clpt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'field': plate_donnell_bardell_field,
                    'matrices': plate_donnell_bardell,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    },
    }
