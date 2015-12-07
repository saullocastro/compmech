r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.stiffener.modelDB


"""
from models import *

db = {
    'bladestiff2d_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'matrices': bladestiff2d_clt_donnell_bardell,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    },
    }
