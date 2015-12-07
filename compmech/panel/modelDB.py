r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.panel.modelDB


"""
from models import *

db = {
    'plate_clpt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'field': plate_clpt_donnell_bardell_field,
                    'matrices': plate_clpt_donnell_bardell,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    },
    }
