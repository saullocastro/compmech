r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.panel.modelDB


"""
from models import *

db = {
    'cpanel_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'field': plate_clt_donnell_bardell_field,
                    'matrices': cpanel_clt_donnell_bardell,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    },
    'plate_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'field': plate_clt_donnell_bardell_field,
                    'matrices': plate_clt_donnell_bardell,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    },
    'plate_clt_donnell_bardell_w': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'field': plate_clt_donnell_bardell_w_field,
                    'matrices': plate_clt_donnell_bardell_w,
                    'dofs': 1,
                    'e_num': 6,
                    'num': 1,
                    },
    }
