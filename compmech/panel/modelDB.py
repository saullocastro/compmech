r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.panel.modelDB


"""
from . models import *


db = {
    'kpanel_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'field': clt_bardell_field,
                    'matrices': kpanel_clt_donnell_bardell,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    },
    'cpanel_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clt_bardell_field,
                    'matrices': cpanel_clt_donnell_bardell,
                    'matrices_num': cpanel_clt_donnell_bardell_num,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    },
    'plate_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clt_bardell_field,
                    'matrices': plate_clt_donnell_bardell,
                    'matrices_num': plate_clt_donnell_bardell_num,
                    'dofs': 3,
                    'e_num': 6,
                    'num': 3,
                    },
    'plate_clt_donnell_bardell_w': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'field': clt_bardell_field_w,
                    'matrices': plate_clt_donnell_bardell_w,
                    'dofs': 1,
                    'e_num': 6,
                    'num': 1,
                    },
    }
