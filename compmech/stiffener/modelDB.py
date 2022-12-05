r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.stiffener.modelDB


"""
from . models import *
from compmech.panel.models import (cpanel_clt_donnell_bardell,
                                   clt_bardell_field,
                                   plate_clt_donnell_bardell,
                                   clt_bardell_field)
db = {
    'bladestiff1d_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'matrices': bladestiff1d_clt_donnell_bardell,
                    'dofs': 3,
                    'e_num': 6,
                    'num1': 3,
                    },
    'bladestiff2d_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'connections': bladestiff2d_clt_donnell_bardell,
                    },
    'tstiff2d_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'connections': tstiff2d_clt_donnell_bardell,
                    },
    }
