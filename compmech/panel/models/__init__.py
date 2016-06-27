r"""
=============================================
Panel models (:mod:`compmech.panel.models`)
=============================================

.. currentmodule:: compmech.panel.models

The first part of the module identification is:

- cpanel - Cylindrical Panels
- kpanel - Conical Panels
- plate - Flat plates

The second part:

- clt - Classical Laminated Theory
- fsdt - First-order Shear Deformation Theory

Third part:

- Donnell - kinematic equations using Donnell's terms
- Sanders - kinematic equations using Sanders' terms

Fourth part:
- field - functions to calculate field variables
- empty - functions to calculate the stiffness matrices

"""
from glob import glob
from os.path import realpath, dirname, join
import inspect
import imp

module_names = [
          'cpanel_clt_donnell_bardell_field',
          'cpanel_clt_donnell_bardell',
          'cpanel_clt_donnell_bardell_num',
          #'kpanel_clt_donnell_bardell_field',
          'kpanel_clt_donnell_bardell',
          'plate_clt_donnell_bardell_field',
          'plate_clt_donnell_bardell',
          'plate_clt_donnell_bardell_num',
          'plate_clt_donnell_bardell_w_field',
          'plate_clt_donnell_bardell_w',
          ]

for module_name in module_names:
    exec('from . import {0}'.format(module_name))
