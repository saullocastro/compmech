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
- empty - functions to integrate the stiffness matrices
- num - functions to numerically integrate the stiffness matrices

"""
module_names = [
          'clt_bardell_field',
          'clt_bardell_field_w',
          'cpanel_clt_donnell_bardell',
          'cpanel_clt_donnell_bardell_num',
          'kpanel_clt_donnell_bardell',
          'plate_clt_donnell_bardell',
          'plate_clt_donnell_bardell_num',
          'plate_clt_donnell_bardell_w',
          ]

for module_name in module_names:
    exec('from . import {0}'.format(module_name))
