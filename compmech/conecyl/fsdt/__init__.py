r"""
==========================================================================
First-order Shear Deformation Theory (FSDT) (:mod:`compmech.conecyl.fsdt`)
==========================================================================

.. currentmodule:: compmech.conecyl.fsdt

The ``compmech.conecyl.fsdt`` module contains all the methods and
functions to calculate the stiffness matrices, the displacement field, the
strain and the stress fields for all models using the FSDT kinematic
assumption.

"""
modules = [
          'fsdt_commons_bc1',
          'fsdt_commons_bc2',
          'fsdt_commons_bc3',
          'fsdt_commons_bc4',
          'fsdt_commons_bcn',

          'fsdt_donnell_bc1_linear',
          'fsdt_donnell_bc2_linear',
          'fsdt_donnell_bc3_linear',
          'fsdt_donnell_bc4_linear',
          'fsdt_donnell_bcn_linear',

          'fsdt_sanders_bcn_linear',

          'fsdt_shadmehri2012_bc2',
          'fsdt_shadmehri2012_bc3',

          'fsdt_geier1997_bc2',

          'fsdt_donnell_bc1_nonlinear',
          'fsdt_donnell_bc2_nonlinear',
          'fsdt_donnell_bc3_nonlinear',
          'fsdt_donnell_bc4_nonlinear',
          'fsdt_donnell_bcn_nonlinear',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))


