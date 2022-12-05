r"""
=======================================================================
Classical Laminated Plate Theory (CLPT) (:mod:`compmech.conecyl.clpt`)
=======================================================================

.. currentmodule:: compmech.conecyl.clpt

The ``compmech.conecyl.clpt`` module contains all the methods and
functions to calculate the stiffness matrices, the displacement field, the
strain and the stress fields for all models using the CLPT kinematic
assumption.

Though the shell rotations about `x` (`\phi_x`) and `\theta` (`\phi_theta`)
are not variables necessary to solve the displacement field for the CLPT,
their values are calculated using the function `fuvw()` using the partial
derivatives `- \tfrac{\partial w}{\partial x}` and
`\tfrac{\partial w}{\partial \theta}`, respectively.


"""
modules = [
          'clpt_commons_bc1',
          'clpt_commons_bc2',
          'clpt_commons_bc3',
          'clpt_commons_bc4',
          'clpt_commons_bcn',

          'clpt_donnell_bc1_linear',
          'clpt_donnell_bc2_linear',
          'clpt_donnell_bc3_linear',
          'clpt_donnell_bc4_linear',
          'clpt_donnell_bcn_linear',
          'clpt_geier1997_bc2',
          'iso_clpt_donnell_bc2_linear',
          'iso_clpt_donnell_bc3_linear',

          'clpt_sanders_bc1_linear',
          'clpt_sanders_bc2_linear',
          'clpt_sanders_bc3_linear',
          'clpt_sanders_bc4_linear',

          'clpt_donnell_bc1_nonlinear',
          'clpt_donnell_bc2_nonlinear',
          'clpt_donnell_bc3_nonlinear',
          'clpt_donnell_bc4_nonlinear',
          'clpt_sanders_bc1_nonlinear',
          'clpt_sanders_bc2_nonlinear',
          'clpt_sanders_bc3_nonlinear',
          'clpt_sanders_bc4_nonlinear',
          'iso_clpt_donnell_bc2_nonlinear',
          'iso_clpt_donnell_bc3_nonlinear',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))
