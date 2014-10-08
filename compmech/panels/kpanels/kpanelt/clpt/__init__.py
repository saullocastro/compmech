r"""
=======================================================================
Classical Laminated Plate Theory (CLPT) (:mod:`compmech.panels.clpt`)
=======================================================================

.. currentmodule:: compmech.conecyl.panels

The ``compmech.conecyl.panels`` module contains all the methods and
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

          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))
