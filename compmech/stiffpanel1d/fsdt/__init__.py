r"""
================================================
CLPT (:mod:`compmech.aero.pistonstiffpanel.fsdt`)
================================================

.. currentmodule:: compmech.aero.pistonstiffpanel.fsdt

The ``compmech.aero.pistonstiffpanel.fsdt`` module contains all the methods
and functions to calculate the stiffness matrices, aerodynamic matrix, mass
matrix, the displacement strain and the stress fields for all models using the
CLPT kinematic assumption.

"""
modules = [
          'fsdt_commons_bc1',

          'fsdt_donnell_bc1_linear',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))


