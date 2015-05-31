r"""
================================================
CLPT (:mod:`compmech.aero.pistonplate.clpt`)
================================================

.. currentmodule:: compmech.aero.pistonplate.clpt

The ``compmech.aero.pistonplate.clpt`` module contains all the methods and
functions to calculate the stiffness matrices, aerodynamic matrix, mass
matrix, the displacement strain and the stress fields for all models using the
CLPT kinematic assumption.

"""
modules = [
          'clpt_commons_free',
          'clpt_donnell_free_linear',
          #'clpt_donnell_bc4_nonlinear',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))


