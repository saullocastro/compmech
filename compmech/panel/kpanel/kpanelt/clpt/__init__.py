r"""
=========================================
CLPT (:mod:`compmech.panels.kpanel.clpt`)
=========================================

.. currentmodule:: compmech.panels.kpanel.clpt

The ``compmech.panels.kpanel.clpt`` module contains all the methods and
functions to calculate the stiffness matrices, the displacement field, the
strain and the stress fields for all models using the CLPT kinematic
assumption.

"""
modules = [
          'clpt_commons_bc4',
          'clpt_donnell_bc4_linear',
          'clpt_donnell_bc4_nonlinear',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))


