r"""
=========================================
FSDT (:mod:`compmech.panels.kpanel.fsdt`)
=========================================

.. currentmodule:: compmech.panels.kpanel.fsdt

The ``compmech.panels.kpanel.fsdt`` module contains all the methods and
functions to calculate the stiffness matrices, the displacement field, the
strain and the stress fields for all models using the FSDT kinematic
assumption.

"""
modules = [
          'fsdt_commons_bc1',
          'fsdt_commons_free',

          'fsdt_donnell_bc1_linear',
          'fsdt_donnell_free_linear',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))


