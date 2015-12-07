r"""
=========================================
CLPT (:mod:`compmech.panel.cpanel.clpt`)
=========================================

.. currentmodule:: compmech.panel.cpanel.clpt

The ``compmech.panel.cpanel.clpt`` module contains all the methods and
functions to calculate the stiffness matrices, the displacement field, the
strain and the stress fields for all models using the CLPT kinematic
assumption.

"""
modules = [
          'clpt_commons_bardell',
          'clpt_donnell_bardell_linear',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))


