r"""
=====================================================
CLPT (:mod:`compmech.panels.kpanels.kpanelcheb.clpt`)
=====================================================

.. currentmodule:: compmech.panels.kpanels.kpanelcheb.clpt

The ``compmech.panels.kpanels.kpanelcheb.clpt`` module contains all the
methods and functions to calculate the stiffness matrices, the displacement
field, the strain and the stress fields for all models using the CLPT
kinematic assumption.

"""
modules = [
          'clpt_commons_cheb',
          'clpt_donnell_cheb_linear',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))


