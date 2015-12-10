r"""
================================================
CLPT (:mod:`compmech.stiffener.models`)
================================================

.. currentmodule:: compmech.stiffener.models

"""
modules = [
          'bladestiff1d_clt_donnell_bardell',
          'bladestiff2d_clt_donnell_bardell',
          ]

for module in modules:
    try:
        exec('from . import {0}'.format(module))
    except:
        print('WARNING - module {0} could not be imported!'.format(module))
        exec('{0} = None'.format(module))

