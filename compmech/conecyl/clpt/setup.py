from __future__ import division, print_function, absolute_import

import os
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    if os.name == 'nt':
        args_linear = ['/openmp']
        args_nonlinear = ['/openmp', '/fp:fast']
    else:
        args_linear = ['-fopenmp']
        args_nonlinear = ['-fopenmp', '-ffast-math']

    config = Configuration('clpt', parent_package, top_path)
    config.add_extension('clpt_commons_bc1', ['clpt_commons_bc1.pyx'],
              extra_compile_args=args_linear,
              depends=['../imperfections/mgi.pyx'])
    config.add_extension('clpt_commons_bc2', ['clpt_commons_bc2.pyx'],
              extra_compile_args=args_linear,
              depends=['../imperfections/mgi.pyx'])
    config.add_extension('clpt_commons_bc3', ['clpt_commons_bc3.pyx'],
              extra_compile_args=args_linear,
              depends=['../imperfections/mgi.pyx'])
    config.add_extension('clpt_commons_bc4', ['clpt_commons_bc4.pyx'],
              extra_compile_args=args_linear,
              depends=['../imperfections/mgi.pyx'])

    config.add_extension('clpt_donnell_bc1_linear',
              ['clpt_donnell_bc1_linear.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('clpt_donnell_bc2_linear',
              ['clpt_donnell_bc2_linear.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('clpt_donnell_bc3_linear',
              ['clpt_donnell_bc3_linear.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('clpt_donnell_bc4_linear',
              ['clpt_donnell_bc4_linear.pyx'],
              extra_compile_args=args_linear)

    config.add_extension('clpt_geier1997_bc2',
              ['clpt_geier1997_bc2.pyx'],
              extra_compile_args=args_linear)

    config.add_extension('clpt_donnell_bc1_nonlinear',
              ['clpt_donnell_bc1_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc1.pyx',
                       '../../integrate/integratev.pyx',
                       '../imperfections/mgi.pyx'])
    config.add_extension('clpt_donnell_bc2_nonlinear',
              ['clpt_donnell_bc2_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc2.pyx',
                       '../../integrate/integratev.pyx',
                       '../imperfections/mgi.pyx'])
    config.add_extension('clpt_donnell_bc3_nonlinear',
              ['clpt_donnell_bc3_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc3.pyx',
                       '../../integrate/integratev.pyx',
                       '../imperfections/mgi.pyx'])
    config.add_extension('clpt_donnell_bc4_nonlinear',
              ['clpt_donnell_bc4_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc4.pyx',
                       '../../integrate/integratev.pyx',
                       '../imperfections/mgi.pyx'])

    config.add_extension('clpt_sanders_bc1_linear',
              ['clpt_sanders_bc1_linear.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('clpt_sanders_bc2_linear',
              ['clpt_sanders_bc2_linear.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('clpt_sanders_bc3_linear',
              ['clpt_sanders_bc3_linear.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('clpt_sanders_bc4_linear',
              ['clpt_sanders_bc4_linear.pyx'],
              extra_compile_args=args_linear)

    config.add_extension('clpt_sanders_bc1_nonlinear',
              ['clpt_sanders_bc1_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc1.pyx',
                       '../../integrate/integratev.pyx',
                       '../imperfections/mgi.pyx'])
    config.add_extension('clpt_sanders_bc2_nonlinear',
              ['clpt_sanders_bc2_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc2.pyx',
                       '../../integrate/integratev.pyx',
                       '../imperfections/mgi.pyx'])
    config.add_extension('clpt_sanders_bc3_nonlinear',
              ['clpt_sanders_bc3_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc3.pyx',
                       '../../integrate/integratev.pyx',
                       '../imperfections/mgi.pyx'])
    config.add_extension('clpt_sanders_bc4_nonlinear',
              ['clpt_sanders_bc4_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc4.pyx',
                       '../../integrate/integratev.pyx',
                       '../imperfections/mgi.pyx'])

    config.add_extension('iso_clpt_donnell_bc2_linear',
              ['iso_clpt_donnell_bc2_linear.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('iso_clpt_donnell_bc3_linear',
              ['iso_clpt_donnell_bc3_linear.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('iso_clpt_donnell_bc2_nonlinear',
              ['iso_clpt_donnell_bc2_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc2.pyx',
                       '../imperfections/mgi.pyx',
                       '../../integrate/integratev.pyx'])
    config.add_extension('iso_clpt_donnell_bc3_nonlinear',
              ['iso_clpt_donnell_bc3_nonlinear.pyx'],
              extra_compile_args=args_nonlinear,
              depends=['clpt_commons_bc2.pyx',
                       '../imperfections/mgi.pyx',
                       '../../integrate/integratev.pyx'])

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
