from __future__ import division, print_function, absolute_import

import os
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    if os.name == 'nt':
        args_linear = ['/openmp']
        args_nonlinear = ['/openmp', '/fp:fast']
    else:
        args_linear = ['-openmp']
        args_nonlinear = ['-openmp', '-fp:fast']

    config = Configuration('clpt', parent_package, top_path)

    config.add_extension('clpt_commons_bc1', ['clpt_commons_bc1.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('clpt_commons_free', ['clpt_commons_free.pyx'],
              extra_compile_args=args_linear)

    config.add_extension('clpt_donnell_bc1_linear', ['clpt_donnell_bc1_linear.pyx'],
              extra_compile_args=args_nonlinear)
    config.add_extension('clpt_donnell_free_linear', ['clpt_donnell_free_linear.pyx'],
              extra_compile_args=args_nonlinear)

    config.add_extension('clpt_commons_bardell', ['clpt_commons_bardell.pyx'],
              extra_compile_args=args_linear,
              include_dirs=['../../include'],
              libraries=['bardell_functions'],
              library_dirs=['../../lib'])
    config.add_extension('clpt_commons_bardell_w', ['clpt_commons_bardell_w.pyx'],
              extra_compile_args=args_linear,
              include_dirs=['../../include'],
              libraries=['bardell_functions'],
              library_dirs=['../../lib'])

    config.add_extension('clpt_donnell_bardell_linear', ['clpt_donnell_bardell_linear.pyx'],
              extra_compile_args=args_linear,
              include_dirs=['../../include'],
              libraries=['bardell'],
              library_dirs=['../../lib'])
    config.add_extension('clpt_donnell_bardell_w_linear', ['clpt_donnell_bardell_w_linear.pyx'],
              extra_compile_args=args_linear,
              include_dirs=['../../include'],
              libraries=['bardell'],
              library_dirs=['../../lib'])

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
