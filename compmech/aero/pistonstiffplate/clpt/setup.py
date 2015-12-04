from __future__ import division, print_function, absolute_import

import os
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    if os.name == 'nt':
        args_linear = ['/openmp']
    else:
        args_linear = ['-openmp']

    config = Configuration('clpt', parent_package, top_path)

    config.add_extension('clpt_commons_bc1', ['clpt_commons_bc1.pyx'],
              extra_compile_args=args_linear)
    config.add_extension('clpt_donnell_bc1_linear', ['clpt_donnell_bc1_linear.pyx'],
              extra_compile_args=args_linear)

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
