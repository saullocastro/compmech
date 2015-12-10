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

    config = Configuration('models', parent_package, top_path)

    config.add_extension('bladestiff1d_clt_donnell_bardell',
                     ['bladestiff1d_clt_donnell_bardell.pyx'],
                     extra_compile_args=args_linear,
                     include_dirs=['../../include'],
                     libraries=['bardell'],
                     library_dirs=['../../lib'])

    config.add_extension('bladestiff2d_clt_donnell_bardell',
                     ['bladestiff2d_clt_donnell_bardell.pyx'],
                     extra_compile_args=args_linear,
                     include_dirs=['../../include'],
                     libraries=['bardell_functions', 'bardell'],
                     library_dirs=['../../lib'])

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
