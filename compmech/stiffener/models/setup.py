from __future__ import division, print_function, absolute_import

import os
from distutils.sysconfig import get_python_lib
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('models', parent_package, top_path)

    include = os.path.join(get_python_lib(), 'compmech', 'include')
    lib = os.path.join(get_python_lib(), 'compmech', 'lib')

    if os.name == 'nt':
        runtime_library_dirs = None
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in MiniConda
            args_linear = []
            args_nonlinear = []
        else:
            args_linear = ['/openmp']
            args_nonlinear = ['/openmp', '/fp:fast']
    else:
        runtime_library_dirs = [lib]
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in MiniConda
            args_linear = []
            args_nonlinear = []
        else:
            args_linear = ['-fopenmp']
            args_nonlinear = ['-fopenmp', '-ffast-math']

    config.add_extension('bladestiff1d_clt_donnell_bardell',
                     ['bladestiff1d_clt_donnell_bardell.pyx'],
                     extra_compile_args=args_linear,
                     runtime_library_dirs=runtime_library_dirs,
                     include_dirs=[include],
                     libraries=['bardell_functions', 'bardell'],
                     library_dirs=[lib])

    config.add_extension('bladestiff2d_clt_donnell_bardell',
                     ['bladestiff2d_clt_donnell_bardell.pyx'],
                     extra_compile_args=args_linear,
                     runtime_library_dirs=runtime_library_dirs,
                     include_dirs=[include],
                     libraries=['bardell_functions', 'bardell'],
                     library_dirs=[lib])

    config.add_extension('tstiff2d_clt_donnell_bardell',
                     ['tstiff2d_clt_donnell_bardell.pyx'],
                     extra_compile_args=args_linear,
                     runtime_library_dirs=runtime_library_dirs,
                     include_dirs=[include],
                     libraries=['bardell_functions', 'bardell',
                         'bardell_12', 'bardell_c0c1'],
                     library_dirs=[lib])

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from setuptools import setup
    setup(**configuration(top_path='').todict())
