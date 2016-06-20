from __future__ import division, print_function, absolute_import

import os
from Cython.Build import cythonize
import numpy

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    if os.name == 'nt':
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in MiniConda
            compile_args = []
        else:
            compile_args = ['/openmp']
    else:
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in MiniConda
            compile_args = []
        else:
            compile_args = ['-fopenmp']

    config = Configuration('imperfections', parent_package, top_path)
    config.add_extension('mgi',
                         sources=['mgi.pyx'],
                        )

    cythonize(config.ext_modules)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from setuptools import setup
    setup(**configuration(top_path='').todict())
