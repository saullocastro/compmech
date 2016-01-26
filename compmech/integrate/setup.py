from __future__ import division, print_function, absolute_import

import os
from Cython.Build import cythonize
import numpy

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    if os.name == 'nt':
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in AppVeyor
            compile_args = []
        else:
            compile_args = ['/openmp']
    else:
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in Travis CI
            compile_args = []
        else:
            compile_args = ['-fopenmp']

    config = Configuration('integrate', parent_package, top_path)
    config.add_extension('integrate',
                         sources=['integrate.pyx'],
                         extra_compile_args=compile_args,
                        )
    config.add_extension('integratev',
                         sources=['integratev.pyx'],
                         extra_compile_args=compile_args,
                        )

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
