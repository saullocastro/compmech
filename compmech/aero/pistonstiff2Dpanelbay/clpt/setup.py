import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import sys
sys.path.append(r'C:\repos\compmech\compmech\func\bardell')

if os.name == 'nt':
    args_linear = ['/openmp', '/O2', '/favor:INTEL64']
    args_nonlinear = ['/openmp', '/O2', '/favor:INTEL64', '/fp:fast']
else:
    args_linear = ['-O3']
    args_nonlinear = ['-O3']

extensions = [Extension('clpt_donnell_bardell_linear',
                        ['clpt_donnell_bardell_linear.pyx',
                         '../../../../C/bardell/bardell.c',
                         '../../../../C/bardell/bardell_12.c',
                         '../../../../C/bardell/bardell_functions.c',
                        ]),
             ]

ext_modules = cythonize(extensions)
setup(
name = 'aeropistonstiff2Dpanel_clpt',
ext_modules = ext_modules
)
