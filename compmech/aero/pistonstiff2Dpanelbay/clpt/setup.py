import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

if os.name == 'nt':
    args_linear = ['/O2', '/openmp', '/favor:INTEL64']
    args_nonlinear = ['/O2', '/openmp', '/favor:INTEL64', '/fp:fast']
else:
    args_linear = ['-O3']
    args_nonlinear = ['-O3']

extensions = [Extension('clpt_donnell_bardell_linear',
                        ['clpt_donnell_bardell_linear.pyx'],
                        libraries=['bardell', 'bardell_12',
                        'bardell_functions'],
                        language='c',
                        extra_compile_args=args_linear,
                        ),
             ]

ext_modules = cythonize(extensions)
setup(
name = 'aeropistonstiff2Dpanel_clpt',
ext_modules = ext_modules
)
