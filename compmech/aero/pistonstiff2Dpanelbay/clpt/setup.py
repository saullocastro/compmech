import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

if os.name == 'nt':
    args_linear = ['/openmp', '/O2', '/favor:INTEL64']
    args_nonlinear = ['/openmp', '/O2', '/favor:INTEL64', '/fp:fast']
else:
    args_linear = ['-O3']
    args_nonlinear = ['-O3']
ext_modules = [
    Extension('clpt_commons_bc1', ['clpt_commons_bc1.pyx'],
              extra_compile_args=args_linear),

    Extension('clpt_donnell_bc1_linear', ['clpt_donnell_bc1_linear.pyx'],
              extra_compile_args=args_linear),
    ]
setup(
name = 'aeropistonstiffpanel_clpt',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
