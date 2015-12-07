from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
args_linear = ['/openmp', '/O2', '/favor:INTEL64']
args_nonlinear = ['/openmp', '/O2', '/favor:INTEL64', '/fp:fast']
ext_modules = [
    Extension('clpt_commons_bc4', ['clpt_commons_bc4.pyx'],
              extra_compile_args=args_linear),
    Extension('clpt_donnell_bc4_linear', ['clpt_donnell_bc4_linear.pyx'],
              extra_compile_args=args_linear),
    Extension('clpt_donnell_bc4_nonlinear', ['clpt_donnell_bc4_nonlinear.pyx'],
              extra_compile_args=args_nonlinear),
    ]
setup(
name = 'fsdt_donnell',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
