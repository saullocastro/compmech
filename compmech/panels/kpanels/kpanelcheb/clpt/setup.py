from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
args_linear = ['/openmp', '/O2', '/favor:INTEL64']
args_linear2 = ['/openmp', '/Od', '/favor:INTEL64']
args_nonlinear = ['/openmp', '/O2', '/favor:INTEL64', '/fp:fast']
ext_modules = [
    Extension('clpt_commons_cheb', ['clpt_commons_cheb.pyx'],
              extra_compile_args=args_linear),
    Extension('clpt_donnell_cheb_linear', ['clpt_donnell_cheb_linear.pyx'],
              extra_compile_args=args_linear2),
    ]
setup(
name = 'clpt_donnell',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
