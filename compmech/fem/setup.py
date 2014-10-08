from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

args = ['/openmp', '/O2', '/favor:INTEL64', '/fp:fast']
extensions = [
    Extension('utils', ['utils.pyx'],
              extra_compile_args=args),
    ]
setup(
name = 'fem',
cmdclass = {'build_ext': build_ext},
ext_modules = extensions
)
