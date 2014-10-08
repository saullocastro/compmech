from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
args = ['/openmp', '/O2', '/favor:INTEL64', '/fp:fast']
extensions = [
    Extension('fsdt_donnell_kquad8', ['fsdt_donnell_kquad8.pyx'],
              extra_compile_args=args),
    ]
setup(
name = 'elements',
cmdclass = {'build_ext': build_ext},
ext_modules = extensions
)
