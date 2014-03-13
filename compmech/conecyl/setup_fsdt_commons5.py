from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('fsdt_commons5',
                         ['fsdt_commons5.pyx'],
                         extra_compile_args=['/openmp',
                             '/O2', '/favor:INTEL64'],
                         )]
setup(
name = 'fsdt_commons5',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
