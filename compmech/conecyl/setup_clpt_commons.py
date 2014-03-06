from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('clpt_commons',
                         ['clpt_commons.pyx'],
                         extra_compile_args=['/openmp',
                             '/O2', '/favor:INTEL64', '/fp:fast'],
                         )]
setup(
name = 'clpt_commons',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
