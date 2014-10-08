from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('clpt_commons_bc1',
                         ['clpt_commons_bc1.pyx'],
                         extra_compile_args=['/openmp',
                             '/O2', '/favor:INTEL64'],
                         )]
setup(
name = 'clpt_commons_bc1',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
