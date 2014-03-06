from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('clpt_linear_donnell',
                         ['clpt_linear_donnell.pyx'],
                   extra_compile_args=['/openmp',
                       '/O2', '/favor:INTEL64', '/fp:fast'],
                   extra_link_args=[],
                         )]
setup(
name = 'clpt_linear_donnell',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
