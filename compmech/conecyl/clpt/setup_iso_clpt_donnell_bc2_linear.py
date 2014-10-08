from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('iso_clpt_donnell_bc2_linear',
                         ['iso_clpt_donnell_bc2_linear.pyx'],
                   extra_compile_args=['/openmp', '/O2', '/favor:INTEL64'],
                   extra_link_args=[],
                         )]
setup(
name = 'iso_clpt_donnell_bc2_linear',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
