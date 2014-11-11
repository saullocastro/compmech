from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('iso_clpt_donnell_bc3_nonlinear',
                   ['iso_clpt_donnell_bc3_nonlinear.pyx'],
                   extra_compile_args=['/openmp',
                       '/O2', '/favor:INTEL64', '/fp:fast'],
                   #extra_link_args=['/STACK:512000'],
                   )]
setup(
name = 'iso_clpt_donnell_bc3_nonlinear',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
