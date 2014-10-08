from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('clpt_sanders_bc1_nonlinear',
                   ['clpt_sanders_bc1_nonlinear.pyx'],
                   extra_compile_args=['/openmp',
                       '/O2', '/favor:INTEL64', '/fp:fast'],
                   #extra_link_args=['/STACK:512000'],
                   )]
setup(
name = 'clpt_sanders_bc1_nonlinear',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
