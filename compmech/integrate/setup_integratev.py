from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('integratev',
                   ['integratev.pyx'],
                   extra_compile_args=['/openmp'],
                   #extra_link_args=['/MANIFEST:NO']
                   )]

setup(
name = 'integratev',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
