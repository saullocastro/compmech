from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('fsdt_geier1997_bc2',
                         ['fsdt_geier1997_bc2.pyx'],
                   extra_compile_args=['/openmp',
                       '/O2', '/favor:INTEL64'],
                   extra_link_args=[],
                         )]
setup(
name = 'fsdt_geier1997_bc2',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
