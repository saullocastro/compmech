from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('clpt_NL_sanders_numerical',
                         ['clpt_NL_sanders_numerical.pyx'],
                   extra_compile_args=['/openmp',
                       '/O2', '/favor:INTEL64', '/fp:fast'],
                   extra_link_args=[],
                         )]
setup( name = 'clpt_NL_sanders_numerical',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
