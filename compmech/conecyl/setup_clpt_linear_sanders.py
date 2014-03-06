from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('clpt_linear_sanders', ['clpt_linear_sanders.pyx'])]
setup(
name = 'clpt_linear_sanders',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
