from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
os.system('rmdir /q /s build')

extensions = [Extension('test',
            sources=['_cython_test_lib.pyx'],
            libraries=[
                'bardell',
                'bardell_c0c1',
                'bardell_functions',
                ],
            library_dirs=['../../compmech/lib'],
            include_dirs=['../../compmech/include'],
            language='c',
            extra_compile_args=['/openmp', '/O2'],

                        ),
             ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions
)
