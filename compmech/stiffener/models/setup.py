from __future__ import division, print_function, absolute_import

import os
from distutils.sysconfig import get_python_lib
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('models', parent_package, top_path)

    include = os.path.join(get_python_lib(), 'compmech', 'include')
    lib = os.path.join(get_python_lib(), 'compmech', 'lib')

    compmech_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
            )
        )
    )
    src_path = os.path.join(compmech_path, 'lib', 'src')

    bardell_sources = [os.path.join(src_path, 'bardell.c')]
    bardell_functions_sources = [os.path.join(src_path, 'bardell_functions.c')]
    bardell_integral_sources = [
        os.path.join(src_path, 'bardell_integral_ff_12.c'),
        os.path.join(src_path, 'bardell_integral_ff_c0c1.c'),
        os.path.join(src_path, 'bardell_integral_ffxi_12.c'),
        os.path.join(src_path, 'bardell_integral_ffxi_c0c1.c'),
        os.path.join(src_path, 'bardell_integral_ffxixi_12.c'),
        os.path.join(src_path, 'bardell_integral_fxif_c0c1.c'),
        os.path.join(src_path, 'bardell_integral_fxifxi_12.c'),
        os.path.join(src_path, 'bardell_integral_fxifxi_c0c1.c'),
        os.path.join(src_path, 'bardell_integral_fxifxixi_12.c'),
        os.path.join(src_path, 'bardell_integral_fxixifxixi_12.c'),
        os.path.join(src_path, 'bardell_integral_fxixifxixi_c0c1.c'),
    ] 
    legendre_gauss_sources = [os.path.join(src_path, 'legendre_gauss_quadrature.c')]

    if os.name == 'nt':
        runtime_library_dirs = None
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in MiniConda
            args_linear = ['-O0']
            args_nonlinear = ['-O0']
        else:
            args_linear = ['/openmp']
            args_nonlinear = ['/openmp', '/fp:fast']
    else:
        runtime_library_dirs = [lib]
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in MiniConda
            args_linear = ['-O0']
            args_nonlinear = ['-O0']
        else:
            args_linear = ['-fopenmp']
            args_nonlinear = ['-fopenmp', '-ffast-math']

    # TODO: check commented keyword arguments
    config.add_extension('bladestiff1d_clt_donnell_bardell',
                     ['bladestiff1d_clt_donnell_bardell.pyx'] + bardell_sources + bardell_functions_sources,
                     #extra_compile_args=args_linear,
                     include_dirs=[include],
                     )
    config.add_extension('bladestiff2d_clt_donnell_bardell',
                     ['bladestiff2d_clt_donnell_bardell.pyx'] + bardell_sources + bardell_functions_sources,
                     #extra_compile_args=args_linear,
                     include_dirs=[include],
                     )
    config.add_extension('tstiff2d_clt_donnell_bardell',
                     ['tstiff2d_clt_donnell_bardell.pyx'] + bardell_sources + bardell_functions_sources + bardell_integral_sources,
                     #extra_compile_args=args_linear,
                     include_dirs=[include],
                     )
    cythonize(config.ext_modules)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
