import os
from distutils.sysconfig import get_python_lib

from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('models', parent_package, top_path)

    include = os.path.join(get_python_lib(), 'compmech', 'include')

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

    if os.name == 'nt':
        if os.environ.get('CYTHON_TRACE_NOGIL') is not None:
            #NOTE removing openmp and optimizations for CI
            args_linear = ['/O0', '/openmp']
        else:
            args_linear = ['/openmp']
    else:
        if os.environ.get('CYTHON_TRACE_NOGIL') is not None:
            #NOTE removing openmp and optimizations for CI
            args_linear = ['-O0', '-fopenmp']
        else:
            args_linear = ['-fopenmp']

    #TODO: check commented keyword arguments
    config.add_extension('bladestiff1d_clt_donnell_bardell',
                     ['bladestiff1d_clt_donnell_bardell.pyx'] + bardell_sources + bardell_functions_sources,
                     extra_compile_args=args_linear,
                     include_dirs=[include],
                     )
    config.add_extension('bladestiff2d_clt_donnell_bardell',
                     ['bladestiff2d_clt_donnell_bardell.pyx'] + bardell_sources + bardell_functions_sources,
                     extra_compile_args=args_linear,
                     include_dirs=[include],
                     )
    config.add_extension('tstiff2d_clt_donnell_bardell',
                     ['tstiff2d_clt_donnell_bardell.pyx'] + bardell_sources + bardell_functions_sources + bardell_integral_sources,
                     extra_compile_args=args_linear,
                     include_dirs=[include],
                     )
    cythonize(config.ext_modules)

    config.make_config_py()

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
