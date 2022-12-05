import os
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('models', parent_package, top_path)

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
        if os.environ.get('CYTHON_TRACE_NOGIL') is not None:
            #NOTE removing openmp and optimizations for CI
            args_linear = ['/O0', '/openmp']
            args_nonlinear = ['/O0', '/openmp', '/fp:fast']
        else:
            args_linear = ['/openmp']
            args_nonlinear = ['/openmp', '/fp:fast']
    else:
        if os.environ.get('CYTHON_TRACE_NOGIL') is not None:
            #NOTE removing openmp and optimizations for CI
            args_linear = ['-O0', '-fopenmp']
            args_nonlinear = ['-O0', '-fopenmp', '-ffast-math']
        else:
            args_linear = ['-fopenmp']
            args_nonlinear = ['-fopenmp', '-ffast-math']

    config.add_extension('clt_bardell_field',
              sources=['clt_bardell_field.pyx'] + bardell_functions_sources,
              extra_compile_args=args_linear,
              )
    config.add_extension('clt_bardell_field_w',
              sources=['clt_bardell_field_w.pyx'] + bardell_functions_sources,
              extra_compile_args=args_linear,
              )
    config.add_extension('kpanel_clt_donnell_bardell',
              sources=['kpanel_clt_donnell_bardell.pyx'] + bardell_sources + bardell_integral_sources,
              extra_compile_args=args_linear,
              )
    config.add_extension('cpanel_clt_donnell_bardell',
              sources=['cpanel_clt_donnell_bardell.pyx'] + bardell_sources + bardell_integral_sources,
              extra_compile_args=args_linear,
              )
    config.add_extension('cpanel_clt_donnell_bardell_num',
              sources=['cpanel_clt_donnell_bardell_num.pyx'] + bardell_functions_sources + legendre_gauss_sources,
              extra_compile_args=args_nonlinear,
              )
    config.add_extension('plate_clt_donnell_bardell',
              sources=['plate_clt_donnell_bardell.pyx'] + bardell_sources + bardell_integral_sources,
              extra_compile_args=args_linear,
              )
    config.add_extension('plate_clt_donnell_bardell_num',
              sources=['plate_clt_donnell_bardell_num.pyx'] + bardell_functions_sources + legendre_gauss_sources,
              extra_compile_args=args_nonlinear,
              )
    config.add_extension('plate_clt_donnell_bardell_w',
              sources=['plate_clt_donnell_bardell_w.pyx'] + bardell_sources + bardell_integral_sources,
              extra_compile_args=args_linear,
              )

    cythonize(config.ext_modules)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
