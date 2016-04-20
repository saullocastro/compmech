from __future__ import division, print_function, absolute_import

import os
from distutils.sysconfig import get_python_lib
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('models', parent_package, top_path)

    include = os.path.join(get_python_lib(), 'compmech', 'include')
    lib = os.path.join(get_python_lib(), 'compmech', 'lib')

    if os.name == 'nt':
        runtime_library_dirs = None
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in MiniConda
            args_linear = []
            args_nonlinear = []
        else:
            args_linear = ['/openmp']
            args_nonlinear = ['/openmp', '/fp:fast']
    else:
        runtime_library_dirs = [lib]
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            #NOTE removing openmp to compile in MiniConda
            args_linear = []
            args_nonlinear = []
        else:
            args_linear = ['-fopenmp']
            args_nonlinear = ['-fopenmp', '-ffast-math']

    config.add_extension('connect_panels',
              sources=['connect_panels.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell', 'bardell_functions'],
              library_dirs=[lib])

    config.add_extension('kpanel_clt_donnell_bardell',
              sources=['kpanel_clt_donnell_bardell.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell', 'bardell_12'],
              library_dirs=[lib])
    #config.add_extension('kpanel_clt_donnell_bardell_field',
              #sources=['kpanel_clt_donnell_bardell_field.pyx'],
              #extra_compile_args=args_linear,
              #runtime_library_dirs=runtime_library_dirs,
              #include_dirs=[include],
              #libraries=['bardell_functions'],
              #library_dirs=[lib])

    config.add_extension('cpanel_clt_donnell_bardell',
              sources=['cpanel_clt_donnell_bardell.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell', 'bardell_12'],
              library_dirs=[lib])
    config.add_extension('cpanel_clt_donnell_bardell_num',
              sources=['cpanel_clt_donnell_bardell_num.pyx'],
              extra_compile_args=args_nonlinear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell_functions', 'legendre_gauss_quadrature'],
              library_dirs=[lib])
    config.add_extension('cpanel_clt_donnell_bardell_field',
              sources=['cpanel_clt_donnell_bardell_field.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell_functions'],
              library_dirs=[lib])

    config.add_extension('plate_clt_donnell_bardell',
              sources=['plate_clt_donnell_bardell.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell', 'bardell_12'],
              library_dirs=[lib])
    config.add_extension('plate_clt_donnell_bardell_num',
              sources=['plate_clt_donnell_bardell_num.pyx'],
              extra_compile_args=args_nonlinear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell_functions', 'legendre_gauss_quadrature'],
              library_dirs=[lib])
    config.add_extension('plate_clt_donnell_bardell_field',
              sources=['plate_clt_donnell_bardell_field.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell_functions'],
              library_dirs=[lib])

    config.add_extension('plate_clt_donnell_bardell_w',
              sources=['plate_clt_donnell_bardell_w.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell', 'bardell_12'],
              library_dirs=[lib])
    config.add_extension('plate_clt_donnell_bardell_w_field',
              sources=['plate_clt_donnell_bardell_w_field.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=[include],
              libraries=['bardell_functions'],
              library_dirs=[lib])

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
