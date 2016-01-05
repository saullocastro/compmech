from __future__ import division, print_function, absolute_import

import os
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('models', parent_package, top_path)

    libpath = os.path.join(os.path.realpath(config.package_path),
                           '..', '..', 'lib')
    if os.name == 'nt':
        runtime_library_dirs = None
        args_linear = ['/openmp']
        args_nonlinear = ['/openmp', '/fp:fast']
    else:
        runtime_library_dirs = [libpath]
        args_linear = []
        args_nonlinear = ['-ffast-math']

    config.add_extension('kpanel_clt_donnell_bardell',
              sources=['kpanel_clt_donnell_bardell.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=['../../include'],
              libraries=['bardell', 'bardell_12'],
              library_dirs=['../../lib'])
    #config.add_extension('kpanel_clt_donnell_bardell_field',
              #sources=['kpanel_clt_donnell_bardell_field.pyx'],
              #extra_compile_args=args_linear,
              #runtime_library_dirs=runtime_library_dirs,
              #include_dirs=['../../include'],
              #libraries=['bardell_functions'],
              #library_dirs=['../../lib'])

    config.add_extension('cpanel_clt_donnell_bardell',
              sources=['cpanel_clt_donnell_bardell.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=['../../include'],
              libraries=['bardell', 'bardell_12'],
              library_dirs=['../../lib'])
    config.add_extension('cpanel_clt_donnell_bardell_field',
              sources=['cpanel_clt_donnell_bardell_field.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=['../../include'],
              libraries=['bardell_functions'],
              library_dirs=['../../lib'])

    config.add_extension('plate_clt_donnell_bardell',
              sources=['plate_clt_donnell_bardell.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=['../../include'],
              libraries=['bardell', 'bardell_12'],
              library_dirs=['../../lib'])
    config.add_extension('plate_clt_donnell_bardell_field',
              sources=['plate_clt_donnell_bardell_field.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=['../../include'],
              libraries=['bardell_functions'],
              library_dirs=['../../lib'])

    config.add_extension('plate_clt_donnell_bardell_w',
              sources=['plate_clt_donnell_bardell_w.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=['../../include'],
              libraries=['bardell', 'bardell_12'],
              library_dirs=['../../lib'])
    config.add_extension('plate_clt_donnell_bardell_w_field',
              sources=['plate_clt_donnell_bardell_w_field.pyx'],
              extra_compile_args=args_linear,
              runtime_library_dirs=runtime_library_dirs,
              include_dirs=['../../include'],
              libraries=['bardell_functions'],
              library_dirs=['../../lib'])

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
