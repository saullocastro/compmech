import os
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('connections', parent_package, top_path)

    compmech_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
            )
        )
    )
    bardell_path = os.path.join(compmech_path, 'lib', 'src', 'bardell.c')
    bardell_functions_path = os.path.join(compmech_path, 'lib', 'src', 'bardell_functions.c')
    bardel_sources = [bardell_path, bardell_functions_path]

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

    # TODO: check commented keyword arguments
    config.add_extension('kCBFycte',
              sources=['kCBFycte.pyx'] + bardel_sources,
              extra_compile_args=args_linear,
              )
    config.add_extension('kCSB',
              sources=['kCSB.pyx'] + bardel_sources,
              extra_compile_args=args_linear,
              )
    config.add_extension('kCSSxcte',
              sources=['kCSSxcte.pyx'] + bardel_sources,
              extra_compile_args=args_linear,
              )
    config.add_extension('kCSSycte',
              sources=['kCSSycte.pyx'] + bardel_sources,
              extra_compile_args=args_linear,
              )

    for ext in config.ext_modules:
        for src in ext.sources:
            cythonize(src)

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
