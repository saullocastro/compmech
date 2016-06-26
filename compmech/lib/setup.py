from __future__ import division, print_function, absolute_import

import sys
import os
from os.path import join, basename, extsep
from distutils.sysconfig import get_python_lib
from subprocess import Popen

#TODO not working... but seems promising
#import setup_patch


def in_appveyor_ci():
    if os.environ.get('APPVEYOR_BUILD_FOLDER') is None:
        return False
    else:
        return True


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    if 'install' in sys.argv:
        install_dir = join(get_python_lib(), 'compmech', 'lib')
    else:
        install_dir = join(config.package_path)

    config = Configuration('lib', parent_package, top_path)

    if sys.version_info[0] == 2:
        extra_args = ['/Od']
    else:
        extra_args = []

    config.add_installed_library('bardell',
            sources=['./src/bardell.c'],
            install_dir=install_dir,
            build_info={'extra_compiler_args': extra_args})

    config.add_installed_library('bardell_12',
            sources=[
                './src/bardell_integral_ff_12.c',
                './src/bardell_integral_ffxi_12.c',
                './src/bardell_integral_ffxixi_12.c',
                './src/bardell_integral_fxifxi_12.c',
                './src/bardell_integral_fxifxixi_12.c',
                './src/bardell_integral_fxixifxixi_12.c',
                ],
            install_dir=install_dir,
            build_info={'extra_compiler_args': extra_args})

    config.add_installed_library('bardell_c0c1',
            sources=[
                './src/bardell_integral_ff_c0c1.c',
                './src/bardell_integral_ffxi_c0c1.c',
                './src/bardell_integral_fxif_c0c1.c',
                './src/bardell_integral_fxifxi_c0c1.c',
                './src/bardell_integral_fxixifxixi_c0c1.c',
                ],
            install_dir=install_dir)

    config.add_installed_library('bardell_functions',
            sources=['./src/bardell_functions.c'],
            install_dir=install_dir,
            build_info={'extra_compiler_args': extra_args})

    config.add_installed_library('legendre_gauss_quadrature',
            sources=['./src/legendre_gauss_quadrature.c'],
            install_dir=install_dir,
            build_info={'extra_compiler_args': extra_args})

    config.make_config_py()

    return config, install_dir


def create_dlls(config, install_dir):
    tmp = config.get_build_temp_dir()
    # generating DLLs
    if os.name == 'nt' and not in_appveyor_ci():
        objext = 'obj'
    else:
        objext = 'o'

    for instlib in config.libraries:
        objs = ''
        for src in instlib[1]['sources']:
            binary = basename(src).split(extsep)[0] + extsep + objext
            binary = join(tmp, 'compmech', 'lib', 'src', binary)
            objs += binary + ' '

        if os.name == 'nt':
            if in_appveyor_ci():
                libpath = join(install_dir, 'lib' + instlib[0] + '.so')
                libpath_a = libpath.replace('.so', '.a')
                p = Popen('gcc -shared {0} -o {1} -Wl,--out-implib,{2}'.format(
                    objs, libpath, libpath_a), shell=True)
            else:
                libpath = join(install_dir, instlib[0] + '.dll')
                p = Popen('link /DLL {0} /OUT:{1}'.format(objs, libpath),
                        shell=True)
        else:
            libpath = join(install_dir, 'lib' + instlib[0] + '.so')
            p = Popen('gcc -shared -o {1} {0}'.format(objs, libpath),
                    shell=True)
        p.wait()
        if p.returncode != 0:
            raise RuntimeError('LINK error with: {0}'.format(libpath))


if __name__ == '__main__':
    from numpy.distutils.core import setup
    config, install_dir = configuration(top_path='')
    setup(**config.todict())
    create_dlls(config, install_dir)
