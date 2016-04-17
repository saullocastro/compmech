from __future__ import division, print_function, absolute_import

import sys
import os
from distutils.sysconfig import get_python_lib
from os.path import basename, dirname, realpath, extsep, join
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial


def in_appveyor_ci():
    if os.environ.get('APPVEYOR_BUILD_FOLDER') is None:
        return False
    else:
        return True


def compile(config, src):
    if config.top_path.endswith('lib'):
        srcpath = join(realpath(config.package_path), src)
    else:
        srcpath = join(realpath(config.top_path), src)
    srcdir = dirname(srcpath)

    hashpath = srcpath + '.hashcode'
    hash_new = hashlib.sha256(os.name + open(srcpath, 'rb').read()).digest()

    if os.name == 'nt' and not in_appveyor_ci():
        objext = 'obj'
    else:
        objext = 'o'
    objpath = join(dirname(srcpath),
                   basename(srcpath).split(extsep)[0] + '.' + objext)

    needscompile = True
    if os.path.isfile(hashpath):
        if os.path.isfile(objpath):
            fsize = os.path.getsize(objpath)
            if fsize > 1L:
                hash_old = open(hashpath, 'rb').read().strip()
                if hash_old == hash_new:
                    needscompile = False

    if needscompile:
        bkpdir = os.getcwd()
        os.chdir(srcdir)
        if os.name == 'nt' and not in_appveyor_ci():
            os.system('cl /Ox /c {0}'.format(basename(srcpath)))
        else:
            os.system('gcc -pthread -g -O3 -fPIC -g -c -Wall {0}'.format(basename(srcpath)))
        fsize = os.path.getsize(basename(srcpath))
        os.chdir(bkpdir)
        if fsize > 1L:
            with open(hashpath, 'wb') as f:
                f.write(hash_new + '\n')
    else:
        print('Source {0} already compiled!'.format(srcpath))


def link(config, instlib):
    objs = ''
    if 'install' in sys.argv:
        libdir = join(get_python_lib(), 'compmech', 'lib')
    else:
        libdir = realpath(config.package_path)
    try:
        os.makedirs(libdir)
    except:
        pass

    if os.name == 'nt' and not in_appveyor_ci():
        objext = 'obj'
    else:
        objext = 'o'

    for src in instlib[1]['sources']:
        if config.top_path.endswith('lib'):
            srcpath = join(realpath(config.package_path), src)
        else:
            srcpath = join(realpath(config.top_path), src)
        objs += srcpath.replace('.c', '.' + objext) + ' '

    if os.name == 'nt':
        if in_appveyor_ci():
            libpath = join(libdir, 'lib' + instlib[0] + '.so')
            libpath_a = libpath.replace('.so', '.a')
            os.system('gcc -shared {0} -o {1} -Wl,--out-implib,{2}'.format(
                objs, libpath, libpath_a))
        else:
            libpath = join(libdir, instlib[0] + '.dll')
            os.system('link /DLL {0} /OUT:{1}'.format(objs, libpath))
    else:
        libpath = join(libdir, 'lib' + instlib[0] + '.so')
        os.system('gcc -shared -o {1} {0}'.format(objs, libpath))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration


    config = Configuration('lib', parent_package, top_path)

    config.add_installed_library('bardell',
            sources=['../../C/src/bardell.c'],
            install_dir=config.package_path)
    config.add_installed_library('bardell_12',
            sources=[
                '../../C/src/bardell_integral_ff_12.c',
                '../../C/src/bardell_integral_ffxi_12.c',
                '../../C/src/bardell_integral_ffxixi_12.c',
                '../../C/src/bardell_integral_fxifxi_12.c',
                '../../C/src/bardell_integral_fxifxixi_12.c',
                '../../C/src/bardell_integral_fxixifxixi_12.c',
                ],
            install_dir=config.package_path)
    config.add_installed_library('bardell_c0c1',
            sources=[
                '../../C/src/bardell_integral_ff_c0c1.c',
                '../../C/src/bardell_integral_ffxi_c0c1.c',
                '../../C/src/bardell_integral_fxifxi_c0c1.c',
                ],
            install_dir=config.package_path)
    config.add_installed_library('bardell_functions',
            sources=['../../C/src/bardell_functions.c'],
            install_dir=config.package_path)

    config.add_installed_library('legendre_gauss_quadrature',
            sources=['../../C/src/legendre_gauss_quadrature.c'],
            install_dir=config.package_path)

    config.options['ignore_setup_xxx_py'] = True
    config.make_config_py()

    for instlib in config.libraries:
        if in_appveyor_ci():
            for src in instlib[1]['sources']:
                compile(config, src)
            link(config, instlib)
        else:
            p = Pool(cpu_count()-1)
            partial_compile = partial(compile, config)
            p.map(partial_compile, instlib[1]['sources'])
            p.close()
            link(config, instlib)

    return config

if __name__ == '__main__':
    from setuptools import setup
    setup(**configuration(top_path='').todict())
