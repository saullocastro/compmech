from __future__ import division, print_function, absolute_import

import os
from os.path import basename, dirname, realpath, extsep, join
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial


def compile(config, src):
    libdir = realpath(config.package_path)

    if config.top_path.endswith('lib'):
        srcpath = join(realpath(config.package_path), src)
    else:
        srcpath = join(realpath(config.top_path), src)
    srcdir = dirname(srcpath)

    hashpath = srcpath + '.hashcode'
    hash_new = hashlib.sha256(os.name + open(srcpath, 'rb').read()).digest()

    if os.name == 'nt':
        objext = 'obj'
    else:
        objext = 'a'
    objpath = join(dirname(srcpath),
                   basename(srcpath).split(extsep)[0] + '.' + objext)

    needscompile = True
    if os.path.isfile(hashpath) and os.path.isfile(objpath):
        hash_old = open(hashpath, 'rb').read().strip()
        if hash_old == hash_new:
            needscompile = False
    if needscompile:
        if os.name == 'nt':
            bkpdir = os.getcwd()
            os.chdir(srcdir)
            os.system('cl /Ox /c {0}'.format(basename(srcpath)))
            os.chdir(bkpdir)
        else:
            raise NotImplementedError('Only Windows supported yet...')
        with open(hashpath, 'wb') as f:
            f.write(hash_new + '\n')
    else:
        print('Source {0} already compiled!'.format(srcpath))


def link(config, instlib):
    if os.name == 'nt':
        objs = ''
        libdir = realpath(config.package_path)
        for src in instlib[1]['sources']:
            if config.top_path.endswith('lib'):
                srcpath = join(realpath(config.package_path), src)
            else:
                srcpath = join(realpath(config.top_path), src)
            objs += srcpath.replace('.c', '.obj') + ' '
        libpath = join(libdir, instlib[0] + '.dll')
        os.system('link /DLL {0} /OUT:{1}'.format(objs, libpath))
    else:
        raise NotImplementedError('Only Windows supported yet...')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration


    config = Configuration('lib', parent_package, top_path)
    config.add_installed_library('bardell',
            sources=['../../C/src/bardell.c'],
            install_dir=config.package_path)
    config.add_installed_library('bardell_12',
            sources=[
                '../../C/src/bardell_12_integral_ff.c',
                '../../C/src/bardell_12_integral_ffxi.c',
                '../../C/src/bardell_12_integral_ffxixi.c',
                '../../C/src/bardell_12_integral_fxifxi.c',
                '../../C/src/bardell_12_integral_fxifxixi.c',
                '../../C/src/bardell_12_integral_fxixifxixi.c',
                ],
            install_dir=config.package_path)
    config.add_installed_library('bardell_functions',
            sources=['../../C/src/bardell_functions.c'],
            install_dir=config.package_path)
    config.options['ignore_setup_xxx_py'] = True
    config.make_config_py()

    for instlib in config.libraries:
        p = Pool(cpu_count())
        partial_compile = partial(compile, config)
        p.map(partial_compile, instlib[1]['sources'])
        p.close()
        link(config, instlib)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
