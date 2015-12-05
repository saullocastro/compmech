from __future__ import division, print_function, absolute_import

import os
from multiprocessing import Pool, cpu_count


def compile(src):
    srcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), src)
    srcdir = os.path.dirname(srcpath)
    if os.name == 'nt':
        bkpdir = os.getcwd()
        os.chdir(srcdir)
        os.system('cl /Ox /c {0}'.format(srcpath))
        os.chdir(bkpdir)
    else:
        raise NotImplementedError('Only Windows supported yet...')

def link(instlib):
    if os.name == 'nt':
        objs = ''
        libdir = os.path.dirname(os.path.realpath(__file__))
        for src in instlib[1]['sources']:
            srcpath = os.path.join(libdir, src)
            objs += srcpath.replace('.c', '.obj') + ' '
        libpath = os.path.join(libdir, instlib[0] + '.dll')
        os.system('link /DLL {0} /OUT:{1}'.format(objs, libpath))
    else:
        raise NotImplementedError('Only Windows supported yet...')


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('lib',parent_package,top_path)
    config.add_installed_library('bardell',
            sources=['../../C/src/bardell.c'],
            install_dir='.')
    config.add_installed_library('bardell_12',
            sources=[
                '../../C/src/bardell_12_integral_ff.c',
                '../../C/src/bardell_12_integral_ffxi.c',
                '../../C/src/bardell_12_integral_ffxixi.c',
                '../../C/src/bardell_12_integral_fxifxi.c',
                '../../C/src/bardell_12_integral_fxifxixi.c',
                '../../C/src/bardell_12_integral_fxixifxixi.c',
                ],
            install_dir='.')
    config.add_installed_library('bardell_functions',
            sources=['../../C/src/bardell_functions.c'],
            install_dir='.')

    for instlib in config.libraries:
        p = Pool(cpu_count())
        p.map(compile, instlib[1]['sources'])
        p.close()
        link(instlib)



    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
