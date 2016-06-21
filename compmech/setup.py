from __future__ import division, print_function, absolute_import

import os
from os.path import join, realpath
from distutils.sysconfig import get_python_lib
from subprocess import Popen
import shutil


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('compmech', parent_package, top_path)

    ###################################################
    #NOTE include and lib must be the first to install
    # include
    print('Copying include files...')
    includedir = join(get_python_lib(), 'compmech', 'include')
    if os.path.isdir(includedir):
        shutil.rmtree(includedir)
    shutil.copytree(join(realpath(config.package_path),
                    'include'), includedir)

    # lib
    print('Building shared libraries...')
    p = Popen('python ' +
              join(realpath(config.package_path),
                           './lib/setup.py') +
              ' install --inplace clean', shell=True)
    p.wait()
    ###################################################
    if 'Python27' in get_python_lib():
        pyversion = '2.7'
    else:
        raise NotImplementedError('Setup not ready for this Python version!')
    libplatdir = join(realpath(top_path), 'build', 'lib.' + os.environ['PLAT']
            + '-' + pyversion)

    libplatinclude = join(libplatdir, 'compmech', 'include')
    if os.path.isdir(libplatinclude):
        shutil.rmtree(libplatinclude)
    shutil.copytree(includedir, libplatinclude)

    libplatlib = join(libplatdir, 'compmech', 'lib')
    if os.path.isdir(libplatlib):
        shutil.rmtree(libplatlib)
    libdir = join(get_python_lib(), 'compmech', 'lib')
    shutil.copytree(libdir, libplatlib)
    ###################################################

    config.add_subpackage('analysis')
    config.add_subpackage('composite')

    if (os.environ.get('APPVEYOR_BUILD_FOLDER') is None
    and os.environ.get('TRAVIS_BUILD_DIR') is None):
        config.add_subpackage('conecyl')

    config.add_subpackage('integrate')

    config.add_subpackage('matplotlib_utils')
    config.add_subpackage('panel')
    config.add_subpackage('stiffener')
    config.add_subpackage('stiffpanelbay')
    config.add_data_dir('stiffpanelbay')
    config.add_subpackage('symbolic')

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
