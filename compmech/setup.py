from __future__ import division, print_function, absolute_import

import os
import sys
from os.path import join, realpath
from distutils.sysconfig import get_python_lib
from subprocess import Popen
import shutil

import setup_patch


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('compmech', parent_package, top_path)

    #_________________________________________________
    #
    #NOTE include and lib must be the first to install
    #_________________________________________________
    # include
    includedir = join(get_python_lib(), 'compmech', 'include')
    print('Copying include files to {0}'.format(includedir))
    if os.path.isdir(includedir):
        shutil.rmtree(includedir)
    shutil.copytree(join(realpath(config.package_path), 'include'), includedir)
    # lib
    setuppath = join(realpath(config.package_path), './lib/setup.py')
    print('Building shared libraries using {0}'.format(setuppath))
    if 'install' in sys.argv:
        p = Popen('python ' + join(setuppath) + ' install',
            shell=True)
    else:
        p = Popen('python ' + join(setuppath) + ' build_ext --inplace clean',
            shell=True)
    p.wait()
    #_________________________________________________

    config.add_subpackage('analysis')
    config.add_subpackage('composite')

    if (os.environ.get('APPVEYOR_BUILD_FOLDER') is None
    and os.environ.get('TRAVIS_BUILD_ID') is None):
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
