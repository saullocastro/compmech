from __future__ import division, print_function, absolute_import

import os
from subprocess import Popen


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('compmech',parent_package,top_path)

    config.add_subpackage('analysis')
    config.add_subpackage('composite')

    if os.environ.get('APPVEYOR_BUILD_FOLDER') is not None:
        config.add_subpackage('conecyl')

    config.add_data_dir('include')
    config.add_subpackage('integrate')

    # lib
    p = Popen('python ' +
              os.path.join(os.path.realpath(config.package_path),
                           './lib/setup.py') +
              ' build_ext --inplace clean', shell=True)
    p.wait()
    config.add_data_dir('lib')

    config.add_subpackage('matplotlib_utils')
    config.add_subpackage('panel')
    config.add_subpackage('stiffener')
    config.add_subpackage('stiffpanelbay')
    config.add_subpackage('symbolic')

    config.make_config_py()


    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
