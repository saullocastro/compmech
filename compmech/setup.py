from __future__ import division, print_function, absolute_import

import sys


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('compmech',parent_package,top_path)
    config.add_subpackage('aero')
    config.add_subpackage('analysis')
    config.add_subpackage('composite')
    config.add_subpackage('conecyl')
    config.add_data_dir('include')
    config.add_subpackage('integrate')
    config.add_subpackage('lib')
    config.add_data_dir('lib')
    config.add_subpackage('matplotlib_utils')
    config.add_subpackage('panels')
    config.add_subpackage('plates')
    config.add_subpackage('symbolic')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
