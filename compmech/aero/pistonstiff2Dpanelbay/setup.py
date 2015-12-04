from __future__ import division, print_function, absolute_import

import sys


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('pistonstiff2Dpanelbay',parent_package,top_path)
    config.add_subpackage('clpt')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
