from __future__ import division, print_function, absolute_import

import sys


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('conecyl', parent_package, top_path)
    config.add_subpackage('clpt')
    config.add_subpackage('fsdt')
    config.add_subpackage('imperfections')
    config.add_subpackage('sympytools')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
