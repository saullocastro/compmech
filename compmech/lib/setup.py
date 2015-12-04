from __future__ import division, print_function, absolute_import

import sys


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('lib',parent_package,top_path)
    config.add_installed_library('bardell',
                                 sources=['../../C/src/bardell.c'],
                                 install_dir='.')
    #config.add_library('bardell_12', sources=['../../C/src/bardell_12.c'])
    config.add_installed_library('bardell_functions',
                       sources=['../../C/src/bardell_functions.c'],
                       install_dir='.')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
