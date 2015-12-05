from __future__ import division, print_function, absolute_import

import os


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    if os.name == 'nt':
        os.system('cl /O2 /LD {0}/C/src/bardell.c -o{0}/compmech/lib/bardell.dll'.format(root))
        os.system('cl /O2 /LD {0}/C/src/bardell_12.c -o{0}/compmech/lib/bardell_12.dll ')
        os.system('cl /O2 /LD {0}/C/src/bardell_functions.c -o{0}/compmech/lib/bardell_functions.dll'.format(root))
    else:
        raise NotImplementedError('Only Windows supported yet...')

    config = Configuration('lib',parent_package,top_path)
    config.add_installed_library('bardell',
         sources=['../../C/src/bardell.c'],
         install_dir='.')
    config.add_installed_library('bardell_12',
          sources=['../../C/src/bardell_12.c'],
          install_dir='.')
    config.add_installed_library('bardell_functions',
          sources=['../../C/src/bardell_functions.c'],
          install_dir='.')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
