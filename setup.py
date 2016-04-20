#!/usr/bin/env python
"""CompMech: Computational Mechanics in Python

This repository was built with the intention to become a collection of
theories and implementations on the field of Mechanics.

This setup is based on SciPy's setup.

"""
import os
import sys
import subprocess

if sys.version_info[:2] < (2, 6) or (3, 0) <= sys.version_info[0:2] < (3, 2):
    raise RuntimeError("Python version 2.7 or >= 3.2 required.")

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: BSD Approved
Programming Language :: FORTRAN
Programming Language :: C
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: Unix

"""

MAJOR = 0
MINOR = 5
MICRO = 5
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')



def write_version_py(filename='compmech/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM CompMech setup.py
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
isreleased = %(isreleased)s
if isreleased:
    __version__ = version
else:
    __version__ = full_version

if not isreleased:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isreleased': str(ISRELEASED)})
    finally:
        a.close()


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    FULLVERSION = VERSION
    GIT_REVISION = git_version()
    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('compmech')
    config.add_data_files(('compmech', '*.txt'))

    config.get_version('compmech/__version__.py')

    return config


def setup_package():
    cmdclass = {}
    # Figure out whether to add ``*_requires = ['numpy']``.
    # We don't want to do that unconditionally, because we risk updating
    # an installed numpy which fails too often.  Just if it's not installed, we
    # may give it a try.  See gh-3379.
    build_requires = ['numpy>=1.6.2', 'scipy>=0.14.0']

    FULLVERSION, GIT_REVISION = get_version_info()

    metadata = dict(
        name='compmech',
        maintainer='CompMech Developers',
        maintainer_email='compmech.project@gmail.com ',
        version=FULLVERSION,
        description=DOCLINES[0],
        long_description='\n'.join(DOCLINES[2:]),
        url='http://compmech.github.io/compmech/',
        download_url='https://github.com/compmech/compmech/releases',
        license='BSD',
        cmdclass=cmdclass,
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=['Windows', 'Linux'],
        #test_suite='nose.collector',
        setup_requires=build_requires,
        install_requires=build_requires,
    )


    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                            'clean')):
        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Scipy when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        FULLVERSION, GIT_REVISION = get_version_info()
        metadata['version'] = FULLVERSION
    else:
        if (len(sys.argv) >= 2 and sys.argv[1] in ('bdist_wheel', 'bdist_egg')) or (
                    'develop' in sys.argv):
            # bdist_wheel/bdist_egg needs setuptools
            import setuptools

        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)

if __name__ == '__main__':
    write_version_py()

    if os.name == 'nt' and os.environ.get('CONDA_DEFAULT_ENV') is None:
        os.environ['DISTUTILS_USE_SDK'] = '1'
        os.environ['MSSdk'] = '1'

    setup_package()
