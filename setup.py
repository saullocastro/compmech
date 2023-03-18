#!/usr/bin/env python
"""CompMech: Computational Mechanics in Python
Collection of theories and implementations on the field of Computational
Mechanics, currently almost exclusively on Solid Mechanics and simple
fluid-structure interaction with the panel flutter analyzes available.
"""
import os
import platform
import inspect
import subprocess
from setuptools import setup, find_packages, Extension

import numpy as np
from Cython.Build import cythonize


DOCLINES = __doc__.split("\n")


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
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info(version, is_released):
    fullversion = version
    if not is_released:
        git_revision = git_version()
        fullversion += '.dev0+' + git_revision[:7]
    return fullversion


def write_version_py(version, is_released, filename='compmech/version.py'):
    fullversion = get_version_info(version, is_released)
    with open("./compmech/version.py", "wb") as f:
        f.write(('__version__ = "%s"\n' % fullversion).encode())
    return fullversion


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    setupdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return open(os.path.join(setupdir, fname)).read()


#_____________________________________________________________________________

install_requires = [
        "numpy >= 1.23.0",
        "scipy",
        "matplotlib",
        ]

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: End Users/Desktop
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Topic :: Education
Topic :: Software Development
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: POSIX :: BSD
Programming Language :: Fortran
Programming Language :: C
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
License :: OSI Approved :: BSD License
"""

is_released = False
version = '0.8.0'

fullversion = write_version_py(version, is_released)

data_files = [('', [
        'README.md',
        'LICENSE',
        'ROADMAP.rst',
        'setup.py',
        'compmech/version.py',
        ])]

package_data = {
        'compmech': ['*.pxd'],
        '': ['tests/*.*'],
        }

link_args = []
if platform.system() == 'Windows':
    #if os.environ.get('CYTHON_TRACE_NOGIL') is not None:
        #compiler_args = ['/openmp']
    compiler_args = ['/openmp']
    compiler_args_NL = compiler_args + ['/fp:fast']
elif platform.system() == 'Darwin': # MAC-OS
    compiler_args = ['-Xclang', '-fopenmp']
    link_args = ['-lomp']
    compiler_args_NL = compiler_args + ['-ffast-math']
else: # Linux
    compiler_args = ['-fopenmp']
    link_args = ['-fopenmp']
    compiler_args_NL = compiler_args + ['-ffast-math']

root_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/compmech'

include_dirs = [
            np.get_include(),
            root_path +  '/include',
            ]

library_dirs = []
if os.name == 'posix': # MAC-OS
    include_dirs.append('/usr/local/opt/libomp/include')
    library_dirs.append('/usr/local/opt/libomp/lib')

legendre_src = root_path + '/lib/src/legendre_gauss_quadrature.c'
bardell_int_src = root_path + '/lib/src/bardell.c'
bardell_func_src = root_path + '/lib/src/bardell_functions.c'
bardell_int12_src = [
    root_path + '/lib/src/bardell_integral_ff_12.c',
    root_path + '/lib/src/bardell_integral_ffxi_12.c',
    root_path + '/lib/src/bardell_integral_ffxixi_12.c',
    root_path + '/lib/src/bardell_integral_fxifxi_12.c',
    root_path + '/lib/src/bardell_integral_fxifxixi_12.c',
    root_path + '/lib/src/bardell_integral_fxixifxixi_12.c',
    ]
bardell_intc0c1_src = [
    root_path + '/lib/src/bardell_integral_ff_c0c1.c',
    root_path + '/lib/src/bardell_integral_ffxi_c0c1.c',
    root_path + '/lib/src/bardell_integral_fxif_c0c1.c',
    root_path + '/lib/src/bardell_integral_fxifxi_c0c1.c',
    root_path + '/lib/src/bardell_integral_fxixifxixi_c0c1.c',
    ]

extensions = [
    Extension('compmech.integrate.integrate',
        sources=[
            root_path + '/integrate/integrate.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.integrate.integratev',
        sources=[
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.imperfections.mgi',
        sources=[
            root_path + '/conecyl/imperfections/mgi.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.clpt.clpt_commons_bc1',
        sources=[
            root_path + '/conecyl/clpt/clpt_commons_bc1.pyx',
            ],
        depends=[
            root_path + '/conecyl/imperfections/mgi.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_commons_bc2',
        sources=[
            root_path + '/conecyl/clpt/clpt_commons_bc2.pyx',
            ],
        depends=[
            root_path + '/conecyl/imperfections/mgi.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_commons_bc3',
        sources=[
            root_path + '/conecyl/clpt/clpt_commons_bc3.pyx',
            ],
        depends=[
            root_path + '/conecyl/imperfections/mgi.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_commons_bc4',
        sources=[
            root_path + '/conecyl/clpt/clpt_commons_bc4.pyx',
            ],
        depends=[
            root_path + '/conecyl/imperfections/mgi.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.clpt.clpt_donnell_bc1_linear',
        sources=[
            root_path + '/conecyl/clpt/clpt_donnell_bc1_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_donnell_bc2_linear',
        sources=[
            root_path + '/conecyl/clpt/clpt_donnell_bc2_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_donnell_bc3_linear',
        sources=[
            root_path + '/conecyl/clpt/clpt_donnell_bc3_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_donnell_bc4_linear',
        sources=[
            root_path + '/conecyl/clpt/clpt_donnell_bc4_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.clpt.clpt_geier1997_bc2',
        sources=[
            root_path + '/conecyl/clpt/clpt_geier1997_bc2.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.clpt.clpt_donnell_bc1_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/clpt_donnell_bc1_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc1.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_donnell_bc2_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/clpt_donnell_bc2_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc2.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_donnell_bc3_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/clpt_donnell_bc3_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc3.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_donnell_bc4_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/clpt_donnell_bc4_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc4.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.clpt.clpt_sanders_bc1_linear',
        sources=[
            root_path + '/conecyl/clpt/clpt_sanders_bc1_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_sanders_bc2_linear',
        sources=[
            root_path + '/conecyl/clpt/clpt_sanders_bc2_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_sanders_bc3_linear',
        sources=[
            root_path + '/conecyl/clpt/clpt_sanders_bc3_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_sanders_bc4_linear',
        sources=[
            root_path + '/conecyl/clpt/clpt_sanders_bc4_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.clpt.clpt_sanders_bc1_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/clpt_sanders_bc1_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc1.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_sanders_bc2_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/clpt_sanders_bc2_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc2.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_sanders_bc3_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/clpt_sanders_bc3_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc3.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.clpt_sanders_bc4_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/clpt_sanders_bc4_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc4.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.iso_clpt_donnell_bc2_linear',
        sources=[
            root_path + '/conecyl/clpt/iso_clpt_donnell_bc2_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.iso_clpt_donnell_bc3_linear',
        sources=[
            root_path + '/conecyl/clpt/iso_clpt_donnell_bc3_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.iso_clpt_donnell_bc2_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/iso_clpt_donnell_bc2_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc2.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.clpt.iso_clpt_donnell_bc3_nonlinear',
        sources=[
            root_path + '/conecyl/clpt/iso_clpt_donnell_bc3_nonlinear.pyx',
            ],
        depends=[
            root_path + '/conecyl/clpt/clpt_commons_bc3.pyx',
            root_path + '/conecyl/imperfections/mgi.pyx',
            root_path + '/integrate/integratev.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.fsdt.fsdt_commons_bc1',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_commons_bc1.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_commons_bc2',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_commons_bc2.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_commons_bc3',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_commons_bc3.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_commons_bc4',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_commons_bc4.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_commons_bcn',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_commons_bcn.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.fsdt.fsdt_donnell_bc1_linear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bc1_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_donnell_bc2_linear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bc2_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_donnell_bc3_linear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bc3_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_donnell_bc4_linear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bc4_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_donnell_bcn_linear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bcn_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.fsdt.fsdt_geier1997_bc2',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_geier1997_bc2.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.fsdt.fsdt_sanders_bcn_linear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_sanders_bcn_linear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_shadmehri2012_bc2',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_shadmehri2012_bc2.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_shadmehri2012_bc3',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_shadmehri2012_bc3.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.conecyl.fsdt.fsdt_donnell_bc1_nonlinear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bc1_nonlinear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_donnell_bc2_nonlinear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bc2_nonlinear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_donnell_bc3_nonlinear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bc3_nonlinear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_donnell_bc4_nonlinear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bc4_nonlinear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),
    Extension('compmech.conecyl.fsdt.fsdt_donnell_bcn_nonlinear',
        sources=[
            root_path + '/conecyl/fsdt/fsdt_donnell_bcn_nonlinear.pyx',
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args_NL, extra_link_args=link_args,
        language='c'),

    Extension('compmech.panel.models.clt_bardell_field',
        sources=[
            root_path + '/panel/models/clt_bardell_field.pyx',
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.panel.models.clt_bardell_field_w',
        sources=[
            root_path + '/panel/models/clt_bardell_field_w.pyx',
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.models.cpanel_clt_donnell_bardell',
        sources=[
            root_path + '/panel/models/cpanel_clt_donnell_bardell.pyx',
            bardell_int_src,
            bardell_func_src,
            ] + bardell_int12_src,
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.models.cpanel_clt_donnell_bardell_num',
        sources=[
            root_path + '/panel/models/cpanel_clt_donnell_bardell_num.pyx',
            legendre_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.models.kpanel_clt_donnell_bardell',
        sources=[
            root_path + '/panel/models/kpanel_clt_donnell_bardell.pyx',
            bardell_int_src,
            ] + bardell_int12_src,
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    #TODO kpanel is not completely developed yet
    #root_path + '/panel/models/kpanel_clt_donnell_bardell_num.pyx',
    Extension('compmech.panel.models.plate_clt_donnell_bardell',
        sources=[
            root_path + '/panel/models/plate_clt_donnell_bardell.pyx',
            bardell_int_src,
            ] + bardell_int12_src,
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.models.plate_clt_donnell_bardell_num',
        sources=[
            root_path + '/panel/models/plate_clt_donnell_bardell_num.pyx',
            legendre_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.models.plate_clt_donnell_bardell_w',
        sources=[
            root_path + '/panel/models/plate_clt_donnell_bardell_w.pyx',
            bardell_int_src,
            ] + bardell_int12_src,
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.panel.connections.kCSSxcte',
        sources=[
            root_path + '/panel/connections/kCSSxcte.pyx',
            bardell_int_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.connections.kCSSycte',
        sources=[
            root_path + '/panel/connections/kCSSycte.pyx',
            bardell_int_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.connections.kCBFycte',
        sources=[
            root_path + '/panel/connections/kCBFycte.pyx',
            bardell_int_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.connections.kCSB',
        sources=[
            root_path + '/panel/connections/kCSB.pyx',
            bardell_int_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.connections.kCBFxcte',
        sources=[
            root_path + '/panel/connections/kCBFxcte.pyx',
            bardell_int_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.panel.connections.kCLTxycte',
        sources=[
            root_path + '/panel/connections/kCLTxycte.pyx',
            bardell_int_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),

    Extension('compmech.stiffener.models.bladestiff1d_clt_donnell_bardell',
        sources=[
            root_path + '/stiffener/models/bladestiff1d_clt_donnell_bardell.pyx',
            legendre_src,
            bardell_int_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.stiffener.models.bladestiff2d_clt_donnell_bardell',
        sources=[
            root_path + '/stiffener/models/bladestiff2d_clt_donnell_bardell.pyx',
            legendre_src,
            bardell_int_src,
            bardell_func_src,
            ],
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    Extension('compmech.stiffener.models.tstiff2d_clt_donnell_bardell',
        sources=[
            root_path + '/stiffener/models/tstiff2d_clt_donnell_bardell.pyx',
            legendre_src,
            bardell_int_src,
            bardell_func_src,
            ] + bardell_int12_src + bardell_intc0c1_src,
        include_dirs=include_dirs, library_dirs=library_dirs,
        extra_compile_args=compiler_args, extra_link_args=link_args,
        language='c'),
    ]

ext_modules = cythonize(extensions,
        compiler_directives={'linetrace': True},
        language_level='3',
        )

s = setup(
    name = "compmech",
    version = fullversion,
    author = "Saullo G. P. Castro",
    author_email = "S.G.P.Castro@tudelft.nl",
    description = DOCLINES[0],
    long_description = read('README.md'),
    long_description_content_type = 'text/markdown',
    license = "BSD",
    download_url='https://github.com/saullocastro/compmech/releases',
    keywords=['computational', 'mechanics', 'structural', 'analysis',
        'analytical', 'Bardell', 'Ritz'],
    url='http://saullocastro.github.io/compmech/',
    package_data = package_data,
    data_files = data_files,
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires = install_requires,
    ext_modules = ext_modules,
    packages = find_packages(),
)
