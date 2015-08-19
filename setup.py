from glob import glob
import sys
import os
from subprocess import Popen

#params = 'build_ext -inplace -IC:\clones\cubature\cubature ' + ' '.join(sys.argv[1:])
params = 'build_ext --inplace ' + ' '.join(sys.argv[1:]) + ' clean'

cwd = os.getcwd()

print('####################')
print('Compiling modules...')
print('####################')
print('')
basedirs = [os.path.join('compmech', 'conecyl', 'clpt'),
            os.path.join('compmech', 'conecyl', 'fsdt'),
            os.path.join('compmech', 'integrate'),
            os.path.join('compmech', 'conecyl', 'imperfections'),
            os.path.join('compmech', 'aero', 'pistonplate', 'clpt'),
           ]

for basedir in basedirs:
    print('Compiling setup.py in %s' % basedir)
    basedir = os.path.sep.join([cwd, basedir])
    os.chdir(basedir)
    for fname in glob('setup_*.py'):
        p = Popen(('python {} '.format(fname) + params), shell=True)
        p.wait()
os.chdir(cwd)


