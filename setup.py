from glob import glob
import sys
import os
from subprocess import Popen

#params = 'build_ext -i -IC:\clones\cubature\cubature ' + ' '.join(sys.argv[1:])
params = 'build_ext -i ' + ' '.join(sys.argv[1:]) + ' clean'

cwd = os.getcwd()

print('####################')
print('Compiling modules...')
print('####################')
print('')
basedirs = [r'\compmech\conecyl\clpt', r'\compmech\conecyl\fsdt',
            r'\compmech\integrate',
            r'\compmech\conecyl\imperfections']
for basedir in basedirs:
    basedir = os.path.sep.join([cwd, basedir])
    os.chdir(basedir)
    for fname in glob('setup_*.py'):
        p = Popen(('python {} '.format(fname) + params), shell=True)
        p.wait()
os.chdir(cwd)


