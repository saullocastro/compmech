import os
import sys
from subprocess import Popen

#params = 'build_ext -i -IC:\clones\cubature\cubature ' + ' '.join(sys.argv[1:])
params = 'build_ext -i ' + ' '.join(sys.argv[1:])
for fname in os.listdir('.'):
    if 'setup_' in fname:
        p = Popen(('python {} '.format(fname) + params), shell=True)
        p.wait()
