import os

os.environ['PATH'] = '../../compmech/lib;' + os.environ['PATH']

import test
test.test()
