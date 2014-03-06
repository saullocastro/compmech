import os
#os.chdir('.')
print os.path.realpath(os.curdir)
old = 'kG'
new = 'kG0'
for name in os.listdir('.'):
    if name.find( old ) > -1:
        namenew = name.replace( old, new )
        print name, namenew
        if os.path.isfile( name ):
            if not os.path.isfile( namenew ):
                os.rename( name, namenew )
            else:
                print 'already exists %s' % namenew

        else:
            print 'not found' % name
import time
time.sleep(2)
