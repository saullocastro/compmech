from matplotlib.pyplot import *
from math import sqrt

m = 1/3.
xs = [+1, +1, -1, -1]
ys = [-1, +1, -1, +1]
figure(figsize=(4, 4))
ax = gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_aspect('equal')
ax.set_xlim(-1.4, +1.6)
ax.set_ylim(-1.4, +1.6)
ax.text(1.8, 0., r'$\xi$', transform=ax.transData, va='center')
ax.text(0., 1.8, r'$\eta$', rotation='horizontal', transform=ax.transData,
        ha='center')
ax.text(+1.1, +1.1, '$n_1$\n' + r'$(+1, +1)$', ha='center', va='bottom',
        fontsize=10)
ax.text(-1.1, +1.1, '$n_2$\n' + r'$(-1, +1)$', ha='center', va='bottom',
        fontsize=10)
ax.text(-1.1, -1.1, '$n_3$\n' + r'$(-1, -1)$', ha='center', va='top'  ,
        fontsize=10)
ax.text(+1.1, -1.1, '$n_4$\n' + r'$(+1, -1)$', ha='center', va='top'  ,
        fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
#ax.set_xticklabels(['-1', '+1'])
#ax.set_yticklabels(['-1', '+1'])
plot([1, -1, -1, 1, 1], [1, 1, -1, -1, 1], '-k')
plot(xs, ys, 'ok', mfc='k')
tight_layout()
#savefig('test.png')
show()
