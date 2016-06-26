import pylab
import os
import _small
import _big
def get_all_defaults(small=True):
    if small:
        return _small.all_defaults
    else:
        return _big.all_defaults

def get_params(small=True):
    if small:
        return _small.params
    else:
        return _big.params

def get_defaults(number, small=True):
    all_defaults = get_all_defaults(small)
    defaults  = {}
    for param, values in all_defaults.iteritems():
        if type(values) == list:
            defaults[param] = values[number % len(values)]
        else:
            defaults[param] = values

    return defaults

def figname(x_label,y_label):
    x_label = x_label.replace('/','')
    y_label = y_label.replace('/','')
    x_label = x_label.replace('\\','')
    y_label = y_label.replace('\\','')
    x_label = x_label.replace('$','')
    y_label = y_label.replace('$','')
    x_label = x_label.replace('{','')
    y_label = y_label.replace('{','')
    x_label = x_label.replace('}','')
    y_label = y_label.replace('}','')
    y_label = y_label.replace('|','')
    ans = 'fig'+x_label[:5]+y_label[:5]+'_000.png'
    while True:
        if ans in os.listdir('.'):
            tmp = ans.split('_')
            ans = '_'.join(tmp[:-1])
            num = tmp[-1]
            num = int(num[:-4])
            num += 1
            ans += '_%03d.png' % num
        else:
            break
    return ans

def savefig(fig, fname='', small=True, figsize=None, **kwargs):
    ax = fig.axes[0]
    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()
    if not fname:
        fname = figname(x_label, y_label)
    if not figsize:
        params = get_params(small)
        figsize = params['figure.figsize']
    pylab.savefig(fname,
                  bbox_inches='tight',
                  figsize=figsize, **kwargs)

def myfig(fig, small=True, marker=False):
    all_defaults = get_all_defaults(small)
    params = get_params(small)
    pylab.rcParams.update(params)
    pylab.rc('font', **all_defaults['font'])
    for i, line in enumerate(fig.axes[0].lines):
        defaults = get_defaults(i)
        keys = defaults.keys()
        keys.sort()
        for k in keys:
            k2=k
            if k.find('_marker') > -1:
                if marker:
                    k2 = k[:-7]
                else:
                    continue
            attr = getattr(line, 'set_' + k2, 'NOTFOUND')
            if attr == 'NOTFOUND':
                continue
            attr(defaults[k])
    leg = fig.axes[0].legend()
    if leg != None:
        leg.set_visible(True)
    pylab.ion()
    return fig

def myplot(xs, ys, x_label='', y_label='',
           defaults=None,
           fig=None,
           label=None,
           defaults_number=0,
           small=True):
    if small:
        params = _small.params
    else:
        params = _big.params
    if fig==None:
        fig = pylab.figure()
    if defaults == None:
        defaults = get_defaults(defaults_number)
    xmin = 1.e6
    xmax = -1.e6
    ymin = 1.e6
    ymax = -1.e6
    pylab.rcParams.update(params)
    pylab.rc('font', **all_defaults['font'])
    pylab.plot(xs, ys,
               color = defaults['color'],
               label = label,
               linewidth = defaults['linewidth'],
               linestyle = defaults['linestyle'],
               marker = defaults['marker'],
               markerfacecolor = defaults['markerfacecolor'],
               markeredgecolor = defaults['markeredgecolor'],
               markeredgewidth = defaults['markeredgewidth'],
               markersize = defaults['markersize'])
    pylab.axis('scaled')
    pylab.legend(loc='upper right', ncol=2)
    pylab.xlabel(x_label, labelpad=10, ha = 'left')
    pylab.ylabel(y_label, labelpad=10, va='bottom')
    # in case we want to rescale the axes
    xmin = min(xmin, min(xs))
    xmax = max(xmax, max(xs))
    ymin = min(ymin, min(ys))
    ymax = max(ymax, max(ys))
    pylab.xlim(xmin,xmax*1.002)
    pylab.ylim(ymin,ymax*1.002)
    figsize = params['figure.figsize']
    pylab.savefig(figname(x_label,y_label),
                  bbox_inches='tight',
                  figsize=figsize)
    return fig



