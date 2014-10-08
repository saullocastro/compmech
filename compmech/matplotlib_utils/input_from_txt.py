from collections import Iterable

import matplotlib.pyplot as plt

import plot_defaults

def input_from_txt(file_name, small = True):
    all_defaults = plot_defaults.get_all_defaults(small)
    file = open(file_name, 'r')
    lines = file.read().splitlines()
    file.close()
    curves = {}
    index = -1
    append_x = False
    append_y = False
    number = -1
    index = -1
    while True:
        index += 1
        if index > len(lines)-1:
            break
        line = lines[index]
        #TODO remove this brute force check
        try:
            line = float(line)
        except:
            line = line.strip().lower()
        if line in all_defaults.keys():
            index += 1
            curves[name][line] = lines[index].strip().lower()
        if line == 'label':
            append_x = False
            append_y = False
            number += 1
            index += 1
            line = lines[index]
            name = '%02d_' % number + line
            curves[name] = {'x':[], 'y':[]}
            continue
        if line == "":
            append_x = False
            append_y = False
        if line == 'x':
            append_x = True
            append_y = False
            continue
        if line == 'y':
            append_x = False
            append_y = True
            continue
        if append_x:
            curves[name]['x'].append(float(line))
        if append_y:
            curves[name]['y'].append(float(line))
    return curves

def create_fig(file_name,
               small=True,
               marker=False,
               figsize=None,
               nrows=1,
               ncols=1,
               sharex=False,
               sharey=False):
    if not isinstance(file_name, Iterable) or isinstance(file_name, str):
        file_name = [file_name]
    all_defaults = plot_defaults.get_all_defaults(small)
    params = plot_defaults.get_params(small)
    plt.rcParams.update(params)
    #plt.rc('font', **all_defaults['font'])
    if figsize:
        if ncols>1 or nrows>1:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                     sharex=sharex, sharey=sharey,
                                     figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)
            axes = fig.add_subplot(111)
    else:
        if ncols>1 or nrows>1:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                     sharex=sharex, sharey=sharey)
        else:
            fig = plt.figure()
            axes = fig.add_subplot(111)
    if not isinstance(axes, Iterable):
        axes = [axes]
    for i, f_name in enumerate(file_name):
        curves = input_from_txt(file_name=f_name, small=False)
        labels = sorted(curves.keys())
        xmin = 1.e6
        xmax = -1.e6
        ymin = 1.e6
        ymax = -1.e6
        for label in labels:
            curve = curves[label]
            number = label.split('_')[0]
            label  = '_'.join(label.split('_')[1:])
            number = int(number)
            defaults = plot_defaults.get_defaults(number, small=False)
            def current_default(key):
                if key in curve.keys():
                    return curve[key]
                else:
                    if key in defaults.keys():
                        return defaults[key]
                    else:
                        if marker:
                            key += '_marker'
                            if key in defaults.keys():
                                return defaults[key]
                            else:
                                return None
            axes[i].plot(curve['x'], curve['y'],
                         color = current_default('color'),
                         label = label,
                         linewidth = current_default('linewidth'),
                         linestyle = current_default('linestyle'),
                         marker = current_default('marker'),
                         markerfacecolor = current_default('markerfacecolor'),
                         markeredgecolor = current_default('markeredgecolor'),
                         markeredgewidth = current_default('markeredgewidth'),
                         markersize = current_default('markersize'))

    return fig

