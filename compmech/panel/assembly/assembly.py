import numpy as np
from numpy import linspace

from compmech.logger import msg, warn
from compmech.constants import DOUBLE
import compmech.panel.modelDB as modelDB


class PanelAssembly(object):
    def __init__(self, panels, connectivity):
        self.panels = panels
        self.connectivity = connectivity
        self.size = None
        self.out_num_cores = 4


    def get_size(self):
        self.size = sum([3*p.m*p.n for p in panels])
        return self.size


    def plot(self, c, group, invert_y=False, vec='w', filename='', ax=None,
            figsize=(3.5, 2.), save=True, title='',
            identify=False, show_boundaries=False,
            boundary_line='--k', boundary_linewidth=1.,
            colorbar=False, cbar_nticks=2, cbar_format=None, cbar_title='',
            cbar_fontsize=10, aspect='equal', clean=True, dpi=400, texts=[],
            xs=None, ys=None, gridx=50, gridy=50, num_levels=400,
            vecmin=None, vecmax=None, calc_data_only=False):
        r"""Contour plot for a Ritz constants vector.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together.
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``
        invert_y : bool, optional
            Inverts the `y` axis of the plot.
        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
        dpi : int, optional
            Resolution of the saved file in dots per inch.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            the `name` parameter of the ``Panel`` object will be used.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it.
        figsize : tuple, optional
            The figure size given by ``(width, height)``.
        title : str, optional
            If any string is given a title is added to the contour plot.
        indentify : bool, optional
            If domains should be identified. If yes, the name of each panel is
            used.
        show_boundaries : bool, optional
            If boundaries between domains should be drawn.
        boundary_line : str, optional
            Matplotlib string to define line type and color.
        boundary_linewidth : float, optional
            Matplotlib float to define line width.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method.
        clean : bool, optional
            Clean axes ticks, grids, spines etc.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
        vecmin : float, optional
            Minimum value for the contour scale (useful to compare with other
            results). If not specified it will be taken from the calculated
            field.
        vecmax : float, optional
            Maximum value for the contour scale.
        calc_data_only : bool, optional
            If only calculated data should be returned.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.
        data : dict
            Data calculated during the plotting procedure.

        """
        msg('Plotting contour...')

        import matplotlib.pyplot as plt
        import matplotlib

        msg('Computing field variables...', level=1)
        displs = ['u', 'v', 'w', 'phix', 'phiy']
        strains = ['exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz']
        stresses = ['Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx']
        if vec in displs:
            res = self.uvw(c, group, gridx=gridx, gridy=gridy)
            field = np.array(res[vec])
        elif vec in strains:
            raise NotImplementedError('Strains not implemented')
        elif vec in stresses:
            raise NotImplementedError('Stresses not implemented')
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))
        msg('Finished!', level=1)

        if vecmin is None:
            vecmin = field.min()
        if vecmax is None:
            vecmax = field.max()

        data = dict(vecmin=vecmin, vecmax=vecmax)

        if calc_data_only:
            return None, data

        levels = linspace(vecmin, vecmax, num_levels)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            if isinstance(ax, matplotlib.axes.Axes):
                ax = ax
                fig = ax.figure
                save = False
            else:
                raise ValueError('ax must be an Axes object')

        if invert_y == True:
            ax.invert_yaxis()
        ax.invert_xaxis()

        count = -1
        for i, panel in enumerate(self.panels):
            if panel.group != group:
                continue
            count += 1
            xplot = res['y'][count] + panel.y0
            yplot = res['x'][count] + panel.x0
            field = res[vec][count]
            contour = ax.contourf(xplot, yplot, field, levels=levels)
            if identify:
                ax.text(xplot.mean(), yplot.mean(), 'P {0:02d}'.format(i+1),
                        transform=ax.transData, ha='center')
            if show_boundaries:
                x1, x2 = xplot.min(), xplot.max()
                y1, y2 = yplot.min(), yplot.max()
                ax.plot((x1, x2), (y1, y1), boundary_line, lw=boundary_linewidth)
                ax.plot((x1, x2), (y2, y2), boundary_line, lw=boundary_linewidth)
                ax.plot((x1, x1), (y1, y2), boundary_line, lw=boundary_linewidth)
                ax.plot((x2, x2), (y1, y2), boundary_line, lw=boundary_linewidth)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbarticks = linspace(vecmin, vecmax, cbar_nticks)
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            cbar.outline.remove()
            cbar.ax.tick_params(labelsize=fsize, pad=0., tick2On=False)

        if title != '':
            ax.set_title(str(title))

        fig.tight_layout()
        ax.set_aspect(aspect)

        ax.grid(False)
        ax.set_frame_on(False)
        if clean:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        else:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        for kwargs in texts:
            ax.text(transform=ax.transAxes, **kwargs)

        if save:
            if not filename:
                filename = group + '.png'
            fig.savefig(filename, transparent=True,
                        bbox_inches='tight', pad_inches=0.05, dpi=dpi)
            plt.close()

        msg('finished!')

        return ax, data


    def uvw(self, c, group, gridx=50, gridy=50):
        r"""Calculates the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the ``Panel`` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.

        Returns
        -------
        out : tuple
            A tuple of ``np.ndarrays`` containing
            ``(xs, ys, u, v, w, phix, phiy)``.

        Notes
        -----
        The returned values ``u```, ``v``, ``w``, ``phix``, ``phiy`` are
        stored as parameters with the same name in the ``Panel`` object.

        """
        def default_field(panel):
            xs = linspace(0, panel.a, gridx)
            ys = linspace(0, panel.b, gridy)
            xs, ys = np.meshgrid(xs, ys, copy=False)
            xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
            ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
            shape = xs.shape
            xs = xs.ravel()
            ys = ys.ravel()
            return xs, ys, shape

        res = dict(x=[], y=[], u=[], v=[], w=[], phix=[], phiy=[])
        for panel in self.panels:
            if panel.group != group:
                continue
            cpanel = c[panel.col_start: panel.col_end]
            cpanel = np.ascontiguousarray(cpanel, dtype=DOUBLE)
            m = panel.m
            n = panel.n
            a = panel.a
            b = panel.b
            model = panel.model
            fuvw = modelDB.db[model]['field'].fuvw
            x, y, shape = default_field(panel)
            u, v, w, phix, phiy = fuvw(cpanel, m, n, a, b,
                    np.ascontiguousarray(x),
                    np.ascontiguousarray(y),
                    self.out_num_cores)
            res['x'].append(x.reshape(shape))
            res['y'].append(y.reshape(shape))
            res['u'].append(u.reshape(shape))
            res['v'].append(v.reshape(shape))
            res['w'].append(w.reshape(shape))
            res['phix'].append(phix.reshape(shape))
            res['phiy'].append(phiy.reshape(shape))

        return res

