r"""
Interpolate (:mod:`compmech.interpolate`)
==================================================

.. currentmodule:: compmech.interpolate

This module includes some interpolation utilities that will be used in other
modules.

.. autofunction:: inv_weighted

.. autofunction:: interp

"""
from collections import Iterable

import numpy as np

from compmech.logger import msg, warn


def inv_weighted(data, mesh, num_sub, col, ncp=5, power_parameter=2):
    r"""Interpolates the values taken at one group of points into
    another using an inverse-weighted algorithm

    In the inverse-weighted algorithm a number of `n_{CP}` measured points
    of the input parameter ``data`` that are closest to a given node in
    the input parameter ``mesh`` are found and the imperfection value of
    this node (represented by the normal displacement `{w_0}_{node}`) is
    calculated as follows:

    .. math::
        {w_0}_{node} = \frac{\sum_{i}^{n_{CP}}{{w_0}_i\frac{1}{w_i}}}
                            {\sum_{i}^{n_{CP}}{\frac{1}{w_i}}}

    where `w_i` is the imperfection at each measured point, calculated as:

    .. math::
        w_i = \left[(x_{node}-x_i)^2+(y_{node}-y_i)^2+(z_{node}-y_i)^2
              \right]^p

    with `p` being a power parameter that when increased will increase the
    relative influence of a closest point.

    Parameters
    ----------
    data : numpy.ndarray, shape (N, ndim+1)
        The data or an array containing the imperfection file. The values
        to be interpolated must be in the last column.
    mesh : numpy.ndarray, shape (M, ndim)
        The new coordinates where the values will be interpolated to.
    num_sub : int
        The number of sub-sets used during the interpolation. The points
        are divided in sub-sets to increase the algorithm's efficiency.
    col : int
        The index of the column to be used in order to divide the data
        in sub-sets. Note that the first column index is ``0``.
    ncp : int, optional
        Number of closest points used in the inverse-weighted interpolation.
    power_parameter : float, optional
        Power of inverse weighted interpolation function.

    Returns
    -------
    ans : numpy.ndarray
        A 1-D array with the interpolated values. The size of this array
        is ``mesh.shape[0]``.

    """
    if mesh.shape[1] != data.shape[1]-1:
        raise ValueError('Invalid input: mesh.shape[1] != data.shape[1]')

    msg('Interpolating... ')
    num_sub = int(num_sub)
    mesh_size = mesh.shape[0]

    # memory control
    mem_limit = 1024*1024*1024*8*2    # 2 GB
    mem_entries = int(mem_limit / 64) # if float64 is used
    sec_size = int(mesh_size/num_sub)
    while sec_size**2*10 > mem_entries:
        num_sub +=1
        sec_size = int(mesh_size/num_sub)
        if sec_size**2*10 <= mem_entries:
            warn('New num_sub: {0}'.format(int(mesh_size/float(sec_size))))
            break

    mesh_seq = np.arange(mesh.shape[0])

    mesh_argsort = np.argsort(mesh[:, col])
    mesh_seq = mesh_seq[mesh_argsort]
    back_argsort = np.argsort(mesh_seq)

    mesh = np.asarray(mesh[mesh_argsort], order='F')

    length = mesh[:, col].max() - mesh[:, col].min()

    data = np.asarray(data[np.argsort(data[:, col])], order='F')

    ans = np.zeros(mesh.shape[0], dtype=mesh.dtype)

    # max_num_limits defines how many times the log will print
    # "processed ... out of ... entries"
    max_num_limits = 10
    for den in range(max_num_limits, 0, -1):
        if num_sub % den == 0:
            limit = int(num_sub/den)
            break

    for i in range(num_sub+1):
        i_inf = sec_size*i
        i_sup = sec_size*(i+1)

        if i % limit == 0:
            msg('\t processed {0:7d} out of {1:7d} entries'.format(
                  min(i_sup, mesh_size), mesh_size))
        sub_mesh = mesh[i_inf : i_sup]
        if not np.any(sub_mesh):
            continue
        inf = sub_mesh[:, col].min()
        sup = sub_mesh[:, col].max()

        tol = 0.03
        if i == 0 or i == num_sub:
            tol = 0.06

        while True:
            cond1 = data[:, col] >= inf - tol*length
            cond2 = data[:, col] <= sup + tol*length
            cond = np.all(np.array((cond1, cond2)), axis=0)
            sub_data = data[cond]
            if not np.any(sub_data):
                tol += 0.01
            else:
                break

        dist = np.subtract.outer(sub_mesh[:, 0], sub_data[:, 0])**2
        for j in range(1, sub_mesh.shape[1]):
            dist += np.subtract.outer(sub_mesh[:, j], sub_data[:, j])**2
        asort = np.argsort(dist, axis=1)
        lenn = sub_mesh.shape[0]
        lenp = sub_data.shape[0]
        asort_mesh = asort + np.meshgrid(np.arange(lenn)*lenp,
                                         np.arange(lenp))[0].transpose()
        # getting the distance of the closest points
        dist_cp = np.take(dist, asort_mesh[:, :ncp])
        # avoiding division by zero
        dist_cp[(dist_cp==0)] == 1.e-12
        # fetching the imperfection of the sub-data
        imp = sub_data[:, -1]
        # taking only the imperfection of the closest points
        imp_cp = np.take(imp, asort[:, :ncp])
        # weight calculation
        total_weight = np.sum(1./(dist_cp**power_parameter), axis=1)
        weight = 1./(dist_cp**power_parameter)
        # computing the new imp
        imp_new = np.sum(imp_cp*weight, axis=1)/total_weight
        # updating the answer array
        ans[i_inf : i_sup] = imp_new

    ans = ans[back_argsort]

    msg('Interpolation completed!')

    return ans


def interp(x, xp, fp, left=None, right=None, period=None):
    """
    One-dimensional linear interpolation

    Returns the one-dimensional piecewise linear interpolant to a function
    with given values at discrete data-points.

    .. note:: This function has been incorporated in NumPy >= 1.10.0 and will be soon
              removed from here.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the interpolated values.
    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument
        ``period`` is not specified. Otherwise, ``xp`` is internally sorted
        after normalizing the periodic boundaries with ``xp = xp % period``.
    fp : 1-D sequence of floats
        The y-coordinates of the data points, same length as ``xp``.
    left : float, optional
        Value to return for ``x < xp[0]``, default is ``fp[0]``.
    right : float, optional
        Value to return for ``x > xp[-1]``, default is ``fp[-1]``.
    period : float, optional
        A period for the x-coordinates. This parameter allows the proper
        interpolation of angular x-coordinates. Parameters ``left`` and
        ``right`` are ignored if ``period`` is specified.

    Returns
    -------
    y : {float, ndarray}
        The interpolated values, same shape as ``x``.

    Raises
    ------
    ValueError
        If ``xp`` and ``fp`` have different length
        If ``xp`` or ``fp`` are not 1-D sequences
        If ``period==0``

    Notes
    -----
    Does not check that the x-coordinate sequence ``xp`` is increasing.
    If ``xp`` is not increasing, the results are nonsense.
    A simple check for increasing is::

        np.all(np.diff(xp) > 0)


    Examples
    --------
    >>> xp = [1, 2, 3]
    >>> fp = [3, 2, 0]
    >>> interp(2.5, xp, fp)
    1.0
    >>> interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
    array([ 3. ,  3. ,  2.5 ,  0.56,  0. ])
    >>> UNDEF = -99.0
    >>> interp(3.14, xp, fp, right=UNDEF)
    -99.0

    Plot an interpolant to the sine function:

    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.sin(x)
    >>> xvals = np.linspace(0, 2*np.pi, 50)
    >>> yinterp = interp(xvals, x, y)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(xvals, yinterp, '-x')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    Interpolation with periodic x-coordinates:

    >>> x = [-180, -170, -185, 185, -10, -5, 0, 365]
    >>> xp = [190, -190, 350, -350]
    >>> fp = [5, 10, 3, 4]
    >>> interp(x, xp, fp, period=360)
    array([7.5, 5., 8.75, 6.25, 3., 3.25, 3.5, 3.75])

    """
    if period is None:
        return np.interp(x, xp, fp, left, right)
    else:
        if period==0:
            raise ValueError('Argument `period` must be a non-zero value')
        period = abs(period)
        if not isinstance(x, Iterable):
            x = [x]
        x = np.asarray(x)
        xp = np.asarray(xp)
        fp = np.asarray(fp)
        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError('Data points must be 1-D sequences')
        if xp.shape[0] != fp.shape[0]:
            raise ValueError('Inputs `xp` and `fp` must have the same shape')
        # eliminating discontinuity between periods
        x = x % period
        xp = xp % period
        asort_xp = np.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = np.concatenate((xp[-1:]-period, xp, xp[0:1]+period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))
        return np.interp(x, xp, fp)

if __name__=='__main__':
    a = np.array([[1.1, 1.2, 10],
                  [1.2, 1.2, 10],
                  [1.3, 1.3, 10],
                  [1.4, 1.3, 10],
                  [1.5, 1.3, 10],
                  [2.6, 2.3, 5],
                  [2.7, 2.3, 5],
                  [2.6, 2.1, 5],
                  [2.7, 2.1, 5],
                  [2.8, 2.2, 5],
                  [2.8, 2.2, 5],
                  [5.6, 5.3, 20],
                  [5.7, 5.3, 20],
                  [5.6, 5.1, 20],
                  [5.7, 5.1, 20],
                  [5.8, 5.2, 20],
                  [5.8, 5.2, 20]])

    b = np.array([[1., 1.],
                  [2., 2.],
                  [4., 4.],
                  [5., 5.]])

    print(inv_weighted(a, b, num_sub=1, col=1, ncp=10))

