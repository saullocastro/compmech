import os
from random import sample

import numpy as np
from numpy import cos
from scipy.linalg import lstsq

from compmech.constants import CMHOME
from compmech.logger import *

def load_c0(name, funcnum, m0, n0):
    path = os.path.join(CMHOME, 'conecyl', 'imperfections', 'c0',
            'c0_{0}_f{1}_m{2:03d}_n{3:03d}.txt'.format(
            name, funcnum, m0, n0))
    if os.path.isfile(path):
        return np.loadtxt(path)
    else:
        raise ValueError('Coefficient file not found!')

def calc_c0(path, m0=40, n0=40, funcnum=2, sample_size=None,
            maxmem=8, save=True, offset_w0=None):
    r"""Find the coefficients `c_0` that best fit the `w_0` function.

    The measured data will be fit using one of the following functions,
    selected using the ``funcnum`` parameter:

    ``funcnum=1``

    .. math::
        w_0 = \sum_{i=1}^{m_0}{ \sum_{j=0}^{n_0}{
                 c_{ij}^a sin{b_x} sin{b_\theta}
                +c_{ij}^b sin{b_x} cos{b_\theta}}}

    ``funcnum=2`` (default)

    .. math::
        w_0 = \sum_{i=0}^{m_0}{ \sum_{j=0}^{n_0}{
                c_{ij}^a cos{b_x} sin{b_\theta}
                +c_{ij}^b cos{b_x} cos{b_\theta}}}

    ``funcnum=3``

    .. math::
        w_0 = \sum_{i=0}^{m_0}{ \sum_{j=0}^{n_0}{
                 c_{ij}^a sin{b_x} sin{b_\theta}
                +c_{ij}^b sin{b_x} cos{b_\theta}
                +c_{ij}^c cos{b_x} sin{b_\theta}
                +c_{ij}^d cos{b_x} cos{b_\theta}}}

    where:

    .. math::
        b_x = i \pi \frac x L_{points}

        b_\theta = j \theta

    where `L_{points}` represents the difference between the maximum and
    the height values in the imperfection file divided by the cosine
    of the semi-vertex angle:

    .. math::
        L_{points} = \frac{H_{max} - H_{min}}{cos(\alpha)}
                   = \frac{H_{points}}{cos(\alpha)}

    In this form `{}^x/_{L_{points}}` will vary from `0.` (at the top)
    to `1.` (at the bottom).

    .. note:: Note that if the measured sample does not
              cover all the height, **it will be stretched**.

    The approximation can be written in matrix form as:

    .. math::
        w_0 = [g] \{c\}

    where `[g]` carries the base functions and `{c}` the respective
    amplitudes. The solution consists on finding the best `\{c\}` that minimizes
    the least-square error between the measured imperfection pattern and the
    `w_0` function.

    Parameters
    ----------
    path : str or numpy.ndarray
        The path of the file containing the data. Can be a full path using
        ``r"C:\Temp\inputfile.txt"``, for example.
        The input file must have 3 columns: `\theta`, `height`, `imp`;
        expressed in Cartesian coordinates.

        This input can also be a ``numpy.ndarray`` object, with
        `\theta`, `height`, `imp` in each corresponding column.
    m0 : int
        Number of terms along the meridian (`x`).
    n0 : int
        Number of terms along the circumference (`\theta`).
    funcnum : int, optional
        As explained above, selects the base functions used for
        the approximation.
    sample_size : int or None, optional
        Specifies how many points of the imperfection file should be used. If
        ``None`` all points will be used in the computations.
    maxmem : int, optional
        Maximum RAM memory in GB allowed to compute the base functions.
        The ``scipy.interpolate.lstsq`` will go beyond this limit.
    save : bool, optional
        If ``True`` saves the calculated coefficients in the
        ``compmech/conecyl/imperfections/c0`` folder.

    Returns
    -------
    out : numpy.ndarray
        A 1-D array with the best-fit coefficients.

    """
    import mgi

    if isinstance(path, np.ndarray):
        input_pts = path
        path = 'unnamed.txt'
    else:
        input_pts = np.loadtxt(path)

    if input_pts.shape[1] != 3:
        raise ValueError('Input does not have the format: "theta, x, imp"')

    log('Finding w0 coefficients for {0},\n\tusing funcnum {1}'.format(
        str(os.path.basename(path)), funcnum))

    if sample_size:
        num = input_pts.shape[0]
        if sample_size < num:
            input_pts = input_pts[sample(range(num), int(sample_size))]

    if funcnum==1:
        size = 2
    elif funcnum==2:
        size = 2
    elif funcnum==3:
        size = 4
    else:
        raise ValueError('Valid values for "funcnum" are 1, 2 or 3')

    maxnum = maxmem*1024*1024*1024*8/(64*size*m0*n0)
    num = input_pts.shape[0]
    if num >= maxnum:
        input_pts = input_pts[sample(range(num), int(maxnum))]
        warn('Reducing sample size from {0} to {1} ' +
             'due to the "maxmem" specified'.format(num, maxnum), level=1)

    thetas = input_pts[:, 0].copy()
    xs = input_pts[:, 1]
    w0pts = input_pts[:, 2]

    if offset_w0:
        w0pts += offset_w0

    # normalizing x
    xs = (xs - xs.min())/(xs.max() - xs.min())

    # inverting x to cope with the coordsys of the semi-analytical model
    xs = 1 - xs

    a = mgi.fa(m0, n0, xs, thetas, funcnum=funcnum)
    log('Base functions calculated', level=1)
    try:
        c0, residues, rank, s = lstsq(a, w0pts)
    except MemoryError:
        error('Reduce the "maxmem" parameter!')
    log('Finished scipy.linalg.lstsq', level=1)

    if save:
        name = '.'.join(os.path.basename(path).split('.')[0:-1])
        outpath = os.path.join(CMHOME, 'conecyl', 'imperfections', 'c0',
                'c0_{0}_f{1}_m{2:03d}_n{3:03d}.txt'.format(
                name, funcnum, m0, n0))
        np.savetxt(outpath, c0)

    return c0, residues

def fw0(m0, n0, c0, xs_norm, ts, funcnum=2):
    r"""Calculates the imperfection field `w_0` for a given input.

    Parameters
    ----------
    m0 : int
        The number of terms along the meridian.
    n0 : int
        The number of terms along the circumference.
    c0 : numpy.ndarray
        The coefficients of the imperfection pattern.
    xs_norm : numpy.ndarray
        The meridian coordinate (`x`) normalized to be between ``0.`` and
        ``1.``.
    ts : numpy.ndarray
        The angles in radians representing the circumferential coordinate
        (`\theta`).
    funcnum : int, optional
        The function used for the approximation (see the ``calc_c0`` function)

    Notes
    -----
    The inputs ``xs_norm`` and ``ts`` must be of the same size.

    If ``funcnum==1 or funcnum==2`` then ``size=2``, if ``funcnum==3`` then
    ``size=4`` and the inputs must satisfy ``c0.shape[0] == size*m0*n0``.

    """
    if xs_norm.shape != ts.shape:
        raise ValueError('xs_norm and ts must have the same shape')
    if funcnum==1:
        size = 2
    elif funcnum==2:
        size = 2
    elif funcnum==3:
        size = 4
    if c0.shape[0] != size*m0*n0:
        raise ValueError('Invalid c0 for the given m0 and n0!')
    import mgi
    w0s = mgi.fw0(m0, n0, c0, xs_norm.ravel(), ts.ravel(), funcnum)
    return w0s.reshape(xs_norm.shape)
