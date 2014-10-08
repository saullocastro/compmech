r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.conecyl.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *
from fsdt import *

db = {
    'fsdt_donnell_bc1': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': fsdt_commons_bc1,
                    'linear': fsdt_donnell_bc1_linear,
                    'non-linear': None,
                    'dofs': 5,
                    'e_num': 8,
                    'num0': 4,
                    'num1': 2,
                    'num2': 2,
                    'num3': 2,
                    'num4': 5,
                    },
    'fsdt_donnell_free': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': fsdt_commons_free,
                    'linear': fsdt_donnell_free_linear,
                    'non-linear': None,
                    'dofs': 5,
                    'e_num': 8,
                    'num0': 1,
                    'num1': 5,
                    'num2': 5,
                    'num3': 5,
                    'num4': 5,
                    },
    }

def get_linear_matrices(cc, combined_load_case=None):
    r"""Obtain the right functions to calculate hte linear matrices
    for a given model.

    The ``model`` parameter of the ``ConeCyl`` object is used to search
    for the functions ``fG0``, ``fG0_cyl``, ``fkG0``, ``fkG0_cyl``,
    and the matrix ``k0edges`` is calculated, when applicable.

    Parameters
    ----------
    cc : compmech.conecyl.ConeCyl
        The ``ConeCyl`` object.
    combined_load_case : int, optional
        As explained in the :meth:`ConeCyl.lb() <compmech.conecyl.ConeCyl.lb>`
        method, the integer indicating
        which combined load case should be used. Default is ``None``.

    Returns
    -------
    out : tuple
        A tuple containing ``(fk0, fk0_cyl, fkG0, fkG0_cyl, k0edges)``.

    """
    r1 = cc.r1
    r2 = cc.r2
    tmin = cc.tminrad
    tmax = cc.tmaxrad
    L = cc.L
    m2 = cc.m2
    n3 = cc.n3
    m4 = cc.m4
    n4 = cc.n4
    model = cc.model

    try:
        fk0edges = db[model]['linear'].fk0edges
    except AttributeError:
        k0edges = None

    if model=='fsdt_donnell_bc1':
        k0edges = fk0edges(m2, n3, m4, n4, r1, r2, L, tmin, tmax,
                           cc.kphixBot, cc.kphixTop,
                           cc.kphitBot, cc.kphitTop,
                           cc.kphixLeft, cc.kphixRight,
                           cc.kphitLeft, cc.kphitRight)
    elif model=='fsdt_donnell_free':
        k0edges = fk0edges(m2, n3, m4, n4, r1, r2, L, tmin, tmax,
                           cc.kuBot, cc.kuTop,
                           cc.kvBot, cc.kvTop,
                           cc.kwBot, cc.kwTop,
                           cc.kphixBot, cc.kphixTop,
                           cc.kphitBot, cc.kphitTop,
                           cc.kuLeft, cc.kuRight,
                           cc.kvLeft, cc.kvRight,
                           cc.kwLeft, cc.kwRight,
                           cc.kphixLeft, cc.kphixRight,
                           cc.kphitLeft, cc.kphitRight)

    fk0 = db[model]['linear'].fk0
    fk0_cyl = db[model]['linear'].fk0_cyl
    fkG0 = db[model]['linear'].fkG0
    fkG0_cyl = db[model]['linear'].fkG0_cyl

    return fk0, fk0_cyl, fkG0, fkG0_cyl, k0edges

