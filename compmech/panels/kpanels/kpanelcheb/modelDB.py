r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.conecyl.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *
from fsdt import *

db = {
    'clpt_donnell_cheb': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_cheb,
                    'linear': clpt_donnell_cheb_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 0,
                    'num1': 75,
                    },
    }

def get_linear_matrices(kp, combined_load_case=None):
    r"""Obtain the right functions to calculate hte linear matrices
    for a given model.

    The ``model`` parameter of the ``ConeCyl`` object is used to search
    for the functions ``fG0``, ``fG0_cyl``, ``fkG0``, ``fkG0_cyl``,
    and the matrix ``k0edges`` is calculated, when applicable.

    Parameters
    ----------
    kp : compmech.conecyl.ConeCyl
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
    r1 = kp.r1
    r2 = kp.r2
    tmin = kp.tminrad
    tmax = kp.tmaxrad
    L = kp.L
    m1 = kp.m1
    n1 = kp.n1
    model = kp.model
    alpharad = kp.alpharad
    s = kp.s

    try:
        fk0edges = db[model]['linear'].fk0edges
    except AttributeError:
        k0edges = None

    if 'cheb' in model:
        if kp.is_cylinder:
            fk0edges_cyl = db[model]['linear'].fk0edges_cyl
            k0edges = fk0edges_cyl(m1, n1, r1, L,
                               tmin, tmax,
                               kp.kuBot, kp.kuTop,
                               kp.kvBot, kp.kvTop,
                               kp.kwBot, kp.kwTop,
                               kp.kphixBot, kp.kphixTop,
                               kp.kphitBot, kp.kphitTop,
                               kp.kuLeft, kp.kuRight,
                               kp.kvLeft, kp.kvRight,
                               kp.kwLeft, kp.kwRight,
                               kp.kphixLeft, kp.kphixRight,
                               kp.kphitLeft, kp.kphitRight)
        else:
            k0edges = fk0edges(m1, n1, alpharad,
                               s, r1, r2, L, tmin, tmax,
                               kp.kuBot, kp.kuTop,
                               kp.kvBot, kp.kvTop,
                               kp.kwBot, kp.kwTop,
                               kp.kphixBot, kp.kphixTop,
                               kp.kphitBot, kp.kphitTop,
                               kp.kuLeft, kp.kuRight,
                               kp.kvLeft, kp.kvRight,
                               kp.kwLeft, kp.kwRight,
                               kp.kphixLeft, kp.kphixRight,
                               kp.kphitLeft, kp.kphitRight)

    fk0 = db[model]['linear'].fk0
    fk0_cyl = db[model]['linear'].fk0_cyl
    fkG0 = db[model]['linear'].fkG0
    fkG0_cyl = db[model]['linear'].fkG0_cyl

    return fk0, fk0_cyl, fkG0, fkG0_cyl, k0edges

