r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.plate.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *
#from fsdt import *

db = {
    'clpt_donnell_bardell_w': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_bardell_w,
                    'linear': clpt_donnell_bardell_w_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 0,
                    'num1': 1,
                    },
    'clpt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_bardell,
                    'linear': clpt_donnell_bardell_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 0,
                    'num1': 3,
                    },
    'clpt_donnell_bc1': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_bc1,
                    'linear': clpt_donnell_bc1_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 2,
                    'num1': 3,
                    },
    'clpt_donnell_free': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_free,
                    'linear': clpt_donnell_free_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 2,
                    'num1': 4,
                    },
    }

def get_linear_matrices(kp):
    r"""Obtain the right functions to calculate hte linear matrices
    for a given model.

    The ``model`` parameter of the ``ConeCyl`` object is used to search for
    the functions ``fG0``, ``fkG0``, and the matrix ``k0edges`` is calculated,
    when applicable.

    Parameters
    ----------
    kp : compmech.plate.Plate
        The ``Plate`` object.

    Returns
    -------
    out : tuple
        A tuple containing ``(fk0, fkG0, k0edges)``.

    """
    a = kp.a
    b = kp.b
    m1 = kp.m1
    n1 = kp.n1
    model = kp.model

    try:
        fk0edges = db[model]['linear'].fk0edges
    except AttributeError:
        k0edges = None

    if 'free' in model:
        fk0edges = db[model]['linear'].fk0edges
        k0edges = fk0edges(m1, n1, a, b,
                           kp.kuBot, kp.kuTop,
                           kp.kvBot, kp.kvTop,
                           kp.kwBot, kp.kwTop,
                           kp.kphixBot, kp.kphixTop,
                           kp.kphiyBot, kp.kphiyTop,
                           kp.kuLeft, kp.kuRight,
                           kp.kvLeft, kp.kvRight,
                           kp.kwLeft, kp.kwRight,
                           kp.kphixLeft, kp.kphixRight,
                           kp.kphiyLeft, kp.kphiyRight)
    elif 'bc1' in model:
        fk0edges = db[model]['linear'].fk0edges
        k0edges = fk0edges(m1, n1, a, b,
                           kp.kphixBot, kp.kphixTop,
                           kp.kphiyLeft, kp.kphiyRight)
    elif 'bardell' in model:
        k0edges = None

    fk0 = db[model]['linear'].fk0
    fkG0 = db[model]['linear'].fkG0

    return fk0, fkG0, k0edges

