r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.aero.pistonplate.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *
#from fsdt import *

db = {
    'clpt_donnell_bc1': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_bc1,
                    'linear': clpt_donnell_bc1_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 0,
                    'num1': 3,
                    },
    'clpt_donnell_bc4': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_bc4,
                    'linear': clpt_donnell_bc4_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 0,
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
                    'num0': 0,
                    'num1': 4,
                    },
    }

def get_linear_matrices(p):
    r"""Obtain the right functions to calculate hte linear matrices
    for a given model.

    The ``model`` parameter of the ``ConeCyl`` object is used to search for
    functions ``fG0``, ``fkG0``, ``fkA``, ``fkM``and the matrix ``k0edges`` is
    calculated, when applicable.

    Parameters
    ----------
    p : compmech.aero.pistonplate.AeroPistonPlate
        The ``AeroPistonPlate`` object.

    Returns
    -------
    out : tuple
        A tuple containing ``(fk0, fkG0, fkA, fkM, k0edges)``.

    """
    a = p.a
    b = p.b
    m1 = p.m1
    n1 = p.n1
    model = p.model

    try:
        fk0edges = db[model]['linear'].fk0edges
    except AttributeError:
        k0edges = None

    fk0edges = db[model]['linear'].fk0edges
    if 'bc1' in model:
        k0edges = fk0edges(m1, n1, a, b,
                           p.kphixBot, p.kphixTop,
                           p.kphiyLeft, p.kphiyRight)
    elif 'bc4' in model:
        k0edges = fk0edges(m1, n1, a, b,
                           p.kuBot, p.kuTop,
                           p.kvBot, p.kvTop,
                           p.kphixBot, p.kphixTop,
                           p.kuLeft, p.kuRight,
                           p.kvLeft, p.kvRight,
                           p.kphiyLeft, p.kphiyRight)
    elif 'free' in model:
        k0edges = fk0edges(m1, n1, a, b,
                           p.kuBot, p.kuTop,
                           p.kvBot, p.kvTop,
                           p.kwBot, p.kwTop,
                           p.kphixBot, p.kphixTop,
                           p.kphiyBot, p.kphiyTop,
                           p.kuLeft, p.kuRight,
                           p.kvLeft, p.kvRight,
                           p.kwLeft, p.kwRight,
                           p.kphixLeft, p.kphixRight,
                           p.kphiyLeft, p.kphiyRight)

    fk0 = db[model]['linear'].fk0
    fkG0 = db[model]['linear'].fkG0
    fkA = db[model]['linear'].fkA
    fkM = db[model]['linear'].fkM

    return fk0, fkG0, fkA, fkM, k0edges

