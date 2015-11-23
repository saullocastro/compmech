r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.panels.cpanel.modelDB


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
    }

def get_linear_matrices(cp):
    r"""Obtain the right functions to calculate hte linear matrices
    for a given model.

    The ``model`` parameter of the ``ConeCyl`` object is used to search for
    the functions ``fG0``, ``fkG0``, and the matrix ``k0edges`` is calculated,
    when applicable.

    Parameters
    ----------
    cp : compmech.panels.cpanel.CPanel
        The ``CPanel`` object.

    Returns
    -------
    out : tuple
        A tuple containing ``(fk0, fkG0, k0edges)``.

    """
    a = cp.a
    b = cp.b
    r = cp.r
    m1 = cp.m1
    n1 = cp.n1
    model = cp.model

    try:
        fk0edges = db[model]['linear'].fk0edges
    except AttributeError:
        k0edges = None

    if 'bardell' in model:
        k0edges = None

    fk0 = db[model]['linear'].fk0
    fkG0 = db[model]['linear'].fkG0

    return fk0, fkG0, k0edges

