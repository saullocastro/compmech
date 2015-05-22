r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.panels.kpanels.kpanelt.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *
#from fsdt import *

db = {
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

def get_linear_matrices(kp):
    r"""Obtain the right functions to calculate hte linear matrices
    for a given model.

    The ``model`` parameter of the ``ConeCyl`` object is used to search
    for the functions ``fG0``, ``fG0_cyl``, ``fkG0``, ``fkG0_cyl``,
    and the matrix ``k0edges`` is calculated, when applicable.

    Parameters
    ----------
    kp : compmech.panels.kpanels.kpanelt.KPanelT
        The ``KPanelT`` object.

    Returns
    -------
    out : tuple
        A tuple containing ``(fk0, fk0_cyl, fkG0, fkG0_cyl, k0edges)``.

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
        fk0edges_cyl = db[model]['linear'].fk0edges
        k0edges = fk0edges_cyl(m1, n1, a, b,
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

    fk0 = db[model]['linear'].fk0
    fkG0 = db[model]['linear'].fkG0

    return fk0, fkG0, k0edges

