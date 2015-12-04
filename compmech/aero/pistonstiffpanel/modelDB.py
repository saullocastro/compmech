r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.aero.pistonstiffpanel.modelDB


"""
import numpy as np
from scipy.sparse import coo_matrix

from clpt import *

db = {
    'clpt_donnell_bc1': {
                    'linear static': False,
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
    'clpt_sanders_bc1': {
                    'linear static': False,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': clpt_commons_bc1,
                    'linear': clpt_sanders_bc1_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'num0': 0,
                    'num1': 3,
                    },
    }

def get_linear_matrices(p):
    r"""Obtain the right functions to calculate hte linear matrices
    for a given model.

    The ``model`` parameter of the ``ConeCyl`` object is used to search for
    functions ``fG0``, ``fkG0``, ``fkAx``, ``fkAy``, ``fkM``and the matrix
    ``k0edges`` is calculated, when applicable.

    Parameters
    ----------
    p : compmech.aero.pistonstiffpanel.AeroPistonPlate
        The ``AeroPistonPlate`` object.

    Returns
    -------
    out : tuple
        A tuple containing ``(fk0, fkG0, fkAx, fcA, fkAy, fkM, fk0edges,
        fk0sb, fk0sf, fkMsb, fkMsf)``.

    """
    model = p.model

    fk0 = db[model]['linear'].fk0
    fkG0 = db[model]['linear'].fkG0
    fkAx = db[model]['linear'].fkAx
    fkAy = db[model]['linear'].fkAy
    fcA = getattr(db[model]['linear'], 'fcA', None)
    fkM = db[model]['linear'].fkM
    fk0edges = db[model]['linear'].fk0edges
    fk0sb = db[model]['linear'].fk0sb
    fk0sf = db[model]['linear'].fk0sf
    fkMsb = db[model]['linear'].fkMsb
    fkMsf = db[model]['linear'].fkMsf

    return (fk0, fkG0, fkAx, fkAy, fcA, fkM, fk0edges, fk0sb, fk0sf, fkMsb,
            fkMsf)

