r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: compmech.conecyl.modelDB


"""
from __future__ import absolute_import

from . clpt import *
from . fsdt import *

db = {
    'clpt_donnell_bc1': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc1,
                    'linear': clpt_donnell_bc1_linear,
                    'non-linear': clpt_donnell_bc1_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'clpt_donnell_bc2': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc2,
                    'linear': clpt_donnell_bc2_linear,
                    'non-linear': clpt_donnell_bc2_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'iso_clpt_donnell_bc2': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc2,
                    'linear': iso_clpt_donnell_bc2_linear,
                    'non-linear': iso_clpt_donnell_bc2_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'clpt_donnell_bc3': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc3,
                    'linear': clpt_donnell_bc3_linear,
                    'non-linear': clpt_donnell_bc3_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'iso_clpt_donnell_bc3': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc3,
                    'linear': iso_clpt_donnell_bc3_linear,
                    'non-linear': iso_clpt_donnell_bc3_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'clpt_donnell_bc4': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc4,
                    'linear': clpt_donnell_bc4_linear,
                    'non-linear': clpt_donnell_bc4_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'clpt_donnell_bcn': {
                    'linear static': True,
                    'linear buckling': False,
                    'non-linear static': None,
                    'commons': clpt_commons_bcn,
                    'linear': clpt_donnell_bcn_linear,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 8,
                    },
    'clpt_sanders_bc1': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc1,
                    'linear': clpt_sanders_bc1_linear,
                    'non-linear': clpt_sanders_bc1_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'clpt_sanders_bc2': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc2,
                    'linear': clpt_sanders_bc2_linear,
                    'non-linear': clpt_sanders_bc2_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'clpt_sanders_bc3': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc3,
                    'linear': clpt_sanders_bc3_linear,
                    'non-linear': clpt_sanders_bc3_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'clpt_sanders_bc4': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': clpt_commons_bc4,
                    'linear': clpt_sanders_bc4_linear,
                    'non-linear': clpt_sanders_bc4_nonlinear,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 3,
                    'num2': 6,
                    },
    'clpt_geier1997_bc2': {
                    'linear static': None,
                    'linear buckling': True,
                    'non-linear static': None,
                    'commons': clpt_geier1997_bc2,
                    'linear': clpt_geier1997_bc2,
                    'non-linear': None,
                    'dofs': 3,
                    'e_num': 6,
                    'i0': 0,
                    'j0': 0,
                    'num0': 0,
                    'num1': 0,
                    'num2': 3,
                    },
    'fsdt_donnell_bcn': {
                    'linear static': True,
                    'linear buckling': False,
                    'non-linear static': True,
                    'commons': fsdt_commons_bcn,
                    'linear': fsdt_donnell_bcn_linear,
                    'non-linear': fsdt_donnell_bcn_nonlinear,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 5,
                    'num2': 10,
                    },
    'fsdt_donnell_bc1': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'commons': fsdt_commons_bc1,
                    'linear': fsdt_donnell_bc1_linear,
                    'non-linear': fsdt_donnell_bc1_nonlinear,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 5,
                    'num2': 10,
                    },
    'fsdt_donnell_bc2': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': fsdt_commons_bc2,
                    'linear': fsdt_donnell_bc2_linear,
                    'non-linear': fsdt_donnell_bc2_nonlinear,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 5,
                    'num2': 10,
                    },
    'fsdt_donnell_bc3': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': fsdt_commons_bc3,
                    'linear': fsdt_donnell_bc3_linear,
                    'non-linear': fsdt_donnell_bc3_nonlinear,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 5,
                    'num2': 10,
                    },
    'fsdt_donnell_bc4': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'commons': fsdt_commons_bc4,
                    'linear': fsdt_donnell_bc4_linear,
                    'non-linear': fsdt_donnell_bc4_nonlinear,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 5,
                    'num2': 10,
                    },
    'fsdt_sanders_bcn': {
                    'linear static': True,
                    'linear buckling': False,
                    'non-linear static': False,
                    'commons': fsdt_commons_bcn,
                    'linear': fsdt_sanders_bcn_linear,
                    'non-linear': None,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 1,
                    'num0': 3,
                    'num1': 5,
                    'num2': 10,
                    },
    'fsdt_shadmehri2012_bc2': {
                    'linear static': None,
                    'linear buckling': True,
                    'non-linear static': None,
                    'commons': fsdt_shadmehri2012_bc2,
                    'linear': fsdt_shadmehri2012_bc2,
                    'non-linear': None,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 0,
                    'num0': 0,
                    'num1': 0,
                    'num2': 5,
                    },
    'fsdt_shadmehri2012_bc3': {
                    'linear static': None,
                    'linear buckling': True,
                    'non-linear static': None,
                    'commons': fsdt_shadmehri2012_bc3,
                    'linear': fsdt_shadmehri2012_bc3,
                    'non-linear': None,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 0,
                    'num0': 0,
                    'num1': 0,
                    'num2': 5,
                    },
    'fsdt_geier1997_bc2': {
                    'linear static': None,
                    'linear buckling': True,
                    'non-linear static': None,
                    'commons': fsdt_geier1997_bc2,
                    'linear': fsdt_geier1997_bc2,
                    'non-linear': None,
                    'dofs': 5,
                    'e_num': 8,
                    'i0': 0,
                    'j0': 0,
                    'num0': 0,
                    'num1': 0,
                    'num2': 5,
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
    L = cc.L
    m1 = cc.m1
    m2 = cc.m2
    n2 = cc.n2
    model = cc.model

    try:
        if 'iso_' in model:
            fk0edges = db[model[4:]]['linear'].fk0edges
        else:
            fk0edges = db[model]['linear'].fk0edges
    except AttributeError:
        k0edges = None

    if model == 'clpt_donnell_bc1':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'clpt_donnell_bc2':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kuBot, cc.kuTop,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'iso_clpt_donnell_bc2':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kuBot, cc.kuTop,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'clpt_donnell_bc3':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kvBot, cc.kvTop,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'iso_clpt_donnell_bc3':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kvBot, cc.kvTop,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'clpt_donnell_bc4':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kuBot, cc.kuTop,
                           cc.kvBot, cc.kvTop,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'clpt_donnell_bcn':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kuBot, cc.kuTop,
                           cc.kvBot, cc.kvTop,
                           cc.kwBot, cc.kwTop,
                           cc.kphixBot, cc.kphixTop,
                           cc.kphitBot, cc.kphitTop)

    elif model == 'clpt_sanders_bc1':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'clpt_sanders_bc2':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kuBot, cc.kuTop,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'clpt_sanders_bc3':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kvBot, cc.kvTop,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'clpt_sanders_bc4':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                           cc.kuBot, cc.kuTop,
                           cc.kvBot, cc.kvTop,
                           cc.kphixBot, cc.kphixTop)

    elif model == 'clpt_geier1997_bc2':
        k0edges = fk0edges(m1, m2, n2, r1, r2, L,
                    cc.kuBot, cc.kuTop,
                    cc.kphixBot, cc.kphixTop)

    elif model == 'fsdt_donnell_bc1':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kphixBot, cc.kphixTop)

    elif model == 'fsdt_donnell_bc2':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kuBot, cc.kuTop,
                    cc.kphixBot, cc.kphixTop)

    elif model == 'fsdt_donnell_bc3':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kvBot, cc.kvTop,
                    cc.kphixBot, cc.kphixTop)

    elif model == 'fsdt_donnell_bc4':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kuBot, cc.kuTop,
                    cc.kvBot, cc.kvTop,
                    cc.kphixBot, cc.kphixTop)

    elif model == 'fsdt_donnell_bcn':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kuBot, cc.kuTop,
                    cc.kvBot, cc.kvTop,
                    cc.kwBot, cc.kwTop,
                    cc.kphixBot, cc.kphixTop,
                    cc.kphitBot, cc.kphitTop)

    elif model == 'fsdt_sanders_bcn':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kuBot, cc.kuTop,
                    cc.kvBot, cc.kvTop,
                    cc.kwBot, cc.kwTop,
                    cc.kphixBot, cc.kphixTop,
                    cc.kphitBot, cc.kphitTop)

    elif model == 'fsdt_shadmehri2012_bc2':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kuBot, cc.kuTop,
                    cc.kphixBot, cc.kphixTop)

    elif model == 'fsdt_shadmehri2012_bc3':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kvBot, cc.kvTop,
                    cc.kphixBot, cc.kphixTop)

    elif model == 'fsdt_geier1997_bc2':
        k0edges = fk0edges(m1, m2, n2, r1, r2,
                    cc.kuBot, cc.kuTop,
                    cc.kphixBot, cc.kphixTop)

    fk0 = db[model]['linear'].fk0
    fk0_cyl = db[model]['linear'].fk0_cyl
    if 'iso_' in model:
        fkG0 = db[model[4:]]['linear'].fkG0
        fkG0_cyl = db[model[4:]]['linear'].fkG0_cyl
    else:
        fkG0 = db[model]['linear'].fkG0
        fkG0_cyl = db[model]['linear'].fkG0_cyl

    return fk0, fk0_cyl, fkG0, fkG0_cyl, k0edges


valid_models = sorted(db.keys())


def get_model(model_name):
    if not model_name in valid_models:
        raise ValueError('ERROR - valid models are:\n    ' +
                 '\n    '.join(valid_models))
    else:
        return db[model_name]
