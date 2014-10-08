import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from compmech.logger import *


def remove_null_cols(*args):
    """Remove null rows and cols of a symmetric, square sparse matrix.

    Parameters
    ----------
    args : list of sparse matrices
        The first matrix in this list will be used to extract the columns
        to be removed from all the other matrices.

    Returns
    -------
    out : list of sparse matrices
        A list with the reduced matrices in the same order of ``args`` plus
        an array containing the removed columns at the last position.

    """
    args = list(args)
    log('Removing null columns...', level=3)
    num_cols = args[0].shape[1]
    m = csr_matrix(args[0])
    used_cols = np.unique(m.indices)

    for i, arg in enumerate(args):
        m = csc_matrix(arg)[:, used_cols]
        m = csr_matrix(m)[used_cols, :]
        args[i] = m
    args.append(used_cols)
    log('{} columns removed'.format(num_cols - used_cols.shape[0]), level=4)
    log('finished!', level=3)

    return args


def solve(a, b, **kwargs):
    a, used_cols = remove_null_cols(a)
    px = spsolve(a, b[used_cols], **kwargs)
    x = np.zeros(b.shape[0], dtype=b.dtype)
    x[used_cols] = px

    return x

