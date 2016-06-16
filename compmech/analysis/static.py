from compmech.sparse import solve
from compmech.logger import msg

def static(K, fext, silent=False):
    """Static Analyses

    Parameters
    ----------

    K : sparse_matrix
        Stiffness matrix. Should include initial stress stiffness matrix,
        aerodynamic matrix and so forth when applicable.
    fext : array-like
        Vector of external loads.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.

    """
    increments = []
    cs = []

    NLgeom=False
    if NLgeom:
        raise NotImplementedError('Independent static function not ready for NLgeom')
    else:
        msg('Started Linear Static Analysis', silent=silent)
        c = solve(K, fext, silent=silent)
        increments.append(1.)
        cs.append(c)
        msg('Finished Linear Static Analysis', silent=silent)

    return increments, cs
