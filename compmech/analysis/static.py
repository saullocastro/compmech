from compmech.sparse import solve
from compmech.logger import msg

def static(k0, fext, silent=False):
    """General solver for static analyses

    Parameters
    ----------

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
        c = solve(k0, fext, silent=silent)
        increments.append(1.)
        cs.append(c)
        msg('Finished Linear Static Analysis', silent=silent)

    return increments, cs
