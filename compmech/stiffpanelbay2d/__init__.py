r"""
==============================================================================
Stiffened Panel Bay (:mod:`compmech.stiffpanelbay2d`)
==============================================================================

.. currentmodule:: compmech.stiffpanelbay2d

A panel bay may contain many panels with different properties. The panels are
separated by their `y` (circumferential) coordinate. Usually there is a
stiffener positioned at the `y` coordinate between two panels. If only one
panel property is desired for the entire bay, module :mod:`compmech.stiffpanel`
is recommended.

.. autoclass:: StiffPanelBay2D
    :members:

.. autoclass:: Panel
    :members:

.. autoclass:: Stiffener2D
    :members:

"""
from stiffpanelbay2d import (load, StiffPanelBay2D, Stiffener2D, Panel)
