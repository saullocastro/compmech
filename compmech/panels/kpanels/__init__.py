r"""
=======================================================
Conical (Konus) Panels (:mod:`compmech.panels.kpanels`)
=======================================================

.. currentmodule:: compmech.panels.kpanels

Can be used to simulate cylindrical panels when `\alpha = 0`. The main
attributes are listed below:

.. figure:: ../../../figures/kpanel.png
   :align: center

=====================  ==================================================
Geometric Attributes   Description
=====================  ==================================================
``r1``                 Radius at the bottom edge
``r2``                 Radius at the top edge
``L``                  Length along the `x` axis
``tmindeg``            Circumferential start angle
``tmaxdeg``            Circumferential end angle
``alphadeg``           Cone semi-vertex angle in degrees
=====================  ==================================================

.. autoclass:: KPanelT
    :members:

.. autoclass:: KPanelCheb
    :members:

"""
from kpanelt import KPanelT
from kpanelcheb import KPanelCheb
