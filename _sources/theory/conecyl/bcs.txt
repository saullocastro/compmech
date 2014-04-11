.. _boundary_conditions:

Boundary conditions
===================

The classification of Som and Deb (2014) [som2014]_ has been used for the
boundary conditions.

The simply-supported boundary conditions are:

.. math::

    \begin{tabular}{l c r}
        Name & Displ. Vector & Elastic Constants \\
        \hline
        SS1  & $u=v=w=0$        & $K^u=K^v=K^w=\infty$ \\
             &                  & $K^{\phi_x}=K^{\phi_\theta}=0$ \\
        \hline
        SS2  & $v=w=0$          & $K^v=K^w=\infty$ \\
             &                  & $K^u=K^{\phi_x}=K^{\phi_\theta}=0$ \\
        \hline
        SS3  & $u=w=0$          & $K^u=K^w=\infty$ \\
             &                  & $K^v=K^{\phi_x}=K^{\phi_\theta}=0$ \\
        \hline
        SS4  & $w=0$            & $K^w=\infty$ \\
             &                  & $K^u=K^v=K^{\phi_x}=K^{\phi_\theta}=0$
    \end{tabular}

and the clamped are:

.. math::
    \begin{tabular}{l c r}
        Name & Displ. Vector & Elastic Constants \\
        \hline
        CC1  & $u=v=w=w_{,x}=0$ & $K^u=K^v=K^w=K^{\phi_x}=\infty$ \\
             &                  & $K^{\phi_\theta}=0$ \\
        \hline
        CC2  & $v=w=w_{,x}=0$   & $K^v=K^w=K^{\phi_x}=\infty$ \\
             &                  & $K^u=K^{\phi_\theta}=0$ \\
        \hline
        CC3  & $u=w=w_{,x}=0$   & $K^u=K^w=K^{\phi_x}=\infty$ \\
             &                  & $K^v=K^{\phi_\theta}=0$ \\
        \hline
        CC4  & $w=w_{,x}=0$     & $K^w=K^{\phi_x}=\infty$ \\
             &                  & $K^u=K^v=K^{\phi_\theta}=0$ 
    \end{tabular}

Using the default boundary conditions
-------------------------------------

The analyst may set the boundary conditions specifying the parameter ``bc``
in the ``ConeCyl`` object, using the same names specified in the
tables above::

    >>> cc = ConeCyl()
    >>> cc.model = 'fsdt_donnell_bc1'
    >>> cc.bc = 'ss1'

Setting a different boundary condition for the bottom and top edges is
possible using a hyphen ``-`` or an underscore ``_`` to separate them,
obtaining ``'bcBot-bcTop'`` or ``bcBot_bcTop``::

    >>> cc.bc = 'ss1-cc1'
    >>> cc.bc = 'ss1-ss2'

.. note:: When using boundary conditions from different types in the same
          model the analyst must select the most flexible model to use, 
          for example when using ``cc.bc = 'ss1-ss2'``, the analyst must
          use ``cc.model = 'fsdt_donnell_bc2'``, a similar one or a more
          flexible model.

The ``model`` selected should be compatible with the boundary conditions that
one whishes to simulate. For example, for the SS4 or CC4 boundary
conditions, it is recommended to use the :ref:`clpt_donnell_bc4` or the
:ref:`fsdt_donnell_bc4` models, and so forth.

The more flexible models can be used to simulate the more rigid boundary 
conditions, since :ref:`elastic constraints <elastic_constraints>` are
ajusted in order to provide the right set of boundary conditions, as shown
in the table above. The table below shows the models that can be used for each
boundary condition:

.. math::
    \begin{tabular}{l c r}
        Name       & Model         \\ \hline
        SS1 / CC1  & bc2, bc3, bc4 \\ \hline
        SS2 / CC2  & bc2, bc4      \\ \hline
        SS3 / CC3  & bc3, bc4      \\ \hline
        SS4 / CC4  & bc4
    \end{tabular}

Note that the models ``bc4`` can be used for all the cases listed above. It
is expected that **more terms are required** in the approximation when 
a model from another group is used.


Using arbitrary boundary conditions
-----------------------------------

When no value is given to the parameter ``bc`` the model will run by
default with the SS1 boundary conditions. The analyst must change
the elastic stiffnesses by changing the following parameters::

    >>> cc = ConeCyl()
    >>> cc.kuBot
    >>> cc.kuTop
    >>> cc.kuBot
    >>> cc.kvTop
    >>> cc.kvBot
    >>> cc.kphixTop
    >>> cc.kphixTop

In order to achieve the desired results.

The other stiffnesses ``kwBot``, ``kwTop``, ``kphitBot`` and ``kphitTop``
will affect only models :ref:`clpt_donnell_bcn` and
:ref:`fsdt_donnell_bcn` (or the counterparts that use the Sander's equations).


