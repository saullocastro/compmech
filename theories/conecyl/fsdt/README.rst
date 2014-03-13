.. section-numbering::

FSDT - First-order Shear Deformation Theory
===========================================

Many approximation functions were tested along the development of the
cylindrical and conical equations using the FSDT. Recalling that for the 
FSDT the displacement field components are:

.. math::

    u, v, w, \phi_x, \phi_t

and that the approximations can be separated as:

.. math::

    u = u_0 + u_1 + u_2\\

    \vdots

    {{\phi}_\theta} = {{\phi}_\theta}_0 + {{\phi}_\theta}_1 + {{\phi}_\theta}_2

where :math:`u_0` contains the approximation functions corresponding to the
prescribed degrees of freedom, :math:`u_1` contains the functions independent
of :math:`\theta` and :math:`u_2` the functions that depend on both :math:`x`
and :math:`\theta`.

The aim is to have a model capable to simulate non-rigid supports, and where
the displacement components :math:`u, \phi_x` can habe a non-costant value
along the edges.

It is of special importance to allow :math:`\phi_x` to be between zero
(clamped) and another value (up to simply supported),
by using elastic stiffnesses for the corresponding degrees of freedom.

.. contents:: The attempts are listed below (same name as the ``linear_kinematics`` parameter of the `ConeCyl` object):
 
.. _fsdt_donnell: 

++++++++++++
fsdt_donnell
++++++++++++

The approximation functions are:

.. math::

    u = u_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}_a^{u} \sin{{b_x}_1}
                                +{c_{i_1}}_b^{u} \cos{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \sin{{b_x}_2} \cos{j_2 \theta}
                    +{c_{i_2 j_2}}_c^{u} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_d^{u} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\    
    v = v_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{v}\sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{v} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{v} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    w = w_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{w}\sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{w} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                \right)
    \\
    \phi_x = {\phi_x}_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{\phi_x}_a\sin{{b_x}_1} 
                                            +{c_{i_1}}^{\phi_x}_b\cos{{b_x}_1} 
        + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
             {c_{i_2 j_2}}_a^{\phi_x} \sin{{b_x}_2} \sin{j_2 \theta}
            +{c_{i_2 j_2}}_b^{\phi_x} \sin{{b_x}_2} \cos{j_2 \theta}
            +{c_{i_2 j_2}}_c^{\phi_x} \cos{{b_x}_2} \sin{j_2 \theta}
            +{c_{i_2 j_2}}_d^{\phi_x} \cos{{b_x}_2} \cos{j_2 \theta}
        \right)
    \\
    \phi_{\theta} = {\phi_{\theta}}_0 +
                \sum_{i_1=1}^{m_1} {c_{i_1}}^{\phi_{\theta}}_a\sin{{b_x}_1} 
                                  +{c_{i_1}}^{\phi_{\theta}}_b\cos{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
         {c_{i_2 j_2}}_a^{\phi_{\theta}} \sin{{b_x}_2} \sin{j_2 \theta}
        +{c_{i_2 j_2}}_b^{\phi_{\theta}} \sin{{b_x}_2} \cos{j_2 \theta}
        +{c_{i_2 j_2}}_c^{\phi_{\theta}} \cos{{b_x}_2} \sin{j_2 \theta}
        +{c_{i_2 j_2}}_d^{\phi_{\theta}} \cos{{b_x}_2} \cos{j_2 \theta}
            \right)

with:

.. math::

    {b_x}_1 = i_1 \pi \frac x L \\
    {b_x}_2 = i_2 \pi \frac x L 

Observations:

    :math:`\checkmark` linear static


    :math:`\checkmark` succcessfuly simulated the linear static loads

    :math:`\checkmark` the flexible boundary conditions were successfully
    simulated

    :math:`\times` rigid body translations in :math:`u`

    :math:`\times` did not work for linear buckling

    :math:`\times` did not work for non-linear analysis


.. _fsdt_donnell2:

fsdt_donnell2
-------------

With the aim to remove the rigid body translation found in fsdt_donnell_,
only the function for :math:`u`  has been changed:

.. math::


    u = u_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{u}\sin{{b_x}_1}
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \sin{{b_x}_2} \cos{j_2 \theta}
                    +{c_{i_2 j_2}}_c^{u} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_d^{u} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)


Observations:

    :math:`\checkmark` rigid body translation removed

    :math:`\times` did not work for linear buckling

    :math:`\times` did not work for non-linear analysis

.. _fsdt_donnell3:

fsdt_donnell3
-------------

Tried to remove some terms of fsdt_donnell2_ in order to make the linear
buckling analysis work. Only the :math:`cos` function of
:math:`{\phi_x}_1` has been removed:

.. math::


    \phi_x = {\phi_x}_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{\phi_x} \sin{{b_x}_1} 
        + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
             {c_{i_2 j_2}}_a^{\phi_x} \sin{{b_x}_2} \sin{j_2 \theta}
            +{c_{i_2 j_2}}_b^{\phi_x} \sin{{b_x}_2} \cos{j_2 \theta}
            +{c_{i_2 j_2}}_c^{\phi_x} \cos{{b_x}_2} \sin{j_2 \theta}
            +{c_{i_2 j_2}}_d^{\phi_x} \cos{{b_x}_2} \cos{j_2 \theta}
        \right)

Observations:

    :math:`\times` did not work for linear buckling

    :math:`\times` did not work for non-linear analysis

.. _fsdt_donnell4:

fsdt_donnell4
-------------

Tried to remove some terms of fsdt_donnell3_. The :math:`cos` functions
of :math:`u_2` was removed, the :math:`cos` function of :math:`{\phi_x}_1` was
put back:

.. math::

    u = u_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{u}\sin{{b_x}_1}
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    \phi_x = {\phi_x}_0 + \sum_{i_1=1}^{m_1}
                              {c_{i_1}}_a^{\phi_x} \sin{{b_x}_1} 
                             +{c_{i_1}}_b^{\phi_x} \sin{{b_x}_1} 
        + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
             {c_{i_2 j_2}}_a^{\phi_x} \sin{{b_x}_2} \sin{j_2 \theta}
            +{c_{i_2 j_2}}_b^{\phi_x} \sin{{b_x}_2} \cos{j_2 \theta}
            +{c_{i_2 j_2}}_c^{\phi_x} \cos{{b_x}_2} \sin{j_2 \theta}
            +{c_{i_2 j_2}}_d^{\phi_x} \cos{{b_x}_2} \cos{j_2 \theta}
        \right)

Observations:

    :math:`\times` removing the :math:`cos` for :math:`u_2` removed the
    capability to simulate non-rigid boundary conditions in :math:`u`

    :math:`\times` did not work for linear buckling

    :math:`\times` did not work for non-linear analysis

.. _fsdt_donnell5:

fsdt_donnell5
-------------

From the four previous attempts, the fsdt_donnell2_ and fsdt_donnell3_ 
give the same results and the fsdt_donnell3_ is preferred because it has less
degrees of freedom (the :math:`cos` of :math:`{\phi_x}_1`). By then it was
learned how the approximations for non-rigid boundary conditions should be and
the current attempt add more flexibility in :math:`v,w,\phi_\theta` using the
previous know-how. The resulting approximation functions are:

.. math::

    u = u_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{u} \sin{{b_x}_1}
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \sin{{b_x}_2} \cos{j_2 \theta}
                    +{c_{i_2 j_2}}_c^{u} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_d^{u} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\    
    v = v_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{v} \sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{v} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{v} \sin{{b_x}_2} \cos{j_2 \theta}
                    +{c_{i_2 j_2}}_c^{v} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_d^{v} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    w = w_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{w} \sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{w} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                    +{c_{i_2 j_2}}_c^{w} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_d^{w} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    \phi_x = {\phi_x}_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{\phi_x} \sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                 {c_{i_2 j_2}}_a^{\phi_x} \sin{{b_x}_2} \sin{j_2 \theta}
                +{c_{i_2 j_2}}_b^{\phi_x} \sin{{b_x}_2} \cos{j_2 \theta}
                +{c_{i_2 j_2}}_c^{\phi_x} \cos{{b_x}_2} \sin{j_2 \theta}
                +{c_{i_2 j_2}}_d^{\phi_x} \cos{{b_x}_2} \cos{j_2 \theta}
              \right)
    \\
    {\phi}_\theta = {\phi_x}_0 + \sum_{i_1=1}^{m_1}
                                 {c_{i_1}}^{{\phi}_\theta} \sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
         {c_{i_2 j_2}}_a^{{\phi}_\theta} \sin{{b_x}_2} \sin{j_2 \theta}
        +{c_{i_2 j_2}}_b^{{\phi}_\theta} \sin{{b_x}_2} \cos{j_2 \theta}
        +{c_{i_2 j_2}}_c^{{\phi}_\theta} \cos{{b_x}_2} \sin{j_2 \theta}
        +{c_{i_2 j_2}}_d^{{\phi}_\theta} \cos{{b_x}_2} \cos{j_2 \theta}
              \right)

.. _fsdt_donnell6:

fsdt_donnell6
-------------

With an attempt to make the linear buckling analysis to work for FSDT, this
uses very simple approximation functions, design to produce results for
a simply supported cone or cylinder, analogous to the CLPT implementation:

.. math::

    u = u_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{u} \sin{{b_x}_1}
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\    
    v = v_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{v} \sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{v} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{v} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    w = w_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{w} \sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{w} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    \phi_x = {\phi_x}_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{\phi_x} \cos{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                 {c_{i_2 j_2}}_a^{\phi_x} \cos{{b_x}_2} \sin{j_2 \theta}
                +{c_{i_2 j_2}}_b^{\phi_x} \cos{{b_x}_2} \cos{j_2 \theta}
              \right)
    \\
    {\phi}_\theta = {\phi_x}_0 + \sum_{i_1=1}^{m_1}
                                 {c_{i_1}}^{{\phi}_\theta} \sin{{b_x}_1} 
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
         {c_{i_2 j_2}}_a^{{\phi}_\theta} \sin{{b_x}_2} \sin{j_2 \theta}
        +{c_{i_2 j_2}}_b^{{\phi}_\theta} \sin{{b_x}_2} \cos{j_2 \theta}
              \right)
