.. _theory-conecyl-clpt:

=======================================
CLPT - Classical Laminated Plate Theory
=======================================

Description
===========

For the CLPT the displacement field components are:

.. math::

    u, v, w

and that the approximations can be separated as:

.. math::

    u = u_0 + u_1 + u_2\\
    v = v_0 + v_1 + v_2\\
    w = w_0 + w_1 + w_2\\

where :math:`u_0` contains the approximation functions corresponding to the
prescribed degrees of freedom, :math:`u_1` contains the functions independent
of :math:`\theta` and :math:`u_2` the functions that depend on both :math:`x`
and :math:`\theta`.

The aim is to have models capable of simulating the displacement field of
cones and cylinders. The approximation functions are the same
for both the Donnell's and the Sanders' models.
The approximation functions are:

.. _clpt_approx_functions:

.. math::

    u = u_0 + \sum_{i_1=1}^{m_1} {c_{i_1}}^{u} \sin{{b_x}_1}
            + \sum_{i_2=1}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \sin{{b_x}_2} \cos{j_2 \theta}
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

with:

.. math::

    {b_x}_1 = i_1 \pi \frac x L \\
    {b_x}_2 = i_2 \pi \frac x L 

Observations:

    :math:`\checkmark` linear static

    :math:`\checkmark` linear buckling

    :math:`\checkmark` non-linear analysis


Implementations
===============

Below it follows a more detailed description of each of the implementations:

- clpt_donnell_
- clpt_donnell2_
- clpt_sanders_
- clpt_sanders2_

 
Each model can be accessed using the ``linear_kinematics`` parameter of the
``ConeCyl`` object.

.. _clpt_donnell:


clpt_donnell
------------

Simply supported edges with the Donnell's equations. The :ref:`approximation
functions <clpt_approx_functions>` are showed above and no elastic 
restraints are imposed to the shell rotation at the edges.


.. _clpt_donnell2:

clpt_donnell2
-------------

Elastic restrained edges with the Donnell's equations. The :ref:`approximation
functions <clpt_approx_functions>` are showed above and the
elastic restraints are imposed
adding to the strain energy rotational springs at the bottom and top edges.

Basically the following terms should be added to the strain energy:

.. math::

    U_{springs} = \int_\theta r_1 \left(
                      K_Bot^u u(x=L, \theta)^2 
                    + K_Bot^v v(x=L, \theta)^2
                    + K_Bot^w w(x=L, \theta)^2
                    + K_Bot^{\phi_x} {\phi_x}(x=L, \theta)^2
                    + K_Bot^{\phi_\theta} {\phi_\theta}(x=L, \theta)^2
                  \right)
                  \\
                + \int_\theta r_2 \left(
                      K_Top^u u(x=0, \theta)^2 
                    + K_Top^v v(x=0, \theta)^2
                    + K_Top^w w(x=0, \theta)^2
                    + K_Top^{\phi_x} {\phi_x}(x=0, \theta)^2
                    + K_Top^{\phi_\theta} {\phi_\theta}(x=0, \theta)^2
                  \right)

Writting in matrix form the following operation is performed to the 
linear stiffness matrix :math:`[K_0]`:

.. math::

    [K_0]_{new} = [K_0] + [K_0]_{edges}

    [K_0]_{edges} = \int_{\theta} { \left(
                        r_1 [g_{new}]_{x=L}^T [K]_{Bot} [g_{new}]_{x=L}^.
                      + r_2 [g_{new}]_{x=0}^T [K]_{Top} [g_{new}]_{x=0}^.
                         \right) d\theta
                        }


with :

.. math::

    [K]_{Bot} = \begin{bmatrix}
          K_Bot^u &       0 &       0 &              0 &             0 \\
                0 & K_Bot^v &       0 &              0 &             0 \\
                0 &       0 & K_Bot^w &              0 &             0 \\
                0 &       0 &       0 & K_Bot^{\phi_x} &             0 \\
                0 &       0 &       0 &              0 &K_Bot^{\phi_\theta} 
                    \end{bmatrix}

and:

.. math::

    [K]_{Top} = \begin{bmatrix}
          K_Top^u &       0 &       0 &              0 &             0 \\
                0 & K_Top^v &       0 &              0 &             0 \\
                0 &       0 & K_Top^w &              0 &             0 \\
                0 &       0 &       0 & K_Top^{\phi_x} &             0 \\
                0 &       0 &       0 &              0 &K_Top^{\phi_\theta} 
                    \end{bmatrix}


The shape functions :math:`[g_{new}]` contains two extra rows that are built
from the relations:

.. math::

    \phi_x = - \frac{\partial w}{\partial x}
    \\
    \phi_t = \frac{\partial w}{\partial \theta}

and therefore:

.. math::

    [g^{\phi_x}] = - \frac {\partial [g^w]} {\partial x}
    \\
    [g^{\phi_\theta}] = \frac {\partial [g^w]} {\partial \theta}
    \\
    [g_{new}]^T = \left[ [g^u], [g^v], [g^w],
                          [g^{\phi_x}], [g^{\phi_\theta}] \right]


.. _clpt_sanders:

clpt_sanders
------------

Simply supported edges with the Sanders's equations. The :ref:`approximation
functions <clpt_approx_functions>` are showed above and no elastic
restraints are imposed.  Analogous to the clpt_donnell_.

.. _clpt_sanders2:

clpt_sanders2
-------------

Analogous to the clpt_donnell2_ using the Sanders non-linear equations.
