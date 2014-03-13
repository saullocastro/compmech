=======================================
CLPT - Classical Laminated Plate Theory
=======================================

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
simply supported cones and cylinders. The approximation functions are the same
for both the Donnell's and the Sanders' models. The models are accessed
using for the ``linear_kinematics`` parameter ``"clpt_donnell"`` or
``"clpt_sanders"``.


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


