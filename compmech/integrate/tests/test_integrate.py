from numpy import isclose, pi, sin
import numpy as np

from compmech.integrate.integrate import (trapz2d_points, simps2d_points,
                                          python_trapz_quad)


def test_integrate_sinsin():
    ans = 4/pi**2
    nx = ny = 40
    xs, ys, alphas, betas = trapz2d_points(0, 1., nx, 0, 1., ny)
    out = 0.
    for x, y, alpha, beta in zip(xs, ys, alphas, betas):
        out = beta*(out) + alpha*(sin(pi*x)*sin(pi*y))
    assert isclose(out, ans, atol=0.001, rtol=0)

    xs, ys, alphas, betas = simps2d_points(0, 1., nx, 0, 1., ny)
    out = 0.
    for x, y, alpha, beta in zip(xs, ys, alphas, betas):
        out = beta*(out) + alpha*(sin(pi*x)*sin(pi*y))
    assert isclose(out, ans, atol=0.00001, rtol=0)


def test_integrate_sinsin_using_trapz_quad():
    ans = 4/pi**2
    nx = ny = 40
    xis = np.zeros(nx, dtype=np.float64)
    weightsxi = np.zeros(nx, dtype=np.float64)
    etas = np.zeros(ny, dtype=np.float64)
    weightseta = np.zeros(ny, dtype=np.float64)

    python_trapz_quad(nx, xis, weightsxi)
    python_trapz_quad(ny, etas, weightseta)

    out = 0.

    for xi, weightxi in zip(xis, weightsxi):
        for eta, weighteta in zip(etas, weightseta):
            out += weightxi*weighteta*(sin(pi*(xi+1)/2.)*sin(pi*(eta+1)/2.))
    out *= 1/2.*1/2.

    assert isclose(out, ans, atol=0.001, rtol=0)


if __name__ == '__main__':
    test_integrate_sinsin()
    test_integrate_sinsin_using_trapz_quad()
