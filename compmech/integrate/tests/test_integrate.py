from numpy import isclose, pi, sin

from compmech.integrate.integrate import trapz2d_points, simps2d_points


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


if __name__ == '__main__':
    test_integrate_sinsin()
