import pyximport; pyximport.install()
import numpy as np

from compmech.integrate.integratev import _test_integratev


def test_integratev_trapz2d(nx=30, ny=30, method='trapz2d'):
    out = _test_integratev(nx, ny, method)
    assert np.allclose(out, (4/np.pi**2, 4/(9*np.pi**2), 4/(25*np.pi**2)),
            atol=1.e-3)

def test_integratev_simps2d(nx=20, ny=20, method='simps2d'):
    out = _test_integratev(nx, ny, method)
    assert np.allclose(out, (4/np.pi**2, 4/(9*np.pi**2), 4/(25*np.pi**2)),
            atol=1.e-4)

if __name__ == '__main__':
    test_integratev_trapz2d()
    test_integratev_simps2d()
