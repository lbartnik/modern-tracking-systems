from tracking.kalman.generic import RbeSpace
from numpy.testing import assert_allclose
import numpy as np


def test_from_cartesian():
    s = RbeSpace()
    r = s.from_cartesian([[np.sqrt(3), 1, 2],
                          [3, 3, 3*np.sqrt(6)]])
    assert_allclose(r, [[np.sqrt(8), np.pi/6, np.pi/4],
                        [6*np.sqrt(2), np.pi/4, np.pi/3]])


def test_to_cartesian():
    s = RbeSpace()
    r = s.to_cartesian([[np.sqrt(8), np.pi/6, np.pi/4],
                        [6*np.sqrt(2), np.pi/4, np.pi/3]])
    assert_allclose(r, [[np.sqrt(3), 1, 2],
                        [3, 3, 3*np.sqrt(6)]])


def test_jacobian():
    s = RbeSpace()
    j = s.jacobian([1, 2, 3])
    assert_allclose(j, [[ 0.267261,  0.534522,  0.801784],
                        [-0.4     ,  0.2     ,  0.      ],
                        [-0.095831, -0.191663,  0.159719]],
                        rtol=1e-4)
