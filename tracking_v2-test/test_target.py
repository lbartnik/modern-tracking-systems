import numpy as np

from tracking_v2.target import ConstantVelocityTarget


def test_constant_velocity():
    t = ConstantVelocityTarget()
    
    s = t.true_states(1, 3)
    assert np.array_equal(s, [[0, 0, 0, 30, 0, 0],
                              [30, 0, 0, 30, 0, 0],
                              [60, 0, 0, 30, 0, 0]])
    
    s = t.true_states([0, 1, 2])
    assert np.array_equal(s, [[0, 0, 0, 30, 0, 0],
                              [30, 0, 0, 30, 0, 0],
                              [60, 0, 0, 30, 0, 0]])
    
    s = t.true_states([1, 3, 6])
    assert np.array_equal(s, [[30, 0, 0, 30, 0, 0],
                              [90, 0, 0, 30, 0, 0],
                              [180, 0, 0, 30, 0, 0]])
    