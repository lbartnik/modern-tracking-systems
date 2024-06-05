from tracking.evaluation import Track, _mask_to_ranges, _calculate_highlight_regions
from numpy.testing import assert_allclose
import numpy as np


def test_interpolate():
    t = Track([1, 4],
              [[1, 1, 1], [4, 4, 4]],
              [[[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[4, 0, 0], [0, 8, 0], [0, 0, 12]]])
    u = t.interpolate([2])
    assert_allclose(u.time, [2])
    assert_allclose(u.positions, [[2, 2, 2]])
    assert_allclose(u.position_covariance, [[[2, 0, 0], [0, 4, 0], [0, 0, 6]]])


def test_interpolate_random():
    t = Track(np.arange(20),
              np.random.normal(size=(20, 3)),
              np.random.normal(size=(20, 3, 3)))
    u = t.interpolate(np.arange(20))
    assert_allclose(u.time, t.time)
    assert_allclose(u.positions, t.positions)
    assert_allclose(u.position_covariance, t.position_covariance)


def test_mask_to_ranges():
    assert_allclose(_mask_to_ranges([0, 0, 1, 1, 0, 1]), [(2, 4), (5, 5)])
    assert_allclose(_mask_to_ranges([1, 1, 0, 1, 0, 1]), [(0, 2), (3, 4), (5, 5)])


def test_calculate_highlight_regions():
    boxes = _calculate_highlight_regions([0, 1, 2, 3, 4], [0, 0, 3, 2, 4], [1, 1, 0, 0, 6])
    assert boxes == [(0, 0, 1.25, 6), (3.5, 0, 4, 6)]