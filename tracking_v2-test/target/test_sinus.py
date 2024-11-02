import numpy as np
from tracking_v2.target.turns import SinusTarget


def test_sinus():
    states = SinusTarget(30, 2).true_states()
    x = states[:, 0]

    # period of 360: U-turn takes 180/2 seconds, times 4
    assert np.abs(x[0] - x[360]) < 0.001

    # U-turn inflection points
    assert np.argmin(x) == 90
    assert np.argmax(x) == 270
